import os
from pathlib import Path

import datetime
from astropy.io import fits
from astropy.cosmology import FlatLambdaCDM
import multiprocessing
import numpy as np
import pandas as pd

from spaxelsleuth.config import settings
from spaxelsleuth.utils.continuum import compute_d4000, compute_continuum_intensity
from spaxelsleuth.utils.dqcut import compute_measured_HALPHA_amplitude_to_noise
from spaxelsleuth.utils.addcolumns import add_columns
from spaxelsleuth.utils.linefns import bpt_num_to_str

import logging
logger = logging.getLogger(__name__)

###############################################################################
# Paths
input_path = Path(settings["lzifu"]["input_path"])
output_path = Path(settings["lzifu"]["output_path"])
data_cube_path = Path(settings["lzifu"]["data_cube_path"])


###############################################################################
def add_metadata(df, df_metadata):
    """Merge an input DataFrame with that was created using make_lzifu_df()."""
    if "ID" not in df_metadata:
        raise ValueError("df_metadata must contain an 'ID' column!")
    df = df.merge(df_metadata, on="ID", how="left")
    return df

###############################################################################
def _process_lzifu(args):

    #######################################################################
    # Parse arguments
    #######################################################################
    _, gal, ncomponents, data_cube_path = args
    lzifu_ncomponents = ncomponents if type(ncomponents) == int else 3

    #######################################################################
    # Scrape outputs from LZIFU output
    #######################################################################
    hdulist_lzifu = fits.open(input_path / f"{gal}_{ncomponents}_comp.fits")
    hdr = hdulist_lzifu[0].header

    # NOTE: for some reason, the SET extension is missing from the merge_comp FITS file so we need to get it from one of the others :/
    if ncomponents == "merge":
        hdulist_lzifu_1comp = fits.open(input_path / f"{gal}_1_comp.fits")
        t = hdulist_lzifu_1comp["SET"].data  # Table storing fit parameters
    else:
        t = hdulist_lzifu["SET"].data  # Table storing fit parameters

    # Path to data cubes used in the fit
    if data_cube_path is None:
        data_cube_path = t["DATA_PATH"][0]

    # Was the fit one-sided or two-sided (i.e., red and blue cubes?)
    # In either case, use SET to find the original datacubes used in the fit.
    onesided = True if t["ONLY_1SIDE"][0] == 1 else False
    if onesided:
        datacube_fname = data_cube_path / f"{gal}.fits"
        datacube_fname_alt = data_cube_path / f"{gal}.fits.gz"
        if not os.path.exists(datacube_fname):
            if os.path.exists(datacube_fname_alt):
                datacube_fname = datacube_fname_alt
            else:
                raise FileNotFoundError(
                    "Cannot find files {datacube_fname} or {datacube_fname_alt}!"
                )
    else:
        datacube_fnames = []
        for side in ["B", "R"]:
            datacube_fname = data_cube_path / f"{gal}_{side}.fits"
            datacube_fname_alt = data_cube_path / f"{gal}_{side}.fits.gz"
            if not os.path.exists(datacube_fname):
                if os.path.exists(datacube_fname_alt):
                    datacube_fname = datacube_fname_alt
                else:
                    raise FileNotFoundError(
                        "Cannot find files {datacube_fname} or {datacube_fname_alt}!"
                    )
            datacube_fnames.append(datacube_fname)

        datacube_B_fname, datacube_R_fname = datacube_fnames

    # Redshift
    z = t["Z"][0]

    # Calculate cosmological distances from the redshift
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    D_A_Mpc = cosmo.angular_diameter_distance(z).value
    D_L_Mpc = cosmo.luminosity_distance(z).value
    kpc_per_arcsec = D_A_Mpc * 1e3 * np.pi / 180.0 / 3600.0

    #######################################################################
    # LOAD THE DATACUBES
    #######################################################################
    if onesided:
        with fits.open(datacube_fname) as hdulist_cube:
            header = hdulist_cube[0].header
            data_cube = hdulist_cube[0].data
            var_cube = hdulist_cube[1].data

            # Plate scale
            if header_B["CUNIT1"].lower().startswith("deg"):
                as_per_px = np.abs(header_B["CDELT1"]) * 3600.
            elif header_B["CUNIT1"].lower().startswith("arc"):
                as_per_px = np.abs(header_B["CDELT1"])

            # Wavelength values
            lambda_0_A = header["CRVAL3"] - header["CRPIX3"] * header["CDELT3"]
            dlambda_A = header["CDELT3"]
            N_lambda = header["NAXIS3"]
            lambda_vals_A = np.array(range(N_lambda)) * dlambda_A + lambda_0_A

            data_cube_B = data_cube.copy()
            var_cube_B = var_cube.copy()
            data_cube_R = data_cube.copy()
            var_cube_R = var_cube.copy()
            lambda_vals_B_A = lambda_vals_A.copy()
            lambda_vals_R_A = lambda_vals_A.copy()
    else:
        # Blue cube
        with fits.open(datacube_B_fname) as hdulist_B_cube:
            header_B = hdulist_B_cube[0].header
            data_cube_B = hdulist_B_cube[0].data
            var_cube_B = hdulist_B_cube[1].data

            # Plate scale
            if header_B["CUNIT1"].lower().startswith("deg"):
                as_per_px = np.abs(header_B["CDELT1"]) * 3600.
            elif header_B["CUNIT1"].lower().startswith("arc"):
                as_per_px = np.abs(header_B["CDELT1"])

            # Wavelength values
            lambda_0_A = header_B[
                "CRVAL3"] - header_B["CRPIX3"] * header_B["CDELT3"]
            dlambda_A = header_B["CDELT3"]
            N_lambda = header_B["NAXIS3"]
            lambda_vals_B_A = np.array(
                range(N_lambda)) * dlambda_A + lambda_0_A

        # Red cube
        with fits.open(datacube_R_fname) as hdulist_R_cube:
            header_R = hdulist_R_cube[0].header
            data_cube_R = hdulist_R_cube[0].data
            var_cube_R = hdulist_R_cube[1].data

            # Wavelength values
            lambda_0_A = header_R[
                "CRVAL3"] - header_R["CRPIX3"] * header_R["CDELT3"]
            dlambda_A = header_R["CDELT3"]
            N_lambda = header_R["NAXIS3"]
            lambda_vals_R_A = np.array(
                range(N_lambda)) * dlambda_A + lambda_0_A

    # Rest-frame wavelength arrays
    lambda_vals_R_rest_A = lambda_vals_R_A / (1 + z)
    lambda_vals_B_rest_A = lambda_vals_B_A / (1 + z)

    # Create masks of empty pixels
    im_empty_B = np.all(np.isnan(data_cube_B), axis=0)
    im_empty_R = np.all(np.isnan(data_cube_R), axis=0)
    im_empty = np.logical_and(im_empty_B, im_empty_R)

    # Get coordinate lists corresponding to non-empty spaxels
    ny, nx = im_empty.shape
    y_c_list, x_c_list = np.where(~im_empty)

    #/////////////////////////////////////////////////////////////////////////
    def _2d_map_to_1d_list(colmap):
        """Returns a 1D array of values extracted from from spaxels in x_c_list and y_c_list in 2D array colmap."""
        if colmap.ndim != 2:
            raise ValueError(
                f"colmap must be a 2D array but has ndim = {colmap.ndim}!")
        row = np.full_like(x_c_list, np.nan, dtype="float")
        for jj, coords in enumerate(zip(x_c_list, y_c_list)):
            x_c, y_c = coords
            y, x = (int(np.round(y_c)), int(np.round(x_c)))
            if x > nx or y > ny:
                x = min([x, nx])
                y = min([y, ny])
            row[jj] = colmap[y, x]
        return row

    #######################################################################
    # SCRAPE LZIFU MEASUREMENTS
    #######################################################################
    # Get extension names
    extnames = [
        hdr[e] for e in hdr if e.startswith("EXT") and type(hdr[e]) == str
    ]  #TODO check whether the [SII]6716,31 lines are here
    quantities = [e.rstrip("_ERR")
                  for e in extnames if e.endswith("_ERR")] + ["CHI2", "DOF"]
    # NOTE: for some stupid reason, the SII6731_ERR extension is MISSING from the FITS file when ncomponents = "merge", so we need to add this in manually.
    if "SII6731" in extnames and "SII6731_ERR" not in extnames:
        quantities += ["SII6731"]
    rows_list = []
    colnames = []
    eline_list = [
        q for q in quantities if q not in ["V", "VDISP", "CHI2", "DOF"]
    ]
    lzifu_ncomponents = hdulist_lzifu["V"].shape[0] - 1

    # Scrape the FITS file: emission line flues, velocity/velocity dispersion, fit quality
    for quantity in quantities:
        data = hdulist_lzifu[quantity].data
        if f"{quantity}_ERR" in hdulist_lzifu:
            err = hdulist_lzifu[f"{quantity}_ERR"].data
        elif quantity == "SII6731" and "SII6731_ERR" not in extnames:
            # For now, let's just copy the SII6716 error.
            err = hdulist_lzifu[f"SII6716_ERR"].data
        # If 'data' has 3 dimensions, then it has been measured for all quantities
        if data.ndim == 3:
            # Total fluxes (only for emission lines)
            if quantity not in ["V", "VDISP"]:
                rows_list.append(_2d_map_to_1d_list(data[0]))
                colnames.append(f"{quantity} (total)")
                if f"{quantity}_ERR" in hdulist_lzifu:
                    rows_list.append(_2d_map_to_1d_list(err[0]))
                    colnames.append(f"{quantity} error (total)")
            # Fluxes/values for individual components
            for nn in range(lzifu_ncomponents):
                rows_list.append(_2d_map_to_1d_list(data[nn + 1]))
                colnames.append(f"{quantity} (component {nn + 1})")
                if f"{quantity}_ERR" in hdulist_lzifu:
                    rows_list.append(_2d_map_to_1d_list(err[nn + 1]))
                    colnames.append(f"{quantity} error (component {nn + 1})")
        # Otherwise it's a 2D map
        elif data.ndim == 2:
            rows_list.append(_2d_map_to_1d_list(data))
            colnames.append(f"{quantity}")
            if f"{quantity}_ERR" in hdulist_lzifu:
                rows_list.append(_2d_map_to_1d_list(err))
                colnames.append(f"{quantity} error")

    ##########################################################
    # COMPUTE QUANTITIES DIRECTLY FROM THE DATACUBES
    ##########################################################
    # NOTE: because we do not have access to stellar velocities, we assume no peculiar velocities within the object when calculating continuum quantities
    v_star_map = np.zeros(data_cube_B.shape[1:])
    # Compute the D4000Å break
    if lambda_vals_B_rest_A[0] <= 3850 and lambda_vals_B_rest_A[-1] >= 4100:
        d4000_map, d4000_map_err = compute_d4000(
            data_cube=data_cube_B,
            var_cube=var_cube_B,
            lambda_vals_rest_A=lambda_vals_B_rest_A,
            v_star_map=v_star_map)
        rows_list.append(_2d_map_to_1d_list(d4000_map))
        colnames.append(f"D4000")
        rows_list.append(_2d_map_to_1d_list(d4000_map_err))
        colnames.append(f"D4000 error")

    # Compute the continuum intensity so that we can compute the Halpha equivalent width.
    # Continuum wavelength range taken from here: https://ui.adsabs.harvard.edu/abs/2019MNRAS.485.4024V/abstract
    if lambda_vals_R_rest_A[0] <= 6500 and lambda_vals_R_rest_A[-1] >= 6540:
        cont_HALPHA_map, cont_HALPHA_map_std, cont_HALPHA_map_err = compute_continuum_intensity(
            data_cube=data_cube_R,
            var_cube=var_cube_R,
            lambda_vals_rest_A=lambda_vals_R_rest_A,
            start_A=6500,
            stop_A=6540,
            v_map=v_star_map)
        rows_list.append(_2d_map_to_1d_list(cont_HALPHA_map))
        colnames.append(f"HALPHA continuum")
        rows_list.append(_2d_map_to_1d_list(cont_HALPHA_map_std))
        colnames.append(f"HALPHA continuum std. dev.")
        rows_list.append(_2d_map_to_1d_list(cont_HALPHA_map_err))
        colnames.append(f"HALPHA continuum error")

    # Compute the approximate B-band continuum
    if lambda_vals_B_rest_A[0] <= 4000 and lambda_vals_B_rest_A[-1] >= 5000:
        cont_B_map, cont_B_map_std, cont_B_map_err = compute_continuum_intensity(
            data_cube=data_cube_B,
            var_cube=var_cube_B,
            lambda_vals_rest_A=lambda_vals_B_rest_A,
            start_A=4000,
            stop_A=5000,
            v_map=v_star_map)
        rows_list.append(_2d_map_to_1d_list(cont_B_map))
        colnames.append(f"B-band continuum")
        rows_list.append(_2d_map_to_1d_list(cont_B_map_std))
        colnames.append(f"B-band continuum std. dev.")
        rows_list.append(_2d_map_to_1d_list(cont_B_map_err))
        colnames.append(f"B-band continuum error")

    # Compute the HALPHA amplitude-to-noise
    if lambda_vals_R_rest_A[0] <= 6562.8 and lambda_vals_R_rest_A[-1] >= 6562.8:
        v_map = hdulist_lzifu["V"].data  # Get velocity field from LZIFU fit
        AN_HALPHA_map = compute_measured_HALPHA_amplitude_to_noise(
            data_cube=data_cube_R,
            var_cube=var_cube_R,
            lambda_vals_rest_A=lambda_vals_R_rest_A,
            v_star_map=v_star_map,
            v_map=v_map[0],
            dv=300)
        rows_list.append(_2d_map_to_1d_list(AN_HALPHA_map))
        colnames.append(f"HALPHA A/N (measured)")

    ##########################################################
    # Add other stuff
    rows_list.append([gal] * len(x_c_list))
    colnames.append("ID")
    rows_list.append([D_A_Mpc] * len(x_c_list))
    colnames.append("D_A (Mpc)")
    rows_list.append([D_L_Mpc] * len(x_c_list))
    colnames.append("D_L (Mpc)")
    rows_list.append([as_per_px] * len(x_c_list))
    colnames.append("as_per_px")
    rows_list.append([kpc_per_arcsec] * len(x_c_list))
    colnames.append("kpc per arcsec")
    rows_list.append([as_per_px**2] * len(x_c_list))
    colnames.append("Bin size (square arcsec)")
    rows_list.append([(as_per_px * kpc_per_arcsec)**2] * len(x_c_list))
    colnames.append("Bin size (square kpc)")
    rows_list.append([nx] * len(x_c_list))
    colnames.append("N_x")
    rows_list.append([ny] * len(x_c_list))
    colnames.append("N_y")
    rows_list.append(np.array(x_c_list).flatten())
    colnames.append("x (pixels)")
    rows_list.append(np.array(y_c_list).flatten())
    colnames.append("y (pixels)")
    rows_list.append(np.array(x_c_list).flatten() * as_per_px)
    colnames.append("x (projected, arcsec)")
    rows_list.append(np.array(y_c_list).flatten() * as_per_px)
    colnames.append("y (projected, arcsec)")

    # Rename columns
    oldcol_v = [f"V (component {nn + 1})" for nn in range(lzifu_ncomponents)]
    oldcol_v_err = [
        f"V error (component {nn + 1})" for nn in range(lzifu_ncomponents)
    ]
    oldcol_sigma = [
        f"VDISP (component {nn + 1})" for nn in range(lzifu_ncomponents)
    ]
    oldcol_sigma_err = [
        f"VDISP error (component {nn + 1})" for nn in range(lzifu_ncomponents)
    ]
    newcol_v = [
        f"v_gas (component {nn + 1})" for nn in range(lzifu_ncomponents)
    ]
    newcol_v_err = [
        f"v_gas error (component {nn + 1})" for nn in range(lzifu_ncomponents)
    ]
    newcol_sigma = [
        f"sigma_gas (component {nn + 1})" for nn in range(lzifu_ncomponents)
    ]
    newcol_sigma_err = [
        f"sigma_gas error (component {nn + 1})"
        for nn in range(lzifu_ncomponents)
    ]
    v_dict = dict(zip(oldcol_v, newcol_v))
    v_err_dict = dict(zip(oldcol_v_err, newcol_v_err))
    sigma_dict = dict(zip(oldcol_sigma, newcol_sigma))
    sigma_err_dict = dict(zip(oldcol_sigma_err, newcol_sigma_err))
    rename_dict = {**v_dict, **v_err_dict, **sigma_dict, **sigma_err_dict}
    for cc in range(len(colnames)):
        if colnames[cc] in rename_dict:
            colnames[cc] = rename_dict[colnames[cc]]

    ##########################################################
    # Transpose so that each row represents a single pixel & each column a measured quantity.
    rows_arr = np.array(rows_list).T
    return rows_arr, colnames, eline_list


###############################################################################
def make_lzifu_df(gals,
                  ncomponents,
                  eline_SNR_min,
                  correct_extinction,
                  sigma_inst_kms,
                  df_fname=None,
                  sigma_gas_SNR_min=3,
                  line_flux_SNR_cut=True,
                  missing_fluxes_cut=True,
                  line_amplitude_SNR_cut=True,
                  flux_fraction_cut=False,
                  sigma_gas_SNR_cut=True,
                  vgrad_cut=False,
                  metallicity_diagnostics=[
                      "N2Ha_PP04",
                      "N2Ha_M13",
                      "O3N2_PP04",
                      "O3N2_M13",
                      "N2S2Ha_D16",
                      "N2O2_KD02",
                      "Rcal_PG16",
                      "Scal_PG16",
                      "ON_P10",
                      "ONS_P10",
                      "N2Ha_K19",
                      "O3N2_K19",
                      "N2O2_K19",
                      "R23_KK04",
                      "N2Ha_PP04",
                      "N2Ha_K19",
                      "R23_KK04",
                  ],
                  nthreads=None):
    """
    Make a DataFrame from LZIFU fitting outputs, where each row represents a 
    single spaxel in a galaxy.

    DESCRIPTION
    ---------------------------------------------------------------------------
    This function is used to create a Pandas DataFrame containing emission line 
    fluxes & kinematics, stellar kinematics, extinction, star formation rates, 
    and other quantities for individual spaxels in galaxies where the emission
    lines have been fitted using LZIFU.

    The output is stored in HDF format as a Pandas DataFrame in which each row 
    corresponds to a given spaxel for every galaxy. 

    USAGE
    ---------------------------------------------------------------------------
    
        >>> from spaxelsleuth.loaddata.lzifu import make_lzifu_df()
        >>> make_lzifu_df(gals=gal_list, ncomponents=1, sigma_inst_kms=30, eline_SNR_min=5)

    will create a DataFrame using the data products from 1-component Gaussian 
    fits to the unbinned datacubes, and will adopt a minimum S/N threshold of 
    5 to mask out unreliable emission line fluxes and associated quantities.
    sigma_inst_kms refers to the Gaussian sigma of the instrumental line function
    in km/s.    

    Other input arguments may be configured to control other aspects of the data 
    quality and S/N cuts made. 

    INPUTS
    ---------------------------------------------------------------------------
    gals:                       list
        List of galaxies on which to run. 

    ncomponents:                str
        Controls which data products are used, depending on the number of 
        Gaussian components fitted to the emission lines. 
        Options are "merge" (the 'recommended' multi-component fits) or 1, 
        2 or 3.

    eline_SNR_min:              int 
        Minimum emission line flux S/N to adopt when making S/N and data 
        quality cuts.

    correct_extinction:         bool 
        If True, correct emission line fluxes for extinction. 

        Note that the Halpha equivalent widths are NOT corrected for extinction if 
        correct_extinction is True. This is because stellar continuum extinction 
        measurements are not available, and so applying the correction only to the 
        Halpha fluxes may over-estimate the true EW.
    
    sigma_inst_kms:             float
        the Gaussian sigma of the instrumental line function in km/s.    

    df_fname:                   str (optional)
        Filename of the output DataFrame. Defaults to 
            lzifu_{ncomponents}-comp_extcorr_minSNR={eline_SNR_min}.hd5
        if correct_extinction is True, otherwise
            lzifu_{ncomponents}-comp_minSNR={eline_SNR_min}.hd5.

    sigma_gas_SNR_min:          float (optional)
        Minimum velocity dipersion S/N to accept. Defaults to 3.

    line_flux_SNR_cut:          bool (optional)
        Whether to NaN emission line components AND total fluxes 
        (corresponding to emission lines in eline_list) below a specified S/N 
        threshold, given by eline_SNR_min. The S/N is simply the flux dividied 
        by the formal 1sigma uncertainty on the flux. Default: True.

    missing_fluxes_cut:         bool (optional)
        Whether to NaN out "missing" fluxes - i.e., cells in which the flux
        of an emission line (total or per component) is NaN, but the error 
        is not for some reason. Default: True.

    line_amplitude_SNR_cut:     bool (optional)
        If True, removes components with Gaussian amplitudes < 3 * RMS of the 
        continuum in the vicinity of Halpha. By default this is set to True
        because this does well at removing shallow components which are most 
        likely due to errors in the stellar continuum fit. Default: True.

    flux_fraction_cut:          bool (optional)
        If True, and if ncomponents > 1, remove intermediate and broad 
        components with line amplitudes < 0.05 that of the narrow componet.
        Set to False by default b/c it's unclear whether this is needed to 
        reject unreliable components. Default: False.

    sigma_gas_SNR_cut:          bool (optional)
        If True, mask component velocity dispersions where the S/N on the 
        velocity dispersion measurement is below sigma_gas_SNR_min. 
        By default this is set to True as it's a robust way to account for 
        emission line widths < instrumental.  Default: True.

    vgrad_cut:                  bool (optional)     
        If True, mask component kinematics (velocity and velocity dispersion)
        that are likely to be affected by beam smearing.
        By default this is set to False because it tends to remove nuclear spaxels 
        which may be of interest to your science case, & because it doesn't 
        reliably remove spaxels with quite large beam smearing components.
        Default: False.
    
    metallicity_diagnostics:    list of str (optional)
        List of strong-line metallicity diagnostics to compute. 
        Options:
            N2Ha_K19    N2Ha diagnostic from Kewley (2019).
            S2Ha_K19    S2Ha diagnostic from Kewley (2019).
            N2S2_K19    N2S2 diagnostic from Kewley (2019).
            S23_K19     S23 diagnostic from Kewley (2019).
            O3N2_K19    O3N2 diagnostic from Kewley (2019).
            O2S2_K19    O2S2 diagnostic from Kewley (2019).
            O2Hb_K19    O2Hb diagnostic from Kewley (2019).
            N2O2_K19    N2O2 diagnostic from Kewley (2019).
            R23_K19     R23 diagnostic from Kewley (2019).
            N2Ha_PP04   N2Ha diagnostic from Pilyugin & Peimbert (2004).
            N2Ha_M13    N2Ha diagnostic from Marino et al. (2013).
            O3N2_PP04   O3N2 diagnostic from Pilyugin & Peimbert (2004).
            O3N2_M13    O3N2 diagnostic from Marino et al. (2013).
            R23_KK04    R23 diagnostic from Kobulnicky & Kewley (2004).
            N2S2Ha_D16  N2S2Ha diagnostic from Dopita et al. (2016).
            N2O2_KD02   N2O2 diagnostic from Kewley & Dopita (2002).
            Rcal_PG16   Rcal diagnostic from Pilyugin & Grebel (2016).
            Scal_PG16   Scal diagnostic from Pilyugin & Grebel (2016).
            ONS_P10     ONS diagnostic from Pilyugin et al. (2010).
            ON_P10      ON diagnostic from Pilyugin et al. (2010).
    
    nthreads:                   int (optional)           
        Maximum number of threads to use. Defaults to os.cpu_count().

    OUTPUTS
    ---------------------------------------------------------------------------
    The resulting DataFrame will be stored as 

        settings["lzifu"]["output_path"]/lzifu_{ncomponents}-comp_extcorr_minSNR={eline_SNR_min}.hd5

    if correct_extinction is True, or else

        settings["lzifu"]["output_path"]/lzifu_{ncomponents}-comp_minSNR={eline_SNR_min}.hd5

    The DataFrame will be stored in CSV format in case saving in HDF format 
    fails for any reason.

    PREREQUISITES
    ---------------------------------------------------------------------------
    The data cubes must be and stored as follows: 

        For one-sided fits:
            settings["lzifu"]["data_cube_path"]/<gal>.fits(.gz)

        For two-sided fits:
            settings["lzifu"]["data_cube_path"]/<gal>_B.fits(.gz)
            settings["lzifu"]["data_cube_path"]/<gal>_R.fits(.gz)

    And the LZIFU outputs must be stored as

        settings["lzifu"]["input_path"]/<gal>_<merge/1/2/3>_comp.fits

    """

    ###############################################################################
    # input checking
    ###############################################################################
    if df_fname is not None:
        if not df_fname.endswith(".hd5"):
            df_fname += ".hd5"
    else:
        # Input file name
        df_fname = f"lzifu_{ncomponents}-comp"
        if correct_extinction:
            df_fname += "_extcorr"
        df_fname += f"_minSNR={eline_SNR_min}.hd5"

    if (type(ncomponents) not in [int, str]) or (type(ncomponents) == str
                                                 and ncomponents != "merge"):
        raise ValueError("ncomponents must be either an integer or 'merge'!")

    logger.info(f"input parameters: ncomponents={ncomponents}, eline_SNR_min={eline_SNR_min}, correct_extinction={correct_extinction}")

    # Determine number of threads
    if nthreads is None:
        nthreads = os.cpu_count()
        logger.warning(f"nthreads not specified: running make_sami_metadata_df() on {nthreads} threads...")

    ###############################################################################
    # Scrape measurements for each galaxy from FITS files
    ###############################################################################
    args_list = [[gg, gal, ncomponents, data_cube_path]
                 for gg, gal in enumerate(gals)]

    if len(gals) == 1:
        res_list = [_process_lzifu(args_list[0])]
    else:
        if nthreads > 1:
            logger.info(f"beginning pool...")
            pool = multiprocessing.Pool(min([nthreads, len(gals)]))
            res_list = pool.map(_process_lzifu, args_list)
            pool.close()
            pool.join()
        else:
            logger.info(f"running sequentially...")
            res_list = []
            for args in args_list:
                res = _process_lzifu(args)
                res_list.append(res)

    ###############################################################################
    # Convert to a Pandas DataFrame
    ###############################################################################
    rows_list_all = [r[0] for r in res_list]
    colnames = res_list[0][1]
    eline_list = res_list[0][2]  #TODO
    df_spaxels = pd.DataFrame(np.vstack(tuple(rows_list_all)),
                              columns=colnames)

    # Cast to float data types
    for col in df_spaxels.columns:
        # Check if column values can be cast to float
        df_spaxels[col] = pd.to_numeric(df_spaxels[col], errors="coerce")

    ###############################################################################
    # Generic stuff: compute additional columns - extinction, metallicity, etc.
    ###############################################################################
    df_spaxels = add_columns(
        df_spaxels.copy(),
        eline_SNR_min=eline_SNR_min,
        sigma_gas_SNR_min=sigma_gas_SNR_min,
        eline_list=eline_list,
        line_flux_SNR_cut=line_flux_SNR_cut,
        missing_fluxes_cut=missing_fluxes_cut,
        line_amplitude_SNR_cut=line_amplitude_SNR_cut,
        flux_fraction_cut=flux_fraction_cut,
        sigma_gas_SNR_cut=sigma_gas_SNR_cut,
        vgrad_cut=vgrad_cut,
        stekin_cut=False,
        correct_extinction=correct_extinction,
        metallicity_diagnostics=metallicity_diagnostics,
        compute_sfr=True,
        sigma_inst_kms=sigma_inst_kms,
        nthreads=nthreads,
        base_missing_flux_components_on_HALPHA=False,  # NOTE: this is important!!
        )

    ###############################################################################
    # Save to file
    ###############################################################################
    logger.info(f"saving to file {output_path / df_fname}...")
    df_spaxels.to_hdf(output_path / df_fname, key="lzifu")
    logger.info(f"finished!")

    return


###############################################################################
def load_lzifu_df(ncomponents=None,
                  correct_extinction=None,
                  eline_SNR_min=None,
                  df_fname=None,
                  key=None):

    #######################################################################
    # INPUT CHECKING
    #######################################################################
    # Input file name
    if df_fname is not None:
        logger.warning(
            f"loading DataFrame from user-provided filename {output_path / df_fname} which may not correspond to the provided ncomponents, etc. Proceed with caution!",
            RuntimeWarning)
        if not df_fname.endswith(".hd5"):
            df_fname += ".hd5"
    else:
        # Input file name
        df_fname = f"lzifu_{ncomponents}-comp"
        if correct_extinction:
            df_fname += "_extcorr"
        df_fname += f"_minSNR={eline_SNR_min}.hd5"

    if not os.path.exists(output_path / df_fname):
        raise FileNotFoundError(
            f"File {output_path / df_fname} does does not exist!")

    # Load the data frame
    t = os.path.getmtime(output_path / df_fname)
    logger.info(
        f"loading DataFrame from file {output_path / df_fname} [last modified {datetime.datetime.fromtimestamp(t)}]..."
    )
    if key is not None:
        df = pd.read_hdf(output_path / df_fname, key=key)
    else:
        df = pd.read_hdf(output_path / df_fname)

    # Add "metadata" columns to the DataFrame
    df["survey"] = "lzifu"
    df["ncomponents"] = ncomponents

    # Add back in object-type columns
    df["x, y (pixels)"] = list(
    zip(df["x (projected, arcsec)"] / 0.5,
        df["y (projected, arcsec)"] / 0.5))
    df["BPT (total)"] = bpt_num_to_str(df["BPT (numeric) (total)"])

    # Return
    logger.info("finished!")
    return df.sort_index()