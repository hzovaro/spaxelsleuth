import os
from pathlib import Path
import warnings

import datetime
from astropy.io import fits
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.cosmology import FlatLambdaCDM
import multiprocessing
import numpy as np
import pandas as pd

from spaxelsleuth.config import settings
from spaxelsleuth.utils.continuum import compute_d4000, compute_continuum_intensity
from spaxelsleuth.utils.dqcut import compute_HALPHA_amplitude_to_noise
from spaxelsleuth.utils.addcolumns import add_columns
from spaxelsleuth.utils.linefns import bpt_num_to_str

import logging
logger = logging.getLogger(__name__)

###############################################################################
# Paths
input_path = Path(settings["s7"]["input_path"])
output_path = Path(settings["s7"]["output_path"])
data_cube_path = Path(settings["s7"]["data_cube_path"])

#/////////////////////////////////////////////////////////////////////////////////
def add_metadata(df, df_metadata):
    """Merge an input DataFrame with that was created using make_s7_df()."""
    if "ID" not in df_metadata:
        raise ValueError("df_metadata must contain an 'ID' column!")

    df = df.merge(df_metadata, on="ID", how="left")

    return df

###############################################################################
def make_s7_metadata_df():
    """
    Create a DataFrame containing "metadata" for all S7 galaxies.

    DESCRIPTION
    ---------------------------------------------------------------------------
    This function is used to create a DataFrame containing "metadata", including
    stellar masses, spectroscopic redshifts, morphologies and other information
    for each galaxy in S7. In addition to the provided values in the input
    catalogues, the angular scale (in kpc per arcsecond) and inclination are 
    computed for each galaxy.

    This script must be run before make_s7_df() as the resulting DataFrame
    is used there.

    USAGE
    ---------------------------------------------------------------------------
            
            >>> from spaxelsleuth.loaddata.s7 import make_s7_metadata_df
            >>> make_s7_metadata_df()

    INPUTS
    ---------------------------------------------------------------------------
    None.

    OUTPUTS
    ---------------------------------------------------------------------------
    The DataFrame is saved to 

        settings["s7"]["output_path"]/s7_metadata.hd5

    PREREQUISITES
    ---------------------------------------------------------------------------

    The table containing metadata for S7 galaxies is required for this script. 
    This has been included in the ../data/ directory but can be downloaded in 
    CSV format from 
        
        https://miocene.anu.edu.au/S7/Data_release_2/S7_DR2_Table_2_Catalogue.csv

    NOTE: the version downloaded from the website has some problematic whitespace
    which has been removed from the version included with spaxelsleuth.
    
    """
    logger.info("creating metadata DataFrame...")
    ###############################################################################
    # Filenames
    input_catalogue_fname = "S7_DR2_Table_2_Catalogue.csv"
    df_metadata_fname = "s7_metadata.hd5"

    ###############################################################################
    # READ IN THE METADATA
    ###############################################################################
    data_path = Path(__file__.split("loaddata")[0]) / "data"
    assert os.path.exists(data_path / input_catalogue_fname),\
        f"File {data_path / input_catalogue_fname} not found!"
    df_metadata = pd.read_csv(data_path / input_catalogue_fname, skiprows=58)
    gals = df_metadata["S7_Name"].values

    ###############################################################################
    # Convert object coordinates to degrees
    ###############################################################################
    coords = SkyCoord(df_metadata["RA_hms"], df_metadata["Dec_sxgsml"],
                 unit=(u.hourangle, u.deg))
    df_metadata["RA (J2000)"] = coords.ra.deg
    df_metadata["Dec (J2000)"] = coords.dec.deg

    ###############################################################################
    # Rename columns
    ###############################################################################
    rename_dict = {
        "S7_Name": "ID",
        "HL_inclination": "i (degrees)",
        "HL_Re": "R_e (arcsec)",
        "HL_Re_err": "R_e error (arcsec)",
        "NED_ax_ratio": "b/a",
        "NED_ax_ratio_err": "b/a error",
        "HL_PA": "PA (degrees)",
        "S7_best_WiFeS_PA": "WiFeS PA",
        "S7_Mstar": "log M_*",
        "S7_Mstar_err": "log M_* error",
        "S7_Sy1_subtraction?": "Sy1 subtraction?",
        "S7_mosaic?": "Mosaic?",
        "S7_BPT_classification": "BPT (global)",
        "S7_z": "z",
        "S7_nucleus_index_x": "x0_px",
        "S7_nucleus_index_y": "y0_px"
    }
    df_metadata = df_metadata.rename(columns=rename_dict)
    df_metadata = df_metadata.set_index(df_metadata["ID"])

    # Get rid of unneeded columns
    good_cols = [rename_dict[k] for k in rename_dict.keys()] + ["RA (J2000)", "Dec (J2000)"]
    df_metadata = df_metadata[good_cols]

    ###############################################################################
    # Add angular scale info
    ###############################################################################
    logger.info(f"computing distances...")
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    for gal in gals:
        D_A_Mpc = cosmo.angular_diameter_distance(df_metadata.loc[gal, "z"]).value
        D_L_Mpc = cosmo.luminosity_distance(df_metadata.loc[gal, "z"]).value
        df_metadata.loc[gal, "D_A (Mpc)"] = D_A_Mpc
        df_metadata.loc[gal, "D_L (Mpc)"] = D_L_Mpc
    df_metadata["kpc per arcsec"] = df_metadata["D_A (Mpc)"] * 1e3 * np.pi / 180.0 / 3600.0

    ###############################################################################
    # Define a "Good?" column
    ###############################################################################
    df_metadata["Sy1 subtraction?"] = [True if x == "Y" else False for x in df_metadata["Sy1 subtraction?"].values]
    df_metadata["Good?"] = ~df_metadata["Sy1 subtraction?"].values
    df_metadata = df_metadata.sort_index()

    ###############################################################################
    # Save to file
    ###############################################################################
    logger.info(f"saving metadata DataFrame to file {output_path / df_metadata_fname}...")
    df_metadata.to_hdf(output_path / df_metadata_fname, key="metadata")

    logger.info(f"finished!")
    return

#/////////////////////////////////////////////////////////////////////////////////
def _process_s7(args):
    """Helper function that is used in make_s7_df() to process S7 galaxies across multiple threads."""
    gal, df_metadata = args

    #######################################################################
    # Scrape outputs from LZIFU output
    #######################################################################
    hdulist_best_components = fits.open(input_path / f"{gal}_best_components.fits")
    hdr = hdulist_best_components[0].header

    # Angular scale
    z = hdr["Z"]
    D_A_Mpc = df_metadata.loc[gal, "D_A (Mpc)"]
    kpc_per_arcsec = D_A_Mpc * 1e3 * np.pi / 180.0 / 3600.0

    # Angular scale
    as_per_px = settings["s7"]["as_per_px"]

    #######################################################################
    # LOAD THE DATACUBES
    # NOTE: the data cubes have units 
    #       erg / cm^2 / s / Angstroem / arcsec^2 
    # So we normalise them by a factor of 1e-16 to put them in the same 
    # units as the emission line fluxes.
    #######################################################################
    datacube_R_fname = data_cube_path / f"{gal}_R.fits"
    datacube_B_fname = data_cube_path / f"{gal}_B.fits"
    
    # Blue cube
    with fits.open(datacube_B_fname) as hdulist_B_cube:
        header_B = hdulist_B_cube[0].header
        data_cube_B = hdulist_B_cube[0].data * 1e16
        var_cube_B = hdulist_B_cube[1].data * 1e32

        # Wavelength values
        lambda_0_A = header_B["CRVAL3"]
        dlambda_A = header_B["CDELT3"]
        N_lambda = header_B["NAXIS3"]
        lambda_vals_B_A = np.array(
            range(N_lambda)) * dlambda_A + lambda_0_A

    # Red cube
    with fits.open(datacube_R_fname) as hdulist_R_cube:
        header_R = hdulist_R_cube[0].header
        data_cube_R = hdulist_R_cube[0].data * 1e16
        var_cube_R = hdulist_R_cube[1].data * 1e32

        # Wavelength values
        lambda_0_A = header_R["CRVAL3"]
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
    ngood_bins = len(x_c_list)

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
    ]
    quantities = [e.rstrip("_ERR")
                  for e in extnames if e.endswith("_ERR")] + ["CHI2", "DOF", "STAR_V", "STAR_VDISP"]
    rows_list = []
    colnames = []
    eline_list = [
        q for q in quantities if q not in ["V", "VDISP", "CHI2", "DOF", "STAR_V", "STAR_VDISP"]
    ]
    lzifu_ncomponents = hdulist_best_components["V"].shape[0] - 1

    # Scrape the FITS file: emission line flues, velocity/velocity dispersion, fit quality
    # NOTE: fluxes are in units of E-16 erg / cm^2 / s
    for quantity in quantities:
        data = hdulist_best_components[quantity].data
        if f"{quantity}_ERR" in hdulist_best_components:
            err = hdulist_best_components[f"{quantity}_ERR"].data
        
        if quantity in ["OII3726", "OII3729", "OIII4363"]:
            # Special case for these lines: only total fluxes are available 
            rows_list.append(_2d_map_to_1d_list(data))
            colnames.append(f"{quantity} (total)")
            rows_list.append(_2d_map_to_1d_list(err))
            colnames.append(f"{quantity} error (total)")
        elif quantity in ["STAR_V", "STAR_VDISP"]:
            # Special case for stellar kinematics: errors stored in the 2nd slice
            rows_list.append(_2d_map_to_1d_list(data[0]))
            colnames.append(f"{quantity}")
            rows_list.append(_2d_map_to_1d_list(data[1]))
            colnames.append(f"{quantity} error")

        else:
            if data.ndim == 3:
                # Total fluxes (only for emission lines)
                if quantity not in ["V", "VDISP"]:
                    rows_list.append(_2d_map_to_1d_list(data[0]))
                    colnames.append(f"{quantity} (total)")
                    if f"{quantity}_ERR" in hdulist_best_components:
                        rows_list.append(_2d_map_to_1d_list(err[0]))
                        colnames.append(f"{quantity} error (total)")
                # Fluxes/values for individual components
                for nn in range(lzifu_ncomponents):
                    rows_list.append(_2d_map_to_1d_list(data[nn + 1]))
                    colnames.append(f"{quantity} (component {nn + 1})")
                    if f"{quantity}_ERR" in hdulist_best_components:
                        rows_list.append(_2d_map_to_1d_list(err[nn + 1]))
                        colnames.append(f"{quantity} error (component {nn + 1})")
            # Otherwise it's a 2D map
            elif data.ndim == 2:
                rows_list.append(_2d_map_to_1d_list(data))
                colnames.append(f"{quantity}")
                if f"{quantity}_ERR" in hdulist_best_components:
                    rows_list.append(_2d_map_to_1d_list(err))
                    colnames.append(f"{quantity} error")

    ##########################################################
    # COMPUTE QUANTITIES DIRECTLY FROM THE DATACUBES
    ##########################################################
    v_star_map = hdulist_best_components["STAR_V"].data[0]
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
        v_map = hdulist_best_components["V"].data  # Get velocity field from LZIFU fit
        AN_HALPHA_map = compute_HALPHA_amplitude_to_noise(
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
    rows_list.append([as_per_px] * len(x_c_list))
    colnames.append("as_per_px")
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

    # Get rid of rows that are all NaNs
    # id_idx = colnames.index("ID")
    bad_rows = np.all(np.isnan(rows_arr), axis=1)
    rows_good = rows_arr[~bad_rows]

    # Append a column with the galaxy ID & other properties
    safe_cols = [
        c for c in df_metadata.columns if (df_metadata[c].dtypes == float) or (df_metadata[c].dtypes == int)
    ] + ["ID"]
    gal_metadata = np.tile(
        df_metadata.loc[df_metadata.loc[:, "ID"] == gal][safe_cols].values,
        (len(x_c_list), 1))
    rows_good = np.hstack((gal_metadata, rows_good))

    logger.info(f"finished processing {gal}")

    return rows_good, colnames, eline_list


#/////////////////////////////////////////////////////////////////////////////////
def make_s7_df(gals=None,
                eline_SNR_min=5,
                sigma_gas_SNR_min=3,
                line_flux_SNR_cut=True,
                missing_fluxes_cut=True,
                line_amplitude_SNR_cut=True,
                flux_fraction_cut=False,
                sigma_gas_SNR_cut=True,
                vgrad_cut=False,
                stekin_cut=False,
                correct_extinction=True,
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
                ],
                nthreads=None,
                debug=False):
    """
    Make the SAMI DataFrame, where each row represents a single spaxel in a SAMI galaxy.

    DESCRIPTION
    ---------------------------------------------------------------------------
    This function is used to create a Pandas DataFrame containing emission line 
    fluxes & kinematics, stellar kinematics, extinction, star formation rates, 
    and other quantities for individual spaxels in SAMI galaxies as taken from 
    SAMI DR3.

    The output is stored in HDF format as a Pandas DataFrame in which each row 
    corresponds to a given spaxel (or Voronoi/sector bin) for every galaxy. 

    USAGE
    ---------------------------------------------------------------------------
    
        >>> from spaxelsleuth.loaddata.sami import make_sami_df()
        >>> make_sami_df(ncomponents="1", bin_type="default", correct_extinction=True, eline_SNR_min=5)

    will create a DataFrame using the data products from 1-component Gaussian 
    fits to the unbinned datacubes, and will adopt a minimum S/N threshold of 
    5 to mask out unreliable emission line fluxes and associated quantities.

    Other input arguments may be configured to control other aspects of the data 
    quality and S/N cuts made.

    Running this function on the full sample takes some time (~10-20 minutes 
    when threaded across 20 threads). Execution time can be sped up by tweaking 
    the nthreads parameter. 

    If you wish to run in debug mode, set the DEBUG flag to True: this will run 
    the script on a subset (by default 10) galaxies to speed up execution. 

    INPUTS
    ---------------------------------------------------------------------------
    ncomponents:                str
        Controls which data products are used, depending on the number of 
        Gaussian components fitted to the emission lines. 
        Options are "recom" (the recommended multi-component fits) or "1" 
        (1-component fits).
        
        NOTE: if __use_lzifu_fits is True, then ncomponents is ONLY used in  
        loading data products that are NOT contained in the output LZIFU files -
        i.e., SFRs/SFR surface densities and HALPHA extinction correction 
        factors. Use parameter __lzifu_ncomponents to control which data 
        derived from the LZIFU fits is loaded. 

    bin_type:                   str
        Spatial binning strategy. Options are "default" (unbinned), "adaptive"
        (Voronoi binning) or "sectors" (sector binning)

    eline_SNR_min:              int 
        Minimum emission line flux S/N to adopt when making S/N and data 
        quality cuts.

    correct_extinction:         bool
        If True, correct emission line fluxes for extinction. 

        Note that the Halpha equivalent widths are NOT corrected for extinction if 
        correct_extinction is True. This is because stellar continuum extinction 
        measurements are not available, and so applying the correction only to the 
        Halpha fluxes may over-estimate the true EW.

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

    stekin_cut:                 bool (optional)
        If True, mask stellar kinematic quantities that do not meet the DQ and 
        S/N requirements specified in Croom et al. (2021). Default: True.

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

    eline_list:                 list of str (optional)
        List of emission lines to use. Defaults to the full list of lines fitted 
        in SAMI DR3.

    nthreads:                   int (optional)           
        Maximum number of threads to use. Defaults to os.cpu_count().

    __use_lzifu_fits:           bool (optional)
        If True, load the DataFrame containing emission line quantities
        (including fluxes, kinematics, etc.) derived directly from the LZIFU
        output FITS files, rather than those included in DR3. Default: False.

    __lzifu_ncomponents:        str  (optional)
        Number of components corresponding to the LZIFU fit, if 
        __use_lzifu_fits is specified. May be '1', '2', '3' or 'recom'. Note 
        that this keyword ONLY affects emission line fluxes and gas kinematics;
        other quantities including SFR/SFR surface densities and HALPHA 
        extinction correction factors are loaded from DR3 data products as per
        the ncomponents keyword. Default: False.

    debug:                      bool (optional)
        If True, run on a subset of the entire sample (10 galaxies) and save
        the output with "_DEBUG" appended to the filename. This is useful for
        tweaking S/N and DQ cuts since running the function on the entire 
        sample is quite slow.  Default: False.

    OUTPUTS
    ---------------------------------------------------------------------------
    The resulting DataFrame will be stored as 

        settings["sami"]["output_path"]/sami_{bin_type}_{ncomponents}-comp_extcorr_minSNR={eline_SNR_min}.hd5

    if correct_extinction is True, or else

        settings["sami"]["output_path"]/sami_{bin_type}_{ncomponents}-comp_minSNR={eline_SNR_min}.hd5

    The DataFrame will be stored in CSV format in case saving in HDF format 
    fails for any reason.

    PREREQUISITES
    ---------------------------------------------------------------------------
    make_sami_metadata_df() must be run first.

    SAMI data products must be downloaded from DataCentral

        https://datacentral.org.au/services/download/

    and stored as follows: 

        settings["sami"]["output_path"]/ifs/<gal>/<gal>_<quantity>_<bin type>_<number of components>-comp.fits

    This is essentially the default file structure when data products are 
    downloaded from DataCentral and unzipped:

        sami/dr3/ifs/<gal>/<gal>_<quantity>_<bin type>_<number of components>-comp.fits

    The following data products are required to run this script:

        Halpha_{bin_type}_{ncomponents}-comp.fits,
        Hbeta_{bin_type}_{ncomponents}-comp.fits,
        NII6583_{bin_type}_{ncomponents}-comp.fits,
        OI6300_{bin_type}_{ncomponents}-comp.fits,
        OII3728_{bin_type}_{ncomponents}-comp.fits,
        OIII5007_{bin_type}_{ncomponents}-comp.fits,
        SII6716_{bin_type}_{ncomponents}-comp.fits,
        SII6731_{bin_type}_{ncomponents}-comp.fits,
        gas-vdisp_{bin_type}_{ncomponents}-comp.fits,
        gas-velocity_{bin_type}_{ncomponents}-comp.fits,
        stellar-velocity-dispersion_{bin_type}_two-moment.fits,
        stellar-velocity_{bin_type}_two-moment.fits,
        extinct-corr_{bin_type}_{ncomponents}-comp.fits,
        sfr-dens_{bin_type}_{ncomponents}-comp.fits,
        sfr_{bin_type}_{ncomponents}-comp.fits

    SAMI data cubes must also be downloaded from DataCentral and stored as follows: 

        settings["sami"]["data_cube_path"]/ifs/<gal>/<gal>_A_cube_<blue/red>.fits.gz

    settings["sami"]["data_cube_path"] can be the same as settings["sami"]["output_path"] (I just have them differently
    in my setup due to storage space limitations).
    """

    df_fname = f"s7_default_merge-comp"
    if correct_extinction:
        df_fname += "_extcorr"
    df_fname += f"_minSNR={eline_SNR_min}"
    if debug:
        df_fname += "_DEBUG"
    df_fname += ".hd5"

    # Load metadata DataFrame
    try:
        df_metadata = pd.read_hdf(output_path / "s7_metadata.hd5", key="metadata")
    except FileNotFoundError:
        raise FileNotFoundError(f"Metadata DataFrame {output_path / 's7_metadata.hd5'} not found - have you run make_s7_metadata_df() first?")

    # Check validity of input galaxies
    if gals is None:
        gals = df_metadata["ID"].unique()
    else:
        for gal in gals:
            if gal not in df_metadata["ID"].unique():
                raise ValueError(f"Galaxy {gal} is not a valid S7 galaxy!")

    # Determine number of threads
    if nthreads is None:
        nthreads = os.cpu_count()
        logger.warning(f"nthreads not specified: running make_sami_metadata_df() on {nthreads} threads...")

    ###############################################################################
    # Scrape measurements for each galaxy from FITS files
    ###############################################################################
    args_list = [[g, df_metadata] for g in gals]
    if len(gals) == 1:
        res_list = [_process_s7(args_list[0])]
    else:
        if nthreads > 1:
            logger.info(f"beginning pool...")
            pool = multiprocessing.Pool(min([nthreads, len(gals)]))
            res_list = np.array((pool.map(_process_s7, args_list)))
            pool.close()
            pool.join()
        else:
            logger.info(f"running sequentially...")
            res_list = []
            for args in args_list:
                res = _process_s7(args)
                res_list.append(res)

    ###############################################################################
    # Convert to a Pandas DataFrame
    ###############################################################################
    rows_list_all = [r[0] for r in res_list]
    colnames = res_list[0][1]
    eline_list = res_list[0][2]
    safe_cols = [
        c for c in df_metadata.columns if (df_metadata[c].dtypes == float) or (df_metadata[c].dtypes == int)
    ] + ["ID"]
    df_spaxels = pd.DataFrame(np.vstack(tuple(rows_list_all)),
                              columns=safe_cols + colnames)

    # Cast to float data types
    for col in [c for c in df_spaxels.columns if c != "ID"]:
        # Check if column values can be cast to float
        df_spaxels[col] = pd.to_numeric(df_spaxels[col], errors="coerce")

    ###############################################################################
    # Generic stuff: compute additional columns - extinction, metallicity, etc.
    ###############################################################################
    df_spaxels = add_columns(
        df_spaxels,
        eline_SNR_min=eline_SNR_min,
        sigma_gas_SNR_min=sigma_gas_SNR_min,
        eline_list=eline_list,
        line_flux_SNR_cut=line_flux_SNR_cut,
        missing_fluxes_cut=missing_fluxes_cut,
        line_amplitude_SNR_cut=line_amplitude_SNR_cut,
        flux_fraction_cut=flux_fraction_cut,
        sigma_gas_SNR_cut=sigma_gas_SNR_cut,
        vgrad_cut=vgrad_cut,
        stekin_cut=stekin_cut,
        correct_extinction=correct_extinction,
        metallicity_diagnostics=metallicity_diagnostics,
        compute_sfr=True,
        sigma_inst_kms=settings["s7"]["sigma_inst_kms"],
        nthreads=nthreads,
        base_missing_flux_components_on_HALPHA=
        False,  # NOTE: this is important!!
        debug=debug)

    ###############################################################################
    # Save to file
    ###############################################################################
    logger.info(f"saving to file {df_fname}...")
    df_spaxels.to_hdf(output_path / df_fname, key=f"s7")
    logger.info(f"finished!")

    return


#/////////////////////////////////////////////////////////////////////////////////
def load_s7_df(correct_extinction=None,
                  eline_SNR_min=None,
                  debug=False):

    #######################################################################
    # INPUT CHECKING
    #######################################################################
    # Input file name
    df_fname = f"s7_default_merge-comp"
    if correct_extinction:
        df_fname += "_extcorr"
    df_fname += f"_minSNR={eline_SNR_min}"
    if debug:
        df_fname += "_DEBUG"
    df_fname += ".hd5"

    if not os.path.exists(output_path / df_fname):
        raise FileNotFoundError(
            f"File {output_path / df_fname} does does not exist!")

    # Load the data frame
    t = os.path.getmtime(output_path / df_fname)
    logger.info(
        f"loading DataFrame from file {output_path / df_fname} [last modified {datetime.datetime.fromtimestamp(t)}]..."
    )
    df = pd.read_hdf(output_path / df_fname)

    # Add "metadata" columns to the DataFrame
    df["survey"] = "s7"
    df["ncomponents"] = "merge"
    df["bin_type"] = "default"
    df["debug"] = debug
    df["flux units"] = "E-16 erg/cm^2/s"  # Units of continuum & emission line flux
    df["continuum units"] = "E-16 erg/cm^2/Å/s"  # Units of continuum & emission line flux

    # Add back in object-type columns
    df["x, y (pixels)"] = list(
        zip(df["x (pixels)"], df["y (pixels)"]))
    df["BPT (total)"] = bpt_num_to_str(df["BPT (numeric) (total)"])

    # Return
    logger.info("finished!")
    return df.sort_index()

###############################################################################
def load_s7_metadata_df():
    """Load the S7 metadata DataFrame, containing "metadata" for each galaxy."""
    if not os.path.exists(Path(settings["s7"]["output_path"]) / "s7_metadata.hd5"):
        raise FileNotFoundError(
            f"File {Path(settings['s7']['output_path']) / 's7_metadata.hd5'} not found. Did you remember to run make_s7_metadata_df() first?"
        )
    return pd.read_hdf(
        Path(settings["s7"]["output_path"]) / "s7_metadata.hd5")
