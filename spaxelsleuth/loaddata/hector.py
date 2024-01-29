# Imports
import datetime
from astropy.io import fits
from astropy.cosmology import FlatLambdaCDM
import multiprocessing
import numpy as np
import os
import pandas as pd
from pathlib import Path

from spaxelsleuth.config import settings
from spaxelsleuth.utils.continuum import compute_d4000, compute_continuum_intensity
from spaxelsleuth.utils.dqcut import compute_measured_HALPHA_amplitude_to_noise
from spaxelsleuth.utils.addcolumns import add_columns
from spaxelsleuth.utils.linefns import bpt_num_to_str
from spaxelsleuth.utils.velocity import compute_v_grad

import logging
logger = logging.getLogger(__name__)

# TODO: put these in config 
H0 = 70.0
OMEGA_0 = 0.3


###############################################################################
# Paths
input_path = Path(settings["hector"]["input_path"])
output_path = Path(settings["hector"]["output_path"])
data_cube_path = Path(settings["hector"]["data_cube_path"])
eline_fit_path = input_path / "emission_cubes"
stekin_path = input_path / "initial_stel_kin"
continuum_fit_path = input_path / "cont_subtracted"


##############################################################################
def _2d_map_to_1d_list(colmap, x_c_list, y_c_list, nx, ny):
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


def _1d_map_to_2d_list(rows, x_c_list, y_c_list, nx, ny):
    """Reconstructs a 2D array of values from row with coordinates specified in x_c_list and y_c_list. DEBUGGING ONLY."""
    colmap = np.full((ny, nx), np.nan)
    for d, x, y in zip(rows, x_c_list, y_c_list):
        colmap[y, x] = d
    return colmap


def check_row(rows, colname, data, x_c_list, y_c_list, nx, ny):
    """Plots data before and after being converted into row form. DEBUGGING ONLY."""
    fig, axs = plt.subplots(ncols=3, figsize=(10, 5))
    colmap = _1d_map_to_2d_list(rows, x_c_list, y_c_list, nx, ny)
    axs[0].imshow(data)
    axs[1].imshow(colmap)
    axs[2].imshow(data - colmap)
    fig.suptitle(colname)
    axs[0].set_title("From FITS file")
    axs[1].set_title("Reconstructed from row")
    axs[2].set_title("Difference")


###############################################################################
def make_hector_metadata_df():
    """Create the Hector "metadata" DataFrame.

    DESCRIPTION
    ---------------------------------------------------------------------------
    This function is used to create a DataFrame containing "metadata", including
    coordinates, redshifts, distances and angular scales for each Hector galaxy. 
    Also stored are paths to the FITS files containing the raw data cubes, 
    emission line and stellar continuum fitting products.

    This script must be run before make_hector_df() as the resulting DataFrame
    is used there.

    Details:
        - Distances are computed from the redshifts assuming a flat ΛCDM cosmology 
    with H0 = 70 km/s/Mpc, ΩM = 0.3 and ΩΛ = 0.7. 

    USAGE
    ---------------------------------------------------------------------------
            
            >>> from spaxelsleuth.loaddata.hector import make_hector_metadata_df
            >>> make_hector_metadata_df()

    INPUTS
    ---------------------------------------------------------------------------
    None.

    OUTPUTS
    ---------------------------------------------------------------------------
    The DataFrame is saved to 

        settings["hector"]["output_path"]/hector_metadata.hd5

    PREREQUISITES
    ---------------------------------------------------------------------------
    TODO: complete this.
    """
    logger.info("creating metadata DataFrame...")

    # Output DataFrame filename
    df_fname = f"hector_metadata.hd5"

    ###########################################################################
    # Get list of unique galaxy IDs 
    ###########################################################################
    gals = [int(g) for g in set([f.split("_")[0] for f in os.listdir(data_cube_path)])]
    df_metadata = pd.DataFrame(data = {"ID": gals,}).set_index("ID")

    ###########################################################################
    # Iterate through all galaxies and scrape data from FITS files 
    ###########################################################################
    for gal in gals:
        # Get the blue & read data cube names 
        datacube_B_fnames = [data_cube_path / f for f in os.listdir(data_cube_path) if f.startswith(str(gal)) and "blue" in f]
        datacube_R_fnames = [data_cube_path / f for f in os.listdir(data_cube_path) if f.startswith(str(gal)) and "red" in f]
        assert len(datacube_B_fnames) == 1
        assert len(datacube_R_fnames) == 1
        datacube_B_fname = datacube_B_fnames[0]
        datacube_R_fname = datacube_R_fnames[0]
        assert os.path.exists(datacube_B_fname)
        assert os.path.exists(datacube_R_fname)

        # Get RA, Dec, spectrograph, bundle, field & tile numbers
        with fits.open(datacube_B_fname) as hdulist_B_cube:
            header_B = hdulist_B_cube[0].header
            spectrograph_B = header_B["INSTRUME"]
            bundle_B = header_B["IFUPROBE"]
            ra_B = header_B["CRVAL1"]
            dec_B = header_B["CRVAL2"]
            plateid_B = header_B["PLATEID"]
            x0_px_B, y0_px_B = np.floor(header_B["CRPIX1"]).astype(int), np.floor(header_B["CRPIX2"]).astype(int)
            nx_B, ny_B = np.floor(header_B["NAXIS1"]).astype(int), np.floor(header_B["NAXIS2"]).astype(int)
            
        with fits.open(datacube_R_fname) as hdulist_R_cube:
            header_R = hdulist_R_cube[0].header
            spectrograph_R = header_R["INSTRUME"]
            bundle_R = header_R["IFUPROBE"]
            ra_R = header_R["CRVAL1"]
            dec_R = header_R["CRVAL2"]
            plateid_R = header_R["PLATEID"]
            x0_px_R, y0_px_R = np.floor(header_R["CRPIX1"]).astype(int), np.floor(header_R["CRPIX2"]).astype(int)
            nx_R, ny_R = np.floor(header_R["NAXIS1"]).astype(int), np.floor(header_R["NAXIS2"]).astype(int)
            
        # Check for consistency between blue & red cubes 
        assert spectrograph_B == spectrograph_R
        assert bundle_B == bundle_R
        assert ra_B == ra_R
        assert dec_B == dec_R
        assert plateid_B == plateid_R
        assert nx_B == nx_R
        assert ny_B == ny_R
        assert x0_px_B == x0_px_R
        assert y0_px_B == y0_px_R
        field, tile = plateid_B.split("_") 

        # FITS filenames for stellar kinematics & continuum fit data products
        stekin_fnames = [stekin_path / f for f in os.listdir(stekin_path) if f.startswith(str(gal))]
        cont_fit_B_fnames = [continuum_fit_path / f for f in os.listdir(continuum_fit_path) if f.startswith(str(gal)) and "blue" in f]
        cont_fit_R_fnames = [continuum_fit_path / f for f in os.listdir(continuum_fit_path) if f.startswith(str(gal)) and "red" in f]
        assert len(stekin_fnames) == 1
        assert len(cont_fit_B_fnames) == 1
        assert len(cont_fit_R_fnames) == 1
        stekin_fname = stekin_fnames[0]
        cont_fit_B_fname = cont_fit_B_fnames[0]
        cont_fit_R_fname = cont_fit_R_fnames[0]
        assert os.path.exists(stekin_fname)
        assert os.path.exists(cont_fit_B_fname)
        assert os.path.exists(cont_fit_R_fname)

        # Get FITS filenames for emission line fit data products
        eline_fit_subdirs = [f for f in os.listdir(eline_fit_path) if f.startswith(str(gal))]
        assert len(eline_fit_subdirs) == 1
        eline_fit_subdir = eline_fit_subdirs[0]
        eline_fit_fnames = []
        for ncomponents in [1, 2, 3, "rec"]:
            eline_component_fit_fnames = [eline_fit_path / eline_fit_subdir / f for f in os.listdir(eline_fit_path / eline_fit_subdir) if f.startswith(str(gal)) and f"{ncomponents}comp" in f]
            assert len(eline_component_fit_fnames) == 1
            eline_fit_fname = eline_component_fit_fnames[0]
            assert os.path.exists(eline_fit_fname)
            eline_fit_fnames.append(eline_fit_fname)

        # Get redshift & calculate distances 
        with fits.open(stekin_fname) as hdulist_stekin:
            z = hdulist_stekin[0].header["Z"]
        cosmo = FlatLambdaCDM(H0=H0, Om0=OMEGA_0)
        D_A_Mpc = cosmo.angular_diameter_distance(z).value
        D_L_Mpc = cosmo.luminosity_distance(z).value
        kpc_per_arcsec = D_A_Mpc * 1e3 * np.pi / 180.0 / 3600.0

        # Store in DataFrame
        df_metadata.loc[gal, "RA (J2000)"] = ra_B
        df_metadata.loc[gal, "Dec (J2000)"] = dec_B
        df_metadata.loc[gal, "z"] = z
        df_metadata.loc[gal, "D_A (Mpc)"] = D_A_Mpc
        df_metadata.loc[gal, "D_L (Mpc)"] = D_L_Mpc
        df_metadata.loc[gal, "N_x"] = int(nx_B)
        df_metadata.loc[gal, "N_y"] = int(ny_B)
        df_metadata.loc[gal, "x0_px"] = int(x0_px_B)
        df_metadata.loc[gal, "y0_px"] = int(y0_px_B)
        df_metadata.loc[gal, "Galaxy centre x0_px (projected, arcsec)"] = x0_px_B * settings["hector"]["as_per_px"]
        df_metadata.loc[gal, "Galaxy centre y0_px (projected, arcsec)"] = y0_px_B * settings["hector"]["as_per_px"]
        df_metadata.loc[gal, "kpc per arcsec"] = kpc_per_arcsec
        df_metadata.loc[gal, "Field"] = field
        df_metadata.loc[gal, "Tile"] = tile
        df_metadata.loc[gal, "Spectrograph"] = spectrograph_B
        df_metadata.loc[gal, "Bundle"] = bundle_B
        df_metadata.loc[gal, "Plate ID"] = plateid_B
        df_metadata.loc[gal, "Blue data cube FITS file"] = str(datacube_B_fname)
        df_metadata.loc[gal, "Red data cube FITS file"] = str(datacube_R_fname)
        df_metadata.loc[gal, "Stellar kinematics FITS file"] = str(stekin_fname)
        df_metadata.loc[gal, "Blue continuum fit FITS file"] = str(cont_fit_B_fname)
        df_metadata.loc[gal, "Red continuum fit FITS file"] = str(cont_fit_R_fname)
        for ncomponents, eline_fit_fname in zip([1, 2, 3, "rec"], eline_fit_fnames): 
            df_metadata.loc[gal, f"{ncomponents}-component fit emission line FITS file"] = str(eline_fit_fname)

    ###########################################################################
    # Save to file
    ###########################################################################
    logger.info(f"saving metadata DataFrame to file {output_path / df_fname}...")
    df_metadata = df_metadata.sort_index()
    df_metadata.to_hdf(output_path / df_fname, key="metadata")
    logger.info(f"finished!")
    return


###############################################################################
def load_hector_metadata_df():
    """Load the Hector metadata DataFrame, containing "metadata" for each galaxy."""
    if not os.path.exists(output_path / "hector_metadata.hd5"):
        raise FileNotFoundError(
            f"File {output_path / 'hector_metadata.hd5'} not found. Did you remember to run make_hector_metadata_df() first?"
        )
    df_metadata = pd.read_hdf(output_path / "hector_metadata.hd5")
    return df_metadata


###############################################################################
def _process_hector(args):
    # Extract input arguments
    gal, ncomponents, df_metadata = args 
    ncomponents = "rec"
    if ncomponents == "rec":
        ncomponents_max = 3
    else:
        ncomponents_max = ncomponents

    z = df_metadata.loc[gal, "z"]
    kpc_per_arcsec = df_metadata.loc[gal, "kpc per arcsec"]
    nx = df_metadata.loc[gal, "N_x"].astype(int)
    ny = df_metadata.loc[gal, "N_y"].astype(int)
    x0_px = df_metadata.loc[gal, "x0_px"].astype(int)
    y0_px = df_metadata.loc[gal, "y0_px"].astype(int)

    #--------------------------------------------------------------------------
    # Filenames
    datacube_B_fname = df_metadata.loc[gal, "Blue data cube FITS file"]
    datacube_R_fname = df_metadata.loc[gal, "Red data cube FITS file"]
    stekin_fname = df_metadata.loc[gal, "Stellar kinematics FITS file"]
    eline_fit_fname = df_metadata.loc[gal, f"{ncomponents}-component fit emission line FITS file"]
    cont_fit_B_fname = df_metadata.loc[gal, "Blue continuum fit FITS file"]
    cont_fit_R_fname = df_metadata.loc[gal, "Red continuum fit FITS file"]
    for file in [stekin_fname, eline_fit_fname, cont_fit_B_fname, cont_fit_R_fname, datacube_B_fname, datacube_R_fname]:
        if not os.path.exists(str(file)):
            raise FileNotFoundError(f"File {file} not found!")

    #--------------------------------------------------------------------------
    # Blue cube & wavelength array
    with fits.open(datacube_B_fname) as hdulist_B_cube:
        header_B = hdulist_B_cube[0].header
        data_cube_B = hdulist_B_cube[0].data
        var_cube_B = hdulist_B_cube[1].data
        # Wavelength information
        lambda_0_A = header_B["CRVAL3"] - header_B["CRPIX3"] * header_B["CDELT3"]
        dlambda_A = header_B["CDELT3"]
        N_lambda = header_B["NAXIS3"]
        lambda_vals_B_A = np.array(range(N_lambda)) * dlambda_A + lambda_0_A

    # Red cube & wavelength array
    with fits.open(datacube_R_fname) as hdulist_R_cube:
        header_R = hdulist_R_cube[0].header
        data_cube_R = hdulist_R_cube[0].data
        var_cube_R = hdulist_R_cube[1].data
        # Wavelength information
        lambda_0_A = header_R["CRVAL3"] - header_R["CRPIX3"] * header_R["CDELT3"]
        dlambda_A = header_R["CDELT3"]
        N_lambda = header_R["NAXIS3"]
        lambda_vals_R_A = np.array(range(N_lambda)) * dlambda_A + lambda_0_A

    # Rest-frame wavelength arrays
    lambda_vals_R_rest_A = lambda_vals_R_A / (1 + z)
    lambda_vals_B_rest_A = lambda_vals_B_A / (1 + z)

    # Get coordinate lists corresponding to non-empty spaxels
    im_empty_B = np.all(np.isnan(data_cube_B), axis=0)
    im_empty_R = np.all(np.isnan(data_cube_R), axis=0)
    im_empty = np.logical_and(im_empty_B, im_empty_R)
    mask = ~im_empty
    y_c_list, x_c_list = np.where(mask)

    #--------------------------------------------------------------------------
    # Construct a dict where the keys are the FINAL column names, and the 
    # values are 2D arrays storing the corresponding quantity
    _2dmap_dict = {}

    #--------------------------------------------------------------------------
    # EMISSION LINES 
    with fits.open(eline_fit_fname) as hdulist_eline_fit:
        # Emission lines: fluxes
        for eline in settings["hector"]["eline_list"]:
            _2dmap_dict[f"{eline} (total)"] = hdulist_eline_fit[eline].data[0]
            _2dmap_dict[f"{eline} error (total)"] = hdulist_eline_fit[eline + "_ERR"].data[0]
            if ncomponents_max > 1:
                for component in range(1, ncomponents_max + 1):
                    _2dmap_dict[f"{eline} (component {component})"] = hdulist_eline_fit[eline].data[component]
                    _2dmap_dict[f"{eline} error (component {component})"] = hdulist_eline_fit[eline + "_ERR"].data[component]
            else:
                _2dmap_dict[f"{eline} (component 1)"] = hdulist_eline_fit[eline].data[0]
                _2dmap_dict[f"{eline} error (component 1)"] = hdulist_eline_fit[eline + "_ERR"].data[0]
        
        # Emission lines: gas kinematics 
        for component in range(1, ncomponents_max + 1):
            _2dmap_dict[f"v_gas (component {component})"] = hdulist_eline_fit["V"].data[component]
            _2dmap_dict[f"v_gas error (component {component})"] = hdulist_eline_fit["V_ERR"].data[component]
            _2dmap_dict[f"sigma_gas (component {component})"] = hdulist_eline_fit["VDISP"].data[component]
            _2dmap_dict[f"sigma_gas error (component {component})"] = hdulist_eline_fit["VDISP_ERR"].data[component]

        # Emission lines: goodness-of-fit  
        _2dmap_dict["chi2 (emission lines)"] = hdulist_eline_fit["CHI2"].data

    #--------------------------------------------------------------------------
    # STELLAR CONTINUUM
    with fits.open(stekin_fname) as hdulist_stekin:
        # Stellar continuum fit: kinematics
        _2dmap_dict[f"v_*"] = hdulist_stekin["V_STAR"].data
        _2dmap_dict[f"v_* error"] = hdulist_stekin["V_STAR_ERR"].data
        _2dmap_dict[f"sigma_*"] = hdulist_stekin["SIG_STAR"].data
        _2dmap_dict[f"sigma_* error"] = hdulist_stekin["SIG_STAR_ERR"].data

        # Stellar continuum fit: goodness-of-fit/continuum S/N
        _2dmap_dict["chi2 (ppxf)"] = hdulist_stekin["CHI2"].data
        _2dmap_dict["Median continuum S/N (ppxf)"] = hdulist_stekin["PRIMARY"].data

    #--------------------------------------------------------------------------
    # CONTINUUM PROPERTIES 
    # Continuum properties: Compute the d4000 Angstrom break 
    d4000_map, d4000_map_err = compute_d4000(
        data_cube=data_cube_B,
        var_cube=var_cube_B,
        lambda_vals_rest_A=lambda_vals_B_rest_A,
        v_star_map=_2dmap_dict["v_*"])
    _2dmap_dict["D4000"] = d4000_map
    _2dmap_dict["D4000 error"] = d4000_map_err

    # Continuum properties: Compute the continuum intensity so that we can compute the Halpha equivalent width.
    cont_HALPHA_map, cont_HALPHA_map_std, cont_HALPHA_map_err = compute_continuum_intensity(
        data_cube=data_cube_R,
        var_cube=var_cube_R,
        lambda_vals_rest_A=lambda_vals_R_rest_A,
        start_A=6500,
        stop_A=6540,
        v_map=_2dmap_dict["v_*"])
    _2dmap_dict["HALPHA continuum"] = cont_HALPHA_map
    _2dmap_dict["HALPHA continuum std. dev."] = cont_HALPHA_map_std
    _2dmap_dict["HALPHA continuum error"] = cont_HALPHA_map_err

    # Continuum properties: Compute the approximate B-band continuum. (from raw cube or fit?)
    cont_B_map, cont_B_map_std, cont_B_map_err = compute_continuum_intensity(
        data_cube=data_cube_B,
        var_cube=var_cube_B,
        lambda_vals_rest_A=lambda_vals_B_rest_A,
        start_A=4000,
        stop_A=5000,
        v_map=_2dmap_dict["v_*"])
    _2dmap_dict[f"B-band continuum"] = cont_B_map
    _2dmap_dict[f"B-band continuum std. dev."] = cont_B_map_std
    _2dmap_dict[f"B-band continuum error"] = cont_B_map_err

    # Compute the continuum intensity so that we can compute the Halpha equivalent width.
    # Continuum wavelength range taken from here: https://ui.adsabs.harvard.edu/abs/2019MNRAS.485.4024V/abstract
    AN_HALPHA_map = compute_measured_HALPHA_amplitude_to_noise(
        data_cube=data_cube_R,
        var_cube=var_cube_R,
        lambda_vals_rest_A=lambda_vals_R_rest_A,
        v_star_map=_2dmap_dict["v_*"],
        v_map=_2dmap_dict["v_gas (component 1)"],
        dv=300)
    _2dmap_dict["HALPHA A/N (measured)"] = AN_HALPHA_map

    # Median value in blue/red clubes 
    for cont_fit_fname, side in zip([cont_fit_B_fname, cont_fit_R_fname], ["blue", "red"]):
        with fits.open(cont_fit_fname) as hdulist_cont_fit:
            _2dmap_dict[f"Median spectral value ({side})"] = hdulist_cont_fit["MED_SPEC"].data

    # Compute v_grad using eqn. 1 of Zhou+2017
    for component in range(1, ncomponents_max + 1):
        _2dmap_dict[f"v_grad (componennt {component})"] = compute_v_grad(_2dmap_dict[f"v_gas (component {component})"])

    #--------------------------------------------------------------------------
    # OTHER QUANTITIES 
    # NOTE: to create the final list of rows, only values at coordinates in x/y_c_list are extracted from the maps in _2dmap_dict.
    ys, xs = np.meshgrid(range(nx), range(ny), indexing="ij")
    _2dmap_dict["x (pixels)"] = xs
    _2dmap_dict["y (pixels)"] = ys
    _2dmap_dict["x (projected, arcsec)"] = xs * settings["hector"]["as_per_px"]
    _2dmap_dict["y (projected, arcsec)"] = ys * settings["hector"]["as_per_px"]
    _2dmap_dict["x (relative to galaxy centre, projected, arcsec)"] = (xs - x0_px) * settings["hector"]["as_per_px"]
    _2dmap_dict["y (relative to galaxy centre, projected, arcsec)"] = (ys - y0_px) * settings["hector"]["as_per_px"]
    _2dmap_dict["r (relative to galaxy centre, projected, arcsec)"] = np.sqrt((ys - y0_px)**2 + (xs - x0_px)**2) * settings["hector"]["as_per_px"]
    _2dmap_dict["x (relative to galaxy centre, projected, kpc)"] = (xs - x0_px) * settings["hector"]["as_per_px"] * kpc_per_arcsec
    _2dmap_dict["y (relative to galaxy centre, projected, kpc)"] = (ys - y0_px) * settings["hector"]["as_per_px"] * kpc_per_arcsec
    _2dmap_dict["r (relative to galaxy centre, projected, kpc)"] = np.sqrt((ys - y0_px)**2 + (xs - x0_px)**2) * settings["hector"]["as_per_px"] * kpc_per_arcsec

    #--------------------------------------------------------------------------
    # Convert 2D maps to 1D rows 
    rows_list = []
    colnames = list(_2dmap_dict.keys())
    for colname in colnames:
        rows = _2d_map_to_1d_list(_2dmap_dict[colname], x_c_list, y_c_list, nx, ny)
        rows_list.append(rows)

    # Add galaxy ID 
    rows_list.append([gal] * len(x_c_list))
    colnames.append("ID")

    # Transpose so that each row represents a single pixel & each column a measured quantity.
    rows_arr = np.array(rows_list).T

    return rows_arr, colnames


def make_hector_df(ncomponents,
                  eline_SNR_min,
                  eline_ANR_min,
                  correct_extinction,
                  gals=None,
                  df_fname=None,
                  sigma_gas_SNR_min=3,
                  line_flux_SNR_cut=True,
                  missing_fluxes_cut=True,
                  missing_kinematics_cut=True,
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
    Make the SAMI DataFrame, where each row represents a single spaxel in a SAMI galaxy.

    DESCRIPTION
    ---------------------------------------------------------------------------
    This function is used to create a Pandas DataFrame containing emission line 
    fluxes & kinematics, stellar kinematics, extinction, star formation rates, 
    and other quantities for individual spaxels in Hector galaxies.

    The output is stored in HDF format as a Pandas DataFrame in which each row 
    corresponds to a given spaxel (or Voronoi/sector bin) for every galaxy. 

    USAGE
    ---------------------------------------------------------------------------
    
        >>> from spaxelsleuth.loaddata.hector import make_hector_df()
        >>> make_hector_df(ncomponents="1", bin_type="default", 
                         eline_SNR_min=5, eline_ANR_min=3, correct_extinction=True)

    will create a DataFrame using the data products from 1-component Gaussian 
    fits to the unbinned datacubes, and will adopt minimum S/N and A/N 
    thresholds of 5 and 3 respectively to mask out unreliable emission line 
    fluxes and associated quantities. sigma_inst_kms refers to the Gaussian 
    sigma of the instrumental line function in km/s.    

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

    bin_type:                   str
        Spatial binning strategy. Options are "default" (unbinned), "adaptive"
        (Voronoi binning) or "sectors" (sector binning)

    eline_SNR_min:              int 
        Minimum emission line flux S/N to adopt when making S/N and data 
        quality cuts.

    eline_ANR_min:          float
        Minimum A/N to adopt for emission lines in each kinematic component,
        defined as the Gaussian amplitude divided by the continuum standard
        deviation in a nearby wavelength range.

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

    missing_kinematics_cut: bool
        Whether to NaN out "missing" values for v_gas/sigma_gas/v_*/sigma_* - 
        i.e., cells in which the measurement itself is NaN, but the error 
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
        List of emission lines to use. Defaults to the full list of lines fitted,
        which are specified in the config file.

    nthreads:                   int (optional)           
        Maximum number of threads to use. Defaults to os.cpu_count().

    debug:                      bool (optional)
        NOTE: not yet implemented.
        If True, run on a subset of the entire sample (10 galaxies) and save
        the output with "_DEBUG" appended to the filename. This is useful for
        tweaking S/N and DQ cuts since running the function on the entire 
        sample is quite slow.  Default: False.

    OUTPUTS
    ---------------------------------------------------------------------------
    The resulting DataFrame will be stored as 

        settings["hector"]["output_path"]/hector_{bin_type}_{ncomponents}-comp_extcorr_minSNR={eline_SNR_min}_minANR={eline_ANR_min}.hd5

    if correct_extinction is True, or else

        settings["hector"]["output_path"]/hector_{bin_type}_{ncomponents}-comp_minSNR={eline_SNR_min}_minANR={eline_ANR_min}.hd5

    PREREQUISITES
    ---------------------------------------------------------------------------
    make_hector_metadata_df() must be run first.

    Hector data products must be stored in the folders specified in the config
    file.
    """
    ###############################################################################
    # input checking
    ###############################################################################
    if df_fname is not None:
        if not df_fname.endswith(".hd5"):
            df_fname += ".hd5"
    else:
        # Input file name
        df_fname = f"hector_{ncomponents}-comp"
        if correct_extinction:
            df_fname += "_extcorr"
        df_fname += f"_minSNR={eline_SNR_min}_minANR={eline_ANR_min}.hd5"

    if (type(ncomponents) not in [int, str]) or (type(ncomponents) == str
                                                 and ncomponents != "rec"):
        raise ValueError("ncomponents must be either an integer or 'rec'!")

    logger.info(f"input parameters: ncomponents={ncomponents}, eline_SNR_min={eline_SNR_min}, eline_ANR_min={eline_ANR_min}, correct_extinction={correct_extinction}")

    # Determine number of threads
    if nthreads is None:
        nthreads = os.cpu_count()
        logger.warning(f"nthreads not specified: running make_hector_df() on {nthreads} threads...")

    ###############################################################################
    # Load metadata DataFrame to get list of galaxies & associated fields and tiles
    ###############################################################################
    df_metadata = load_hector_metadata_df()
    if gals is None:
        gals = df_metadata.index.values

    ###############################################################################
    # Scrape measurements for each galaxy from FITS files
    ###############################################################################
    args_list = [[gal, ncomponents, df_metadata] for gal in gals]
    if len(gals) == 1:
        res_list = [_process_hector(args_list[0])]
    else:
        if nthreads > 1:
            logger.info(f"beginning pool...")
            pool = multiprocessing.Pool(min([nthreads, len(gals)]))
            res_list = pool.map(_process_hector, args_list)
            pool.close()
            pool.join()
        else:
            logger.info(f"running sequentially...")
            res_list = []
            for args in args_list:
                res = _process_hector(args)
                res_list.append(res)

    ###############################################################################
    # Convert to a Pandas DataFrame
    ###############################################################################
    rows_list_all = [r[0] for r in res_list]
    colnames = res_list[0][1]
    df_spaxels = pd.DataFrame(np.vstack(tuple(rows_list_all)), columns=colnames)

    # Merge with sub-columns of df_metadata
    cols_to_merge = [
        "RA (J2000)",
        "Dec (J2000)",
        "z",
        "D_A (Mpc)",
        "D_L (Mpc)",
        "N_x",
        "N_y",
        "x0_px",
        "y0_px",
        "Galaxy centre x0_px (projected, arcsec)",
        "Galaxy centre y0_px (projected, arcsec)",
        "kpc per arcsec",
    ]
    df_spaxels = df_spaxels.merge(df_metadata.loc[:, cols_to_merge].reset_index(), how="left", on="ID")

    ###############################################################################
    # Generic stuff: compute additional columns - extinction, metallicity, etc.
    ###############################################################################
    df_spaxels = add_columns(
        df_spaxels.copy(),
        eline_SNR_min=eline_SNR_min,
        eline_ANR_min=eline_ANR_min,
        sigma_gas_SNR_min=sigma_gas_SNR_min,
        eline_list=settings["hector"]["eline_list"],
        line_flux_SNR_cut=line_flux_SNR_cut,
        missing_fluxes_cut=missing_fluxes_cut,
        missing_kinematics_cut=missing_kinematics_cut,
        line_amplitude_SNR_cut=line_amplitude_SNR_cut,
        flux_fraction_cut=flux_fraction_cut,
        sigma_gas_SNR_cut=sigma_gas_SNR_cut,
        vgrad_cut=vgrad_cut,
        stekin_cut=False,
        correct_extinction=correct_extinction,
        metallicity_diagnostics=metallicity_diagnostics,
        compute_sfr=True,
        sigma_inst_kms=settings["hector"]["sigma_inst_kms"],  # TODO make this flexible 
        nthreads=nthreads,
        base_missing_flux_components_on_HALPHA=False,  # NOTE: this is important!!
        )

    ###############################################################################
    # Save to file
    ###############################################################################
    logger.info(f"saving to file {output_path / df_fname}...")
    df_spaxels.to_hdf(output_path / df_fname, key="hector")
    logger.info(f"finished!")

    return

def load_hector_df(ncomponents,
                  eline_SNR_min,
                  eline_ANR_min,
                  correct_extinction,
                  df_fname=None,
                  key=None):
    """
    DESCRIPTION
    ---------------------------------------------------------------------------
    Load and return the Pandas DataFrame containing spaxel-by-spaxel 
    information for all Hector galaxies which was created using make_hector_df().

    INPUTS
    ---------------------------------------------------------------------------
    ncomponents:        int or str
        Number of components; may either be 1, 2, 3... N where N is the number 
        of Gaussian components fitted to the emission lines or "rec" 
        (corresponding to the "recommendend"-component Gaussian fits where 
        the number of components in each spaxel is determined using the 
        Bayesian Information Criterion.

    eline_SNR_min:      int 
        Minimum flux S/N to accept. Fluxes below the threshold (plus associated
        data products) are set to NaN.

    eline_ANR_min:      float
        Minimum A/N to adopt for emission lines in each kinematic component,
        defined as the Gaussian amplitude divided by the continuum standard
        deviation in a nearby wavelength range.
   
    correct_extinction: bool
        If True, load the DataFrame in which the emission line fluxes (but not 
        EWs) have been corrected for intrinsic extinction.

    df_fname:           str
        (Optional) If specified, load DataFerame from file <df_fname>.hd5. 
        Otherwise, the DataFrame filename is automatically determined using 
        the other input arguments.

    key:                str
        (Optional) key used to read the HDF file if df_fname is specified.
    
    USAGE
    ---------------------------------------------------------------------------
    load_hector_df() is called as follows:

        >>> from spaxelsleuth.loaddata.hector import load_hector_df
        >>> df = load_hector_df(ncomponents, eline_SNR_min, eline_ANR_min, 
                               correct_extinction)

    OUTPUTS
    ---------------------------------------------------------------------------
    The Dataframe.
    """

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
        df_fname = f"hector_{ncomponents}-comp"
        if correct_extinction:
            df_fname += "_extcorr"
        df_fname += f"_minSNR={eline_SNR_min}_minANR={eline_ANR_min}.hd5"

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
    df["survey"] = "hector"
    df["ncomponents"] = ncomponents
    df["as_per_px"] = settings["hector"]["as_per_px"]
    df["bin_type"] = "default"
    df["flux units"] = "E-16 erg/cm^2/s"  # TODO check! Units of continuum & emission line flux
    df["continuum units"] = "E-16 erg/cm^2/Å/s"  # TODO check! Units of continuum & emission line flux

    # Add back in object-type columns
    df["BPT (total)"] = bpt_num_to_str(df["BPT (numeric) (total)"])

    # Return
    logger.info("finished!")
    return df.sort_index()


if __name__ == "__main__":

    """
    Place to test the functions in this submodule.
    """
    import matplotlib.pyplot as plt 

    from spaxelsleuth import load_user_config
    load_user_config("/Users/u5708159/Desktop/spaxelsleuth_test/.myconfig.json")
    from spaxelsleuth.plotting.plot2dmap import plot2dmap

    from .. import make_hector_metadata_df, make_hector_df, load_hector_df, load_hector_metadata_df

    ncomponents = "rec"
    nthreads = 3
    eline_SNR_min=0
    eline_ANR_min=0
    correct_extinction=True

    make_hector_metadata_df()

    make_hector_df(ncomponents=ncomponents, 
                   eline_SNR_min=eline_SNR_min, 
                   eline_ANR_min=eline_ANR_min, 
                   correct_extinction=correct_extinction, 
                   sigma_inst_kms=30, 
                   metallicity_diagnostics=["N2Ha_K19", "N2S2Ha_D16"],
                   nthreads=nthreads)
    
    df_spaxels = load_hector_df(ncomponents=ncomponents, 
                                eline_SNR_min=eline_SNR_min, 
                                eline_ANR_min=eline_ANR_min, 
                                correct_extinction=correct_extinction)

    for gal in df_spaxels["ID"].unique():
        fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(8, 8))
        fig.subplots_adjust(wspace=0.4, hspace=0.4, top=0.95, bottom=0.05)
        plot2dmap(df_spaxels, gal=gal, col_z="HALPHA (total)", vmin=0, vmax="auto", ax=axs.flat[0])
        plot2dmap(df_spaxels, gal=gal, col_z="HALPHA EW (total)", ax=axs.flat[1])
        plot2dmap(df_spaxels, gal=gal, col_z="log(O/H) + 12 (N2Ha_K19/O3O2_K19) (total)", ax=axs.flat[2])
        plot2dmap(df_spaxels, gal=gal, col_z="BPT (total)", ax=axs.flat[3])


