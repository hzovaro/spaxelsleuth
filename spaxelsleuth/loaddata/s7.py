"""
File:       s7.py
Author:     Henry Zovaro
Email:      henry.zovaro@anu.edu.au

DESCRIPTION
------------------------------------------------------------------------------
This script contains the following functions:

    make_s7_metadata_df():
        This function is used to create a DataFrame containing "metadata", including
        stellar masses, spectroscopic redshifts, morphologies and other information
        for each galaxy in s7. In addition to the provided values in the input
        catalogues, the angular scale (in kpc per arcsecond) and inclination are 
        computed for each galaxy. 

        This script must be run before make_s7_df() as the resulting DataFrame
        is used there.
    
    make_s7_df():     
        This function is used to create a Pandas DataFrame containing emission line 
        fluxes & kinematics, stellar kinematics, extinction, star formation rates, 
        and other quantities for individual spaxels in s7 galaxies as taken from 
        S7 DR2.

        The output is stored in HDF format as a Pandas DataFrame in which each row 
        corresponds to a given spaxel (or Voronoi bin) for every galaxy. 

    load_s7_df():
        load a DataFrame containing emission line fluxes, etc. that was created 
        using make_s7_df().

PREREQUISITES
------------------------------------------------------------------------------
S7_DIR must be defined as an environment variable.

See function docstrings for specific prerequisites.

------------------------------------------------------------------------------
Copyright (C) 2022 Henry Zovaro
"""
###############################################################################
# Imports
import os, inspect
import pandas as pd
import numpy as np
from itertools import product
from scipy import constants
from astropy.cosmology import FlatLambdaCDM
from astropy.io import fits
from astropy import units as u
from astropy.coordinates import SkyCoord

from spaxelsleuth.utils import linefns, dqcut, metallicity, extcorr

from IPython.core.debugger import Tracer

###############################################################################
# Paths
s7_data_path = os.environ["S7_DIR"]
assert "S7_DIR" in os.environ, "Environment variable S7_DIR is not defined!"

###############################################################################
def make_s7_metadata_df():
    """
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

        S7_DIR/s7_metadata.hd5

    PREREQUISITES
    ---------------------------------------------------------------------------
    S7_DIR must be defined as an environment variable.

    The table containing metadata for S7 galaxies is required for this script. 
    This has been included in the ../data/ directory but can be downloaded in 
    CSV format from 
        
        https://miocene.anu.edu.au/S7/Data_release_2/S7_DR2_Table_2_Catalogue.csv
    
    """
    print("In make_s7_metadata_df(): Creating metadata DataFrame...")
    ###############################################################################
    # Filenames
    df_metadata_fname = "S7_DR2_Table_2_Catalogue.csv"
    df_fname = "s7_metadata.hd5"

    ###############################################################################
    # READ IN THE METADATA
    ###############################################################################
    data_path = os.path.join(__file__.split("loaddata")[0], "data")
    assert os.path.exists(os.path.join(data_path, df_metadata_fname)),\
        f"File {os.path.join(data_path, df_metadata_fname)} not found!"
    df_metadata = pd.read_csv(os.path.join(data_path, df_metadata_fname), skiprows=58)
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
        "S7_log M_*": "log M_*",
        "S7_log M_*_err": "log M_* error",
        "S7_Sy1_subtraction?": "Sy1 subtraction?",
        "S7_mosaic?": "Mosaic?",
        "S7_BPT_classification": "BPT (global)",
        "S7_z": "z",
        "S7_nucleus_index_x": "x0 (pixels)",
        "S7_nucleus_index_y": "y0 (pixels)"
    }
    df_metadata = df_metadata.rename(columns=rename_dict)
    df_metadata = df_metadata.set_index(df_metadata["ID"])

    # Get rid of unneeded columns
    good_cols = [rename_dict[k] for k in rename_dict.keys()] + ["RA (J2000)", "Dec (J2000)"]
    df_metadata = df_metadata[good_cols]

    ###############################################################################
    # Add angular scale info
    ###############################################################################
    print(f"In make_s7_metadata_df(): Computing distances...")
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

    ###############################################################################
    # Save to file
    ###############################################################################
    print(f"In make_s7_metadata_df(): Saving metadata DataFrame to file {os.path.join(s7_data_path, df_fname)}...")
    df_metadata.to_hdf(os.path.join(s7_data_path, df_fname), key="metadata")

    print(f"In make_s7_metadata_df(): Finished!")
    return

###############################################################################
def make_s7_df(bin_type="default", ncomponents="recom", 
               line_flux_SNR_cut=True, eline_SNR_min=5,
               vgrad_cut=False ,
               sigma_gas_SNR_cut=True, sigma_gas_SNR_min=3,
               line_amplitude_SNR_cut=True,
               flux_fraction_cut=False,
               stekin_cut=True,
               met_diagnostic_list=["Dopita+2016", "N2O2"], logU = -3.0,
               eline_list=["HALPHA", "HBETA", "NII6583", "OI6300", "OII3726+OII3729", "OIII5007", "SII6716", "SII6731"],
               nthreads_max=20, debug=False):
    """
    DESCRIPTION
    ---------------------------------------------------------------------------
    This function is used to create a Pandas DataFrame containing emission line 
    fluxes & kinematics, stellar kinematics, extinction, star formation rates, 
    and other quantities for individual spaxels in S7 galaxies as taken from 
    S7 DR2.

    The output is stored in HDF format as a Pandas DataFrame in which each row 
    corresponds to a given spaxel (or Voronoi bin) for every galaxy. 

    USAGE
    ---------------------------------------------------------------------------
    
        >>> from spaxelsleuth.loaddata.s7 import make_s7_df()
        >>> make_s7_df(ncomponents="recom", bin_type="default", eline_SNR_min=5)

    will create a DataFrame comprising the multi-component Gaussian fits to the 
    unbinned datacubes, and will use a minimum S/N threshold of 5 to mask out 
    unreliable emission line fluxes and associated quantities.

    Other input arguments may be set in the script to control other aspects
    of the data quality and S/N cuts made, however the default values can be
    left as-is.

    Note that due to the relatively small sample size, multithreading is NOT
    used to process the galaxies, unlike in the equivalent function in sami.py
    (make_df_sami()). Here, the nthreads_max only controls the number of 
    threads used during the extinction computation.

    If you wish to run in debug mode, set the DEBUG flag to True: this will run 
    the script on a subset (by default 10) galaxies to speed up execution. 

    INPUTS
    ---------------------------------------------------------------------------
    ncomponents:        str
        Which number of Gaussian components to assume. Must be "recom" (the
        recommended multi-component fits).

    bin_type:           str
        Spatial binning strategy. Must be "default" (unbinned) for now.

    eline_SNR_min:      int 
        Minimum emission line flux S/N to assume.

    line_flux_SNR_cut:          bool
        If True, make a S/N cut on all emission line components and associated
        quantities with flux S/N below eline_SNR_min.

    vgrad_cut:                  bool         
        If True, mask component kinematics (velocity and velocity dispersion)
        that are likely to be affected by beam smearing.
        By default this is set to False b/c it tends to remove nuclear spaxels 
        which may be of interest to your science case, & because it doesn't 
        reliably remove spaxels with quite large beam smearing components

    sigma_gas_SNR_cut:          bool
        If True, mask component velocity dispersions where the S/N on the 
        velocity dispersion measurement is below sigma_gas_SNR_min. 
        By default this is set to True b/c it's a robust way to account for 
        emission line widths < instrumental.

    sigma_gas_SNR_min:          int
        Minimum velocity dipersion S/N to accept.

    line_amplitude_SNR_cut:     bool
        If True, removes components with Gaussian amplitudes < 3 * RMS of the 
        continuum in the vicinity of Halpha. By default this is set to True
        because this does well at removing shallow components which are most 
        likely due to errors in the stellar continuum fit.

    flux_fraction_cut:          bool
        If True, and if ncomponents > 1, remove intermediate and broad 
        components with line amplitudes < 0.05 that of the narrow componet.
        Set to False by default b/c it's unclear whether this is needed to 
        reject unreliable components.

    stekin_cut:
        If True, mask stellar kinematic quantities that do not meet the DQ and 
        S/N requirements specified in Croom et al. (2021). True by default.

    eline_list:                 list of str
        Default SAMI emission lines - don't change this!

    met_diagnostic_list:        list of str
        Which metallicity diagnostics to compute. Good options are "Dopita+2016"
        and "N2O2".

    logU:                       float            
        Constant ionisation parameter to assume in metallicity calculation.
        Default value is -3.0.

    nthreads_max:               int            
        Maximum number of threds to use. 

    debug:                      bool
        If True, run on a subset of the entire sample (10 galaxies) and save
        the output with "_DEBUG" appended to the filename. This is useful for
        tweaking S/N and DQ cuts since running the function on the entire 
        sample is quite slow.

    OUTPUTS
    ---------------------------------------------------------------------------
    Each time the function is run, TWO DataFrames are produced - with and without 
    extinction correction applied to the emission line fluxes. 

    The resulting DataFrame will be stored as 

        S7_DIR/s7_spaxels_{bin_type}_{ncomponents}-comp_extcorr_minSNR={eline_SNR_min}.hd5
    and S7_DIR/s7_spaxels_{bin_type}_{ncomponents}-comp_minSNR={eline_SNR_min}.hd5

    The DataFrames will be stored in CSV format in case saving in HDF format fails
    for any reason.

    Note that the Halpha equivalent widths are NOT corrected for extinction in 
    either case. This is because stellar continuum extinction measurements are 
    not available, and so applying the correction only to the Halpha fluxes may 
    over-estimate the true EW.

    PREREQUISITES
    ---------------------------------------------------------------------------
    S7_DIR must be defined as an environment variable.

    make_s7_metadata_df() must be run first.

    SAMI data products must be downloaded from DataCentral

        https://datacentral.org.au/services/download/

    or from Miocene 

        https://miocene.anu.edu.au/S7/Data_release_2/

    The red and blue data cubes must also be downloaded and stored as follows: 

        S7_DIR/0_Cubes/<gal>_R.fits
        S7_DIR/0_Cubes/<gal>_B.fits

    And the data products (including emission line fluxes, kinematics, etc.) 
    for each galaxy must be downloaded and stored as follows:

        S7_DIR/2_Post-processed_mergecomps/<gal>_best_components.fits

    """
    #######################################################################
    # INPUT CHECKING
    #######################################################################
    assert ncomponents == "recom", "ncomponents must be 'recom'!!"
    assert bin_type == "default", "bin_type must be 'default'!!"

    # For printing to stdout
    status_str = f"In s7.make_s7_df() [bin_type={bin_type}, ncomponents={ncomponents}, debug={debug}, eline_SNR_min={eline_SNR_min}]"

    ###############################################################################
    # FILENAMES
    #######################################################################
    df_metadata_fname = "s7_metadata.hd5"

    # Output file names
    df_fname = f"s7_spaxels_{bin_type}_{ncomponents}-comp_minSNR={eline_SNR_min}"
    df_fname_extcorr = f"s7_spaxels_{bin_type}_{ncomponents}-comp_extcorr_minSNR={eline_SNR_min}"
    if debug:
        df_fname += "_DEBUG"
        df_fname_extcorr += "_DEBUG"
    df_fname += ".hd5"
    df_fname_extcorr += ".hd5"

    print(f"{status_str}: saving to files {df_fname} and {df_fname_extcorr}...")

    ###############################################################################
    # READ IN THE METADATA
    ###############################################################################
    try:
        df_metadata = pd.read_hdf(os.path.join(s7_data_path, df_metadata_fname), key="metadata")
    except FileNotFoundError:
        print(f"ERROR: metadata DataFrame file not found ({os.path.join(sami_data_path, s7_data_path)}). Please run make_sami_metadata_df.py first!")

    gal_ids_dq_cut = df_metadata[df_metadata["Good?"] == True].index.values
    if debug: 
        gal_ids_dq_cut = gal_ids_dq_cut[:10]
    df_metadata["Good?"] = df_metadata["Good?"].astype("float")

    ###############################################################################
    # PROCESS GALAXIES SEQUENTIALLY
    ###############################################################################
    df_spaxels = pd.DataFrame()
    for gal in gal_ids_dq_cut:
        hdulist_processed_cube = fits.open(os.path.join(s7_data_path, "2_Post-processed_mergecomps", f"{gal}_best_components.fits"))
        hdulist_R_cube = fits.open(os.path.join(s7_data_path, "0_Cubes", f"{gal}_R.fits"))
        hdulist_B_cube = fits.open(os.path.join(s7_data_path, "0_Cubes", f"{gal}_B.fits"))

        # Other quantities: redshift, etc.
        z = hdulist_processed_cube[0].header["Z"]

        ###############################################################################
        # Calculate Halpha EW

        # Open the fitted continuum
        cont_cube = hdulist_processed_cube["R_CONTINUUM"].data
        line_cube = hdulist_processed_cube["R_LINE"].data

        # Open the original red cube to calculate the continuum intensity
        # Units of 10**(-16) erg /s /cm**2 /angstrom /pixel
        # Continuum wavelength range taken from here: https://ui.adsabs.harvard.edu/abs/2019MNRAS.485.4024V/abstract
        header = hdulist_R_cube[0].header
        data_cube_R = hdulist_R_cube[0].data 
        var_cube_R = hdulist_R_cube[1].data  

        # Wavelength axis values
        lambda_vals_A = np.array(range(header["NAXIS3"])) * header["CDELT3"] + header["CRVAL3"] 

        # Calculate the start & stop indices of the wavelength range
        start_idx = np.nanargmin(np.abs(lambda_vals_A / (1 + z) - 6500))
        stop_idx = np.nanargmin(np.abs(lambda_vals_A / (1 + z) - 6540))

        # Make a 2D map of the continuum intensity
        cont_HALPHA_map = np.nanmean(data_cube_R[start_idx:stop_idx], axis=0)
        cont_HALPHA_map_std = np.nanstd(data_cube_R[start_idx:stop_idx], axis=0)
        cont_HALPHA_map_err = 1 / (stop_idx - start_idx) * np.sqrt(np.nansum(var_cube_R[start_idx:stop_idx], axis=0))
        hdulist_R_cube.close() 

        #######################################################################
        # Use the blue cube to calculate the approximate B-band continuum.
        # Units of 10**(-16) erg /s /cm**2 /angstrom /pixel
        header = hdulist_B_cube[0].header
        data_cube_B = hdulist_B_cube[0].data 
        var_cube_B = hdulist_B_cube[1].data  

        # Wavelength values
        lambda_vals_A = np.array(range(header["NAXIS3"])) * header["CDELT3"] + header["CRVAL3"] 

        # Compute continuum intensity
        start_idx = np.nanargmin(np.abs(lambda_vals_A / (1 + z) - 4000))
        stop_idx = np.nanargmin(np.abs(lambda_vals_A / (1 + z) - 5000))

        # Make a 2D map of the continuum intensity
        cont_B_map = np.nanmean(data_cube_B[start_idx:stop_idx], axis=0)
        cont_B_map_std = np.nanstd(data_cube_B[start_idx:stop_idx], axis=0)
        cont_B_map_err = 1 / (stop_idx - start_idx) * np.sqrt(np.nansum(var_cube_B[start_idx:stop_idx], axis=0))
        hdulist_B_cube.close() 

        #######################################################################
        # Compute the d4000 Angstrom break.
        header = hdulist_B_cube[0].header
        data_cube_B = hdulist_B_cube[0].data
        var_cube_B = hdulist_B_cube[1].data
        hdulist_B_cube.close()

        # Wavelength values
        lambda_vals_A = np.array(range(header["NAXIS3"])) * header["CDELT3"] + header["CRVAL3"] 

        # Compute the D4000Å break
        # Definition from Balogh+1999 (see here: https://arxiv.org/pdf/1611.07050.pdf, page 3)
        start_b_idx = np.nanargmin(np.abs(lambda_vals_A / (1 + z) - 3850))
        stop_b_idx = np.nanargmin(np.abs(lambda_vals_A / (1 + z) - 3950))
        start_r_idx = np.nanargmin(np.abs(lambda_vals_A / (1 + z) - 4000))
        stop_r_idx = np.nanargmin(np.abs(lambda_vals_A / (1 + z) - 4100))
        N_b = stop_b_idx - start_b_idx
        N_r = stop_r_idx - start_r_idx

        # Convert datacube & variance cubes to units of F_nu
        data_cube_B_Hz = data_cube_B * lambda_vals_A[:, None, None]**2 / (constants.c * 1e10)
        var_cube_B_Hz2 = var_cube_B * (lambda_vals_A[:, None, None]**2 / (constants.c * 1e10))**2

        num = np.nanmean(data_cube_B_Hz[start_r_idx:stop_r_idx], axis=0)
        denom = np.nanmean(data_cube_B_Hz[start_b_idx:stop_b_idx], axis=0)
        err_num = 1 / N_r * np.sqrt(np.nansum(var_cube_B_Hz2[start_r_idx:stop_r_idx], axis=0))
        err_denom = 1 / N_b * np.sqrt(np.nansum(var_cube_B_Hz2[start_b_idx:stop_b_idx], axis=0))
        
        d4000_map = num / denom
        d4000_map_err = d4000_map * np.sqrt((err_num / num)**2 + (err_denom / denom)**2)

        ###############################################################################
        # # Emission line strengths
        flux_dict = {}
        flux_err_dict = {}
        ext_names = [hdu.name for hdu in hdulist_processed_cube]
        eline_list = ["HALPHA", "HBETA", "NII6583", "OI6300", "OII3726", "OII3729", "OIII5007", "SII6716", "SII6731"]
        for eline in eline_list:
            if eline in ext_names:
                flux_dict[eline] = hdulist_processed_cube[f"{eline}"].data
                flux_err_dict[eline] = hdulist_processed_cube[f"{eline}_ERR"].data

        ###############################################################################
        # Gas & stellar kinematics
        vdisp_map = hdulist_processed_cube["VDISP"].data
        vdisp_err_map = hdulist_processed_cube["VDISP_ERR"].data
        stellar_vdisp_map = hdulist_processed_cube["STAR_VDISP"].data[0]
        stellar_vdisp_err_map = hdulist_processed_cube["STAR_VDISP"].data[1]
        v_map = hdulist_processed_cube["V"].data
        v_err_map = hdulist_processed_cube["V_ERR"].data
        stellar_v_map = hdulist_processed_cube["STAR_V"].data[0]
        stellar_v_err_map = hdulist_processed_cube["STAR_V"].data[1]
        n_y, n_x = stellar_v_map.shape 

        ###############################################################################
        # Compute v_grad using eqn. 1 of Zhou+2017
        v_grad_map = np.full_like(v_map, np.nan)

        # Compute v_grad for each spaxel in each component
        # in units of km/s/pixel
        for yy, xx in product(range(1, v_map.shape[1] - 1), range(1, v_map.shape[2] - 1)):
            v_grad_map[:, yy, xx] = np.sqrt(((v_map[:, yy, xx + 1] - v_map[:, yy, xx - 1]) / 2)**2 +\
                                            ((v_map[:, yy + 1, xx] - v_map[:, yy - 1, xx]) / 2)**2)

        ###############################################################################
        # Make a radius map
        radius_map = np.zeros((n_y, n_x))
        x_0 = df_metadata.loc[df_metadata["ID"] == gal, "x0 (pixels)"].values[0]
        y_0 = df_metadata.loc[df_metadata["ID"] == gal, "y0 (pixels)"].values[0]
        try:
            i_rad = np.deg2rad(float(df_metadata.loc[df_metadata["ID"] == gal, "i (degrees)"].values[0]))
        except:
            i_rad = 0  # Assume face-on if inclination isn't defined
        try:
            PA_deg = float(df_metadata.loc[df_metadata["ID"] == gal, "PA (degrees)"].values[0])
        except:
            PA_deg = 0  # Assume NE if inclination isn't defined
        PA_obs_deg = float(df_metadata.loc[df_metadata["ID"] == gal, "WiFeS PA"].values[0])
        beta_rad = np.deg2rad(PA_deg - 90 - PA_obs_deg)
        for xx, yy in product(range(n_x), range(n_y)):
            # De-shift, de-rotate & de-incline
            x_cc = xx - x_0
            y_cc = yy - y_0
            x_prime = x_cc * np.cos(beta_rad) + y_cc * np.sin(beta_rad)
            y_prime_projec = (- x_cc * np.sin(beta_rad) + y_cc * np.cos(beta_rad))
            y_prime = (- x_cc * np.sin(beta_rad) + y_cc * np.cos(beta_rad)) / np.cos(i_rad)
            r_prime = np.sqrt(x_prime**2 + y_prime**2)
            radius_map[yy, xx] = r_prime

        ###############################################################################
        # Store in DataFrame
        rows_list = []
        for xx, yy in product(range(n_x), range(n_y)):
            thisrow = {}
            thisrow["x (projected, arcsec)"] = xx 
            thisrow["y (projected, arcsec)"] = yy
            thisrow["r (relative to galaxy centre, deprojected, arcsec)"] = radius_map[yy, xx]
            thisrow["HALPHA continuum"] = cont_HALPHA_map[yy, xx] * 1e16
            thisrow["HALPHA continuum std. dev."] = cont_HALPHA_map_std[yy, xx] * 1e16
            thisrow["HALPHA continuum error"] = cont_HALPHA_map_err[yy, xx] * 1e16
            thisrow["B-band continuum"] = cont_B_map[yy, xx] * 1e16
            thisrow["B-band continuum std. dev."] = cont_B_map_std[yy, xx] * 1e16
            thisrow["B-band continuum error"] = cont_B_map_err[yy, xx] * 1e16
            thisrow["D4000"] = d4000_map[yy, xx]
            thisrow["D4000 error"] = d4000_map_err[yy, xx]
            # thisrow[f"A_V (total)"] = A_V_map[yy, xx]
            # thisrow[f"A_V error (total)"] = A_V_map_err[yy, xx]
            # thisrow[f"A_V error (total)"] = A_V_map_err[yy, xx]

            for nn, component_str in enumerate(["total", "component 1", "component 2", "component 3"]):

                # Add OII doublet flux 
                for eline in ["OII3726", "OII3729"]:
                    if eline in flux_dict.keys() and component_str == "total":
                        thisrow[f"{eline} ({component_str})"] = flux_dict[eline][yy, xx] if flux_dict[eline][yy, xx] > 0 else np.nan
                        thisrow[f"{eline} error ({component_str})"] = flux_err_dict[eline][yy, xx] if flux_dict[eline][yy, xx] > 0 else np.nan
                        thisrow[f"{eline} SNR ({component_str})"] = flux_dict[eline][yy, xx] / flux_err_dict[eline][yy, xx] if (flux_dict[eline][yy, xx] > 0) and (flux_err_dict[eline][yy, xx] > 0) else np.nan

                # emission line fluxes
                for eline in [e for e in eline_list if not e.startswith("OII372") and e in flux_dict.keys()]:
                    thisrow[f"{eline} ({component_str})"] = flux_dict[eline][nn, yy, xx] if flux_dict[eline][nn, yy, xx] > 0 else np.nan
                    thisrow[f"{eline} error ({component_str})"] = flux_err_dict[eline][nn, yy, xx] if flux_dict[eline][nn, yy, xx] > 0 else np.nan
                    thisrow[f"{eline} SNR ({component_str})"] = flux_dict[eline][nn, yy, xx] / flux_err_dict[eline][nn, yy, xx] if (flux_dict[eline][nn, yy, xx] > 0) and (flux_err_dict[eline][nn, yy, xx] > 0) else np.nan

                # Add gas & stellar kinematics
                if component_str == "total":
                    # Then use the maximum velocity dispersion among all components.
                    try:
                        max_idx = np.nanargmax(vdisp_map[:, yy, xx], axis=0)
                        vdisp = vdisp_map[max_idx, yy, xx]
                        vdisp_err = vdisp_err_map[max_idx, yy, xx]
                        v = v_map[max_idx, yy, xx]
                        v_err = v_err_map[max_idx, yy, xx]
                        v_grad = v_grad_map[max_idx, yy, xx]
                    except ValueError as e:
                        vdisp = np.nan
                        vdisp_err = np.nan
                        v = np.nan
                        v_err = np.nan
                        v_grad = np.nan
                    thisrow[f"sigma_gas ({component_str})"] = vdisp
                    thisrow[f"sigma_gas error ({component_str})"] = vdisp_err
                    thisrow[f"v_gas ({component_str})"] = v
                    thisrow[f"v_gas error ({component_str})"] = v_err
                    thisrow[f"v_grad ({component_str})"] = v_grad
                else:
                    thisrow[f"sigma_gas ({component_str})"] = vdisp_map[nn, yy, xx]
                    thisrow[f"sigma_gas error ({component_str})"] = vdisp_err_map[nn, yy, xx]
                    thisrow[f"v_gas ({component_str})"] = v_map[nn, yy, xx]
                    thisrow[f"v_gas error ({component_str})"] = v_err_map[nn, yy, xx]
                    thisrow[f"v_grad ({component_str})"] = v_grad_map[nn, yy, xx]

                # Stellar kinematics
                thisrow["sigma_*"] = stellar_vdisp_map[yy, xx]
                thisrow["sigma_* error"] = stellar_vdisp_err_map[yy, xx]
                thisrow["v_*"] = stellar_v_map[yy, xx]
                thisrow["v_* error"] = stellar_v_err_map[yy, xx]

            # Append these rows to the rows list
            rows_list.append(thisrow)

        # Append to the "master" data frane
        df_gal = pd.DataFrame(rows_list)
        df_gal["ID"] = gal
        df_spaxels = df_spaxels.append(df_gal)

    ###############################################################################
    # Reset index, because at this point the index is multiply-valued!
    ###############################################################################
    df_spaxels = df_spaxels.reset_index()

    ###############################################################################
    # Merge with metadata
    ###############################################################################
    df_spaxels = df_spaxels.merge(df_metadata, left_on="ID", right_index=True)

    ###############################################################################
    # Compute the ORIGINAL number of components
    ###############################################################################
    df_spaxels["Number of components (original)"] =\
        (~df_spaxels["sigma_gas (component 1)"].isna()).astype(int) +\
        (~df_spaxels["sigma_gas (component 2)"].isna()).astype(int) +\
        (~df_spaxels["sigma_gas (component 3)"].isna()).astype(int)

    ###############################################################################
    # Calculate equivalent widths
    ###############################################################################
    for col in ["HALPHA continuum", "HALPHA continuum error"]:
        df_spaxels[col] = pd.to_numeric(df_spaxels[col])

    df_spaxels.loc[df_spaxels["HALPHA continuum"] < 0, "HALPHA continuum"] = 0
    for nn in range(3):
        # Cast to float
        df_spaxels[f"HALPHA (component {nn + 1})"] = pd.to_numeric(df_spaxels[f"HALPHA (component {nn + 1})"])
        df_spaxels[f"HALPHA error (component {nn + 1})"] = pd.to_numeric(df_spaxels[f"HALPHA error (component {nn + 1})"])
        # Compute EWs
        df_spaxels[f"HALPHA EW (component {nn + 1})"] = df_spaxels[f"HALPHA (component {nn + 1})"] / df_spaxels["HALPHA continuum"]
        df_spaxels.loc[np.isinf(df_spaxels[f"HALPHA EW (component {nn + 1})"].astype(float)), f"HALPHA EW (component {nn + 1})"] = np.nan  # If the continuum level == 0, then the EW is undefined, so set to NaN.
        df_spaxels[f"HALPHA EW error (component {nn + 1})"] = df_spaxels[f"HALPHA EW (component {nn + 1})"] *\
            np.sqrt((df_spaxels[f"HALPHA error (component {nn + 1})"] / df_spaxels[f"HALPHA (component {nn + 1})"])**2 +\
                    (df_spaxels[f"HALPHA continuum error"] / df_spaxels[f"HALPHA continuum"])**2) 

    # Calculate total EWs
    df_spaxels["HALPHA EW (total)"] = np.nansum([df_spaxels[f"HALPHA EW (component {nn + 1})"] for nn in range(3)], axis=0)
    df_spaxels["HALPHA EW error (total)"] = np.sqrt(np.nansum([df_spaxels[f"HALPHA EW error (component {nn + 1})"]**2 for nn in range(3)], axis=0))

    # If all HALPHA EWs are NaN, then make the total HALPHA EW NaN too
    df_spaxels.loc[df_spaxels["HALPHA EW (component 1)"].isna() &\
                   df_spaxels["HALPHA EW (component 2)"].isna() &\
                   df_spaxels["HALPHA EW (component 3)"].isna(), 
                   ["HALPHA EW (total)", "HALPHA EW error (total)"]] = np.nan

    ######################################################################
    # Add radius-derived value columns
    ######################################################################
    df_spaxels["r/R_e"] = df_spaxels["r (relative to galaxy centre, deprojected, arcsec)"] / df_spaxels["R_e (arcsec)"]
    df_spaxels["R_e (kpc)"] = df_spaxels["R_e (arcsec)"] * df_spaxels["kpc per arcsec"]
    df_spaxels["log(M/R_e)"] = df_spaxels["log M_*"] - np.log10(df_spaxels["R_e (kpc)"])

    ###############################################################################
    # Add spaxel scale
    ###############################################################################
    df_spaxels["Bin size (pixels)"] = 1.0
    df_spaxels["Bin size (square arcsec)"] = 1.0
    df_spaxels["Bin size (square kpc)"] = df_spaxels["kpc per arcsec"]**2

    ######################################################################
    # Compute S/N in all lines, in all components
    # Compute TOTAL line fluxes
    ######################################################################
    for eline in ["HALPHA", "HBETA", "NII6583", "OI6300", "OII3726", "OII3729", "OIII5007", "SII6716", "SII6731"]:
        # Compute S/N 
        for nn in range(3):
            if f"{eline} (component {nn + 1})" in df_spaxels.columns:
                df_spaxels[f"{eline} S/N (component {nn + 1})"] = df_spaxels[f"{eline} (component {nn + 1})"] / df_spaxels[f"{eline} error (component {nn + 1})"]
        
        # Compute total line fluxes, if the total fluxes are not given
        if f"{eline} (total)" not in df_spaxels.columns:
            df_spaxels[f"{eline} (total)"] = np.nansum([df_spaxels[f"{eline} (component {nn + 1})"] for nn in range(3)], axis=0)
            df_spaxels[f"{eline} error (total)"] = np.sqrt(np.nansum([df_spaxels[f"{eline} error (component {nn + 1})"]**2 for nn in range(3)], axis=0))

        # Compute the S/N in the TOTAL line flux
        df_spaxels[f"{eline} S/N (total)"] = df_spaxels[f"{eline} (total)"] / df_spaxels[f"{eline} error (total)"]

    ######################################################################
    # DQ and S/N CUTS
    ######################################################################
    # For WiFes
    FWHM_inst_A = 0.9 # Based on skyline at 6950 A
    sigma_inst_A = FWHM_inst_A / ( 2 * np.sqrt( 2 * np.log(2) ))
    sigma_inst_km_s = sigma_inst_A * constants.c / 1e3 / 6562.8  # Defined at Halpha
    print(f"{status_str}: WARNING: estimating instrumental dispersion from my own WiFeS observations - may not be consistent with assumed value in LZIFU!")

    df_spaxels = dqcut.dqcut(df=df_spaxels, 
                  ncomponents=3 if ncomponents == "recom" else 1,
                  line_flux_SNR_cut=line_flux_SNR_cut,
                  eline_SNR_min=eline_SNR_min, eline_list=eline_list,
                  sigma_gas_SNR_cut=sigma_gas_SNR_cut,
                  sigma_gas_SNR_min=sigma_gas_SNR_min,
                  sigma_inst_kms=sigma_inst_km_s,
                  vgrad_cut=vgrad_cut,
                  line_amplitude_SNR_cut=line_amplitude_SNR_cut,
                  flux_fraction_cut=flux_fraction_cut,
                  stekin_cut=stekin_cut)

    ######################################################################
    # Make a copy of the DataFrame with EXTINCTION CORRECTION
    # Correct emission line fluxes (but not EWs!)
    # NOTE: extinction.fm07 assumes R_V = 3.1 so do not change R_V from 
    # this value!!!
    ######################################################################
    print(f"{status_str}: Correcting emission line fluxes (but not EWs) for extinction...")
    df_spaxels_extcorr = df_spaxels.copy()
    df_spaxels_extcorr = extcorr.extinction_corr_fn(df_spaxels_extcorr, 
                                    eline_list=eline_list,
                                    reddening_curve="fm07", 
                                    balmer_SNR_min=5, nthreads=nthreads_max,
                                    s=f" (total)")
    df_spaxels_extcorr["Corrected for extinction?"] = True
    df_spaxels["Corrected for extinction?"] = False

    # Sort so that both DataFrames have the same order
    df_spaxels_extcorr = df_spaxels_extcorr.sort_index()
    df_spaxels = df_spaxels.sort_index()

    ######################################################################
    # EVALUATE LINE RATIOS & SPECTRAL CLASSIFICATIONS
    ######################################################################
    df_spaxels = linefns.ratio_fn(df_spaxels, s=f" (total)")
    df_spaxels = linefns.bpt_fn(df_spaxels, s=f" (total)")
    df_spaxels_extcorr = linefns.ratio_fn(df_spaxels_extcorr, s=f" (total)")
    df_spaxels_extcorr = linefns.bpt_fn(df_spaxels_extcorr, s=f" (total)")

    ######################################################################
    # EVALUATE ADDITIONAL COLUMNS - log quantites, etc.
    ######################################################################
    df_spaxels = dqcut.compute_extra_columns(df_spaxels, ncomponents=3 if ncomponents=="recom" else 1)
    df_spaxels_extcorr = dqcut.compute_extra_columns(df_spaxels_extcorr, ncomponents=3 if ncomponents=="recom" else 1)

    ######################################################################
    # EVALUATE METALLICITY
    ######################################################################
    for met_diagnostic in met_diagnostic_list:
        df_spaxels = metallicity.metallicity_fn(df_spaxels, met_diagnostic, logU, s=" (total)")
        df_spaxels_extcorr = metallicity.metallicity_fn(df_spaxels_extcorr, met_diagnostic, logU, s=" (total)")

    ###############################################################################
    # Save input flags to the DataFrame so that we can keep track
    ###############################################################################
    df_spaxels["Extinction correction applied"] = False
    df_spaxels["line_flux_SNR_cut"] = line_flux_SNR_cut
    df_spaxels["eline_SNR_min"] = eline_SNR_min
    df_spaxels["sigma_gas_SNR_min"] = sigma_gas_SNR_min
    df_spaxels["vgrad_cut"] = vgrad_cut
    df_spaxels["sigma_gas_SNR_cut"] = sigma_gas_SNR_cut
    df_spaxels["sigma_gas_SNR_min"] = sigma_gas_SNR_min
    df_spaxels["line_amplitude_SNR_cut"] = line_amplitude_SNR_cut
    df_spaxels["flux_fraction_cut"] = flux_fraction_cut
    df_spaxels["stekin_cut"] = stekin_cut
    df_spaxels["log(U) (const.)"] = logU

    df_spaxels_extcorr["Extinction correction applied"] = True
    df_spaxels_extcorr["line_flux_SNR_cut"] = line_flux_SNR_cut
    df_spaxels_extcorr["eline_SNR_min"] = eline_SNR_min
    df_spaxels_extcorr["sigma_gas_SNR_min"] = sigma_gas_SNR_min
    df_spaxels_extcorr["vgrad_cut"] = vgrad_cut
    df_spaxels_extcorr["sigma_gas_SNR_cut"] = sigma_gas_SNR_cut
    df_spaxels_extcorr["sigma_gas_SNR_min"] = sigma_gas_SNR_min
    df_spaxels_extcorr["line_amplitude_SNR_cut"] = line_amplitude_SNR_cut
    df_spaxels_extcorr["flux_fraction_cut"] = flux_fraction_cut
    df_spaxels_extcorr["stekin_cut"] = stekin_cut
    df_spaxels_extcorr["log(U) (const.)"] = logU

    ###############################################################################
    # Save to .hd5 & .csv
    ###############################################################################
    print(f"{status_str}: Saving to file...")

    # No extinction correction
    df_spaxels.to_csv(os.path.join(s7_data_path, df_fname.split("hd5")[0] + "csv"))
    try:
        df_spaxels.to_hdf(os.path.join(s7_data_path, df_fname), key=f"{bin_type}, {ncomponents}-comp")
    except:
        print(f"{status_str}: Unable to save to HDF file... sigh...")

    # With extinction correction
    df_spaxels_extcorr.to_csv(os.path.join(s7_data_path, df_fname_extcorr.split("hd5")[0] + "csv"))
    try:
        df_spaxels_extcorr.to_hdf(os.path.join(s7_data_path, df_fname_extcorr), key=f"{bin_type}, {ncomponents}-comp")
    except:
        print(f"{status_str}: Unable to save to HDF file... sigh...")

    print(f"{status_str}: Finished!")
    return

###############################################################################
def load_s7_df(ncomponents, bin_type, correct_extinction, eline_SNR_min,
               debug=False):

    """
    DESCRIPTION
    ---------------------------------------------------------------------------
    Load and return the Pandas DataFrame containing spaxel-by-spaxel 
    information for all S7 galaxies which was created using make_s7_df().

    INPUTS
    ---------------------------------------------------------------------------
    ncomponents:        str
        Number of components; must be "recom" (corresponding to the multi-
        component Gaussian fits).

    bin_type:           str
        Binning scheme used. Must be one of 'default'.

    correct_extinction: bool
        If True, load the DataFrame in which the emission line fluxes (but not 
        EWs) have been corrected for intrinsic extinction.

    eline_SNR_min:      int 
        Minimum flux S/N to accept. Fluxes below the threshold (plus associated
        data products) are set to NaN.

    debug:              bool
        If True, load the "debug" version of the DataFrame created when 
        running make_s7_df() with debug=True.
    
    USAGE
    ---------------------------------------------------------------------------
    load_s7_df() is called as follows:

        >>> from spaxelsleuth.loaddata.s7 import load_s7_df
        >>> df = load_s7_df(ncomponents, bin_type, correct_extinction, 
                              eline_SNR_min, debug)

    OUTPUTS
    ---------------------------------------------------------------------------
    The Dataframe.

    """
    #######################################################################
    # INPUT CHECKING
    #######################################################################
    assert ncomponents == "recom", "ncomponents must be 'recom'!!"
    assert bin_type == "default", "bin_type must be 'default'!!"

    # Input file name 
    df_fname = f"s7_spaxels_{bin_type}_{ncomponents}-comp"
    if correct_extinction:
        df_fname += "_extcorr"
    df_fname += f"_minSNR={eline_SNR_min}"
    if debug:
        df_fname += "_DEBUG"
    df_fname += ".hd5"

    assert os.path.exists(os.path.join(s7_data_path, df_fname)),\
        f"File {os.path.join(s7_data_path, df_fname)} does does not exist!"

    # Load the data frame
    print("In load_s7_df(): Loading DataFrame...")
    df = pd.read_hdf(os.path.join(s7_data_path, df_fname))

    # Return
    print("In load_s7_df(): Finished!")
    return df.sort_index()
    