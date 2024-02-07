import os
from pathlib import Path
import pkgutil

from astropy.io import fits
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.cosmology import FlatLambdaCDM
import numpy as np
import pandas as pd

from spaxelsleuth.config import settings
from spaxelsleuth.utils.continuum import compute_d4000, compute_continuum_intensity
from spaxelsleuth.utils.dqcut import compute_measured_HALPHA_amplitude_to_noise

import logging
logger = logging.getLogger(__name__)

# Paths
input_path = Path(settings["s7"]["input_path"])
output_path = Path(settings["s7"]["output_path"])
data_cube_path = Path(settings["s7"]["data_cube_path"])


def make_metadata_df():
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
            
            >>> from spaxelsleuth.io.s7 import make_metadata_df
            >>> make_metadata_df()

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

    # Filenames
    input_catalogue_fname = "S7_DR2_Table_2_Catalogue.csv"
    df_metadata_fname = "s7_metadata.hd5"

    # Read in the metadata
    data_path = Path(pkgutil.get_loader(__name__).get_filename()).parent.parent / "data"
    assert os.path.exists(data_path / input_catalogue_fname),\
        f"File {data_path / input_catalogue_fname} not found!"
    df_metadata = pd.read_csv(data_path / input_catalogue_fname, skiprows=58)
    gals_str = df_metadata["S7_Name"].values
    gals = list(range(len(gals_str)))

    # Convert object coordinates to degrees
    coords = SkyCoord(df_metadata["RA_hms"], df_metadata["Dec_sxgsml"],
                 unit=(u.hourangle, u.deg))
    df_metadata["RA (J2000)"] = coords.ra.deg
    df_metadata["Dec (J2000)"] = coords.dec.deg

    # Rename columns
    rename_dict = {
        "S7_Name": "ID (string)",
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
        "S7_nucleus_index_x": "x_0 (pixels)",
        "S7_nucleus_index_y": "y_0 (pixels)"
    }
    df_metadata = df_metadata.rename(columns=rename_dict)
    df_metadata["ID"] = list(range(df_metadata.shape[0]))

    # Get rid of unneeded columns
    good_cols = [rename_dict[k] for k in rename_dict.keys()] + ["RA (J2000)", "Dec (J2000)", "ID"]
    df_metadata = df_metadata[good_cols]

    # Add angular scale info
    logger.info(f"computing distances...")
    cosmo = FlatLambdaCDM(H0=settings["H_0"], Om0=settings["Omega_0"])
    for gal in gals:
        D_A_Mpc = cosmo.angular_diameter_distance(df_metadata.loc[gal, "z"]).value
        D_L_Mpc = cosmo.luminosity_distance(df_metadata.loc[gal, "z"]).value
        df_metadata.loc[gal, "D_A (Mpc)"] = D_A_Mpc
        df_metadata.loc[gal, "D_L (Mpc)"] = D_L_Mpc
    df_metadata["kpc per arcsec"] = df_metadata["D_A (Mpc)"] * 1e3 * np.pi / 180.0 / 3600.0

    # Define a "Good?" column
    df_metadata["Sy1 subtraction?"] = [True if x == "Y" else False for x in df_metadata["Sy1 subtraction?"].values]
    df_metadata["Good?"] = ~df_metadata["Sy1 subtraction?"].values
    df_metadata = df_metadata.sort_index()

    # Save to file
    logger.info(f"saving metadata DataFrame to file {output_path / df_metadata_fname}...")
    df_metadata.to_hdf(output_path / df_metadata_fname, key="metadata")

    logger.info(f"finished!")
    return


def load_metadata_df():
    """Load the S7 metadata DataFrame, containing "metadata" for each galaxy."""
    if not os.path.exists(Path(settings["s7"]["output_path"]) / "s7_metadata.hd5"):
        raise FileNotFoundError(
            f"File {Path(settings['s7']['output_path']) / 's7_metadata.hd5'} not found. Did you remember to run make_metadata_df() first?"
        )
    return pd.read_hdf(
        Path(settings["s7"]["output_path"]) / "s7_metadata.hd5")


def _process_gals(args):
    """Helper function that is used in make_s7_df() to process S7 galaxies across multiple threads."""
    gal_idx, _, ncomponents, bin_type, df_metadata, kwargs = args

    # Get the gal name 
    gal = df_metadata.loc[gal_idx, "ID (string)"]

    # Scrape outputs from LZIFU output
    hdulist_best_components = fits.open(input_path / f"{gal}_best_components.fits")
    hdr = hdulist_best_components[0].header

    # Angular scale
    z = hdr["Z"]
    D_A_Mpc = df_metadata.loc[gal_idx, "D_A (Mpc)"]
    kpc_per_arcsec = D_A_Mpc * 1e3 * np.pi / 180.0 / 3600.0

    # Angular scale
    as_per_px = settings["s7"]["as_per_px"]

    # LOAD THE DATACUBES
    # NOTE: the data cubes have units 
    #       erg / cm^2 / s / Angstroem / arcsec^2 
    # So we normalise them by a factor of 1e-16 to put them in the same 
    # units as the emission line fluxes.
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

    # SCRAPE LZIFU MEASUREMENTS
    rows_list = []
    colnames = []
    lzifu_ncomponents = hdulist_best_components["V"].shape[0] - 1

    # Scrape the FITS file: emission line flues, velocity/velocity dispersion, fit quality
    # NOTE: fluxes are in units of E-16 erg / cm^2 / s
    eline_list = settings["s7"]["eline_list"]
    quantities = eline_list + ["V", "VDISP", "CHI2", "DOF", "STAR_V", "STAR_VDISP"]
    for quantity in quantities:
        if quantity in eline_list and quantity not in hdulist_best_components:
            missing_eline = True
            # Make dummy data 
            if (lzifu_ncomponents == 1) or (quantity in ["OII3726", "OII3729", "OIII4363"]):
                data = np.full((ny, nx), np.nan)
                err = np.full((ny, nx), np.nan)
            else:
                data = np.full((lzifu_ncomponents + 1, ny, nx), np.nan)
                err = np.full((lzifu_ncomponents + 1, ny, nx), np.nan)
        else:
            missing_eline = False
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
                    if (f"{quantity}_ERR" in hdulist_best_components) or missing_eline:
                        rows_list.append(_2d_map_to_1d_list(err[0]))
                        colnames.append(f"{quantity} error (total)")
                # Fluxes/values for individual components
                for nn in range(lzifu_ncomponents):
                    rows_list.append(_2d_map_to_1d_list(data[nn + 1]))
                    colnames.append(f"{quantity} (component {nn + 1})")
                    if (f"{quantity}_ERR" in hdulist_best_components) or missing_eline:
                        rows_list.append(_2d_map_to_1d_list(err[nn + 1]))
                        colnames.append(f"{quantity} error (component {nn + 1})")
            # Otherwise it's a 2D map
            elif data.ndim == 2:
                rows_list.append(_2d_map_to_1d_list(data))
                colnames.append(f"{quantity}")
                if f"{quantity}_ERR" in hdulist_best_components:
                    rows_list.append(_2d_map_to_1d_list(err))
                    colnames.append(f"{quantity} error")

    # COMPUTE QUANTITIES DIRECTLY FROM THE DATACUBES
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
    else:
        rows_list.append(_2d_map_to_1d_list(np.full((ny, nx), np.nan)))
        colnames.append(f"D4000")
        rows_list.append(_2d_map_to_1d_list(np.full((ny, nx), np.nan)))
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
    else:
        rows_list.append(_2d_map_to_1d_list(np.full((ny, nx), np.nan)))
        colnames.append(f"HALPHA continuum")
        rows_list.append(_2d_map_to_1d_list(np.full((ny, nx), np.nan)))
        colnames.append(f"HALPHA continuum std. dev.")
        rows_list.append(_2d_map_to_1d_list(np.full((ny, nx), np.nan)))
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
    else:
        rows_list.append(_2d_map_to_1d_list(np.full((ny, nx), np.nan)))
        colnames.append(f"B-band continuum")
        rows_list.append(_2d_map_to_1d_list(np.full((ny, nx), np.nan)))
        colnames.append(f"B-band continuum std. dev.")
        rows_list.append(_2d_map_to_1d_list(np.full((ny, nx), np.nan)))
        colnames.append(f"B-band continuum error")

    # Compute the HALPHA amplitude-to-noise
    if lambda_vals_R_rest_A[0] <= 6562.8 and lambda_vals_R_rest_A[-1] >= 6562.8:
        v_map = hdulist_best_components["V"].data  # Get velocity field from LZIFU fit
        AN_HALPHA_map = compute_measured_HALPHA_amplitude_to_noise(
            data_cube=data_cube_R,
            var_cube=var_cube_R,
            lambda_vals_rest_A=lambda_vals_R_rest_A,
            v_star_map=v_star_map,
            v_map=v_map[0],
            dv=300)
        rows_list.append(_2d_map_to_1d_list(AN_HALPHA_map))
        colnames.append(f"HALPHA A/N (measured)")
    else:
        rows_list.append(_2d_map_to_1d_list(np.full((ny, nx), np.nan)))
        colnames.append(f"HALPHA A/N (measured)")

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


    # Add galaxy ID (numerical index)
    rows_list.append([gal_idx] * len(x_c_list))
    colnames.append("ID")

    # Transpose so that each row represents a single pixel & each column a measured quantity.
    rows_arr = np.array(rows_list).T
 
    # Get rid of rows that are all NaNs
    bad_rows = np.all(np.isnan(rows_arr), axis=1)
    rows_good = rows_arr[~bad_rows]

    logger.info(f"finished processing {gal} ({gal_idx})")

    return rows_good, colnames