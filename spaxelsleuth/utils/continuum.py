import numpy as np
from scipy import constants
import warnings

from spaxelsleuth.utils.velocity import get_slices_in_velocity_range
from spaxelsleuth.utils.misc import in_dataframe

import logging
logger = logging.getLogger(__name__)

###############################################################################
def compute_d4000(data_cube, var_cube, lambda_vals_rest_A, v_star_map):
    """Compute the D4000Å break strength and its associated error map from input data and variance cubes.
    Definition from Balogh+1999 (see here: https://arxiv.org/pdf/1611.07050.pdf, page 3)
    
    Parameters:
    - data_cube (numpy.ndarray): 3D array representing the data cube.
    - var_cube (numpy.ndarray): 3D array representing the variance cube.
    - lambda_vals_rest_A (numpy.ndarray): 1D array containing rest wavelength values in Angstroms.
    - v_star_map (numpy.ndarray): 2D array representing the velocity map for stellar absorption features.

    Returns:
    - d4000_map, d4000_map_err (numpy.ndarray): arrays containing the D4000 index map and its associated error map.

    Note:
    - The input data and variance cubes are converted to units of F_nu using the rest wavelength values.
    - The D4000 index is computed by masking slices within specified wavelength ranges for blue and red edges.
    - Error propagation is performed, considering the number of valid pixels in each wavelength range.

    NOTE: docstring written with help from ChatGPT 3.5.
    """
    logger.debug(f"computing D4000 break strengths...")

    # Convert datacube & variance cubes to units of F_nu
    data_cube_Hz = data_cube * lambda_vals_rest_A[:, None, None]**2 / (constants.c * 1e10)
    var_cube_Hz2 = var_cube * (lambda_vals_rest_A[:, None, None]**2 / (constants.c * 1e10))**2

    data_cube_b_masked, var_cube_b_masked = get_slices_in_velocity_range(data_cube_Hz, var_cube_Hz2, lambda_vals_rest_A, 3850, 3950, v_star_map)
    data_cube_r_masked, var_cube_r_masked = get_slices_in_velocity_range(data_cube_Hz, var_cube_Hz2, lambda_vals_rest_A, 4000, 4100, v_star_map)

    N_b = np.nansum(~np.isnan(data_cube_b_masked), axis=0)
    N_r = np.nansum(~np.isnan(data_cube_r_masked), axis=0)

    with warnings.catch_warnings():
        warnings.filterwarnings(action="ignore", message="Mean of empty slice")
        num = np.nanmean(data_cube_r_masked, axis=0)
        denom = np.nanmean(data_cube_b_masked, axis=0)
        
    with warnings.catch_warnings():
        warnings.filterwarnings(action="ignore", category=RuntimeWarning, message="divide by zero encountered in divide")
        warnings.filterwarnings(action="ignore", category=RuntimeWarning, message="invalid value encountered in multiply")
        err_num = 1 / N_r * np.sqrt(np.nansum(var_cube_r_masked, axis=0))
        err_denom = 1 / N_b * np.sqrt(np.nansum(var_cube_b_masked, axis=0))
        d4000_map = num / denom
        d4000_map_err = d4000_map * np.sqrt((err_num / num)**2 + (err_denom / denom)**2)
    
    return d4000_map, d4000_map_err

###############################################################################
def compute_continuum_intensity(data_cube, var_cube, lambda_vals_rest_A, start_A, stop_A, v_map):
    """Compute the mean, std. dev. and error of the mean of the continuum between start_A and stop_A.
    
    Parameters:
    - data_cube (numpy.ndarray): 3D array representing the data cube.
    - var_cube (numpy.ndarray): 3D array representing the variance cube.
    - lambda_vals_rest_A (numpy.ndarray): 1D array containing rest wavelength values in Angstroms.
    - start_A (float): Start wavelength of the desired range in Angstroms.
    - stop_A (float): Stop wavelength of the desired range in Angstroms.
    - v_map (numpy.ndarray): 2D array representing the velocity map.

    Returns:
    - cont_map, cont_map_std, cont_map_err (numpy.ndarray): Arrays containing the continuum intensity map,
      standard deviation, and associated error map.

    NOTE: docstring written with help from ChatGPT 3.5.
    """
    logger.debug(f"computing continuum intensities...")
    data_cube_masked, var_cube_masked = get_slices_in_velocity_range(data_cube, var_cube, lambda_vals_rest_A, start_A, stop_A, v_map)
    with warnings.catch_warnings():
        warnings.filterwarnings(action="ignore", category=RuntimeWarning, message="Mean of empty slice")
        warnings.filterwarnings(action="ignore", category=RuntimeWarning, message="Degrees of freedom <= 0 for slice.")
        cont_map = np.nanmean(data_cube_masked, axis=0)
        cont_map_std = np.nanstd(data_cube_masked, axis=0)
        N = np.nansum(~np.isnan(data_cube_masked), axis=0)
        
    with warnings.catch_warnings():
        warnings.filterwarnings(action="ignore", category=RuntimeWarning, message="invalid value encountered in multiply")
        warnings.filterwarnings(action="ignore", category=RuntimeWarning, message="divide by zero encountered in divide")
        cont_map_err = 1 / N * np.sqrt(np.nansum(var_cube_masked, axis=0))
    # NOTE: N = 0 in the outskirts of the image, so dividing by 1 / N replaces these elements with NaN (which is what we want!)
    return cont_map, cont_map_std, cont_map_err

######################################################################
def compute_continuum_luminosity(df):
    """Compute HALPHA continuum luminosity.

    NOTE: this calculation assumes that HALPHA continuum is in units of 1e-16 erg s^-1 cm^-2 Å-1.
    
    The output is in units of erg s^-1 Å-1.

    Parameters:
    - df (pandas.DataFrame): Input DataFrame containing HALPHA continuum and relevant information.

    Returns:
    - df (pandas.DataFrame): Updated DataFrame with added columns for HALPHA continuum luminosity and its error.

    Note:
    - The following columns are required: "D_L (Mpc)", "Bin size (square kpc)", "HALPHA continuum", and "HALPHA continuum error". 
    - Adds columns "HALPHA continuum luminosity" and "HALPHA continuum luminosity error" to the DataFrame.

    NOTE: docstring written with help from ChatGPT 3.5.
    """
    logger.debug(f"computing continuum luminosities...")
    # HALPHA cont. luminosity: units of erg s^-1 Å-1 kpc^-2
    if all([col in df for col in ["D_L (Mpc)", "Bin size (square kpc)"]]):
        if all([col in df for col in ["HALPHA continuum", "HALPHA continuum error"]]):
            df[f"HALPHA continuum luminosity"] = df[f"HALPHA continuum"] * 1e-16 * 4 * np.pi * (df["D_L (Mpc)"] * 1e6 * 3.086e18)**2 * 1 / df["Bin size (square kpc)"]
            df[f"HALPHA continuum luminosity error"] = df[f"HALPHA continuum error"] * 1e-16 * 4 * np.pi * (df["D_L (Mpc)"] * 1e6 * 3.086e18)**2 * 1 / df["Bin size (square kpc)"]

    return df

###############################################################################
def compute_EW(df, ncomponents_max, eline_list):
    """Calculate equivalent widths (EWs) for the emission lines.
        The EW is defined as 
            
            EW = emission line flux / continuum level in some wavelength range
        
        This function looks looks for the following columns in df:
            <eline> (component <n>/total) -- emission line flux 
            <eline> continuum -- corresponding continuum level

        Errors on the EWs are also computed if errors are available for the 
        emission line flux and continuum level. 
        
        Adds the following columns to df:
            <eline> EW (component <n>/total)
            <eline> EW error (component <n>/total) (if errors on the emission line flux/continuum exist in df)

    Parameters:
    - df (pandas.DataFrame): Input DataFrame containing emission line flux and continuum level information.
    - ncomponents_max (int): Maximum number of components for each emission line.
    - eline_list (list): List of emission lines for which EWs are to be calculated.

    Returns:
    - df (pandas.DataFrame): Updated DataFrame with added columns for EWs and associated errors.

    Note:
    - If the continuum level or emission line flux is <= 0, the corresponding EW is set to NaN.

    NOTE: docstring written with help from ChatGPT 3.5.
    """
    logger.debug(f"computing equivalent widths...")
    for eline in eline_list:
        if in_dataframe(df, [f"{eline} continuum"]):
            # Zero out -ve continuum values
            df.loc[df[f"{eline} continuum"] < 0, f"{eline} continuum"] = 0

            # Compute EW in each component
            for nn in range(ncomponents_max):
                if in_dataframe(df, f"{eline} (component {nn + 1})"):
                    df[f"{eline} EW (component {nn + 1})"] = df[f"{eline} (component {nn + 1})"] / df[f"{eline} continuum"]
                    # Compute associated errors
                    if in_dataframe(df, [f"{eline} error (component {nn + 1})", f"{eline} continuum error"]):
                        df[f"{eline} EW error (component {nn + 1})"] = df[f"{eline} EW (component {nn + 1})"] *\
                            np.sqrt((df[f"{eline} error (component {nn + 1})"] / df[f"{eline} (component {nn + 1})"])**2 +\
                                    (df[f"{eline} continuum error"] / df[f"{eline} continuum"])**2)

                    # If the continuum level <= 0, then the EW is undefined, so set to NaN.
                    df.loc[df[f"{eline} continuum"] <= 0, [f"{eline} EW (component {nn + 1})"]] = np.nan
                    if in_dataframe(df, [f"{eline} EW error (component {nn + 1})"]):
                        df.loc[df[f"{eline} continuum"] <= 0, [f"{eline} EW error (component {nn + 1})"]] = np.nan

                    # If the emission line flux <= 0, then the EW is undefined, so set to NaN.
                    df.loc[df[f"{eline} (component {nn + 1})"] <= 0, [f"{eline} EW (component {nn + 1})"]] = np.nan
                    if in_dataframe(df, [f"{eline} EW error (component {nn + 1})"]):
                        df.loc[df[f"{eline} (component {nn + 1})"] <= 0, [f"{eline} EW error (component {nn + 1})"]] = np.nan

            # Calculate total EW
            if in_dataframe(df, f"{eline} (total)"):
                df[f"{eline} EW (total)"] = df[f"{eline} (total)"] / df[f"{eline} continuum"]
                if in_dataframe(df, [f"{eline} error (total)", f"{eline} continuum error"]):
                    df[f"{eline} EW error (total)"] = df[f"{eline} EW (total)"] *\
                        np.sqrt((df[f"{eline} error (total)"] / df[f"{eline} (total)"])**2 +\
                                (df[f"{eline} continuum error"] / df[f"{eline} continuum"])**2)

                # If the continuum level <= 0, then the EW is undefined, so set to NaN.
                df.loc[df[f"{eline} continuum"] <= 0, [f"{eline} EW (total)"]] = np.nan
                if in_dataframe(df, [f"{eline} EW error (total)"]):
                    df.loc[df[f"{eline} continuum"] <= 0, [f"{eline} EW error (total)"]] = np.nan
                
                # If the emission line flux <= 0, then the EW is undefined, so set to NaN.
                df.loc[df[f"{eline} (total)"] <= 0, [f"{eline} EW (total)"]] = np.nan
                if in_dataframe(df, [f"{eline} EW error (total)"]):
                    df.loc[df[f"{eline} (total)"] <= 0, [f"{eline} EW error (total)"]] = np.nan


    return df