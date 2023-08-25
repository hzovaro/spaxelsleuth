import numpy as np
from scipy import constants
import warnings

from spaxelsleuth.utils.velocity import get_slices_in_velocity_range
from spaxelsleuth.utils.misc import in_dataframe

###############################################################################
def compute_d4000(data_cube, var_cube, lambda_vals_rest_A, v_star_map):
    """Compute the D4000Å break strength in a given data cube.
    Definition from Balogh+1999 (see here: https://arxiv.org/pdf/1611.07050.pdf, page 3)"""

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
    """Compute the mean, std. dev. and error of the mean of the continuum between start_A and stop_A."""
    data_cube_masked, var_cube_masked = get_slices_in_velocity_range(data_cube, var_cube, lambda_vals_rest_A, start_A, stop_A, v_map)
    with warnings.catch_warnings():
        warnings.filterwarnings(action="ignore", message="Mean of empty slice")
        warnings.filterwarnings(action="ignore", message="RuntimeWarning: Degrees of freedom <= 0 for slice.")
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
    """Compute HALPHA continuum luminosity."""
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
            <eline> (component <n>) -- emission line flux 
            <eline> continuum -- corresponding continuum level

    """
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

            # Calculate total EW
            if in_dataframe(df, f"{eline} (total)"):
                df[f"{eline} EW (total)"] = df[f"{eline} (total)"] / df[f"{eline} continuum"]
                if in_dataframe(df, [f"{eline} error (total)", f"{eline} continuum error"]):
                    df[f"{eline} EW error (total)"] = df[f"{eline} EW (total)"] *\
                        np.sqrt((df[f"{eline} error (total)"] / df[f"{eline} (total)"])**2 +\
                                (df[f"{eline} continuum error"] / df[f"{eline} continuum"])**2)

                # If the continuum level <= 0, then the EW is undefined, so set to NaN.
                df.loc[df[f"{eline} continuum"] <= 0, [f"{eline} EW (total)", f"{eline} EW error (total)"]] = np.nan

    return df