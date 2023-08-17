from itertools import product
import numpy as np
from scipy import constants
import warnings

###############################################################################
def get_wavelength_from_velocity(lambda_rest, v, units):
    """Compute the Doppler-shifted wavelength given a velocity and a rest-frame wavelength."""
    if units not in ['m/s', 'km/s',]:
        raise ValueError("units must be m/s or km/s!")
    if units == 'm/s':
        v_m_s = v
    elif units == 'km/s':
        v_m_s = v * 1e3
    lambda_obs = lambda_rest * np.sqrt((1 + v_m_s / constants.c) /
                                       (1 - v_m_s / constants.c))
    return lambda_obs

###############################################################################
def deproject_coordinates(x_c_list,
                          y_c_list,
                          x0_px,
                          y0_px,
                          PA_deg,
                          i_deg,
                          plotit=False,
                          im=None):
    """Deproject coordinates x_c_list, y_c_list given a galaxy inclination, PA and centre coordinates."""
    i_rad = np.deg2rad(i_deg)
    beta_rad = np.deg2rad(PA_deg - 90)

    # De-project the centroids to the coordinate system of the galaxy plane
    x_prime_list = np.full_like(x_c_list, np.nan, dtype="float")
    y_prime_list = np.full_like(x_c_list, np.nan, dtype="float")
    y_prime_projec_list = np.full_like(x_c_list, np.nan, dtype="float") #NOTE I'm not sure why I calculated this?
    r_prime_list = np.full_like(x_c_list, np.nan, dtype="float")
    for jj, coords in enumerate(zip(x_c_list, y_c_list)):
        x_c, y_c = coords
        # De-shift, de-rotate & de-incline
        x_cc = x_c - x0_px
        y_cc = y_c - y0_px
        x_prime = x_cc * np.cos(beta_rad) + y_cc * np.sin(beta_rad)
        y_prime_projec = (- x_cc * np.sin(beta_rad) + y_cc * np.cos(beta_rad))
        y_prime = (- x_cc * np.sin(beta_rad) + y_cc * np.cos(beta_rad)) / np.cos(i_rad)
        r_prime = np.sqrt(x_prime**2 + y_prime**2)

        # Add to list
        x_prime_list[jj] = x_prime
        y_prime_list[jj] = y_prime
        y_prime_projec_list[jj] = y_prime_projec
        r_prime_list[jj] = r_prime

    # For plotting
    if plotit:
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(ncols=2, figsize=(10, 5))
        axs[0].imshow(im, origin="lower")
        axs[1].axhline(0)
        axs[1].axvline(0)
        axs[0].scatter(x_c_list, y_c_list, color="k")
        axs[0].scatter(x0_px, y0_px, color="white")
        axs[1].scatter(x_prime_list, y_prime_list, color="r")
        axs[1].scatter(x_prime_list, y_prime_projec_list, color="r", alpha=0.3)
        axs[1].axis("equal")
        fig.canvas.draw()

    return x_prime_list, y_prime_list, r_prime_list

###############################################################################
def get_slices_in_velocity_range(data_cube, var_cube, lambda_vals_rest_A, lambda_rest_start_A, lambda_rest_stop_A, v_map):
    """Returns a copy of the data/variance cubes with slices outside those in a specified wavelength range masked out."""
    # 3D array containing wavelength values in each spaxel
    lambda_vals_rest_A_cube = np.zeros(data_cube.shape)
    lambda_vals_rest_A_cube[:] = lambda_vals_rest_A[:, None, None]

    # For indices where the velocity is NaN - assume that it's zero
    v_map[np.isnan(v_map)] = 0

    # Min/max wavelength values taking into account the velocities in each spaxel
    lambda_min_A = get_wavelength_from_velocity(lambda_rest_start_A, v_map, units="km/s")
    lambda_max_A = get_wavelength_from_velocity(lambda_rest_stop_A, v_map, units="km/s")

    # Indices within the desired wavelength window, after accounting for velocities in each spaxel
    slice_mask = (lambda_vals_rest_A_cube > lambda_min_A) & (lambda_vals_rest_A_cube < lambda_max_A)

    # Copies of datacubes with slices other than those in the wavelength window NaN'd out 
    data_cube_masked = np.copy(data_cube)
    data_cube_masked[~slice_mask] = np.nan
    var_cube_masked = np.copy(var_cube)
    var_cube_masked[~slice_mask] = np.nan

    return data_cube_masked, var_cube_masked

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

    num = np.nanmean(data_cube_r_masked, axis=0)
    denom = np.nanmean(data_cube_b_masked, axis=0)
    err_num = 1 / N_r * np.sqrt(np.nansum(var_cube_r_masked, axis=0))
    err_denom = 1 / N_b * np.sqrt(np.nansum(var_cube_b_masked, axis=0))

    d4000_map = num / denom
    d4000_map_err = d4000_map * np.sqrt((err_num / num)**2 + (err_denom / denom)**2)
    return d4000_map, d4000_map_err

###############################################################################
def compute_continuum_intensity(data_cube, var_cube, lambda_vals_rest_A, start_A, stop_A, v_map):
    """Compute the mean, std. dev. and error of the mean of the continuum between start_A and stop_A."""
    data_cube_masked, var_cube_masked = get_slices_in_velocity_range(data_cube, var_cube, lambda_vals_rest_A, start_A, stop_A, v_map)
    cont_map = np.nanmean(data_cube_masked, axis=0)
    cont_map_std = np.nanstd(data_cube_masked, axis=0)
    N = np.nansum(~np.isnan(data_cube_masked), axis=0)
    cont_map_err = 1 / N * np.sqrt(np.nansum(var_cube_masked, axis=0))
    # NOTE: N = 0 in the outskirts of the image, so dividing by 1 / N replaces these elements with NaN (which is what we want!)
    return cont_map, cont_map_std, cont_map_err

###############################################################################
def compute_HALPHA_amplitude_to_noise(data_cube, var_cube, lambda_vals_rest_A, v_star_map, v_map, dv):
    """Measure the HALPHA amplitude-to-noise.
        We measure this as
              (peak spectral value in window around Ha - mean R continuum flux density) / standard deviation in R continuum flux density
        As such, this value can be negative."""
    lambda_vals_rest_A_cube = np.zeros(data_cube.shape)
    lambda_vals_rest_A_cube[:] = lambda_vals_rest_A[:, None, None]

    # Get the HALPHA continuum & std. dev.
    cont_HALPHA_map, cont_HALPHA_map_std, cont_HALPHA_map_err = compute_continuum_intensity(data_cube=data_cube, var_cube=var_cube, lambda_vals_rest_A=lambda_vals_rest_A, start_A=6500, stop_A=6540, v_map=v_star_map)

    # Wavelength window in which to compute A/N
    lambda_max_A = get_wavelength_from_velocity(6562.8, v_map + dv, units="km/s")
    lambda_min_A = get_wavelength_from_velocity(6562.8, v_map - dv, units="km/s")

    # Measure HALPHA amplitude-to-noise
    A_HALPHA_mask = (lambda_vals_rest_A_cube > lambda_min_A) & (lambda_vals_rest_A_cube < lambda_max_A)
    data_cube_masked_R = np.copy(data_cube)
    data_cube_masked_R[~A_HALPHA_mask] = np.nan
    A_HALPHA_map = np.nanmax(data_cube_masked_R, axis=0)
    AN_HALPHA_map = (A_HALPHA_map - cont_HALPHA_map) / cont_HALPHA_map_std

    return AN_HALPHA_map

###############################################################################
def compute_v_grad(v_map):
    """Compute v_grad using eqn. 1 of Zhou+2017."""
    v_grad = np.full_like(v_map, np.nan)
    if v_map.ndim == 2:
        ny, nx = v_map.shape
        for yy, xx in product(range(1, ny - 1), range(1, nx - 1)):
            v_grad[yy, xx] = np.sqrt(((v_map[yy, xx + 1] - v_map[yy, xx - 1]) / 2)**2 +\
                                        ((v_map[yy + 1, xx] - v_map[yy - 1, xx]) / 2)**2)
    elif v_map.ndim == 3:
        ny, nx = v_map.shape[1:]
        for yy, xx in product(range(1, ny - 1), range(1, nx - 1)):
            v_grad[:, yy, xx] = np.sqrt(((v_map[:, yy, xx + 1] - v_map[:, yy, xx - 1]) / 2)**2 +\
                                        ((v_map[:, yy + 1, xx] - v_map[:, yy - 1, xx]) / 2)**2)

    return v_grad

######################################################################
def compute_continuum_luminosity(df):
    """Compute HALPHA continuum luminosity."""
    # HALPHA cont. luminosity: units of erg s^-1 Å-1 kpc^-2
    if all([col in df for col in ["D_L (Mpc)", "Bin size (square kpc)"]]):
        if all([col in df for col in ["HALPHA continuum", "HALPHA continuum error"]]):
            df[f"HALPHA continuum luminosity"] = df[f"HALPHA continuum"] * 1e-16 * 4 * np.pi * (df["D_L (Mpc)"] * 1e6 * 3.086e18)**2 * 1 / df["Bin size (square kpc)"]
            df[f"HALPHA continuum luminosity error"] = df[f"HALPHA continuum error"] * 1e-16 * 4 * np.pi * (df["D_L (Mpc)"] * 1e6 * 3.086e18)**2 * 1 / df["Bin size (square kpc)"]

    return df

######################################################################
def compute_eline_luminosity(df, eline_list):
    """Compute emission line luminosities."""
    # Line luminosity: units of erg s^-1 kpc^-2
    if all([col in df for col in ["D_L (Mpc)", "Bin size (square kpc)"]]):
        for eline in eline_list:
            if all([col in df for col in [f"{eline} (total)", f"{eline} error (total)"]]):
                df[f"{eline} luminosity (total)"] = df[f"{eline} (total)"] * 1e-16 * 4 * np.pi * (df["D_L (Mpc)"] * 1e6 * 3.086e18)**2 * 1 / df["Bin size (square kpc)"]
                df[f"{eline} luminosity error (total)"] = df[f"{eline} error (total)"] * 1e-16 * 4 * np.pi * (df["D_L (Mpc)"] * 1e6 * 3.086e18)**2 * 1 / df["Bin size (square kpc)"]
            for nn in range(3):
                if all([col in df for col in [f"{eline} (component {nn + 1})", f"{eline} error (component {nn + 1})"]]):
                    df[f"{eline} luminosity (component {nn + 1})"] = df[f"{eline} (component {nn + 1})"] * 1e-16 * 4 * np.pi * (df["D_L (Mpc)"] * 1e6 * 3.086e18)**2 * 1 / df["Bin size (square kpc)"]
                    df[f"{eline} luminosity error (component {nn + 1})"] = df[f"{eline} error (component {nn + 1})"] * 1e-16 * 4 * np.pi * (df["D_L (Mpc)"] * 1e6 * 3.086e18)**2 * 1 / df["Bin size (square kpc)"]

    return df

######################################################################
def compute_FWHM(df):
    """Compute the Full-Width at Half Maximum from the velocity dispersion."""
    for nn in range(3):
        if f"sigma_gas (component {nn + 1})" in df:
            df[f"FWHM_gas (component {nn + 1})"] = df[f"sigma_gas (component {nn + 1})"] * 2 * np.sqrt(2 * np.log(2))
        if f"sigma_gas error (component {nn + 1})" in df:
            df[f"FWHM_gas error (component {nn + 1})"] = df[f"sigma_gas error (component {nn + 1})"] * 2 * np.sqrt(2 * np.log(2))

    return df

######################################################################
# Compute offsets between gas & stellar kinematics
def compute_gas_stellar_offsets(df):    
    if "v_*" in df and "sigma_*" in df:
        for nn in range(3):
            #//////////////////////////////////////////////////////////////////////
            # Velocity offsets
            if f"v_gas (component {nn + 1})" in df:
                df[f"v_gas - v_* (component {nn + 1})"] = df[f"v_gas (component {nn + 1})"] - df["v_*"]
            if f"v_gas error (component {nn + 1})" in df:
                df[f"v_gas - v_* error (component {nn + 1})"] = np.sqrt(df[f"v_gas error (component {nn + 1})"]**2 + df["v_* error"]**2)

            #//////////////////////////////////////////////////////////////////////
            # Velocity dispersion offsets
            if f"sigma_gas (component {nn + 1})" in df:
                df[f"sigma_gas - sigma_* (component {nn + 1})"] = df[f"sigma_gas (component {nn + 1})"] - df["sigma_*"]
                df[f"sigma_gas^2 - sigma_*^2 (component {nn + 1})"] = df[f"sigma_gas (component {nn + 1})"]**2 - df["sigma_*"]**2
                df[f"sigma_gas/sigma_* (component {nn + 1})"] = df[f"sigma_gas (component {nn + 1})"] / df["sigma_*"]

            if f"sigma_gas error (component {nn + 1})" in df:
                df[f"sigma_gas - sigma_* error (component {nn + 1})"] = np.sqrt(df[f"sigma_gas error (component {nn + 1})"]**2 + df["sigma_* error"]**2)
                df[f"sigma_gas^2 - sigma_*^2 error (component {nn + 1})"] = 2 * np.sqrt(df[f"sigma_gas (component {nn + 1})"]**2 * df[f"sigma_gas error (component {nn + 1})"]**2 +\
                                                                                df["sigma_*"]**2 * df["sigma_* error"]**2)
                df[f"sigma_gas/sigma_* error (component {nn + 1})"] =\
                    df[f"sigma_gas/sigma_* (component {nn + 1})"] *\
                    np.sqrt((df[f"sigma_gas error (component {nn + 1})"] / df[f"sigma_gas (component {nn + 1})"])**2 +\
                            (df["sigma_* error"] / df["sigma_*"])**2)
        
    return df

######################################################################
# Compute differences in Halpha EW, sigma_gas between different components
def compute_component_offsets(df):
    
    for nn_2, nn_1 in ([2, 1], [3, 2], [3, 1]):

        #//////////////////////////////////////////////////////////////////////
        # Difference between gas velocity dispersion between components
        if all([col in df for col in [f"sigma_gas (component {nn_1})", f"sigma_gas (component {nn_2})"]]):
            df[f"delta sigma_gas ({nn_2}/{nn_1})"] = df[f"sigma_gas (component {nn_2})"] - df[f"sigma_gas (component {nn_1})"]
        
        # Error in the difference between gas velocity dispersion between components   
        if all([col in df for col in [f"sigma_gas error (component {nn_1})", f"sigma_gas error (component {nn_2})"]]):
            df[f"delta sigma_gas error ({nn_2}/{nn_1})"] = np.sqrt(df[f"sigma_gas error (component {nn_2})"]**2 +\
                                                                   df[f"sigma_gas error (component {nn_1})"]**2)

        #//////////////////////////////////////////////////////////////////////
        # DIfference between gas velocity between components
        if all([col in df for col in [f"v_gas (component {nn_1})", f"v_gas (component {nn_2})"]]):     
            df[f"delta v_gas ({nn_2}/{nn_1})"] = df[f"v_gas (component {nn_2})"] - df[f"v_gas (component {nn_1})"]
        if all([col in df for col in [f"v_gas error (component {nn_2})", f"v_gas error (component {nn_1})"]]):  
            df[f"delta v_gas error ({nn_2}/{nn_1})"] = np.sqrt(df[f"v_gas error (component {nn_2})"]**2 +\
                                                               df[f"v_gas error (component {nn_1})"]**2)
        
        #//////////////////////////////////////////////////////////////////////
        # Ratio of HALPHA EWs between components   
        if all([col in df for col in [f"HALPHA EW (component {nn_1})", f"HALPHA EW (component {nn_2})"]]):     
            df[f"HALPHA EW ratio ({nn_2}/{nn_1})"] = df[f"HALPHA EW (component {nn_2})"] / df[f"HALPHA EW (component {nn_1})"]
        if all([col in df for col in [f"HALPHA EW error (component {nn_1})", f"HALPHA EW error (component {nn_2})"]]):     
            df[f"HALPHA EW ratio error ({nn_2}/{nn_1})"] = df[f"HALPHA EW ratio ({nn_2}/{nn_1})"] *\
                np.sqrt((df[f"HALPHA EW error (component {nn_2})"] / df[f"HALPHA EW (component {nn_2})"])**2 +\
                        (df[f"HALPHA EW error (component {nn_1})"] / df[f"HALPHA EW (component {nn_1})"])**2)

        #//////////////////////////////////////////////////////////////////////
        # Ratio of HALPHA EWs between components (log)
        if all([col in df for col in [f"log HALPHA EW (component {nn_2})", f"log HALPHA EW (component {nn_1})"]]):     
            df[f"Delta HALPHA EW ({nn_2}/{nn_1})"] = df[f"log HALPHA EW (component {nn_2})"] - df[f"log HALPHA EW (component {nn_1})"]

        #//////////////////////////////////////////////////////////////////////
        # Forbidden line ratios:
        for col in ["log O3", "log N2", "log S2", "log O1"]:
            if f"{col} (component {nn_1})" in df and f"{col} (component {nn_2})" in df:
                df[f"delta {col} ({nn_2}/{nn_1})"] = df[f"{col} (component {nn_2})"] - df[f"{col} (component {nn_1})"]
            if f"{col} error (component {nn_2})" in df and f"{col} error (component {nn_1})" in df:
                df[f"delta {col} ({nn_2}/{nn_1}) error"] = np.sqrt(df[f"{col} error (component {nn_2})"]**2 + df[f"{col} error (component {nn_1})"]**2)

    #//////////////////////////////////////////////////////////////////////
    # Fractional of total Halpha EW in each component
    for nn in range(3):
        if all([col in df.columns for col in [f"HALPHA EW (component {nn + 1})", f"HALPHA EW (total)"]]):
            df[f"HALPHA EW/HALPHA EW (total) (component {nn + 1})"] = df[f"HALPHA EW (component {nn + 1})"] / df[f"HALPHA EW (total)"]

    return df

######################################################################
# Compute log quantities + errors for Halpha EW, sigma_gas and SFRs
def compute_log_columns(df):

    # Halpha flux and EW for individual components
    for col in ["HALPHA luminosity", "HALPHA continuum", "HALPHA EW", "sigma_gas", "S2 ratio"]:
        for s in ["(total)"] + [f"(component {nn})" for nn in [1, 2, 3]]:
            # Compute log quantities for total 
            if f"{col} {s}" in df:
                df[f"log {col} {s}"] = np.log10(df[f"{col} {s}"])
            if f"{col} error {s}" in df:
                df[f"log {col} error (lower) {s}"] = df[f"log {col} {s}"] - np.log10(df[f"{col} {s}"] - df[f"{col} error {s}"])
                df[f"log {col} error (upper) {s}"] = np.log10(df[f"{col} {s}"] + df[f"{col} error {s}"]) -  df[f"log {col} {s}"]

    # Compute log quantities for total SFR
    for col in ["SFR", "SFR surface density", "sSFR"]:
        for s in ["(total)"] + [f"(component {nn})" for nn in [1, 2, 3]]:
            if f"{col} {s}" in df:
                cond = ~np.isnan(df[f"{col} {s}"])
                cond &= df[f"{col} {s}"] > 0
                df.loc[cond, f"log {col} {s}"] = np.log10(df.loc[cond, f"{col} {s}"])
                if f"{col} error {s}" in df:
                    df.loc[cond, f"log {col} error (lower) {s}"] = df.loc[cond, f"log {col} {s}"] - np.log10(df.loc[cond, f"{col} {s}"] - df.loc[cond, f"{col} error {s}"])
                    df.loc[cond, f"log {col} error (upper) {s}"] = np.log10(df.loc[cond, f"{col} {s}"] + df.loc[cond, f"{col} error {s}"]) -  df.loc[cond, f"log {col} {s}"]
                
    return df

