"""
In here: take a "generic" DataFrame & calculate ALL THINGS that are NOT specific to certain surveys 
e.g. metallicities, extinction, etc. 

Input: Pandas DataFrame 

Output: same Pandas DataFrame but with additional columns added. 

Steps to include:
- Calculate equivalent widths
- Compute S/N in all lines
- Fix SFR columns
- DQ and S/N CUTS
- NaN out SFR quantities if the HALPHA flux is NaN 
- EXTINCTION CORRECTION
- EVALUATE LINE RATIOS & SPECTRAL CLASSIFICATIONS
- EVALUATE ADDITIONAL COLUMNS - log quantites, etc.
- EVALUATE METALLICITY (only for spaxels with extinction correction)
- Save input flags to the DataFrame so that we can keep track
- Save to .hd5 & .csv.
"""
from itertools import product
import numpy as np
from scipy import constants

# TODO: figure out how to tidy up this import
from spaxelsleuth.utils import dqcut, linefns, metallicity, extcorr

#//////////////////////////////////////////////////////////////////////////////
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

#//////////////////////////////////////////////////////////////////////////////
def compute_d4000(data_cube, var_cube, lambda_vals_rest_A):
    """Compute the D4000Å break strength in a given data cube.
    Definition from Balogh+1999 (see here: https://arxiv.org/pdf/1611.07050.pdf, page 3)"""
    start_b_idx = np.nanargmin(np.abs(lambda_vals_rest_A - 3850))
    stop_b_idx = np.nanargmin(np.abs(lambda_vals_rest_A - 3950))
    start_r_idx = np.nanargmin(np.abs(lambda_vals_rest_A - 4000))
    stop_r_idx = np.nanargmin(np.abs(lambda_vals_rest_A - 4100))
    N_b = stop_b_idx - start_b_idx
    N_r = stop_r_idx - start_r_idx

    # Convert datacube & variance cubes to units of F_nu
    data_cube_Hz = data_cube * lambda_vals_rest_A[:, None, None]**2 / (constants.c * 1e10)
    var_cube_Hz2 = var_cube * (lambda_vals_rest_A[:, None, None]**2 / (constants.c * 1e10))**2

    num = np.nanmean(data_cube_Hz[start_r_idx:stop_r_idx], axis=0)
    denom = np.nanmean(data_cube_Hz[start_b_idx:stop_b_idx], axis=0)
    err_num = 1 / N_r * np.sqrt(np.nansum(var_cube_Hz2[start_r_idx:stop_r_idx], axis=0))
    err_denom = 1 / N_b * np.sqrt(np.nansum(var_cube_Hz2[start_b_idx:stop_b_idx], axis=0))

    d4000_map = num / denom
    d4000_map_err = d4000_map * np.sqrt((err_num / num)**2 + (err_denom / denom)**2)
    return d4000_map, d4000_map_err

#//////////////////////////////////////////////////////////////////////////////
def compute_continuum_intensity(data_cube, var_cube, lambda_vals_rest_A, start_A, stop_A):
    """Compute the mean, std. dev. and error of the mean of the continuum between start_A and stop_A."""
    start_idx = np.nanargmin(np.abs(lambda_vals_rest_A - start_A))
    stop_idx = np.nanargmin(np.abs(lambda_vals_rest_A - stop_A))
    cont_map = np.nanmean(data_cube[start_idx:stop_idx], axis=0)
    cont_map_std = np.nanstd(data_cube[start_idx:stop_idx], axis=0)
    cont_map_err = 1 / (stop_idx - start_idx) * np.sqrt(np.nansum(var_cube[start_idx:stop_idx], axis=0))
    return cont_map, cont_map_std, cont_map_err

#//////////////////////////////////////////////////////////////////////////////
def compute_HALPHA_amplitude_to_noise(data_cube, var_cube, lambda_vals_rest_A, v_map, dv):
    """Measure the HALPHA amplitude-to-noise.
        We measure this as
              (peak spectral value in window around Ha - mean R continuum flux density) / standard deviation in R continuum flux density
        As such, this value can be negative."""
    lambda_vals_rest_A_cube = np.zeros(data_cube.shape)
    lambda_vals_rest_A_cube[:] = lambda_vals_rest_A[:, None, None]

    # Get the HALPHA continuum & std. dev.
    cont_HALPHA_map, cont_HALPHA_map_std, cont_HALPHA_map_err = compute_continuum_intensity(data_cube=data_cube, var_cube=var_cube, lambda_vals_rest_A=lambda_vals_rest_A, start_A=6500, stop_A=6540)

    # Wavelength window in which to compute A/N
    lambda_max_A = dqcut.get_wavelength_from_velocity(6562.8, v_map + dv, units="km/s")
    lambda_min_A = dqcut.get_wavelength_from_velocity(6562.8, v_map - dv, units="km/s")

    # Measure HALPHA amplitude-to-noise
    A_HALPHA_mask = (lambda_vals_rest_A_cube > lambda_min_A) & (lambda_vals_rest_A_cube < lambda_max_A)
    data_cube_masked_R = np.copy(data_cube)
    data_cube_masked_R[~A_HALPHA_mask] = np.nan
    A_HALPHA_map = np.nanmax(data_cube_masked_R, axis=0)
    AN_HALPHA_map = (A_HALPHA_map - cont_HALPHA_map) / cont_HALPHA_map_std
    return AN_HALPHA_map

#//////////////////////////////////////////////////////////////////////////////
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

#//////////////////////////////////////////////////////////////////////////////
def add_columns(df, **kwargs):
    """Computes quantities such as metallicities, extinctions, etc. for each row in df."""

    # Utility function for checking necessary columns
    def in_dataframe(cols) -> bool:
        """Returns True if all colums in cols are present in DataFrame df."""
        if type(cols) == list:
            return all([c in df for c in cols])
        elif type(cols) == str:
            return cols in df
        else:
            raise ValueError("cols must be str or list of str!")

    status_str = "In generic.add_columns():"

    ###############################################################################
    # Compute the ORIGINAL number of components
    ###############################################################################
    # Figure out the maximum number of components that has been fitted to each spaxel
    ncomponents_max = 0
    while True:
        if not in_dataframe(f"sigma_gas (component {ncomponents_max + 1})"):
            break
        ncomponents_max += 1

    # Compute the ORIGINAL number of components in each spaxel: define these as those in which sigma_gas is not NaN
    ncomponents_original = (~df[f"sigma_gas (component 1)"].isna()).astype(int)
    for nn in range(1, ncomponents_max):
        ncomponents_original += (~df[f"sigma_gas (component {nn + 1})"].isna()).astype(int)
    df["Number of components (original)"] = ncomponents_original

    ###############################################################################
    # Calculate equivalent widths
    ###############################################################################
    if in_dataframe(["HALPHA continuum"]):
        # Zero out -ve continuum values
        df.loc[df["HALPHA continuum"] < 0, "HALPHA continuum"] = 0

        # Compute EW in each component
        for nn in range(ncomponents_max):
            if in_dataframe(f"HALPHA (component {nn + 1})"):
                df[f"HALPHA EW (component {nn + 1})"] = df[f"HALPHA (component {nn + 1})"] / df["HALPHA continuum"]
                # Compute associated errors
                if in_dataframe([f"HALPHA error (component {nn + 1})", "HALPHA continuum error"]):
                    df[f"HALPHA EW error (component {nn + 1})"] = df[f"HALPHA EW (component {nn + 1})"] *\
                        np.sqrt((df[f"HALPHA error (component {nn + 1})"] / df[f"HALPHA (component {nn + 1})"])**2 +\
                                (df[f"HALPHA continuum error"] / df[f"HALPHA continuum"])**2)

                # If the continuum level <= 0, then the EW is undefined, so set to NaN.
                df.loc[df["HALPHA continuum"] <= 0, [f"HALPHA EW (component {nn + 1})"]] = np.nan
                if in_dataframe([f"HALPHA EW error (component {nn + 1})"]):
                    df.loc[df["HALPHA continuum"] <= 0, [f"HALPHA EW error (component {nn + 1})"]] = np.nan

        # Calculate total EW
        if in_dataframe("HALPHA (total)"):
            df[f"HALPHA EW (total)"] = df[f"HALPHA (total)"] / df["HALPHA continuum"]
            if in_dataframe(["HALPHA error (total)", "HALPHA continuum error"]):
                df[f"HALPHA EW error (total)"] = df[f"HALPHA EW (total)"] *\
                    np.sqrt((df[f"HALPHA error (total)"] / df[f"HALPHA (total)"])**2 +\
                            (df[f"HALPHA continuum error"] / df[f"HALPHA continuum"])**2)

            # If the continuum level <= 0, then the EW is undefined, so set to NaN.
            df.loc[df["HALPHA continuum"] <= 0, [f"HALPHA EW (total)", f"HALPHA EW error (total)"]] = np.nan

    ######################################################################
    # Compute S/N in all lines
    ######################################################################
    for eline in kwargs["eline_list"]:
        # Compute S/N
        for nn in range(ncomponents_max):
            if in_dataframe([f"{eline} (component {nn + 1})", f"{eline} error (component {nn + 1})"]):
                df[f"{eline} S/N (component {nn + 1})"] = df[f"{eline} (component {nn + 1})"] / df[f"{eline} error (component {nn + 1})"]

        # Compute the S/N in the TOTAL line flux
        if in_dataframe([f"{eline} (total)", f"{eline} error (total)"]):
            df[f"{eline} S/N (total)"] = df[f"{eline} (total)"] / df[f"{eline} error (total)"]

    ######################################################################
    # DQ and S/N CUTS
    ######################################################################
    df = dqcut.set_flags(df=df, **kwargs)
    df = dqcut.apply_flags(df=df, **kwargs)

    ######################################################################
    # Fix SFR columns
    ######################################################################
    # NaN the SFR surface density if the inclination is undefined
    if in_dataframe("i (degrees)"):
        cond_NaN_inclination = np.isnan(df["i (degrees)"])
        cols = [c for c in df.columns if "SFR surface density" in c]
        df.loc[cond_NaN_inclination, cols] = np.nan

    # NaN the SFR if the SFR == 0
    # Note: I'm not entirely sure why there are spaxels with SFR == 0
    # in the first place.
    if in_dataframe("SFR (total)"):
        cond_zero_SFR = df["SFR (total)"]  == 0
        cols = [c for c in df.columns if "SFR" in c]
        df.loc[cond_zero_SFR, cols] = np.nan

    # NaN out SFR quantities if the HALPHA flux is NaN
    # need to do this AFTER applying S/N and DQ cuts above.
    if in_dataframe("HALPHA (total)"):
        cond_Ha_isnan = df["HALPHA (total)"].isna()
        cols_sfr = [c for c in df.columns if "SFR" in c]
        for col in cols_sfr:
            df.loc[cond_Ha_isnan, col] = np.nan

    ######################################################################
    # EXTINCTION CORRECTION
    # Compute A_V & correct emission line fluxes (but not EWs!)
    ######################################################################
    if kwargs["correct_extinction"]:
        print(f"{status_str}: Correcting emission line fluxes (but not EWs) for extinction...")
        # Compute A_V using total Halpha and Hbeta emission line fluxes
        df = extcorr.compute_A_V(df,
                                         reddening_curve="fm07",
                                         balmer_SNR_min=5,
                                         s=f" (total)")

        # Apply the extinction correction to total emission line fluxes
        df = extcorr.apply_extinction_correction(df,
                                        reddening_curve="fm07",
                                        eline_list=[e for e in kwargs["eline_list"] if f"{e} (total)" in df],
                                        a_v_col_name="A_V (total)",
                                        nthreads=kwargs["nthreads_max"],
                                        s=f" (total)")

        # Apply the extinction correction to fluxes of  individual components
        for nn in range(ncomponents_max):
            df = extcorr.apply_extinction_correction(df,
                                            reddening_curve="fm07",
                                            eline_list=[e for e in kwargs["eline_list"] if f"{e} (component {nn + 1})" in df],
                                            a_v_col_name="A_V (total)",
                                            nthreads=kwargs["nthreads_max"],
                                            s=f" (component {nn + 1})")

        df["Corrected for extinction?"] = True
    else:
        df["Corrected for extinction?"] = False
    df = df.sort_index()

    ######################################################################
    # EVALUATE LINE RATIOS & SPECTRAL CLASSIFICATIONS
    ######################################################################
    df = linefns.ratio_fn(df, s=f" (total)")
    df = linefns.bpt_fn(df, s=f" (total)")

    ######################################################################
    # EVALUATE ADDITIONAL COLUMNS - log quantites, etc.
    ######################################################################
    df = dqcut.compute_extra_columns(df)

    ######################################################################
    # COMPUTE THE SFR
    ######################################################################
    if kwargs["compute_sfr"]:
        df = linefns.sfr_fn(df, s=f" (total)")

    ######################################################################
    # EVALUATE METALLICITY (only for spaxels with extinction correction)
    ######################################################################
    if not kwargs["debug"]:
        df = metallicity.calculate_metallicity(met_diagnostic="N2Ha_PP04", compute_errors=True, niters=1000, df=df, s=" (total)")
        df = metallicity.calculate_metallicity(met_diagnostic="N2Ha_M13", compute_errors=True, niters=1000, df=df, s=" (total)")
        df = metallicity.calculate_metallicity(met_diagnostic="O3N2_PP04", compute_errors=True, niters=1000, df=df, s=" (total)")
        df = metallicity.calculate_metallicity(met_diagnostic="O3N2_M13", compute_errors=True, niters=1000, df=df, s=" (total)")
        df = metallicity.calculate_metallicity(met_diagnostic="N2S2Ha_D16", compute_errors=True, niters=1000, df=df, s=" (total)")
        df = metallicity.calculate_metallicity(met_diagnostic="N2O2_KD02", compute_errors=True, niters=1000, df=df, s=" (total)")
        df = metallicity.calculate_metallicity(met_diagnostic="Rcal_PG16", compute_errors=True, niters=1000, df=df, s=" (total)")
        df = metallicity.calculate_metallicity(met_diagnostic="Scal_PG16", compute_errors=True, niters=1000, df=df, s=" (total)")
        df = metallicity.calculate_metallicity(met_diagnostic="ON_P10", compute_errors=True, niters=1000, df=df, s=" (total)")
        df = metallicity.calculate_metallicity(met_diagnostic="ONS_P10", compute_errors=True, niters=1000, df=df, s=" (total)")
        df = metallicity.calculate_metallicity(met_diagnostic="N2Ha_K19", compute_logU=True, ion_diagnostic="O3O2_K19", compute_errors=True, niters=1000, df=df, s=" (total)")
        df = metallicity.calculate_metallicity(met_diagnostic="O3N2_K19", compute_logU=True, ion_diagnostic="O3O2_K19", compute_errors=True, niters=1000, df=df, s=" (total)")
        df = metallicity.calculate_metallicity(met_diagnostic="N2O2_K19", compute_logU=True, ion_diagnostic="O3O2_K19", compute_errors=True, niters=1000, df=df, s=" (total)")
        df = metallicity.calculate_metallicity(met_diagnostic="R23_KK04", compute_logU=True, ion_diagnostic="O3O2_KK04", compute_errors=True, niters=1000, df=df, s=" (total)")
    else:
        df = metallicity.calculate_metallicity(met_diagnostic="N2Ha_PP04", compute_errors=True, niters=1000, df=df, s=" (total)")
        df = metallicity.calculate_metallicity(met_diagnostic="N2Ha_K19", compute_logU=True, ion_diagnostic="O3O2_K19", compute_errors=True, niters=1000, df=df, s=" (total)")
        # df = metallicity.calculate_metallicity(met_diagnostic="R23_KK04", compute_logU=True, ion_diagnostic="O3O2_KK04", compute_errors=True, niters=1000, df=df, s=" (total)")

    ###############################################################################
    # Save input flags to the DataFrame
    ###############################################################################
    for flag in ["eline_SNR_min", "sigma_gas_SNR_min",
                 "line_flux_SNR_cut", "missing_fluxes_cut", "line_amplitude_SNR_cut",
                 "flux_fraction_cut", "vgrad_cut", "sigma_gas_SNR_cut", "stekin_cut"]:
        df[flag] = kwargs[flag]
    df["Extinction correction applied"] = kwargs["correct_extinction"]

    return df