import os, sys
import numpy as np
from itertools import product
from astropy.io import fits
import pandas as pd
from scipy import constants
from tqdm import tqdm

import extinction
from eline_diagnostics import eline_lambdas_A

import matplotlib.pyplot as plt
plt.ion()
plt.close("all")

from IPython.core.debugger import Tracer

import warnings
warnings.filterwarnings(action="ignore", message="Mean of empty slice")
warnings.filterwarnings(action="ignore", message="invalid value encountered in sqrt")

"""
From https://miocene.anu.edu.au/S7/:

"The emission line information is stored in multiple-extension fits files. These 
fits files were produced using the line-fitting code LZIFU (see Ho et al. 2014) 
and an artificial neural network called The Machine (Hampton et al., in prep). 
LZIFU uses pPXF (Cappellari & Emsellem 2004) to subtract the stellar continuum 
and then uses MPFIT (Markwardt 2009) to fit each emission line spectrum with 1, 
2 and 3 Gaussian emission line components. The Machine then takes the 1, 2 and 
3 component fits and determines how many components are required for each spectrum. 
During the emission line fitting process the fluxes for [NII]6548 are constrained 
to be one third of the fluxes for [NII]6583, and the fluxes for [OIII]4959 are 
constrained to be one third of the fluxes for [OIII]5007 (based on expectations 
from quantum mechanics).

Each LZIFU fits file contains 45 extensions. These extensions contain information 
about the emission-line fluxes and errors, the stellar and gas kinematics and 
the shape of the stellar continuum. The most relevant extensions are those named 
for an emission line, e.g. "HALPHA" and "HALPHA_ERR". The data for MOST of the 
emission line extensions are arranged in 25×38×4 cubes. The 0th 25×38 slice 
contains the total flux for the emission line in each of the 25×38 WiFeS spaxels. 
The 1st(2nd(3rd)) 25×38 slice contains the emission line fluxes for the 1st(2nd(3rd)) 
kinematic component.

In general, we do not have sufficient S/N to constrain the relative intensities 
of the individual kinematic components in [OII]3726,3729 or [OIII]4363. The flux 
and err data for these emission lines are each arranged in a single 25×38 image 
(instead of a 25×38×4 cube), and this image contains the total flux or error of 
the relevant line in each spaxel. Hbeta is also generally too weak to allow us 
to constrain the relative intensities of the individual kinematic components, 
but instead of providing only the total Hbeta flux in each spaxel, we re-distribute 
the total Hbeta flux into components in the same ratio as the Halpha flux components. 
This ensures that the Balmer decrement is consistent across all kinematic components 
within each spaxel.

We are unable to model the true shapes of the Balmer line profiles when a significant 
fraction of the emission line flux is contained within a Seyfert 1 broad component. 
In these spaxels, only the total flux is reported for Halpha and Hbeta."


"""

###############################################################################
# Paths
s7_data_path = "/priv/meggs3/u5708159/S7/"

###############################################################################
# User options
debug = False
if debug:
    print("WARNING: running in debug mode...")

###############################################################################
# Filenames
df_fname = "s7_spaxels_DEBUG.hd5" if debug else "s7_spaxels.hd5"
df_metadata_fname = "s7_metadata.hd5"

###############################################################################
# READ IN THE METADATA
###############################################################################
df_metadata = pd.read_hdf(os.path.join(s7_data_path, df_metadata_fname))
gal_ids_dq_cut = df_metadata[df_metadata["Good?"] == True].index.values
if debug: 
    gal_ids_dq_cut = gal_ids_dq_cut[:5]
df_metadata["Good?"] = df_metadata["Good?"].astype("float")

###############################################################################
# STORING IFS DATA
###############################################################################
df_spaxels = pd.DataFrame()
# for gal in tqdm(gals):
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
    cont_map = np.nanmean(data_cube_R[start_idx:stop_idx], axis=0)
    cont_map_std = np.nanstd(data_cube_R[start_idx:stop_idx], axis=0)
    cont_map_err = 1 / (stop_idx - start_idx) * np.sqrt(np.nansum(var_cube_R[start_idx:stop_idx], axis=0))
    hdulist_R_cube.close() 

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
    # Make A_V maps
    ratio_map = hdulist_processed_cube["HALPHA"].data[0] / hdulist_processed_cube["HBETA"].data[0] 
    ratio_map_err = ratio_map * np.sqrt((hdulist_processed_cube["HALPHA_ERR"].data[0] / hdulist_processed_cube["HALPHA"].data[0])**2 + (hdulist_processed_cube["HBETA_ERR"].data[0] / hdulist_processed_cube["HBETA"].data[0])**2)
    E_ba_map = 2.5 * (np.log10(ratio_map)) - 2.5 * np.log10(2.86)
    E_ba_map_err = 2.5 / np.log(10) * ratio_map_err / ratio_map

    # Calculate ( A(Ha) - A(Hb) ) / E(B-V) from extinction curve
    R_V = 3.1
    E_ba_over_E_BV = float(extinction.fm07(np.array([4861.325]), a_v=1.0) - extinction.fm07(np.array([6562.800]), a_v=1.0)) /  1.0 * R_V
    E_BV_map = 1 / E_ba_over_E_BV * E_ba_map
    E_BV_map_err = 1 / E_ba_over_E_BV * E_ba_map_err

    # Calculate A(V)
    A_V_map = R_V * E_BV_map
    A_V_map_err = R_V * E_BV_map_err

    A_V_map[np.isinf(A_V_map)] = np.nan
    A_V_map[A_V_map < 0] = np.nan

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
    x_0 = df_metadata.loc[df_metadata["catid"] == gal, "x0 (pixels)"].values[0]
    y_0 = df_metadata.loc[df_metadata["catid"] == gal, "y0 (pixels)"].values[0]
    try:
        i_rad = np.deg2rad(float(df_metadata.loc[df_metadata["catid"] == gal, "Inclination i (degrees)"].values[0]))
    except:
        i_rad = 0  # Assume face-on if inclination isn't defined
    try:
        PA_deg = float(df_metadata.loc[df_metadata["catid"] == gal, "pa"].values[0])
    except:
        PA_deg = 0  # Assume NE if inclination isn't defined
    PA_obs_deg = float(df_metadata.loc[df_metadata["catid"] == gal, "WiFeS PA"].values[0])
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
        thisrow["HALPHA continuum"] = cont_map[yy, xx] * 1e16
        thisrow["HALPHA continuum std. dev."] = cont_map_std[yy, xx] * 1e16
        thisrow["HALPHA continuum error"] = cont_map_err[yy, xx] * 1e16
        thisrow["D4000"] = d4000_map[yy, xx]
        thisrow["D4000 error"] = d4000_map_err[yy, xx]
        thisrow[f"A_V (total)"] = A_V_map[yy, xx]
        thisrow[f"A_V error (total)"] = A_V_map_err[yy, xx]
        thisrow[f"A_V error (total)"] = A_V_map_err[yy, xx]

        for nn, component_str in enumerate(["total", "component 0", "component 1", "component 2"]):

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

            # Correct for extinction
            # Commenting this out for now, because we should only really do this
            # if the classification is star-forming.
            # for eline in [e for e in eline_list if f"{e} ({component_str})" in thisrow.keys()]:
            #     A = extinction.fm07(np.array([eline_lambdas_A[eline]]), a_v=A_V_map[nn, yy, xx])[0] if ~np.isnan(A_V_map[nn, yy, xx]) else 0
            #     thisrow[f"{eline} ({component_str})"] *= 10**(0.4 * A)
            #     thisrow[f"{eline} error ({component_str})"] *= 10**(0.4 * A)

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
    df_gal["catid"] = gal
    df_spaxels = df_spaxels.append(df_gal)

###############################################################################
# Merge with metadata
###############################################################################
df_spaxels = df_spaxels.merge(df_metadata, left_on="catid", right_index=True)

###############################################################################
# Compute the ORIGINAL number of components
###############################################################################
df_spaxels["Number of components (original)"] =\
    (~df_spaxels["sigma_gas (component 0)"].isna()).astype(int) +\
    (~df_spaxels["sigma_gas (component 1)"].isna()).astype(int) +\
    (~df_spaxels["sigma_gas (component 2)"].isna()).astype(int)

###############################################################################
# Calculate equivalent widths
###############################################################################
for col in ["HALPHA continuum", "HALPHA continuum error"]:
    df_spaxels[col] = pd.to_numeric(df_spaxels[col])

df_spaxels.loc[df_spaxels["HALPHA continuum"] < 0, "HALPHA continuum"] = 0
for nn in range(3):
    # Cast to float
    df_spaxels[f"HALPHA (component {nn})"] = pd.to_numeric(df_spaxels[f"HALPHA (component {nn})"])
    df_spaxels[f"HALPHA error (component {nn})"] = pd.to_numeric(df_spaxels[f"HALPHA error (component {nn})"])
    # Compute EWs
    df_spaxels[f"HALPHA EW (component {nn})"] = df_spaxels[f"HALPHA (component {nn})"] / df_spaxels["HALPHA continuum"]
    df_spaxels.loc[np.isinf(df_spaxels[f"HALPHA EW (component {nn})"].astype(float)), f"HALPHA EW (component {nn})"] = np.nan  # If the continuum level == 0, then the EW is undefined, so set to NaN.
    df_spaxels[f"HALPHA EW error (component {nn})"] = df_spaxels[f"HALPHA EW (component {nn})"] *\
        np.sqrt((df_spaxels[f"HALPHA error (component {nn})"] / df_spaxels[f"HALPHA (component {nn})"])**2 +\
                (df_spaxels[f"HALPHA continuum error"] / df_spaxels[f"HALPHA continuum"])**2) 

######################################################################
# Add radius-derived value columns
######################################################################
df_spaxels["r/R_e"] = df_spaxels["r (relative to galaxy centre, deprojected, arcsec)"] / df_spaxels["R_e (arcsec)"]
df_spaxels["R_e (kpc)"] = df_spaxels["R_e (arcsec)"] * df_spaxels["kpc per arcsec"]
df_spaxels["log(M/R_e)"] = df_spaxels["mstar"] - np.log10(df_spaxels["R_e (kpc)"])

######################################################################
# Compute S/N in all lines, in all components
# Compute TOTAL line fluxes
######################################################################
for eline in ["HALPHA", "HBETA", "NII6583", "OI6300", "OII3726", "OII3729", "OIII5007", "SII6716", "SII6731"]:
    # Compute S/N 
    for ii in range(3):
        if f"{eline} (component {ii})" in df_spaxels.columns:
            df_spaxels[f"{eline} S/N (component {ii})"] = df_spaxels[f"{eline} (component {ii})"] / df_spaxels[f"{eline} error (component {ii})"]
    
    # Compute total line fluxes, if the total fluxes are not given
    if f"{eline} (total)" not in df_spaxels.columns:
        df_spaxels[f"{eline} (total)"] = np.nansum([df_spaxels[f"{eline} (component {ii})"] for ii in range(3)], axis=0)
        df_spaxels[f"{eline} error (total)"] = np.sqrt(np.nansum([df_spaxels[f"{eline} error (component {ii})"]**2 for ii in range(3)], axis=0))

    # Compute the S/N in the TOTAL line flux
    df_spaxels[f"{eline} S/N (total)"] = df_spaxels[f"{eline} (total)"] / df_spaxels[f"{eline} error (total)"]

######################################################################
# Calculate the TOTAL Halpha EW
######################################################################
df_spaxels["HALPHA EW (total)"] = np.nansum([df_spaxels[f"HALPHA EW (component {ii})"] for ii in range(3)], axis=0)
df_spaxels["HALPHA EW error (total)"] = np.sqrt(np.nansum([df_spaxels[f"HALPHA EW error (component {ii})"]**2 for ii in range(3)], axis=0))

###############################################################################
# Add spaxel scale
###############################################################################
df_spaxels["Bin size (pixels)"] = 1.0
df_spaxels["Bin size (square arcsec)"] = 1.0
df_spaxels["Bin size (square kpc)"] = df_spaxels["kpc per arcsec"]**2

###############################################################################
# Save to .hd5 & .csv
###############################################################################
df_spaxels.to_csv(os.path.join(s7_data_path, df_fname.split("hd5")[0] + "csv"))
try:
    df_spaxels.to_hdf(os.path.join(s7_data_path, df_fname), key=f"S7")
except:
    print("ERROR: unable to save to .hd5... sigh...")
