import os, sys
import numpy as np
from itertools import product
from astropy.io import fits
import pandas as pd
from scipy import constants
from tqdm import tqdm

from spaxelsleuth.loaddata.sami import load_sami_galaxies

import matplotlib.pyplot as plt
plt.ion()
plt.close("all")

from IPython.core.debugger import Tracer

import warnings
warnings.filterwarnings(action="ignore", message="Mean of empty slice")
warnings.filterwarnings(action="ignore", message="invalid value encountered in sqrt")

###############################################################################
"""
This script stores emission line fluxes & other measurements from LZIFU in a 
DataFrame.

Specific galaxies can be specified in command line arguments, e.g.,

    %run make_dfs_lzifu 572402 209807

Separate DataFrames will be saved for each individual galaxy, following the
name format 

    <SAMI ID>_lzifu.hd5

in the directory specified by lzifu_data_path.

If no input arguments are specified, this script will create DataFrames for 
ALL LZIFU galaxies that meet the minimum red S/N requiremnt of

    Median SNR (R, 2R_e) >= 10.0

As above, separate DataFrames for each galaxy will be stored in lzifu_data_path, 
plus a single DataFrame containing all galaxies in the subsample.
"""
###############################################################################
# Paths
sami_data_path = "/priv/meggs3/u5708159/SAMI/sami_dr3/"
sami_datacube_path = "/priv/myrtle1/sami/sami_data/Final_SAMI_data/cube/sami/dr3/"
lzifu_data_path = "/priv/meggs3/u5708159/LZIFU/products/"

###############################################################################
# User options
plotit = False

###############################################################################
# READ IN THE METADATA
###############################################################################
df_metadata_fname = "sami_dr3_metadata_extended.hd5"
df_metadata = pd.read_hdf(os.path.join(sami_data_path, df_metadata_fname))
df_metadata["Good?"] = df_metadata["Good?"].astype("float")

###############################################################################
# List of galaxies
if len(sys.argv) > 1:
    make_master_df = False
    print("WARNING: creating DFs for provided galaxies")
    gals = sys.argv[1:]
    for gal in gals:
        assert gal.isdigit(), "gal must be an integer!"
        fname = os.path.join(lzifu_data_path, f"{gal}_merge_lzcomp.fits")
        assert os.path.exists(fname), f"File {fname} not found!"
    gals = [int(g) for g in gals]
else:
    make_master_df = True
    print("WARNING: creating DFs for all galaxies in the subsample")

    ###############################################################################
    # Make DataFrames for all galaxies in our high-S/N sample
    ###############################################################################
    # List of ALL galaxies we have LZIFU data for
    gals_merge_lzcomp = [int(f.split("_merge_lzcomp.fits")[0]) for f in os.listdir(lzifu_data_path) if f.endswith("merge_lzcomp.fits") and not f.startswith("._")]    
    gals_merge_comp = [int(f.split("_merge_comp.fits")[0]) for f in os.listdir(lzifu_data_path) if f.endswith("merge_comp.fits") and not f.startswith("._")]    
    gals_not_merged = [g for g in gals_merge_comp if g not in gals_merge_lzcomp]
    if len(gals_not_merged) > 0:
        print("WARNING: there are galaxies with LZIFU data that has not been re-component-ized using merge_datacubes()!")

    # Load the DataFrame that gives us the continuum S/N, so we can select a subset
    df_info = pd.read_hdf(os.path.join(sami_data_path, "sami_dr3_metadata_extended.hd5"))

    # Shortlist: median R S/N in 2R_e > 10
    cond_subsample = df_info["Median SNR (R, 2R_e)"] >= 10
    cond_subsample &= df_info["Good?"] == True
    gals_subsample = df_info[cond_subsample].index.values

    # List of galaxies with only 1 component, which we will get from the SAMI data instead
    gals_good_1comp = [g for g in gals_subsample if df_info.loc[g, "Maximum number of components"] == 1]
    gals_good_0comp = [g for g in gals_subsample if df_info.loc[g, "Maximum number of components"] == 0]

    # List of galaxies with LZIFU 
    gals_good_lzifu = [g for g in gals_subsample if g in gals_merge_lzcomp and g not in gals_good_1comp and g not in gals_good_0comp]

    # List of galaxies WITHOUT LZIFU in our sample
    gals_good_missing_lzifu = [g for g in gals_subsample if (g not in gals_merge_lzcomp) and (g not in gals_good_1comp) and (g not in gals_good_0comp)]
    assert len(gals_good_missing_lzifu) == 0,\
        f"{len(gals_good_missing_lzifu)} are missing LZIFU data!"

    # Want to run this script on the gals in the subsample that have LZIFU data.
    gals = gals_good_lzifu

###############################################################################
# Save a separate data frame for each galaxy      
if make_master_df:
    df_all = None  # "big" DF to store all LZIFU galaxies

for gal in tqdm(gals):
    # Filenames
    df_fname = f"lzifu_{gal}.hd5"

    ###############################################################################
    # STORING IFS DATA
    ###############################################################################
    # X, Y pixel coordinates
    ys, xs = np.meshgrid(np.arange(50), np.arange(50), indexing="ij")
    as_per_px = 0.5
    ys_as = ys * as_per_px
    xs_as = xs * as_per_px

    # Centre galaxy coordinates (see p16 of Croom+2021)
    x0_px = 25.5
    y0_px = 25.5

    #######################################################################
    # Open the required FITS files.
    hdulist_B_cube = fits.open(os.path.join(sami_datacube_path, f"ifs/{gal}/{gal}_A_cube_blue.fits.gz"))
    hdulist_R_cube = fits.open(os.path.join(sami_datacube_path, f"ifs/{gal}/{gal}_A_cube_red.fits.gz"))
    hdulist_lzifu = fits.open(os.path.join(lzifu_data_path, f"{gal}_merge_lzcomp.fits"))

    #######################################################################
    # Compute the d4000 Angstrom break.
    header = hdulist_B_cube[0].header
    data_cube_B = hdulist_B_cube[0].data
    var_cube_B = hdulist_B_cube[1].data
    hdulist_B_cube.close()

    # Wavelength values
    lambda_0_A = header["CRVAL3"] - header["CRPIX3"] * header["CDELT3"]
    dlambda_A = header["CDELT3"]
    N_lambda = header["NAXIS3"]
    lambda_vals_A = np.array(range(N_lambda)) * dlambda_A + lambda_0_A 

    # Compute the D4000?? break
    # Definition from Balogh+1999 (see here: https://arxiv.org/pdf/1611.07050.pdf, page 3)
    start_b_idx = np.nanargmin(np.abs(lambda_vals_A / (1 + df_metadata.loc[gal, "z_spec"]) - 3850))
    stop_b_idx = np.nanargmin(np.abs(lambda_vals_A / (1 + df_metadata.loc[gal, "z_spec"]) - 3950))
    start_r_idx = np.nanargmin(np.abs(lambda_vals_A / (1 + df_metadata.loc[gal, "z_spec"]) - 4000))
    stop_r_idx = np.nanargmin(np.abs(lambda_vals_A / (1 + df_metadata.loc[gal, "z_spec"]) - 4100))
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

    #######################################################################
    # Use the red cube to calculate the continuum intensity so 
    # that we can compute the HALPHA equivalent width.
    # Units of 10**(-16) erg /s /cm**2 /angstrom /pixel
    # Continuum wavelength range taken from here: https://ui.adsabs.harvard.edu/abs/2019MNRAS.485.4024V/abstract
    header = hdulist_R_cube[0].header
    data_cube_R = hdulist_R_cube[0].data 
    var_cube_R = hdulist_R_cube[1].data  

    # Wavelength values
    lambda_0_A = header["CRVAL3"] - header["CRPIX3"] * header["CDELT3"]
    dlambda_A = header["CDELT3"]
    N_lambda = header["NAXIS3"]
    lambda_vals_A = np.array(range(N_lambda)) * dlambda_A + lambda_0_A 

    # Compute continuum intensity
    start_idx = np.nanargmin(np.abs(lambda_vals_A / (1 + df_metadata.loc[gal, "z_spec"]) - 6500))
    stop_idx = np.nanargmin(np.abs(lambda_vals_A / (1 + df_metadata.loc[gal, "z_spec"]) - 6540))
    cont_map = np.nanmean(data_cube_R[start_idx:stop_idx], axis=0)
    cont_map_err = 1 / (stop_idx - start_idx) * np.sqrt(np.nansum(var_cube_R[start_idx:stop_idx], axis=0))
    hdulist_R_cube.close() 

    #######################################################################
    # Compute v_grad using eqn. 1 of Zhou+2017
    # Open the velocity & velocity dispersion FITS files
    v = hdulist_lzifu["V"].data.astype(np.float64)[1:]
    v_grad = np.full_like(v, np.nan)

    # Compute v_grad for each spaxel in each component
    # in units of km/s/pixel
    for yy, xx in product(range(1, 49), range(1, 49)):
        v_grad[:, yy, xx] = np.sqrt(((v[:, yy, xx + 1] - v[:, yy, xx - 1]) / 2)**2 +\
                                    ((v[:, yy + 1, xx] - v[:, yy - 1, xx]) / 2)**2)

    #######################################################################
    # Create an image from the datacube to figure out where are "good" spaxels
    im = np.nansum(data_cube_B, axis=0)
    if np.any(im.flatten() < 0): # NaN out -ve spaxels. Most galaxies seem to have *some* -ve pixels
        im[im <= 0] = np.nan

    # Compute the coordinates of "good" spaxels, store in arrays
    y_c_list, x_c_list = np.argwhere(~np.isnan(im)).T
    ngood_bins = len(x_c_list)

    #######################################################################
    # Calculate the inclination
    # I think beta = 90 - PA...
    # Transform coordinates into the galaxy plane
    e = df_metadata.loc[gal, "ellip"]
    PA = df_metadata.loc[gal, "pa"]
    beta_rad = np.deg2rad(PA - 90)
    b_over_a = 1 - e
    q0 = 0.2
    # How to deal with scenario in which b_over_a**2 - q0**2 < 0?
    # This looks kind of dodgy for many galaxies - but the fraction we end up throwing away is about the same as Henry's DR2 work. So leave it for now.
    i_rad = np.arccos(np.sqrt((b_over_a**2 - q0**2) / (1 - q0**2)))  # Want to store this!

    #######################################################################
    # De-project the centroids to the coordinate system of the galaxy plane
    x_prime_list = np.full_like(x_c_list, np.nan, dtype="float")
    y_prime_list = np.full_like(x_c_list, np.nan, dtype="float")
    y_prime_projec_list = np.full_like(x_c_list, np.nan, dtype="float")
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

    #######################################################################
    # For plotting
    if plotit:
        for ax in axs:
            ax.clear()
        fig.suptitle(gal)
        axs[0].imshow(im, origin="lower")
        axs[1].axhline(0)
        axs[1].axvline(0)
        axs[0].scatter(x_c_list, y_c_list, color="k")
        axs[0].scatter(x0_px, y0_px, color="white")
        axs[1].scatter(x_prime_list, y_prime_list, color="r")
        axs[1].scatter(x_prime_list, y_prime_projec_list, color="r", alpha=0.3)
        # Plot circles showing 
        axs[1].axis("equal")
        fig.canvas.draw()

    #######################################################################
    # Open the LZIFU FITS file, extract the values from the maps in each 
    # extension & append
    rows_list = []
    colnames = []

    eline_list = ["HALPHA", "HBETA", "OIII5007", "OI6300", "NII6583", "SII6716", "SII6731"]
    extnames = eline_list + ["V", "VDISP"]

    # Number of kinematic components in each bin
    ncomponent_map = hdulist_lzifu["COMP_MAP"].data

    for ext in extnames:
        data = hdulist_lzifu[ext].data.astype(np.float64)[1:]
        data_err = hdulist_lzifu[f"{ext}_ERR"].data.astype(np.float64)[1:]

        # Extract values from maps in each spaxel, store in a row
        for nn in range(data.shape[0]):
            thisrow = np.full_like(x_c_list, np.nan, dtype="float")
            thisrow_err = np.full_like(x_c_list, np.nan, dtype="float")
            for jj, coords in enumerate(zip(x_c_list, y_c_list)):
                x_c, y_c = coords
                y, x = (int(np.round(y_c)), int(np.round(x_c)))
                thisrow[jj] = data[nn, y, x]
                thisrow_err[jj] = data_err[nn, y, x]
            rows_list.append(thisrow)
            rows_list.append(thisrow_err)
            colnames.append(f"{ext} (component {nn})")
            colnames.append(f"{ext}_ERR (component {nn})")

    hdulist_lzifu.close()

    #######################################################################
    # Do the same but with the stellar kinematics
    hdulist_v_star = fits.open(os.path.join(sami_data_path, f"ifs/{gal}/{gal}_A_stellar-velocity_default_two-moment.fits"))
    v_star = hdulist_v_star[0].data.astype(np.float64)
    v_star_err = hdulist_v_star[1].data.astype(np.float64)
    hdulist_v_star.close()

    hdulist_vdisp_star = fits.open(os.path.join(sami_data_path, f"ifs/{gal}/{gal}_A_stellar-velocity-dispersion_default_two-moment.fits"))
    vdisp_star = hdulist_vdisp_star[0].data.astype(np.float64)
    vdisp_star_err = hdulist_vdisp_star[1].data.astype(np.float64)
    hdulist_vdisp_star.close()

    thisrow = np.full_like(x_c_list, np.nan, dtype="float")
    thisrow_err = np.full_like(x_c_list, np.nan, dtype="float")
    for jj, coords in enumerate(zip(x_c_list, y_c_list)):
        x_c, y_c = coords
        y, x = (int(np.round(y_c)), int(np.round(x_c)))
        thisrow[jj] = v_star[y, x]
        thisrow_err[jj] = v_star_err[y, x]
    rows_list.append(thisrow)
    rows_list.append(thisrow_err)
    colnames.append("v_*")
    colnames.append("v_* error")        

    thisrow = np.full_like(x_c_list, np.nan, dtype="float")
    thisrow_err = np.full_like(x_c_list, np.nan, dtype="float")
    for jj, coords in enumerate(zip(x_c_list, y_c_list)):
        x_c, y_c = coords
        y, x = (int(np.round(y_c)), int(np.round(x_c)))
        thisrow[jj] = vdisp_star[y, x]
        thisrow_err[jj] = vdisp_star_err[y, x]
    rows_list.append(thisrow)
    rows_list.append(thisrow_err)
    colnames.append("sigma_*")
    colnames.append("sigma_* error")  

    #######################################################################
    # Do the same but with the HALPHA extinction correction factor
    hdulist_ext_corr = fits.open(os.path.join(sami_data_path, f"ifs/{gal}/{gal}_A_extinct-corr_default_recom-comp.fits"))
    ext_corr = hdulist_ext_corr[0].data.astype(np.float64)
    ext_corr_err = hdulist_ext_corr[1].data.astype(np.float64)
    hdulist_ext_corr.close()

    thisrow = np.full_like(x_c_list, np.nan, dtype="float")
    thisrow_err = np.full_like(x_c_list, np.nan, dtype="float")
    for jj, coords in enumerate(zip(x_c_list, y_c_list)):
        x_c, y_c = coords
        y, x = (int(np.round(y_c)), int(np.round(x_c)))
        thisrow[jj] = ext_corr[y, x]
        thisrow_err[jj] = ext_corr_err[y, x]
    rows_list.append(thisrow)
    rows_list.append(thisrow_err)
    colnames.append("HALPHA extinction correction")
    colnames.append("HALPHA extinction correction error")        

    #######################################################################
    # Do the same but with the SFR and SFR surface density
    hdulist_sfr = fits.open(os.path.join(sami_data_path, f"ifs/{gal}/{gal}_A_sfr_default_recom-comp.fits"))
    sfr = hdulist_sfr[0].data.astype(np.float64)[1]
    sfr_err = hdulist_sfr[1].data.astype(np.float64)[1]
    hdulist_sfr.close()

    thisrow = np.full_like(x_c_list, np.nan, dtype="float")
    thisrow_err = np.full_like(x_c_list, np.nan, dtype="float")
    for jj, coords in enumerate(zip(x_c_list, y_c_list)):
        x_c, y_c = coords
        y, x = (int(np.round(y_c)), int(np.round(x_c)))
        thisrow[jj] = sfr[y, x]
        thisrow_err[jj] = sfr_err[y, x]
    rows_list.append(thisrow)
    rows_list.append(thisrow_err)
    colnames.append("SFR (total)")
    colnames.append("SFR error (total)")   

    #######################################################################
    # Do the same but with the SFR and SFR surface density
    hdulist_sfrdens = fits.open(os.path.join(sami_data_path, f"ifs/{gal}/{gal}_A_sfr-dens_default_recom-comp.fits"))
    sfrdens = hdulist_sfrdens[0].data.astype(np.float64)[1]
    sfrdens_err = hdulist_sfrdens[1].data.astype(np.float64)[1]
    hdulist_sfrdens.close()

    thisrow = np.full_like(x_c_list, np.nan, dtype="float")
    thisrow_err = np.full_like(x_c_list, np.nan, dtype="float")
    for jj, coords in enumerate(zip(x_c_list, y_c_list)):
        x_c, y_c = coords
        y, x = (int(np.round(y_c)), int(np.round(x_c)))
        thisrow[jj] = sfrdens[y, x]
        thisrow_err[jj] = sfrdens_err[y, x]
    rows_list.append(thisrow)
    rows_list.append(thisrow_err)
    colnames.append("SFR surface density (total)")
    colnames.append("SFR surface density error (total)")

    ####################################################################### 
    # Do the same but with v_grad
    for nn in range(v_grad.shape[0]):
        thisrow = np.full_like(x_c_list, np.nan, dtype="float")
        thisrow_err = np.full_like(x_c_list, np.nan, dtype="float")
        for jj, coords in enumerate(zip(x_c_list, y_c_list)):
            x_c, y_c = coords
            y, x = (int(np.round(y_c)), int(np.round(x_c)))
            thisrow[jj] = v_grad[nn, y, x]
        rows_list.append(thisrow)
        colnames.append(f"v_grad (component {nn})")       

    #######################################################################
    # Do the same but with the continuum intensity for calculating the HALPHA EW
    thisrow = np.full_like(x_c_list, np.nan, dtype="float")
    thisrow_err = np.full_like(x_c_list, np.nan, dtype="float")
    for jj, coords in enumerate(zip(x_c_list, y_c_list)):
        x_c, y_c = coords
        y, x = (int(np.round(y_c)), int(np.round(x_c)))
        thisrow[jj] = cont_map[y, x]
        thisrow_err[jj] = cont_map_err[y, x]
    rows_list.append(thisrow)
    rows_list.append(thisrow_err)
    colnames.append("HALPHA continuum")
    colnames.append("HALPHA continuum error")        

    #######################################################################
    # Do the same but with the D4000?? break
    thisrow = np.full_like(x_c_list, np.nan, dtype="float")
    thisrow_err = np.full_like(x_c_list, np.nan, dtype="float")
    for jj, coords in enumerate(zip(x_c_list, y_c_list)):
        x_c, y_c = coords
        y, x = (int(np.round(y_c)), int(np.round(x_c)))
        thisrow[jj] = d4000_map[y, x]
        thisrow_err[jj] = d4000_map_err[y, x]
    rows_list.append(thisrow)
    rows_list.append(thisrow_err)
    colnames.append("D4000")
    colnames.append("D4000 error")        

    #######################################################################
    # Do the same but with the number of kinematic components
    thisrow = np.full_like(x_c_list, np.nan, dtype="float")
    thisrow_err = np.full_like(x_c_list, np.nan, dtype="float")
    for jj, coords in enumerate(zip(x_c_list, y_c_list)):
        x_c, y_c = coords
        y, x = (int(np.round(y_c)), int(np.round(x_c)))
        thisrow[jj] = ncomponent_map[y, x]
    rows_list.append(thisrow)
    colnames.append("Number of components")   

    #######################################################################
    # Add pixel coordinates
    rows_list.append(np.array([np.rad2deg(i_rad)] * ngood_bins))
    rows_list.append(np.array([x0_px] * ngood_bins) * as_per_px)
    rows_list.append(np.array([y0_px] * ngood_bins) * as_per_px)
    rows_list.append(np.array(x_c_list).flatten() * as_per_px)
    rows_list.append(np.array(y_c_list).flatten() * as_per_px)
    rows_list.append(np.array(x_prime_list).flatten() * as_per_px)
    rows_list.append(np.array(y_prime_list).flatten() * as_per_px)
    rows_list.append(np.array(r_prime_list).flatten() * as_per_px)
    rows_list.append(np.array([1] * ngood_bins))
    rows_list.append(np.array([as_per_px**2] * ngood_bins))
    rows_list.append(np.array([as_per_px**2 * df_metadata.loc[gal, "kpc per arcsec"]**2] * ngood_bins))
    colnames.append("Inclination i (degrees)")
    colnames.append("Galaxy centre x0_px (projected, arcsec)")
    colnames.append("Galaxy centre y0_px (projected, arcsec)")
    colnames.append("x (projected, arcsec)")
    colnames.append("y (projected, arcsec)")
    colnames.append("x (relative to galaxy centre, deprojected, arcsec)")
    colnames.append("y (relative to galaxy centre, deprojected, arcsec)")
    colnames.append("r (relative to galaxy centre, deprojected, arcsec)")
    colnames.append("Bin size (pixels)")
    colnames.append("Bin size (square arcsec)")
    colnames.append("Bin size (square kpc)")

    # Transpose so that each row represents a single pixel & each column a measured quantity.
    rows_arr = np.array(rows_list).T

    # Get rid of rows that are all NaNs
    bad_rows = np.all(np.isnan(rows_arr), axis=1)
    rows_good = rows_arr[~bad_rows]
    ngood = rows_good.shape[0]

    # Append a column with the galaxy ID & other properties
    safe_cols = [c for c in df_metadata.columns if c != "Morphology"]
    gal_metadata = np.tile(df_metadata[df_metadata.loc[:, "catid"] == gal][safe_cols].values, (ngood_bins, 1))
    rows_good = np.hstack((gal_metadata, rows_good))

    ###############################################################################
    # Convert to a Pandas DataFrame
    ###############################################################################
    df_spaxels = pd.DataFrame(rows_good, columns=safe_cols + colnames)

    ###############################################################################
    # Add the morphology column back in
    ###############################################################################
    morph_dict = {
        "0.0": "E",
        "0.5": "E/S0",
        "1.0": "S0",
        "1.5": "S0/Early-spiral",
        "2.0": "Early-spiral",
        "2.5": "Early/Late spiral",
        "3.0": "Late spiral",
        "5.0": "?",
        "-9.0": "no agreement",
        "-0.5": "Unknown"
    }
    df_spaxels["Morphology"] = [morph_dict[str(m)] for m in df_spaxels["Morphology (numeric)"]]
   
    ###############################################################################
    # Rename some columns
    ###############################################################################
    rename_dict = {}
    for eline in eline_list:
        for ii in range(3):
            rename_dict[f"{eline}_ERR (component {ii})"] = f"{eline} error (component {ii})"
    for ii in range(3):
        rename_dict[f"V (component {ii})"] = f"v_gas (component {ii})"
        rename_dict[f"V_ERR (component {ii})"] = f"v_gas error (component {ii})"
    for ii in range(3):
        rename_dict[f"VDISP (component {ii})"] = f"sigma_gas (component {ii})"
        rename_dict[f"VDISP_ERR (component {ii})"] = f"sigma_gas error (component {ii})"

    # R_e
    rename_dict["r_e"] = "R_e (arcsec)"

    # Rename columns
    df_spaxels = df_spaxels.rename(columns=rename_dict)

    ###############################################################################
    # Calculate equivalent widths
    ###############################################################################
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
    for eline in ["HALPHA", "HBETA", "NII6583", "OI6300", "OIII5007", "SII6716", "SII6731"]:
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
    # Calculate the TOTAL HALPHA EW
    ######################################################################
    df_spaxels["HALPHA EW (total)"] = np.nansum([df_spaxels[f"HALPHA EW (component {ii})"] for ii in range(3)], axis=0)
    df_spaxels["HALPHA EW error (total)"] = np.sqrt(np.nansum([df_spaxels[f"HALPHA EW error (component {ii})"]**2 for ii in range(3)], axis=0))

    ######################################################################
    # Compute the SFR and SFR surface density from the 0th component ONLY
    ######################################################################
    df_spaxels["SFR surface density (component 0)"] = df_spaxels["SFR surface density (total)"] * df_spaxels["HALPHA (component 0)"] / df_spaxels["HALPHA (total)"]
    df_spaxels["SFR surface density error (component 0)"] = df_spaxels["SFR surface density error (total)"] * df_spaxels["HALPHA (component 0)"] / df_spaxels["HALPHA (total)"]
    df_spaxels["SFR (component 0)"] = df_spaxels["SFR (total)"] * df_spaxels["HALPHA (component 0)"] / df_spaxels["HALPHA (total)"]
    df_spaxels["SFR error (component 0)"] = df_spaxels["SFR error (total)"] * df_spaxels["HALPHA (component 0)"] / df_spaxels["HALPHA (total)"]

    ######################################################################
    # Save to .hd5
    ######################################################################
    # Add catid column
    df_spaxels["catid"] = gal
    df_spaxels.to_hdf(os.path.join(sami_data_path, df_fname), key=f"LZIFU")

    ######################################################################
    # Concatenate 
    ######################################################################
    if make_master_df:
        if df_all is not None:
            df_all = pd.concat([df_all, df_spaxels], ignore_index=True)
        else:
            df_all = df_spaxels

######################################################################
# Tack on the 1 and 0-component galaxies to the master DataFrame
######################################################################
if make_master_df:
    # Load the SAMI sample
    df_sami = pd.read_hdf(os.path.join(sami_data_path, "sami_default_recom-comp.hd5"), key=f"default, recom-comp")

    # Add the information for the 0- and 1-component galaxies
    for gal in gals_good_1comp + gals_good_0comp:
        df_gal = df_sami[df_sami["catid"] == gal] 
        df_all = pd.concat([df_all, df_gal], ignore_index=True)

    ######################################################################
    # Save to file 
    ######################################################################
    df_all.to_hdf(os.path.join(sami_data_path, f"lzifu_subsample.hd5"), key="LZIFU")


