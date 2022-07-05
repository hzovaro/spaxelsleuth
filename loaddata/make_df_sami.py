import os, sys
import numpy as np
from itertools import product
from astropy.io import fits
import pandas as pd
from scipy import constants
from tqdm import tqdm
import multiprocessing

import linefns

import matplotlib.pyplot as plt
plt.ion()
plt.close("all")

from IPython.core.debugger import Tracer

import warnings
warnings.filterwarnings(action="ignore", message="Mean of empty slice")
warnings.filterwarnings(action="ignore", message="invalid value encountered in sqrt")


"""
This script is used to create a Pandas DataFrame containing emission line 
fluxes & kinematics, stellar kinematics, extinction, star formation rates, 
and other quantities for all galaxies as taken from SAMI DR3. 

FITS files containing each quantity for each galaxy
    e.g., "<gal>_Halpha_<bin type>_<number of components>-comp.fits"
must be stored here:
    SAMI_DIR/ifs/<gal>/

The output is stored as a Pandas DataFrame in which each row corresponds
to a given spaxel (or Voronoi bin) for every galaxy.

"""

###############################################################################
# Paths
sami_data_path = os.environ["SAMI_DIR"]
assert "SAMI_DIR" in os.environ, "Environment variable SAMI_DIR is not defined!"
sami_datacube_path = os.environ["SAMI_DATACUBE_DIR"]
assert "SAMI_DATACUBE_DIR" in os.environ, "Environment variable SAMI_DATACUBE_DIR is not defined!"

###############################################################################
# User options
ncomponents = sys.argv[1]       # Options: "1" or "recom"
bin_type = sys.argv[2]          # Options: "default" (i.e. no binning) or "adaptive" (i.e. Voronoi binning)
assert bin_type in ["default", "adaptive"], "bin_type must be 'default' or 'adaptive'!"
nthreads_max = 20               # Maximum number of threds to use 

# For printing to stdout
status_str = f"In sami.load_sami_galaxies() [bin_type={bin_type}, ncomponents={ncomponents}]"

# If run in debug mode, the script will only be run for one galaxy.
debug = False 
plotit = False
if debug:
    print(f"{status_str}: WARNING: running in debug mode...")

###############################################################################
# Filenames
df_metadata_fname = "sami_dr3_metadata.hd5"

# Output file name 
df_fname = f"sami_{bin_type}_{ncomponents}-comp"
df_fname_extcorr = f"sami_{bin_type}_{ncomponents}-comp_extcorr"
if debug:
    df_fname += "_DEBUG"
    df_fname_extcorr += "_DEBUG"
df_fname += ".hd5"
df_fname_extcorr += ".hd5"

print(f"{status_str}: saving to files {df_fname} and {df_fname_extcorr}...")

# ###############################################################################
# # Seeing why the extinction correction breaks...
# ###############################################################################
# df_spaxels = pd.read_hdf(os.path.join(sami_data_path, df_fname))
# df_spaxels_extcorr = df_spaxels.copy()
# df_spaxels_extcorr = linefns.extcorr.extinction_corr_fn(df_spaxels_extcorr, 
#                                 eline_list=["HALPHA", "HBETA", "NII6583", "OI6300", "OII3726+OII3729", "OIII5007", "SII6716", "SII6731"],
#                                 reddening_curve="fm07", 
#                                 balmer_SNR_min=5, nthreads=nthreads_max,
#                                 s=f" (total)")
# df_spaxels_extcorr["Corrected for extinction?"] = True
# df_spaxels["Corrected for extinction?"] = False

# sys.exit()

###############################################################################
# READ IN THE METADATA
###############################################################################
df_metadata = pd.read_hdf(os.path.join(sami_data_path, df_metadata_fname), key="metadata")
gal_ids_dq_cut = df_metadata[df_metadata["Good?"] == True].index.values
if debug: 
    gal_ids_dq_cut = gal_ids_dq_cut[:10]
df_metadata["Good?"] = df_metadata["Good?"].astype("float")

# Turn off plotting if more than 1 galaxy is to be run
if plotit and len(gal_ids_dq_cut) > 1:
    plotit = False

###############################################################################
# STORING IFS DATA
###############################################################################

# List of filenames
fname_list = [
    f"Halpha_{bin_type}_{ncomponents}-comp",
    f"Hbeta_{bin_type}_{ncomponents}-comp",
    f"NII6583_{bin_type}_{ncomponents}-comp",
    f"OI6300_{bin_type}_{ncomponents}-comp",
    f"OII3728_{bin_type}_{ncomponents}-comp",
    f"OIII5007_{bin_type}_{ncomponents}-comp",
    f"SII6716_{bin_type}_{ncomponents}-comp",
    f"SII6731_{bin_type}_{ncomponents}-comp",
    f"gas-vdisp_{bin_type}_{ncomponents}-comp",
    f"gas-velocity_{bin_type}_{ncomponents}-comp",
    f"stellar-velocity-dispersion_{bin_type}_two-moment",
    f"stellar-velocity_{bin_type}_two-moment",
    f"extinct-corr_{bin_type}_{ncomponents}-comp",
    f"sfr-dens_{bin_type}_{ncomponents}-comp",
    f"sfr_{bin_type}_{ncomponents}-comp"
]

# X, Y pixel coordinates
ys, xs = np.meshgrid(np.arange(50), np.arange(50), indexing="ij")
as_per_px = 0.5
ys_as = ys * as_per_px
xs_as = xs * as_per_px

# Centre galaxy coordinates (see p16 of Croom+2021)
x0_px = 25.5
y0_px = 25.5

################################################################################
def process_gals(args):
    gal_idx, gal = args
    fnames = [os.path.join(sami_data_path, f"ifs/{gal}/{gal}_A_{f}.fits") for f in fname_list]
    rows_list = []

    #######################################################################
    # Open the red & blue cubes.
    hdulist_B_cube = fits.open(os.path.join(sami_datacube_path, f"ifs/{gal}/{gal}_A_cube_blue.fits.gz"))
    hdulist_R_cube = fits.open(os.path.join(sami_datacube_path, f"ifs/{gal}/{gal}_A_cube_red.fits.gz"))

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

    # Compute the D4000Å break
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
    # that we can compute the Halpha equivalent width.
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
    cont_map_std = np.nanstd(data_cube_R[start_idx:stop_idx], axis=0)
    cont_map_err = 1 / (stop_idx - start_idx) * np.sqrt(np.nansum(var_cube_R[start_idx:stop_idx], axis=0))
    hdulist_R_cube.close() 

    #######################################################################
    # Compute v_grad using eqn. 1 of Zhou+2017
    hdulist_v = fits.open(os.path.join(
        sami_data_path, f"ifs/{gal}/{gal}_A_gas-velocity_{bin_type}_{ncomponents}-comp.fits"))

    # Open the velocity & velocity dispersion FITS files
    v = hdulist_v[0].data.astype(np.float64)
    v_grad = np.full_like(v, np.nan)

    # Compute v_grad for each spaxel in each component
    # in units of km/s/pixel
    for yy, xx in product(range(1, 49), range(1, 49)):
        v_grad[:, yy, xx] = np.sqrt(((v[:, yy, xx + 1] - v[:, yy, xx - 1]) / 2)**2 +\
                                    ((v[:, yy + 1, xx] - v[:, yy - 1, xx]) / 2)**2)

    hdulist_v.close()

    #######################################################################
    # Compute the spaxel or bin coordinates, depending on the binning scheme 
    im = np.nansum(data_cube_B, axis=0)
    if bin_type == "default":
        # Create an image from the datacube to figure out where are "good" spaxels
        if np.any(im.flatten() < 0): # NaN out -ve spaxels. Most galaxies seem to have *some* -ve pixels
            im[im <= 0] = np.nan

        # Compute the coordinates of "good" spaxels, store in arrays
        y_c_list, x_c_list = np.argwhere(~np.isnan(im)).T
        ngood_bins = len(x_c_list)

        # List of bin sizes, in pixels
        bin_size_list_px = [1] * ngood_bins
        bin_number_list = np.arange(1, ngood_bins + 1)

    # Compute the light-weighted bin centres, based on the blue unbinned
    # data cube
    elif bin_type == "adaptive":
        # Open the binned blue cube. Get the bin mask extension.
        hdulist_binned_cube = fits.open(os.path.join(sami_data_path, f"ifs/{gal}/{gal}_A_adaptive_blue.fits.gz"))
        bin_map = hdulist_binned_cube[2].data.astype("float")
        bin_map[bin_map==0] = np.nan

        bin_number_list = np.array([nn for nn in np.unique(bin_map) if ~np.isnan(nn)])
        nbins = len(bin_number_list)
        x_c_list = np.full(nbins, np.nan)
        y_c_list = np.full(nbins, np.nan)
        bin_size_list_px = np.full(nbins, np.nan)
        for ii, nn in enumerate(bin_number_list):
            # generate a bin mask.
            bin_mask = bin_map == nn
            bin_size_list_px[ii] = len(bin_mask[bin_mask == True])
            # compute the centroid of the bin.
            x_c = np.nansum(xs * bin_mask * im) / np.nansum(bin_mask * im)
            y_c = np.nansum(ys * bin_mask * im) / np.nansum(bin_mask * im)
            # Don't add the centroids if they are out of bounds.
            if (x_c < 0 or x_c >= 50 or y_c < 0 or y_c >= 50):
                x_c_list[ii] = np.nan
                y_c_list[ii] = np.nan
            else:
                x_c_list[ii] = x_c
                y_c_list[ii] = y_c

        #######################################################################
        # Bin numbers corresponding to bins actually present in the image
        good_bins = np.argwhere(~np.isnan(x_c_list)).flatten()
        ngood_bins = len(good_bins)
        x_c_list = x_c_list[good_bins]
        y_c_list = y_c_list[good_bins]
        bin_size_list_px = bin_size_list_px[good_bins]
        bin_number_list = bin_number_list[good_bins]  

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

    #######################################################################
    # Open each FITS file, extract the values from the maps in each bin & append
    rows_list = []
    colnames = []
    for ff, fname in enumerate(fnames):
        hdu = fits.open(fname)
        data = hdu[0].data.astype(np.float64)
        data_err = hdu[1].data.astype(np.float64)
        hdu.close()

        # Extract values from maps at the bin centroids, store in a row
        if data.ndim == 2:
            # Applies to emission lines, extinction
            thisrow = np.full_like(x_c_list, np.nan, dtype="float")
            thisrow_err = np.full_like(x_c_list, np.nan, dtype="float")
            for jj, coords in enumerate(zip(x_c_list, y_c_list)):
                x_c, y_c = coords
                y, x = (int(np.round(y_c)), int(np.round(x_c)))
                thisrow[jj] = data[y, x]
                thisrow_err[jj] = data_err[y, x]
            rows_list.append(thisrow)
            rows_list.append(thisrow_err)
            colnames.append(fname_list[ff])
            colnames.append(fname_list[ff] + "_error")

        else:
            # The 0th dimension of the Halpha & SFR maps contain the total values, 
            # which we can calculate later
            if "Halpha" in fname or "sfr" in fname:
                data = data[1:]
                data_err = data_err[1:]
            # Add individual components 
            # Applies to Halpha flux, gas velocity & velocity dispersion
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
                colnames.append(f"{fname_list[ff]} (component {nn})")
                colnames.append(f"{fname_list[ff]}_error (component {nn})")

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
    thisrow_std = np.full_like(x_c_list, np.nan, dtype="float")
    thisrow_err = np.full_like(x_c_list, np.nan, dtype="float")
    for jj, coords in enumerate(zip(x_c_list, y_c_list)):
        x_c, y_c = coords
        y, x = (int(np.round(y_c)), int(np.round(x_c)))
        thisrow[jj] = cont_map[y, x]
        thisrow_std[jj] = cont_map_std[y, x]
        thisrow_err[jj] = cont_map_err[y, x]
    rows_list.append(thisrow)
    rows_list.append(thisrow_std)
    rows_list.append(thisrow_err)
    colnames.append("HALPHA continuum")
    colnames.append("HALPHA continuum std. dev.")
    colnames.append("HALPHA continuum error")        

    #######################################################################
    # Do the same but with the D4000Å break
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
    # Add pixel coordinates
    rows_list.append(np.array([np.rad2deg(i_rad)] * ngood_bins))
    rows_list.append(np.array([x0_px] * ngood_bins) * as_per_px)
    rows_list.append(np.array([y0_px] * ngood_bins) * as_per_px)
    rows_list.append(np.array(x_c_list).flatten() * as_per_px)
    rows_list.append(np.array(y_c_list).flatten() * as_per_px)
    rows_list.append(np.array(x_prime_list).flatten() * as_per_px)
    rows_list.append(np.array(y_prime_list).flatten() * as_per_px)
    rows_list.append(np.array(r_prime_list).flatten() * as_per_px)
    rows_list.append(np.array(bin_number_list))
    rows_list.append(np.array(bin_size_list_px))
    rows_list.append(np.array(bin_size_list_px) * as_per_px**2)
    rows_list.append(np.array(bin_size_list_px) * as_per_px**2 * df_metadata.loc[gal, "kpc per arcsec"]**2)
    colnames.append("Inclination i (degrees)")
    colnames.append("Galaxy centre x0_px (projected, arcsec)")
    colnames.append("Galaxy centre y0_px (projected, arcsec)")
    colnames.append("x (projected, arcsec)")
    colnames.append("y (projected, arcsec)")
    colnames.append("x (relative to galaxy centre, deprojected, arcsec)")
    colnames.append("y (relative to galaxy centre, deprojected, arcsec)")
    colnames.append("r (relative to galaxy centre, deprojected, arcsec)")
    colnames.append("Bin number")
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
    gal_metadata = np.tile(df_metadata.loc[df_metadata.loc[:, "catid"] == gal][safe_cols].values, (ngood_bins, 1))
    rows_good = np.hstack((gal_metadata, rows_good))

    print(f"{status_str}: Finished processing {gal} ({gal_idx}/{len(gal_ids_dq_cut)})")

    return rows_good, colnames 

###############################################################################
# Run in parallel
###############################################################################
args_list = [[ii, g] for ii, g in enumerate(gal_ids_dq_cut)]

if len(gal_ids_dq_cut) == 1:
    res_list = [process_gals(args_list[0])]
else:
    print(f"{status_str}: Beginning pool...")
    pool = multiprocessing.Pool(min([nthreads_max, len(gal_ids_dq_cut)]))
    res_list = np.array((pool.map(process_gals, args_list)))
    pool.close()
    pool.join()

###############################################################################
# Convert to a Pandas DataFrame
###############################################################################
rows_list_all = [r[0] for r in res_list]
colnames = res_list[0][1]
safe_cols = [c for c in df_metadata.columns if c != "Morphology"]
df_spaxels = pd.DataFrame(np.vstack(tuple(rows_list_all)), columns=safe_cols + colnames)

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
# Rename columns
###############################################################################
# Only necessary for the adaptively-binned and unbinned data - not necessary for the aperture data
print(f"{status_str}: Renaming columns...")
rename_dict = {}
sami_colnames = [col.split(f"_{bin_type}_{ncomponents}-comp")[0].upper() for col in df_spaxels.columns if col.endswith(f"_{bin_type}_{ncomponents}-comp")]

# Emission lines except for Halpha
for col in [c for c in df_spaxels.columns if c.endswith(f"_{bin_type}_{ncomponents}-comp")]:
    eline = col.split(f"_{bin_type}_{ncomponents}-comp")[0].upper()
    if eline == "OII3728":
        eline = "OII3726+OII3729"
    if eline in ["HALPHA", "HBETA", "NII6583", "OI6300", "OII3726+OII3729", "OIII5007", "SII6716", "SII6731"]:
        rename_dict[col] = f"{eline} (total)"
        rename_dict[f"{col}_error"] = f"{eline} error (total)"

# Halpha & kinematics
for ii in range(3 if ncomponents == "recom" else 1):
    rename_dict[f"Halpha_{bin_type}_{ncomponents}-comp (component {ii})"] = f"HALPHA (component {ii})"
    rename_dict[f"Halpha_{bin_type}_{ncomponents}-comp_error (component {ii})"] = f"HALPHA error (component {ii})"
    rename_dict[f"gas-vdisp_{bin_type}_{ncomponents}-comp (component {ii})"] = f"sigma_gas (component {ii})"
    rename_dict[f"gas-vdisp_{bin_type}_{ncomponents}-comp_error (component {ii})"] = f"sigma_gas error (component {ii})"
    rename_dict[f"gas-velocity_{bin_type}_{ncomponents}-comp (component {ii})"] = f"v_gas (component {ii})"
    rename_dict[f"gas-velocity_{bin_type}_{ncomponents}-comp_error (component {ii})"] = f"v_gas error (component {ii})"

    # Star formation rate
    rename_dict[f"sfr_{bin_type}_{ncomponents}-comp (component {ii})"] = f"SFR (component {ii})"
    rename_dict[f"sfr_{bin_type}_{ncomponents}-comp_error (component {ii})"] = f"SFR error (component {ii})"
    rename_dict[f"sfr-dens_{bin_type}_{ncomponents}-comp (component {ii})"] = f"SFR surface density (component {ii})"
    rename_dict[f"sfr-dens_{bin_type}_{ncomponents}-comp_error (component {ii})"] = f"SFR surface density error (component {ii})"

# Stellar kinematics
rename_dict[f"stellar-velocity-dispersion_{bin_type}_two-moment"] = "sigma_*"
rename_dict[f"stellar-velocity-dispersion_{bin_type}_two-moment_error"] = "sigma_* error"
rename_dict[f"stellar-velocity_{bin_type}_two-moment"] = "v_*"
rename_dict[f"stellar-velocity_{bin_type}_two-moment_error"] = "v_* error"

# Halpha attenuation correction
rename_dict[f"extinct-corr_{bin_type}_{ncomponents}-comp"] = "HALPHA extinction correction"
rename_dict[f"extinct-corr_{bin_type}_{ncomponents}-comp_error"] = "HALPHA extinction correction error"

# R_e
rename_dict["r_e"] = "R_e (arcsec)"

# Rename columns
df_spaxels = df_spaxels.rename(columns=rename_dict)

###############################################################################
# Compute the ORIGINAL number of components
###############################################################################
if ncomponents == "recom":
    df_spaxels["Number of components (original)"] =\
        (~df_spaxels["sigma_gas (component 0)"].isna()).astype(int) +\
        (~df_spaxels["sigma_gas (component 1)"].isna()).astype(int) +\
        (~df_spaxels["sigma_gas (component 2)"].isna()).astype(int)
elif ncomponents == "1":
    df_spaxels["Number of components (original)"] =\
        (~df_spaxels["sigma_gas (component 0)"].isna()).astype(int)

###############################################################################
# Calculate equivalent widths
###############################################################################
for col in ["HALPHA continuum", "HALPHA continuum error", "HALPHA continuum std. dev."]:
    df_spaxels[col] = pd.to_numeric(df_spaxels[col])

df_spaxels.loc[df_spaxels["HALPHA continuum"] < 0, "HALPHA continuum"] = 0
for nn in range(3 if ncomponents == "recom" else 1):
    # Cast to float
    df_spaxels[f"HALPHA (component {nn})"] = pd.to_numeric(df_spaxels[f"HALPHA (component {nn})"])
    df_spaxels[f"HALPHA error (component {nn})"] = pd.to_numeric(df_spaxels[f"HALPHA error (component {nn})"])
   
    # Compute EWs
    df_spaxels[f"HALPHA EW (component {nn})"] = df_spaxels[f"HALPHA (component {nn})"] / df_spaxels["HALPHA continuum"]
    df_spaxels[f"HALPHA EW error (component {nn})"] = df_spaxels[f"HALPHA EW (component {nn})"] *\
        np.sqrt((df_spaxels[f"HALPHA error (component {nn})"] / df_spaxels[f"HALPHA (component {nn})"])**2 +\
                (df_spaxels[f"HALPHA continuum error"] / df_spaxels[f"HALPHA continuum"])**2) 
    
    # If the continuum level <= 0, then the EW is undefined, so set to NaN.
    df_spaxels.loc[df_spaxels["HALPHA continuum"] <= 0, 
                   [f"HALPHA EW (component {nn})", 
                    f"HALPHA EW error (component {nn})"]] = np.nan  

# Calculate total EWs
df_spaxels["HALPHA EW (total)"] = np.nansum([df_spaxels[f"HALPHA EW (component {ii})"] for ii in range(3 if ncomponents == "recom" else 1)], axis=0)
df_spaxels["HALPHA EW error (total)"] = np.sqrt(np.nansum([df_spaxels[f"HALPHA EW error (component {ii})"]**2 for ii in range(3 if ncomponents == "recom" else 1)], axis=0))

# If all HALPHA EWs are NaN, then make the total HALPHA EW NaN too
if ncomponents == "recom":
    df_spaxels.loc[df_spaxels["HALPHA EW (component 0)"].isna() &\
                   df_spaxels["HALPHA EW (component 1)"].isna() &\
                   df_spaxels["HALPHA EW (component 2)"].isna(), 
                   ["HALPHA EW (total)", "HALPHA EW error (total)"]] = np.nan
elif ncomponents == "1":
    df_spaxels.loc[df_spaxels["HALPHA EW (component 0)"].isna(),
                   ["HALPHA EW (total)", "HALPHA EW error (total)"]] = np.nan

######################################################################
# SFR and SFR surface density
######################################################################
# Drop components 1 and 2 because they are always zero
if ncomponents == "recom":
    for ii in [1, 2]:
        df_spaxels = df_spaxels.drop(
            columns=[f"SFR (component {ii})", f"SFR error (component {ii})",
                     f"SFR surface density (component {ii})", f"SFR surface density error (component {ii})"])

# NOW we can rename SFR (compnent 0) to SFR (total)
rename_dict = {}
rename_dict["SFR (component 0)"] = "SFR (total)"
rename_dict["SFR error (component 0)"] = "SFR error (total)"
rename_dict["SFR surface density (component 0)"] = "SFR surface density (total)"
rename_dict["SFR surface density error (component 0)"] = "SFR surface density error (total)"

df_spaxels = df_spaxels.rename(columns=rename_dict)

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
for eline in ["HALPHA", "HBETA", "NII6583", "OI6300", "OII3726+OII3729", "OIII5007", "SII6716", "SII6731"]:
    # Compute S/N 
    for ii in range(3 if ncomponents == "recom" else 1):
        if f"{eline} (component {ii})" in df_spaxels.columns:
            df_spaxels[f"{eline} S/N (component {ii})"] = df_spaxels[f"{eline} (component {ii})"] / df_spaxels[f"{eline} error (component {ii})"]
    
    # Compute total line fluxes, if the total fluxes are not given
    if f"{eline} (total)" not in df_spaxels.columns:
        df_spaxels[f"{eline} (total)"] = np.nansum([df_spaxels[f"{eline} (component {ii})"] for ii in range(3 if ncomponents == "recom" else 1)], axis=0)
        df_spaxels[f"{eline} error (total)"] = np.sqrt(np.nansum([df_spaxels[f"{eline} error (component {ii})"]**2 for ii in range(3 if ncomponents == "recom" else 1)], axis=0))

    # Compute the S/N in the TOTAL line flux
    df_spaxels[f"{eline} S/N (total)"] = df_spaxels[f"{eline} (total)"] / df_spaxels[f"{eline} error (total)"]

######################################################################
# Compute the SFR and SFR surface density from the 0th component ONLY
######################################################################
df_spaxels["SFR surface density (component 0)"] = df_spaxels["SFR surface density (total)"] * df_spaxels["HALPHA (component 0)"] / df_spaxels["HALPHA (total)"]
df_spaxels["SFR surface density error (component 0)"] = df_spaxels["SFR surface density error (total)"] * df_spaxels["HALPHA (component 0)"] / df_spaxels["HALPHA (total)"]
df_spaxels["SFR (component 0)"] = df_spaxels["SFR (total)"] * df_spaxels["HALPHA (component 0)"] / df_spaxels["HALPHA (total)"]
df_spaxels["SFR error (component 0)"] = df_spaxels["SFR error (total)"] * df_spaxels["HALPHA (component 0)"] / df_spaxels["HALPHA (total)"]

######################################################################
# Make a copy of the DataFrame with EXTINCTION CORRECTION
# Correct emission line fluxes (but not EWs!)
# NOTE: extinction.fm07 assumes R_V = 3.1 so do not change R_V from 
# this value!!!
######################################################################
print(f"{status_str}: Correcting emission line fluxes (but not EWs) for extinction...")
df_spaxels_extcorr = df_spaxels.copy()
df_spaxels_extcorr = extcorr.extinction_corr_fn(df_spaxels_extcorr, 
                                eline_list=["HALPHA", "HBETA", "NII6583", "OI6300", "OII3726+OII3729", "OIII5007", "SII6716", "SII6731"],
                                reddening_curve="fm07", 
                                balmer_SNR_min=5, nthreads=nthreads_max,
                                s=f" (total)")
df_spaxels_extcorr["Corrected for extinction?"] = True
df_spaxels["Corrected for extinction?"] = False

# Sort so that both DataFrames have the same order
df_spaxels_extcorr = df_spaxels_extcorr.sort_index()
df_spaxels = df_spaxels.sort_index()

###############################################################################
# Save to .hd5 & .csv
###############################################################################
print(f"{status_str}: Saving to file...")

# No extinction correction
df_spaxels.to_csv(os.path.join(sami_data_path, df_fname.split("hd5")[0] + "csv"))
try:
    df_spaxels.to_hdf(os.path.join(sami_data_path, df_fname), key=f"{bin_type}, {ncomponents}-comp")
except:
    print(f"{status_str}: Unable to save to HDF file... sigh...")

# With extinction correction
df_spaxels_extcorr.to_csv(os.path.join(sami_data_path, df_fname_extcorr.split("hd5")[0] + "csv"))
try:
    df_spaxels_extcorr.to_hdf(os.path.join(sami_data_path, df_fname_extcorr), key=f"{bin_type}, {ncomponents}-comp")
except:
    print(f"{status_str}: Unable to save to HDF file... sigh...")

