"""
File:       make_sami_metadata_df_extended.py
Author:     Henry Zovaro
Email:      henry.zovaro@anu.edu.au

DESCRIPTION
------------------------------------------------------------------------------
This script is used to create an extended version of the DataFrame containing 
metadata for each galaxy. 

Additional quantities computed in this script include 
- compute continuum S/N values in both blue & red SAMI cubes in a variety of 
  different apertures, making it simpler to select high-S/N targets for your
  science case;
- total SFRs computed from the SFR maps provided in SAMI DR3;
- galaxy inclinations 
- the maximum number of components fitted in any spaxel in each galaxy.

The extended metadata DataFrame is saved to 
SAMI_DR/sami_dr3_metadata_extended.hd5.

When run the first time, an additional DataFrame is created ("aperture_snrs.hd5")
to store the SNR measurements. Additional runs of this script will load 
this DataFrame if it exists rather than re-computing the SNRs.

USAGE
------------------------------------------------------------------------------
Run from the command line as follows:

    >>> python make_sami_metadata_df_extended.py 

OUTPUTS
------------------------------------------------------------------------------
The DataFrame is saved to 

    SAMI_DIR/sami_dr3_metadata_extended.hd5

PREREQUISITES
------------------------------------------------------------------------------
Both make_sami_metadata_df.py and make_df_sami.py must be run 
first, as this script requires both the metadata DataFrame (located at 
SAMI_DIR/sami_dr3_metadata.hd5) and the DataFrame containing spaxel-by-spaxel
information for every SAMI galaxy.

------------------------------------------------------------------------------
Copyright (C) 2022 Henry Zovaro
"""
import os, sys
from astropy.io import fits
import numpy as np
import pandas as pd
from tqdm import tqdm
import multiprocessing

from spaxelsleuth.loaddata.sami import load_sami_galaxies

import matplotlib.pyplot as plt
plt.ion()
plt.close()

from IPython.core.debugger import Tracer

###############################################################################
# Paths
sami_data_path = os.environ["SAMI_DIR"]
assert "SAMI_DIR" in os.environ, "Environment variable SAMI_DIR is not defined!"
sami_datacube_path = "/priv/myrtle1/sami/sami_data/Final_SAMI_data/cube/sami/dr3/ifs/"
assert "SAMI_DATACUBE_DIR" in os.environ, "Environment variable SAMI_DATACUBE_DIR is not defined!"

###############################################################################
# User options
plotit = False  # If True, plot histograms showing the distribution in S/N for each aperture.

###############################################################################
# Load the metadata
df_metadata = pd.read_hdf(os.path.join(sami_data_path, "sami_dr3_metadata.hd5"), key="metadata")
df_fname = "sami_dr3_metadata_extended.hd5"

# Obtain list of galaxies
gals = df_metadata[df_metadata["Good?"] == True].index.values

###############################################################################
# For multithreading
def compute_snr(gal, plotit=False):
    # Load the red & blue data cubes.
    hdulist_R_cube = fits.open(os.path.join(sami_datacube_path, f"ifs/{gal}/{gal}_A_cube_red.fits.gz"))
    hdulist_B_cube = fits.open(os.path.join(sami_datacube_path, f"ifs/{gal}/{gal}_A_cube_blue.fits.gz"))
    data_cube_B = hdulist_B_cube[0].data
    var_cube_B = hdulist_B_cube[1].data
    data_cube_R = hdulist_R_cube[0].data
    var_cube_R = hdulist_R_cube[1].data
    hdulist_R_cube.close()
    hdulist_B_cube.close()

    # Compute an image showing the median S/N in each spaxel.
    im_SNR_B = np.nanmedian(data_cube_B / np.sqrt(var_cube_B), axis=0)
    im_SNR_R = np.nanmedian(data_cube_R / np.sqrt(var_cube_R), axis=0)

    #######################################################################
    # Use R_e to compute the median S/N within 1, 1.5, 2 R_e. 
    # Transform coordinates into the galaxy plane
    e = df_metadata.loc[gal, "ellip"]
    PA = df_metadata.loc[gal, "pa"]
    beta_rad = np.deg2rad(PA - 90)
    b_over_a = 1 - e
    q0 = 0.2
    i_rad = np.arccos(np.sqrt((b_over_a**2 - q0**2) / (1 - q0**2)))  # Want to store this!
    i_rad = 0 if np.isnan(i_rad) else i_rad

    # De-project the centroids to the coordinate system of the galaxy plane
    x0_px = 25.5
    y0_px = 25.5
    as_per_px = 0.5
    ys, xs = np.meshgrid(np.arange(50), np.arange(50), indexing="ij")
    x_cc = xs - x0_px  # pixels
    y_cc = ys - y0_px  # pixels
    x_prime = x_cc * np.cos(beta_rad) + y_cc * np.sin(beta_rad)
    y_prime_projec = (- x_cc * np.sin(beta_rad) + y_cc * np.cos(beta_rad))
    y_prime = (- x_cc * np.sin(beta_rad) + y_cc * np.cos(beta_rad)) / np.cos(i_rad)
    r_prime = np.sqrt(x_prime**2 + y_prime**2)

    # Convert to arcsec
    r_prime_as = r_prime * as_per_px

    # Masks enclosing differen multiples of R_e 
    mask_1Re = r_prime_as < df_metadata.loc[gal, "r_e"]
    mask_15Re = r_prime_as < 1.5 * df_metadata.loc[gal, "r_e"]
    mask_2Re = r_prime_as < 2 * df_metadata.loc[gal, "r_e"]

    # Compute median SNR within 1, 1.5, 2R_e
    SNR_full_B = np.nanmedian(im_SNR_B)
    SNR_full_R = np.nanmedian(im_SNR_R)
    SNR_1Re_B = np.nanmedian(im_SNR_B[mask_1Re])
    SNR_1Re_R = np.nanmedian(im_SNR_R[mask_1Re])
    SNR_15Re_B = np.nanmedian(im_SNR_B[mask_15Re])
    SNR_15Re_R = np.nanmedian(im_SNR_R[mask_15Re])
    SNR_2Re_B = np.nanmedian(im_SNR_B[mask_2Re])
    SNR_2Re_R = np.nanmedian(im_SNR_R[mask_2Re])

    #######################################################################
    # Plot
    if plotit:
        # Set up figure
        fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(18, 5))
        ax_B, ax_R = axs[:2]

        bbox = ax_B.get_position()
        cax_B = fig.add_axes([bbox.x0 + bbox.width, bbox.y0, 0.03, bbox.height])
        bbox = ax_R.get_position()
        cax_R = fig.add_axes([bbox.x0 + bbox.width, bbox.y0, 0.03, bbox.height])

        m = ax_B.imshow(im_SNR_B, cmap="GnBu_r", vmin=0, vmax=50, origin="lower")
        plt.colorbar(mappable=m, cax=cax_B)
        m = ax_R.imshow(im_SNR_R, cmap="YlOrRd_r", vmin=0, vmax=50, origin="lower")
        plt.colorbar(mappable=m, cax=cax_R)

        ax_B.text(x=0.05, y=0.95, 
            s=f"SNR (1R_e) = {SNR_1Re_B:.2f}\nSNR (1.5R_e) = {SNR_15Re_B:.2f}\nSNR (2R_e) = {SNR_2Re_B:.2f}\n",
            transform=ax_B.transAxes, horizontalalignment="left", verticalalignment="top")
        ax_R.text(x=0.05, y=0.95, 
            s=f"SNR (1R_e) = {SNR_1Re_R:.2f}\nSNR (1.5R_e) = {SNR_15Re_R:.2f}\nSNR (2R_e) = {SNR_2Re_R:.2f}\n",
            transform=ax_R.transAxes, horizontalalignment="left", verticalalignment="top")

        # Plot radius, just to check...
        axs[-1].imshow(r_prime_as, cmap="jet_r", origin="lower")
        axs[-1].imshow(mask_1Re, cmap="gray", alpha=0.15, origin="lower")
        axs[-1].imshow(mask_2Re, cmap="gray", alpha=0.15, origin="lower")

        fig.suptitle(f"{gal}")
        fig.canvas.draw()
        Tracer()()
        plt.close(fig)

    #######################################################################
    # End
    print(f"Finished processing {gal}")
    return [gal, SNR_full_B, SNR_full_R, 
                 SNR_1Re_B, SNR_1Re_R, 
                 SNR_15Re_B, SNR_15Re_R, 
                 SNR_2Re_B, SNR_2Re_R]

###############################################################################
# Compute SNRs
###############################################################################
if os.path.exists(os.path.join(sami_data_path, "sami_dr3_aperture_snrs.hd5")):
    print(f"WARNING: file {os.path.join(sami_data_path, 'sami_dr3_aperture_snrs.hd5')} found; loading SNRs from existing DataFrame...")
    df_snr = pd.read_hdf(os.path.join(sami_data_path, "sami_dr3_aperture_snrs.hd5"), key="SNR")
else:
    nthreads = 20
    print(f"WARNING: file {os.path.join(sami_data_path, 'sami_dr3_aperture_snrs.hd5')} not found; computing continuum SNRs on {nthreads} threads...")
    print("Beginning pool...")
    args_list = gals
    pool = multiprocessing.Pool(nthreads)
    res_list = np.array((pool.map(compute_snr, args_list)))
    pool.close()
    pool.join()

    ###########################################################################
    # Create DataFrame from results
    ###############################################################################
    df_snr = pd.DataFrame(np.vstack(res_list), columns=["catid",
                                                        "Median SNR (B, full field)",
                                                        "Median SNR (R, full field)",
                                                        "Median SNR (B, 1R_e)",
                                                        "Median SNR (R, 1R_e)",
                                                        "Median SNR (B, 1.5R_e)",
                                                        "Median SNR (R, 1.5R_e)",
                                                        "Median SNR (B, 2R_e)",
                                                        "Median SNR (R, 2R_e)"])
    df_snr["catid"] = df_snr["catid"].astype(int)
    df_snr.set_index("catid")

    # Save 
    df_snr.to_hdf(os.path.join(sami_data_path, "sami_dr3_aperture_snrs.hd5"), key="SNR")

###############################################################################
# Merge with the metadata DataFrame
###############################################################################
common_cols = [c for c in df_metadata.columns if c not in df_snr.columns]
df_snr = df_snr.set_index("catid")
df_snr = pd.concat([df_snr, df_metadata], axis=1)

###############################################################################
# Plot: histograms showing the S/N distributions within different apertures
###############################################################################
if plotit:
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))

    axs[0].hist(df_snr["Median SNR (B, full field)"], histtype="step", range=(0, 50), bins=25, label="Full field")
    axs[0].hist(df_snr["Median SNR (B, 1R_e)"], histtype="step", range=(0, 50), bins=25, label="1R_e")
    axs[0].hist(df_snr["Median SNR (B, 1.5R_e)"], histtype="step", range=(0, 50), bins=25, label="1.5R_e")
    axs[0].hist(df_snr["Median SNR (B, 2R_e)"], histtype="step", range=(0, 50), bins=25, label="2R_e")

    axs[1].hist(df_snr["Median SNR (R, full field)"], histtype="step", range=(0, 50), bins=25, label="Full field")
    axs[1].hist(df_snr["Median SNR (R, 1R_e)"], histtype="step", range=(0, 50), bins=25, label="1R_e")
    axs[1].hist(df_snr["Median SNR (R, 1.5R_e)"], histtype="step", range=(0, 50), bins=25, label="1.5R_e")
    axs[1].hist(df_snr["Median SNR (R, 2R_e)"], histtype="step", range=(0, 50), bins=25, label="2R_e")
    axs[1].legend()

    # Decorations
    axs[0].set_xlabel("Median continuum S/N (blue)")
    axs[1].set_xlabel("Median continuum S/N (red)")
    axs[0].set_ylabel(r"$N$")
    axs[0].set_ylabel(r"$N$")

###############################################################################
# Load the SAMI sample
###############################################################################
df_sami = load_sami_df(ncomponents="recom",
                             bin_type="default",
                             eline_SNR_min=5, 
                             correct_extinction=True)

for gal in tqdm(df_sami.catid.unique()):

    ###########################################################################
    # Compute total SFRs from HALPHA emission
    ###########################################################################
    df_gal = df_sami[df_sami["catid"] == gal]
    sfr_comp0 = df_gal["SFR (component 0)"].sum()
    df_snr.loc[gal, "SFR (component 0)"] = sfr_comp0 if sfr_comp0 > 0 else np.nan
    sfr_tot = df_gal["SFR (total)"].sum()
    df_snr.loc[gal, "SFR (total)"] = sfr_tot if sfr_tot > 0 else np.nan

    ###########################################################################
    # Compute maximum number of components fitted in each galaxy
    ###########################################################################
    df_snr.loc[gal, "Maximum number of components"] = np.nanmax(df_gal["Number of components"])

    ###########################################################################
    # Compute inclination
    ###########################################################################
    e = df_metadata.loc[gal, "ellip"]
    PA = df_metadata.loc[gal, "pa"]
    beta_rad = np.deg2rad(PA - 90)
    b_over_a = 1 - e
    q0 = 0.2
    i_rad = np.arccos(np.sqrt((b_over_a**2 - q0**2) / (1 - q0**2)))  # Want to store this!
    df_snr.loc[gal, "Inclination i (degrees)"] = np.rad2deg(i_rad)

###############################################################################
# Save to .csv 
###############################################################################
df_snr.to_hdf(os.path.join(sami_data_path, df_fname), key="Extended metadata")
