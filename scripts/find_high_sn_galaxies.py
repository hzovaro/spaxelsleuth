import os
from astropy.io import fits
import numpy as np
import pandas as pd
from tqdm import tqdm
import multiprocessing

import matplotlib.pyplot as plt
plt.ion()
plt.close()

from IPython.core.debugger import Tracer

"""
# Find a subset of high-S/N galaxies from the SAMI sample which we can re-fit
# with LZIFU.
"""

###############################################################################
# Paths
sami_data_path = "/priv/meggs3/u5708159/SAMI/sami_dr3/"
sami_datacube_path = "/priv/myrtle1/sami/sami_data/Final_SAMI_data/cube/sami/dr3/ifs/"

###############################################################################
# Load the metadata
df_metadata = pd.read_hdf(os.path.join(sami_data_path, "sami_dr3_metadata.hd5"), key="metadata")

# Obtain list of galaxies
gals = df_metadata[df_metadata["Good?"] == True].index.values

###############################################################################

###############################################################################
# For multithreading
def compute_snr(gal, plotit=False):
    # Load the red & blue data cubes.
    hdulist_R_cube = fits.open(os.path.join(sami_datacube_path, f"{gal}/{gal}_A_cube_red.fits.gz"))
    hdulist_B_cube = fits.open(os.path.join(sami_datacube_path, f"{gal}/{gal}_A_cube_blue.fits.gz"))
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

    print(f"Finished processing {gal}")
    return [gal, SNR_full_B, SNR_full_R, 
                 SNR_1Re_B, SNR_1Re_R, 
                 SNR_15Re_B, SNR_15Re_R, 
                 SNR_2Re_B, SNR_2Re_R]

###############################################################################
# Run in parallel
if compute_snrs:
    print("Beginning pool...")
    args_list = gals
    pool = multiprocessing.Pool(40)
    res_list = np.array((pool.map(compute_snr, args_list)))
    pool.close()
    pool.join()

    ###########################################################################
    # Create DataFrame from results
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
    df_snr.to_hdf(os.path.join(sami_data_path, "aperture_snrs.hd5"), key="SNR")

else:
    # Otherwise, just load the dataframe
    df_snr = pd.read_hdf(os.path.join(sami_data_path, "aperture_snrs.hd5"))

###############################################################################
# Plot: 
