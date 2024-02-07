# Imports
import sys
import os 
import numpy as np
import pandas as pd
from astropy.visualization import hist
from astropy.io import fits

from spaxelsleuth.io.lzifu import load_lzifu_galaxy
from spaxelsleuth.io.sami import load_sami_galaxies
from spaxelsleuth.plotting.plot2dmap import plot2dmap
from spaxelsleuth.plotting.sdssimg import plot_sdss_image
from spaxelsleuth.plotting.plottools import plot_empty_BPT_diagram, plot_BPT_lines, bpt_colours, bpt_labels
from spaxelsleuth.plotting.plottools import vmin_fn, vmax_fn, label_fn, cmap_fn, component_colours
from spaxelsleuth.plotting.plotgalaxies import plot2dscatter, plot2dhistcontours

import matplotlib
from matplotlib import rc, rcParams
from matplotlib.patches import Patch
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from IPython.core.debugger import Tracer

rc("text", usetex=False)
rc("font",**{"family": "serif", "size": 12})
rcParams["savefig.bbox"] = "tight"
rcParams["savefig.format"] = "pdf"
plt.ion()
plt.close("all")

"""
Make a nice figure for the WiFeS proposal.
"""
sami_data_path = os.environ["SAMI_DIR"]
assert "SAMI_DIR" in os.environ, "Environment variable SAMI_DIR is not defined!"
sami_datacube_path = os.environ["SAMI_DATACUBE_DIR"]
assert "SAMI_DATACUBE_DIR" in os.environ, "Environment variable SAMI_DATACUBE_DIR is not defined!"

###########################################################################
# Options
###########################################################################
fig_path = "/priv/meggs3/u5708159/SAMI/figs/wifes_proposal/"
savefigs = False
bin_type = "default"    # Options: "default" or "adaptive" for Voronoi binning
ncomponents = "recom"   # Options: "1" or "recom"
eline_SNR_min = 5       # Minimum S/N of emission lines to accept

###########################################################################
# Load the SAMI sample
###########################################################################
df_sami = load_sami_df(ncomponents="recom",
                             bin_type="default",
                             eline_SNR_min=eline_SNR_min, 
                             vgrad_cut=False,
                             correct_extinction=False,
                             sigma_gas_SNR_cut=True)

###########################################################################
# Make summary plots
###########################################################################
if len(sys.argv) > 1:
    gals = sys.argv[1:]
    for gal in gals:
        assert gal.isdigit(), "each gal given must be an integer!"
        assert int(gal) in df_sami["ID"].values, f"{gal} not found in SAMI sample!"
    gals = [int(g) for g in gals]
else:
    # Load the SNR DataFrame.
    df_snr = pd.read_csv(os.path.join(sami_data_path, "sample_summary.csv"))

    # Sort by median red S/N in 2R_e
    df_snr = df_snr.sort_values("Median SNR (R, 2R_e)", ascending=False)

    # Make a redshift cut to ensure that Na D is in the wavelength range 
    df_snr = df_snr[df_snr["z"] > 0.072035]

    df_snr = df_snr.set_index("ID")
    gals = df_snr.index.values

###########################################################################
# X, Y pixel coordinates for extracting spectra
###########################################################################
ys, xs = np.meshgrid(np.arange(50), np.arange(50), indexing="ij")
as_per_px = 0.5
ys_as = ys * as_per_px
xs_as = xs * as_per_px

# Centre galaxy coordinates (see p16 of Croom+2021)
x0_px = 25.5
y0_px = 25.5

# Create a mask 
mask = (xs - x0_px)**2 + (ys - y0_px)**2 <= 3**2
mask_area_px = len(mask[mask])
mask_area_arcsec2 = mask_area_px * as_per_px**2

for gal in gals:
    df_gal = df_sami[df_sami["ID"] == gal]

    ###########################################################################
    # Figure 
    ###########################################################################
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))
    fig.subplots_adjust(wspace=0)

    ###########################################################################
    # Axis 1: SDSS image
    ###########################################################################
    plot_sdss_image(df_gal, ax=axs[0])
    ax = fig.get_axes()[1]
    ax.text(s=f"{gal}", x=0.1, y=0.9, 
            color="white",
            horizontalalignment="left", verticalalignment="top", 
            transform=ax.transAxes)

    # Turn off RA, dec 
    lon = ax.coords[0]
    lat = ax.coords[1]
    lon.set_ticks_visible(False)
    lon.set_ticklabel_visible(False)
    lat.set_ticks_visible(False)
    lat.set_ticklabel_visible(False)
    lon.set_axislabel('')
    lat.set_axislabel('')

    ###########################################################################
    # Axis 2: BPT classification 
    ###########################################################################
    plot2dmap(df_gal=df_gal, bin_type="default", survey="sami",
              PA_deg=0,
              col_z="BPT (numeric) (total)", 
              ax=axs[1],
              plot_colorbar=False, show_title=False)
    
    # Turn off RA, dec 
    ax = fig.get_axes()[1]
    lon = ax.coords[0]
    lat = ax.coords[1]
    lon.set_ticks_visible(False)
    lon.set_ticklabel_visible(False)
    lat.set_ticks_visible(False)
    lat.set_ticklabel_visible(False)
    lon.set_axislabel('')
    lat.set_axislabel('')

    # Custom legend
    bpt_labels = ["Not classified", "Star formation", "Composite", "LINER", "Seyfert", "Ambiguous"]
    legend_elements = []
    for fc, l in zip(bpt_colours, bpt_labels):
        legend_elements.append(Patch(facecolor=fc, label=l))
    ax.legend(handles=legend_elements, loc="upper center", ncol=2, fontsize="x-small")

    # Save
    fig.savefig(os.path.join(fig_path, f"{gal}_im.ps"), format="ps", bbox_inches="tight")

    ###########################################################################
    # Extract the spectrum from the red data cube 
    ###########################################################################
    hdulist_R_cube = fits.open(os.path.join(sami_datacube_path, f"{gal}/{gal}_A_cube_red.fits.gz"))
    header = hdulist_R_cube[0].header
    data_cube_R = hdulist_R_cube[0].data
    var_cube_R = hdulist_R_cube[1].data

    # Get wavelength values 
    z = df_snr.loc[gal, "z"]
    lambda_0_A = header["CRVAL3"] - header["CRPIX3"] * header["CDELT3"]
    dlambda_A = header["CDELT3"]
    N_lambda = header["NAXIS3"]
    lambda_vals_A = np.array(range(N_lambda)) * dlambda_A + lambda_0_A 
    lambda_rest_A = lambda_vals_A / (1 + z)

    # Extract spectrum
    spec = np.nansum(data_cube_R[:, mask], axis=1)
    spec_err = np.sqrt(np.nansum(var_cube_R[:, mask], axis=1))

    ###########################################################################
    # Create a new figure
    ###########################################################################
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
    fig.subplots_adjust(wspace=0)

    ###########################################################################
    # Axis 3: Halpha, [NII] line profiles 
    ###########################################################################
    start = np.nanargmin(np.abs(lambda_rest_A - (6562.8 - 40)))
    stop = np.nanargmin(np.abs(lambda_rest_A - (6562.8 + 40)))

    # Divide by pixel area in arcsec2
    spec /= mask_area_arcsec2
    spec_err /= mask_area_arcsec2

    # Plot 
    axs[0].errorbar(x=lambda_rest_A, y=spec, yerr=spec_err, color="k")
    axs[0].set_xlim([(6562.8 - 40), (6562.8 + 40)])
    axs[0].set_ylim([0.9 * np.nanmin(spec[start:stop]), 1.1 * np.nanmax(spec[start:stop])])
    axs[0].axvline(6562.8, color="r")
    axs[0].axvline(6583, color="g")
    axs[0].axvline(6548, color="g")
    axs[0].set_xlabel(r"Rest-frame wavelength $\lambda \,\rm (\AA)$")
    axs[0].set_ylabel(r"$F_\lambda(\lambda)$ (arb. scaling)")
    axs[0].set_yticks([])
    axs[0].set_yticklabels([])
    axs[0].text(s=r"H$\alpha$ & [N II]", x=0.05, y=0.95, 
                horizontalalignment="left", verticalalignment="top",
                transform=axs[0].transAxes)

    ###########################################################################
    # Axis 4: Na D line profiles 
    ###########################################################################
    start = np.nanargmin(np.abs(lambda_rest_A - (5889 - 30)))
    stop = np.nanargmin(np.abs(lambda_rest_A - (5896 + 30)))

    # Divide by pixel area in arcsec2
    spec /= mask_area_arcsec2
    spec_err /= mask_area_arcsec2

    # Plot 
    axs[1].errorbar(x=lambda_rest_A, y=spec, yerr=spec_err, color="k")
    axs[1].set_xlim([(5889 - 30), (5896 + 30)])
    axs[1].set_ylim([0.9 * np.nanmin(spec[start:stop]), 1.1 * np.nanmax(spec[start:stop])])
    axs[1].axvline(5889, color="orange")
    axs[1].axvline(5896, color="orange")
    axs[1].axvline(5876, color="cyan")
    axs[1].set_xlabel(r"Rest-frame wavelength $\lambda \,\rm (\AA)$")
    axs[1].set_yticks([])
    axs[1].set_yticklabels([])
    axs[1].text(s=r"He I $\lambda 5876\rm\AA$ & Na D", x=0.05, y=0.95, 
                horizontalalignment="left", verticalalignment="top",
                transform=axs[1].transAxes)

    # Save
    fig.savefig(os.path.join(fig_path, f"{gal}_spec.ps"), format="ps", bbox_inches="tight")
