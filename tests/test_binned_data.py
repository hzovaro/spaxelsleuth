# Imports
import sys
import os 
import numpy as np
import pandas as pd
from astropy.visualization import hist
from astropy.io import fits
from tqdm import tqdm
from scipy import constants
from scipy.stats import ks_2samp, anderson_ksamp, spearmanr

from spaxelsleuth.loaddata.lzifu import load_lzifu_galaxies
from spaxelsleuth.loaddata.sami import load_sami_galaxies
from spaxelsleuth.plotting.plottools import plot_empty_BPT_diagram
from spaxelsleuth.plotting.plottools import vmin_fn, vmax_fn, label_fn, cmap_fn, fname_fn
from spaxelsleuth.plotting.plottools import bpt_colours, bpt_labels, whav_colors, whav_labels
from spaxelsleuth.plotting.plottools import morph_labels, morph_ticks
from spaxelsleuth.plotting.plottools import ncomponents_labels, ncomponents_colours
from spaxelsleuth.plotting.plottools import component_labels, component_colours
from spaxelsleuth.plotting.plotgalaxies import plot2dhistcontours, plot2dscatter, plot2dcontours
from spaxelsleuth.plotting.plot2dmap import plot2dmap
from spaxelsleuth.plotting.sdssimg import plot_sdss_image
from spaxelsleuth.plotting.plottools import plot_empty_BPT_diagram, plot_BPT_lines

import matplotlib
from matplotlib import rc, rcParams
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from IPython.core.debugger import Tracer

rc("text", usetex=False)
rc("font",**{"family": "serif", "size": 11})
rcParams["savefig.bbox"] = "tight"
rcParams["savefig.format"] = "pdf"
plt.ion()
plt.close("all")

###########################################################################
# Paths
sami_data_path = os.environ["SAMI_DIR"]
assert "SAMI_DIR" in os.environ, "Environment variable SAMI_DIR is not defined!"
sami_datacube_path = os.environ["SAMI_DATACUBE_DIR"]
assert "SAMI_DATACUBE_DIR" in os.environ, "Environment variable SAMI_DATACUBE_DIR is not defined!"
sami_fig_path = os.environ["SAMI_FIG_DIR"]
assert "SAMI_FIG_DIR" in os.environ, "Environment variable SAMI_FIG_DIR is not defined!"
sami_fig_path = os.environ["SAMI_FIG_DIR"]
assert "SAMI_FIG_DIR" in os.environ, "Environment variable SAMI_FIG_DIR is not defined!"

###########################################################################
# Options
eline_SNR_min = 5       # Minimum S/N of emission lines to accept

###########################################################################
# Load the data
###########################################################################
# Load the ubinned data 
df_unbinned = load_sami_galaxies(ncomponents="recom",
                                 bin_type="default",
                                 eline_SNR_min=eline_SNR_min,
                                 correct_extinction=True,
                                 debug=True)

# Load the binned data 
df_binned = load_sami_galaxies(ncomponents="recom",
                               bin_type="adaptive",
                               eline_SNR_min=eline_SNR_min,
                               correct_extinction=True,
                               debug=True)

###########################################################################
# Load the DataFrame containing S/N metadata
###########################################################################
# Load the SNR DataFrame.
df_snr = pd.read_csv(os.path.join(sami_data_path, "sample_summary.csv"))

# Sort by median red S/N in 2R_e
df_snr = df_snr.sort_values("Median SNR (R, 2R_e)", ascending=False)

# Set index to catid for ease of indexing
df_snr = df_snr.set_index("catid")

###########################################################################
# Plot to check
###########################################################################
gal = int(sys.argv[1])

###########################################################################
# Collage figure 1: coloured by number of components
###########################################################################
markers = ["o", ">", "D"]
l = 0.05
b = 0.05
dw = 0.05
dh = 0.1
w = (1 - 2 * l - 2 * dw) / 5
h = (1 - 2 * b - dh) / 2

###########################################################################
# Define galaxy
###########################################################################
# Check validity
assert gal in df_unbinned.catid.unique(), f"{gal} not found in SAMI sample!"

# Get the unbinned DataFrame for this galaxy
df_gal_unbinned = df_unbinned[df_unbinned["catid"] == gal]
df_gal_unbinned.loc[df_gal_unbinned["Number of components"] == 0, "Number of components"] = np.nan

# Get the binned DataFrame for this galaxy
df_gal_binned = df_binned[df_binned["catid"] == gal]
df_gal_binned.loc[df_gal_binned["Number of components"] == 0, "Number of components"] = np.nan

###########################################################################
# Unbinned data
###########################################################################
fig, axs = plt.subplots(nrows=4, ncols=3, figsize=(13, 17))
fig.subplots_adjust(wspace=0.05, hspace=0.05)

# SDSS image 
ax = plot_sdss_image(df_gal_unbinned, ax=axs[0][0])
if ax is not None:
    ax.set_title(f"GAMA{gal}")
    lon = ax.coords[0]
    lon.set_ticklabel_visible(False)

# Number of components
_, ax = plot2dmap(df_gal=df_gal_unbinned, bin_type="default", survey="sami",
          PA_deg=0,
          col_z="Number of components",
          ax=axs[0][1], 
          plot_colorbar=True, cax=None, cax_orientation="horizontal", 
          show_title=False)
lon = ax.coords[0]
lon.set_ticklabel_visible(False)
lat = ax.coords[1]
lat.set_ticklabel_visible(False)


# BPT classifications 
_, ax = plot2dmap(df_gal=df_gal_unbinned, bin_type="default", survey="sami",
          PA_deg=0,
          col_z="BPT (numeric) (total)",
          ax=axs[0][2], 
          plot_colorbar=True, cax=None, cax_orientation="vertical", 
          show_title=False)
lon = ax.coords[0]
lon.set_ticklabel_visible(False)
lat = ax.coords[1]
lat.set_ticklabel_visible(False)

# v_gas
for ii in range(3):
    _, ax = plot2dmap(df_gal=df_gal_unbinned, bin_type="default", survey="sami",
              PA_deg=0,
              col_z=f"v_gas (component {ii})",
              ax=axs[1][ii], 
              plot_colorbar=True if ii == 2 else False, cax=None, cax_orientation="vertical", 
              vmin=-200, vmax=+200,
              show_title=False)
    ax.text(s=f"Component {ii + 1}", x=0.05, y=0.95, transform=axs[1][ii].transAxes, verticalalignment="top")
    if ii > 0:
        lat = ax.coords[1]
        lat.set_ticklabel_visible(False)
    lon = ax.coords[0]
    lon.set_ticklabel_visible(False)


# delta sigma 
for ii in range(3):
    _, ax = plot2dmap(df_gal=df_gal_unbinned, bin_type="default", survey="sami",
              PA_deg=0,
              col_z=f"sigma_gas - sigma_* (component {ii})",
              ax=axs[2][ii], 
              plot_colorbar=True if ii == 2 else False, cax=None, cax_orientation="vertical", 
              vmin=-200, vmax=+200,
              show_title=False)
    ax.text(s=f"Component {ii + 1}", x=0.05, y=0.95, transform=axs[1][ii].transAxes, verticalalignment="top")
    if ii > 0:
        lat = ax.coords[1]
        lat.set_ticklabel_visible(False)
    lon = ax.coords[0]
    lon.set_ticklabel_visible(False)

# EW 
for ii in range(3):
    _, ax = plot2dmap(df_gal=df_gal_unbinned, bin_type="default", survey="sami",
              PA_deg=0,
              col_z=f"HALPHA EW (component {ii})",
              ax=axs[3][ii], 
              plot_colorbar=True if ii == 2 else False, cax=None, cax_orientation="vertical", 
              show_title=False)
    ax.text(s=f"Component {ii + 1}", x=0.05, y=0.95, transform=axs[2][ii].transAxes, verticalalignment="top")
    if ii > 0:
        lat = ax.coords[1]
        lat.set_ticklabel_visible(False)
        
###########################################################################
# Binned data
###########################################################################
fig, axs = plt.subplots(nrows=4, ncols=3, figsize=(13, 17))
fig.subplots_adjust(wspace=0.05, hspace=0.05)

# SDSS image 
ax = plot_sdss_image(df_gal_binned, ax=axs[0][0])
if ax is not None:
    ax.set_title(f"GAMA{gal}")
    lon = ax.coords[0]
    lon.set_ticklabel_visible(False)

# Number of components
_, ax = plot2dmap(df_gal=df_gal_binned, bin_type="adaptive", survey="sami",
          PA_deg=0,
          col_z="Number of components",
          ax=axs[0][1], 
          plot_colorbar=True, cax=None, cax_orientation="horizontal", 
          show_title=False)
lon = ax.coords[0]
lon.set_ticklabel_visible(False)
lat = ax.coords[1]
lat.set_ticklabel_visible(False)


# BPT classifications 
_, ax = plot2dmap(df_gal=df_gal_binned, bin_type="adaptive", survey="sami",
          PA_deg=0,
          col_z="BPT (numeric) (total)",
          ax=axs[0][2], 
          plot_colorbar=True, cax=None, cax_orientation="vertical", 
          show_title=False)
lon = ax.coords[0]
lon.set_ticklabel_visible(False)
lat = ax.coords[1]
lat.set_ticklabel_visible(False)

# v_gas
for ii in range(3):
    _, ax = plot2dmap(df_gal=df_gal_binned, bin_type="adaptive", survey="sami",
              PA_deg=0,
              col_z=f"v_gas (component {ii})",
              ax=axs[1][ii], 
              plot_colorbar=True if ii == 2 else False, cax=None, cax_orientation="vertical", 
              vmin=-200, vmax=+200,
              show_title=False)
    ax.text(s=f"Component {ii + 1}", x=0.05, y=0.95, transform=axs[1][ii].transAxes, verticalalignment="top")
    if ii > 0:
        lat = ax.coords[1]
        lat.set_ticklabel_visible(False)
    lon = ax.coords[0]
    lon.set_ticklabel_visible(False)


# delta sigma 
for ii in range(3):
    _, ax = plot2dmap(df_gal=df_gal_binned, bin_type="adaptive", survey="sami",
              PA_deg=0,
              col_z=f"sigma_gas - sigma_* (component {ii})",
              ax=axs[2][ii], 
              plot_colorbar=True if ii == 2 else False, cax=None, cax_orientation="vertical", 
              vmin=-200, vmax=+200,
              show_title=False)
    ax.text(s=f"Component {ii + 1}", x=0.05, y=0.95, transform=axs[1][ii].transAxes, verticalalignment="top")
    if ii > 0:
        lat = ax.coords[1]
        lat.set_ticklabel_visible(False)
    lon = ax.coords[0]
    lon.set_ticklabel_visible(False)

# EW 
for ii in range(3):
    _, ax = plot2dmap(df_gal=df_gal_binned, bin_type="adaptive", survey="sami",
              PA_deg=0,
              col_z=f"HALPHA EW (component {ii})",
              ax=axs[3][ii], 
              plot_colorbar=True if ii == 2 else False, cax=None, cax_orientation="vertical", 
              show_title=False)
    ax.text(s=f"Component {ii + 1}", x=0.05, y=0.95, transform=axs[2][ii].transAxes, verticalalignment="top")
    if ii > 0:
        lat = ax.coords[1]
        lat.set_ticklabel_visible(False)
        

"""
###########################################################################
# Create the figure
###########################################################################
fig_collage = plt.figure(figsize=(18, 7))
ax_sdss = fig_collage.add_axes([l, b, w, h])
ax_im = fig_collage.add_axes([l, b + h + dh, w, h])
bbox = ax_im.get_position()
cax_im = fig_collage.add_axes([bbox.x0 + bbox.width * 0.035, bbox.y0 + bbox.height, bbox.width * 0.93, 0.025])
axs_bpt = []
axs_bpt.append(fig_collage.add_axes([l + w + dw, b + h + dh, w, h]))
axs_bpt.append(fig_collage.add_axes([l + w + dw + w, b + h + dh, w, h]))
axs_bpt.append(fig_collage.add_axes([l + w + dw + 2 * w, b + h + dh, w, h]))
axs_whav = []
axs_whav.append(fig_collage.add_axes([l + w + dw, b, w, h]))
axs_whav.append(fig_collage.add_axes([l + w + dw + w, b, w, h]))
axs_whav.append(fig_collage.add_axes([l + w + dw + 2 * w, b, w, h]))

###########################################################################
# Plot SDSS image and component map
###########################################################################
col_z = "Number of components"

# SDSS image
res = plot_sdss_image(df_gal_unbinned, ax=ax_sdss)
if res is None:
    ax_sdss.text(s="Galaxy not in SDSS footprint",
                 x=0.5, y=0.5, horizontalalignment="center",
                 transform=ax_sdss.transAxes)

# Plot the number of components fitted.
plot2dmap(df_gal=df_gal_unbinned, bin_type="default", survey="sami",
          PA_deg=0,
          col_z=col_z, 
          ax=ax_im, cax=cax_im, cax_orientation="horizontal", show_title=False)

# Text string showing basic info
sfr = df_snr.loc[gal, 'SFR (component 0)']
mstar = df_gal_unbinned["mstar"].unique()[0]
if np.isnan(sfr):
    sfr = "n/a"
else:
    sfr = f"{sfr:.3f}" + r" $\rm M_\odot\,yr^{-1}$"
    
if np.isnan(mstar):
    mstar = r"$\log \rm \, M_\odot = $ n/a"
else:
    mstar = r"$\log \rm \, M_\odot = $" + f"{mstar:.2f}"

t = axs_bpt[0].text(s=f"{gal}, {df_snr.loc[gal, 'Morphology']}, SFR = {sfr}, {mstar}, SNR = {df_snr.loc[gal, 'Median SNR (R, 2R_e)']:.2f}", 
    x=0.0, y=1.02, transform=axs_bpt[0].transAxes)

# Plot BPT diagram
col_y = "log O3"
for cc, col_x in enumerate(["log N2", "log S2", "log O1"]):
    # Plot full SAMI sample
    plot2dhistcontours(df=df_unbinned, 
                       col_x=f"{col_x} (total)",
                       col_y=f"{col_y} (total)", col_z="count", log_z=True,
                       alpha=0.5, cmap="gray_r",
                       ax=axs_bpt[cc], plot_colorbar=False)

    # Add BPT functions
    plot_BPT_lines(ax=axs_bpt[cc], col_x=col_x)    

    # Plot measurements for this galaxy
    plot2dscatter(df=df_gal_unbinned,
                  col_x=f"{col_x} (total)",
                  col_y=f"{col_y} (total)",
                  col_z=None if col_z == "Number of components" else col_z,
                  marker=markers[0], ax=axs_bpt[cc], 
                  cax=None,
                  markersize=20, 
                  markerfacecolor=component_colours[0] if col_z == "Number of components" else None, 
                  markeredgecolor="black",
                  plot_colorbar=False)

# Decorations
[ax.grid() for ax in axs_bpt]
[ax.set_ylabel("") for ax in axs_bpt[1:]]
[ax.set_yticklabels([]) for ax in axs_bpt[1:]]
[ax.set_xticks(ax.get_xticks()[:-1]) for ax in axs_bpt[:-1]]
for ax in axs_bpt:
    _ = [c.set_rasterized(True) for c in ax.collections]
_ = [c.set_rasterized(True) for c in ax_im.collections]

###########################################################################
# Plot WHAN, WHAV and WHAV* diagrams.
###########################################################################
# Plot LZIFU measurements
for cc, col_x in enumerate(["log N2", "sigma_gas - sigma_*", "v_gas - v_*"]):
    # Plot full SAMI sample
    plot2dhistcontours(df=df_unbinned, 
                       col_x=f"{col_x} (total)" if col_x == "log N2" else f"{col_x}",
                       col_y=f"log HALPHA EW (total)" if col_x == "log N2" else f"log HALPHA EW",
                       col_z="count", log_z=True,
                       alpha=0.5, cmap="gray_r", ax=axs_whav[cc],
                       plot_colorbar=False)

# WHAN diagram
plot2dscatter(df=df_gal_unbinned,
              col_x=f"log N2 (total)",
              col_y=f"log HALPHA EW (total)",
              col_z=None if col_z == "Number of components" else col_z,
              marker=markers[0], ax=axs_whav[0], 
              cax=None,
              markersize=20, 
              markerfacecolor=component_colours[0] if col_z == "Number of components" else None, 
              markeredgecolor="black",
              plot_colorbar=False)

# Kinematics 
for cc, col_x in enumerate(["sigma_gas - sigma_*", "v_gas - v_*"]):
    # Plot the data for this galaxy
    for ii in range(3):
        plot2dscatter(df=df_gal_unbinned,
                      col_x=f"{col_x} (component {ii})",
                      col_y=f"log HALPHA EW (component {ii})",
                      col_z=None if col_z == "Number of components" else col_z,
                      marker=markers[ii], ax=axs_whav[cc + 1], 
                      cax=None,
                      markersize=20, 
                      markerfacecolor=component_colours[ii] if col_z == "Number of components" else None, 
                      markeredgecolor="black",
                      plot_colorbar=False)

# Decorations
[ax.grid() for ax in axs_whav]
[ax.set_ylabel("") for ax in axs_whav[1:]]
[ax.set_yticklabels([]) for ax in axs_whav[1:]]
[ax.set_xticks(ax.get_xticks()[:-1]) for ax in axs_whav[:-1]]
[ax.axvline(0, ls="--", color="k") for ax in axs_whav[1:]]
for ax in axs_whav:
    _ = [c.set_rasterized(True) for c in ax.collections]

# Legend
legend_elements = [Line2D([0], [0], marker=markers[ii], 
                          color="none", markeredgecolor="black",
                          label=f"Component {ii}",
                          markerfacecolor=component_colours[ii], markersize=5) for ii in range(3)]
axs_bpt[-1].legend(handles=legend_elements, fontsize="x-small", loc="upper right")

###########################################################################
# Define galaxy
###########################################################################
# Check validity
assert gal in df_binned.catid.values, f"{gal} not found in SAMI sample!"

# Load the DataFrame
df_gal_binned = df_binned[df_binned["catid"] == gal]
df_gal_binned.loc[df_gal_binned["Number of components"] == 0, "Number of components"] = np.nan

###########################################################################
# Create the figure
###########################################################################
fig_collage = plt.figure(figsize=(18, 7))
ax_sdss = fig_collage.add_axes([l, b, w, h])
ax_im = fig_collage.add_axes([l, b + h + dh, w, h])
bbox = ax_im.get_position()
cax_im = fig_collage.add_axes([bbox.x0 + bbox.width * 0.035, bbox.y0 + bbox.height, bbox.width * 0.93, 0.025])
axs_bpt = []
axs_bpt.append(fig_collage.add_axes([l + w + dw, b + h + dh, w, h]))
axs_bpt.append(fig_collage.add_axes([l + w + dw + w, b + h + dh, w, h]))
axs_bpt.append(fig_collage.add_axes([l + w + dw + 2 * w, b + h + dh, w, h]))
axs_whav = []
axs_whav.append(fig_collage.add_axes([l + w + dw, b, w, h]))
axs_whav.append(fig_collage.add_axes([l + w + dw + w, b, w, h]))
axs_whav.append(fig_collage.add_axes([l + w + dw + 2 * w, b, w, h]))

###########################################################################
# Plot SDSS image and component map
###########################################################################
col_z = "Bin number"

# SDSS image
res = plot_sdss_image(df_gal_binned, ax=ax_sdss)
if res is None:
    ax_sdss.text(s="Galaxy not in SDSS footprint",
                 x=0.5, y=0.5, horizontalalignment="center",
                 transform=ax_sdss.transAxes)

# Plot the number of components fitted.
plot2dmap(df_gal=df_gal_binned, bin_type="adaptive", survey="sami",
          PA_deg=0,
          col_z=col_z, 
          ax=ax_im, cax=cax_im, cax_orientation="horizontal", show_title=False)

# Text string showing basic info
sfr = df_snr.loc[gal, 'SFR (component 0)']
mstar = df_gal_binned["mstar"].unique()[0]
if np.isnan(sfr):
    sfr = "n/a"
else:
    sfr = f"{sfr:.3f}" + r" $\rm M_\odot\,yr^{-1}$"
    
if np.isnan(mstar):
    mstar = r"$\log \rm \, M_\odot = $ n/a"
else:
    mstar = r"$\log \rm \, M_\odot = $" + f"{mstar:.2f}"

t = axs_bpt[0].text(s=f"{gal}, {df_snr.loc[gal, 'Morphology']}, SFR = {sfr}, {mstar}, SNR = {df_snr.loc[gal, 'Median SNR (R, 2R_e)']:.2f}", 
    x=0.0, y=1.02, transform=axs_bpt[0].transAxes)

# Plot BPT diagram
col_y = "log O3"
for cc, col_x in enumerate(["log N2", "log S2", "log O1"]):
    # Plot full SAMI sample
    plot2dhistcontours(df=df_binned, 
                       col_x=f"{col_x} (total)",
                       col_y=f"{col_y} (total)", col_z="count", log_z=True,
                       alpha=0.5, cmap="gray_r",
                       ax=axs_bpt[cc], plot_colorbar=False)

    # Add BPT functions
    plot_BPT_lines(ax=axs_bpt[cc], col_x=col_x)    

    # Plot measurements for this galaxy
    plot2dscatter(df=df_gal_binned,
                  col_x=f"{col_x} (total)",
                  col_y=f"{col_y} (total)",
                  col_z=None if col_z == "Number of components" else col_z,
                  marker=markers[0], ax=axs_bpt[cc], 
                  cax=None,
                  markersize=20, 
                  markerfacecolor=component_colours[0] if col_z == "Number of components" else None, 
                  markeredgecolor="black",
                  plot_colorbar=False)

# Decorations
[ax.grid() for ax in axs_bpt]
[ax.set_ylabel("") for ax in axs_bpt[1:]]
[ax.set_yticklabels([]) for ax in axs_bpt[1:]]
[ax.set_xticks(ax.get_xticks()[:-1]) for ax in axs_bpt[:-1]]
for ax in axs_bpt:
    _ = [c.set_rasterized(True) for c in ax.collections]
_ = [c.set_rasterized(True) for c in ax_im.collections]

###########################################################################
# Plot WHAN, WHAV and WHAV* diagrams.
###########################################################################
# Plot LZIFU measurements
for cc, col_x in enumerate(["log N2", "sigma_gas - sigma_*", "v_gas - v_*"]):
    # Plot full SAMI sample
    plot2dhistcontours(df=df_binned, 
                       col_x=f"{col_x} (total)" if col_x == "log N2" else f"{col_x}",
                       col_y=f"log HALPHA EW (total)" if col_x == "log N2" else f"log HALPHA EW",
                       col_z="count", log_z=True,
                       alpha=0.5, cmap="gray_r", ax=axs_whav[cc],
                       plot_colorbar=False)

# WHAN diagram
plot2dscatter(df=df_gal_binned,
              col_x=f"log N2 (total)",
              col_y=f"log HALPHA EW (total)",
              col_z=None if col_z == "Number of components" else col_z,
              marker=markers[0], ax=axs_whav[0], 
              cax=None,
              markersize=20, 
              markerfacecolor=component_colours[0] if col_z == "Number of components" else None, 
              markeredgecolor="black",
              plot_colorbar=False)

# Kinematics 
for cc, col_x in enumerate(["sigma_gas - sigma_*", "v_gas - v_*"]):
    # Plot the data for this galaxy
    for ii in range(3):
        plot2dscatter(df=df_gal_binned,
                      col_x=f"{col_x} (component {ii})",
                      col_y=f"log HALPHA EW (component {ii})",
                      col_z=None if col_z == "Number of components" else col_z,
                      marker=markers[ii], ax=axs_whav[cc + 1], 
                      cax=None,
                      markersize=20, 
                      markerfacecolor=component_colours[ii] if col_z == "Number of components" else None, 
                      markeredgecolor="black",
                      plot_colorbar=False)

# Decorations
[ax.grid() for ax in axs_whav]
[ax.set_ylabel("") for ax in axs_whav[1:]]
[ax.set_yticklabels([]) for ax in axs_whav[1:]]
[ax.set_xticks(ax.get_xticks()[:-1]) for ax in axs_whav[:-1]]
[ax.axvline(0, ls="--", color="k") for ax in axs_whav[1:]]
for ax in axs_whav:
    _ = [c.set_rasterized(True) for c in ax.collections]

# Legend
legend_elements = [Line2D([0], [0], marker=markers[ii], 
                          color="none", markeredgecolor="black",
                          label=f"Component {ii}",
                          markerfacecolor=component_colours[ii], markersize=5) for ii in range(3)]
axs_bpt[-1].legend(handles=legend_elements, fontsize="x-small", loc="upper right")


"""