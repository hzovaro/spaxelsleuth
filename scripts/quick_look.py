

# Imports
import sys
import os 
import numpy as np
import pandas as pd
from astropy.visualization import hist

from spaxelsleuth.loaddata.lzifu import load_lzifu_galaxy
from spaxelsleuth.loaddata.sami import load_sami_galaxies
from spaxelsleuth.plotting.plot2dmap import plot2dmap
from spaxelsleuth.plotting.sdssimg import plot_sdss_image
from spaxelsleuth.plotting.plottools import plot_empty_BPT_diagram, plot_BPT_lines
from spaxelsleuth.plotting.plottools import vmin_fn, vmax_fn, label_fn, cmap_fn, component_colours
from spaxelsleuth.plotting.plotgalaxies import plot2dscatter, plot2dhistcontours

import matplotlib
from matplotlib import rc, rcParams
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
Take a quick look at SAMI galaxies.
"""
sami_data_path = "/priv/meggs3/u5708159/SAMI/sami_dr3/"

###########################################################################
# Options
###########################################################################
fig_path = "/priv/meggs3/u5708159/SAMI/figs/"
savefigs = True
bin_type = "default"    # Options: "default" or "adaptive" for Voronoi binning
ncomponents = "recom"   # Options: "1" or "recom"
eline_SNR_min = 5       # Minimum S/N of emission lines to accept

###########################################################################
# Load the SAMI sample
###########################################################################
df_sami = load_sami_galaxies(ncomponents="recom",
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
        assert gal in df_sami.catid, f"{gal} not found in SAMI sample!"
else:
    # Load the SNR DataFrame. Order galaxies by their median red S/N in 2R_e.
    df_snr = pd.read_csv(os.path.join(sami_data_path, "sample_summary.csv"))

    # Sort by median red S/N in 2R_e
    df_snr = df_snr.sort_values("Median SNR (R, 2R_e)", ascending=False)
    df_snr = df_snr[df_snr["Median SNR (R, 2R_e)"] > 10]
    df_snr = df_snr.set_index("catid")
    gals = df_snr.index.values

###########################################################################
# Collage figure 1: coloured by number of components
###########################################################################
markers = ["o", ">", "D"]
l = 0.05
b = 0.05
dw = 0.1
dh = 0.1
w = (1 - 2 * l - dw) / 4
h = (1 - 2 * b - dh) / 2

# Multi-page pdf
if savefigs:
    pp = PdfPages(os.path.join(fig_path, "quicklook.pdf"))

for gal in gals:

    # Load the DataFrame
    df_gal = df_sami[df_sami["catid"] == gal]
    df_gal.loc[df_gal["Number of components"] == 0, "Number of components"] = np.nan

    ###########################################################################
    # Create the figure
    ###########################################################################
    fig_collage = plt.figure(figsize=(15, 7))
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
    res = plot_sdss_image(df_gal, ax=ax_sdss)
    if res is None:
        ax_sdss.text(s="Galaxy not in SDSS footprint",
                     x=0.5, y=0.5, horizontalalignment="center",
                     transform=ax_sdss.transAxes)

    # Plot the number of components fitted.
    plot2dmap(df_gal=df_gal, bin_type="default", survey="sami",
              PA_deg=0,
              col_z=col_z, 
              ax=ax_im, cax=cax_im, cax_orientation="horizontal", show_title=False)

    # Plot BPT diagram
    col_y = "log O3"
    t = axs_bpt[0].text(s=f"{gal}, {df_snr.loc[gal, 'Morphology']}, SFR = {df_snr.loc[gal, 'SFR (component 0)']:.3f}" + r" $\rm M_\odot\,yr^{-1}$" + f", SNR = {df_snr.loc[gal, 'Median SNR (R, 2R_e)']:.2f}", 
        x=0.0, y=1.02, transform=axs_bpt[0].transAxes)
    for cc, col_x in enumerate(["log N2", "log S2", "log O1"]):
        # Plot full SAMI sample
        plot2dhistcontours(df=df_sami, 
                           col_x=f"{col_x} (total)",
                           col_y=f"{col_y} (total)", col_z="count", log_z=True,
                           alpha=0.5, cmap="gray_r",
                           ax=axs_bpt[cc], plot_colorbar=False)

        # Add BPT functions
        plot_BPT_lines(ax=axs_bpt[cc], col_x=col_x)    

        # Plot measurements for this galaxy
        plot2dscatter(df=df_gal,
                      col_x=f"{col_x} (total)",
                      col_y=f"{col_y} (total)",
                      col_z=None if col_z == "Number of components" else col_z,
                      marker=markers[0], ax=axs_bpt[cc], 
                      cax=None,
                      markersize=20, 
                      markerfacecolour=component_colours[0] if col_z == "Number of components" else None, 
                      edgecolors="black",
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
        plot2dhistcontours(df=df_sami, 
                           col_x=f"{col_x} (total)" if col_x == "log N2" else f"{col_x}",
                           col_y=f"log HALPHA EW (total)" if col_x == "log N2" else f"log HALPHA EW",
                           col_z="count", log_z=True,
                           alpha=0.5, cmap="gray_r", ax=axs_whav[cc],
                           plot_colorbar=False)

    # WHAN diagram
    plot2dscatter(df=df_gal,
                  col_x=f"log N2 (total)",
                  col_y=f"log HALPHA EW (total)",
                  col_z=None if col_z == "Number of components" else col_z,
                  marker=markers[0], ax=axs_whav[0], 
                  cax=None,
                  markersize=20, 
                  markerfacecolour=component_colours[0] if col_z == "Number of components" else None, 
                  edgecolors="black",
                  plot_colorbar=False)

    # Kinematics 
    for cc, col_x in enumerate(["sigma_gas - sigma_*", "v_gas - v_*"]):
        # Plot the data for this galaxy
        for ii in range(3):
            plot2dscatter(df=df_gal,
                          col_x=f"{col_x} (component {ii})",
                          col_y=f"log HALPHA EW (component {ii})",
                          col_z=None if col_z == "Number of components" else col_z,
                          marker=markers[ii], ax=axs_whav[cc + 1], 
                          cax=None,
                          markersize=20, 
                          markerfacecolour=component_colours[ii] if col_z == "Number of components" else None, 
                          edgecolors="black",
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

    if savefigs:
        # fname = "ncomponents" if col_z == "Number of components" else "radius"
        # fig_collage.savefig(os.path.join(fig_path, f"{gal}_SAMI_summary_{fname}.pdf"), format="pdf", bbox_inches="tight")
        pp.savefig(fig_collage, bbox_inches="tight")

    ###########################################################################
    # Pause
    ###########################################################################
    # fig_collage.canvas.draw()
    # Tracer()()

    # # Clear axes 
    # t.remove()  # remove text
    # for ax in axs_bpt:
    #     [ln.remove() for ln in ax.collections[12:]]
    # for ax in axs_whav:
    #     [ln.remove() for ln in ax.collections[12:]]

    # fig_collage.canvas.draw()
    # plt.close(fig_ims)

    plt.close(fig_collage)

if savefigs:
    pp.close()
