# Imports
import sys
import os 
import numpy as np
import pandas as pd
from astropy.visualization import hist
from tqdm import tqdm

from spaxelsleuth.loaddata.lzifu import load_lzifu_galaxies
from spaxelsleuth.loaddata.sami import load_sami_galaxies
from spaxelsleuth.plotting.plot2dmap import plot2dmap
from spaxelsleuth.plotting.sdssimg import plot_sdss_image
from spaxelsleuth.plotting.plottools import plot_empty_BPT_diagram, plot_BPT_lines
from spaxelsleuth.plotting.plottools import vmin_fn, vmax_fn, label_fn, cmap_fn, fname_fn, component_colours
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

###########################################################################
sami_data_path = "/priv/meggs3/u5708159/SAMI/sami_dr3/"

###########################################################################
# Options
###########################################################################
fig_path = "/priv/meggs3/u5708159/SAMI/figs/paper/"
lzifu_data_path = "/priv/meggs3/u5708159/LZIFU/products"
savefigs = True
bin_type = "default"    # Options: "default" or "adaptive" for Voronoi binning
ncomponents = "recom"   # Options: "1" or "recom"
eline_SNR_min = 5       # Minimum S/N of emission lines to accept
debug = False

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
        assert os.path.exists(os.path.join(lzifu_data_path, f"{gal}_merge_lzcomp.fits"))
    df_all = None
else:
    # gals = [int(f.split("_merge_lzcomp.fits")[0]) for f in os.listdir(lzifu_data_path) if f.endswith("merge_lzcomp.fits") and not f.startswith("._")]
    # Load galaxies in the "good" sample
    # Load the DataFrame that gives us the continuum S/N, to define the subset
    # df_info = pd.read_hdf(os.path.join(sami_data_path, "sami_dr3_metadata_extended.hd5"))

    # # Shortlist: median R S/N in 2R_e > 10
    # gals = df_info[df_info["Median SNR (R, 2R_e)"] >= 10].index.values.astype(int)
    # gals = [g for g in gals if df_info.loc[g, "Maximum number of components"] > 0]

    # Load the full sample
    df_all = load_lzifu_galaxies(bin_type=bin_type, ncomponents=ncomponents,
                                 eline_SNR_min=eline_SNR_min,
                                 sigma_gas_SNR_cut=True,
                                 vgrad_cut=False,
                                 stekin_cut=True)    
    gals = df_all.catid.unique()

    if debug:
        gals = gals[:1]

for gal in tqdm(gals):

    try:
        # Load the DataFrame
        if df_all is not None:
            df_gal = df_all[df_all["catid"] == gal]
        else:
            df_gal = load_lzifu_galaxies(gal=gal, 
                                     bin_type=bin_type, ncomponents=ncomponents,
                                     eline_SNR_min=eline_SNR_min,
                                     sigma_gas_SNR_cut=True,
                                     vgrad_cut=False,
                                     stekin_cut=True)    

        df_gal.loc[df_gal["Number of components"] == 0, "Number of components"] = np.nan

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

        ###########################################################################
        # Collage figure 2: coloured by r/R_e
        ###########################################################################
        for col_z in ["Number of components", "r/R_e", "BPT (numeric)", "log N2", "HALPHA EW (total)", "WHAV* (numeric)"]:
            # Create the figure
            fig_collage = plt.figure(figsize=(15, 7))
            ax_sdss = fig_collage.add_axes([l, b, w, h])
            ax_im = fig_collage.add_axes([l, b + h + dh, w, h])
            bbox = ax_im.get_position()
            if col_z != "WHAV* (numeric)":
                cax_im = fig_collage.add_axes([bbox.x0 + bbox.width * 0.035, bbox.y0 + bbox.height, bbox.width * 0.93, 0.025])
            else:
                cax_im = None
            axs_bpt = []
            axs_bpt.append(fig_collage.add_axes([l + w + dw, b + h + dh, w, h]))
            axs_bpt.append(fig_collage.add_axes([l + w + dw + w, b + h + dh, w, h]))
            axs_bpt.append(fig_collage.add_axes([l + w + dw + 2 * w, b + h + dh, w, h]))
            cax_bpt = fig_collage.add_axes(([l + w + dw + 3 * w, b + h + dh, 0.025, h])) if col_z != "Number of components" else None
            axs_whav = []
            axs_whav.append(fig_collage.add_axes([l + w + dw, b, w, h]))
            axs_whav.append(fig_collage.add_axes([l + w + dw + w, b, w, h]))
            axs_whav.append(fig_collage.add_axes([l + w + dw + 2 * w, b, w, h]))
            # cax_whav = fig_collage.add_axes(([l + w + dw + 3 * w, b, 0.025, h])) if col_z != "Number of components" else None

            # SDSS image
            plot_sdss_image(df_gal, ax=ax_sdss)

            # Plot the number of components fitted.
            plot2dmap(df_gal=df_gal, bin_type="default", survey="sami",
                      PA_deg=0,
                      col_z="Number of components" if f"{col_z} (component 0)" in df_gal.columns else col_z,
                      ax=ax_im, 
                      plot_colorbar=False if col_z == "WHAV* (numeric)" else True, cax=cax_im, cax_orientation="horizontal", 
                      show_title=False)

            # Plot BPT diagram
            col_y = "log O3"
            axs_bpt[0].text(s=f"GAMA{gal}", x=0.1, y=0.9, transform=axs_bpt[0].transAxes)
            for cc, col_x in enumerate(["log N2", "log S2", "log O1"]):
                # Plot full SAMI sample
                plot2dhistcontours(df=df_sami, 
                                   col_x=f"{col_x} (total)",
                                   col_y=f"{col_y} (total)", col_z="count", log_z=True,
                                   alpha=0.5, cmap="gray_r",
                                   ax=axs_bpt[cc], plot_colorbar=False)

                # Add BPT functions
                plot_BPT_lines(ax=axs_bpt[cc], col_x=col_x)    

                # Plot LZIFU measurements
                for ii in range(3):
                    if col_z == "Number of components":
                        colz = None
                    elif col_z in df_gal.columns:
                        colz = col_z 
                    elif f"{col_z} (component 0)" in df_gal.columns:
                        colz = f"{col_z} (component {ii})"
                    plot2dscatter(df=df_gal,
                                  col_x=f"{col_x} (component {ii})",
                                  col_y=f"{col_y} (component {ii})",
                                  col_z=colz,
                                  marker=markers[ii], ax=axs_bpt[cc], 
                                  cax=cax_bpt,
                                  markersize=20, 
                                  markerfacecolor=component_colours[ii] if col_z == "Number of components" else None, 
                                  markeredgecolor="black",
                                  plot_colorbar=True if (ii == 0) and (col_z != "Number of components") else False)

                # axis limits
                axs_bpt[cc].set_xlim(
                    [np.nanmin([vmin_fn(col_x), 
                               np.nanmin([df_gal[f"{col_x} (component 0)"].min(), 
                                          df_gal[f"{col_x} (component 1)"].min(), 
                                          df_gal[f"{col_x} (component 2)"].min()]) - 0.1]),
                     np.nanmax([vmax_fn(col_x), 
                               np.nanmax([df_gal[f"{col_x} (component 0)"].max(), 
                                          df_gal[f"{col_x} (component 1)"].max(), 
                                          df_gal[f"{col_x} (component 2)"].max()]) + 0.1])])
                # axis limits
                axs_bpt[cc].set_ylim(
                    [np.nanmin([vmin_fn(col_y), 
                               np.nanmin([df_gal[f"{col_y} (component 0)"].min(), 
                                          df_gal[f"{col_y} (component 1)"].min(), 
                                          df_gal[f"{col_y} (component 2)"].min()]) - 0.1]),
                     np.nanmax([vmax_fn(col_y), 
                               np.nanmax([df_gal[f"{col_y} (component 0)"].max(), 
                                          df_gal[f"{col_y} (component 1)"].max(), 
                                          df_gal[f"{col_y} (component 2)"].max()]) + 0.1])])

            # Decorations
            [ax.grid() for ax in axs_bpt]
            [ax.set_ylabel("") for ax in axs_bpt[1:]]
            [ax.set_yticklabels([]) for ax in axs_bpt[1:]]
            [ax.set_xticks(ax.get_xticks()[:-1]) for ax in axs_bpt[:-1]]
            [ax.collections[0].set_rasterized(True) for ax in axs_bpt]

            ###########################################################################
            # Plot WHAN, WHAV and WHAV* diagrams.
            ###########################################################################
            col_y = "log HALPHA EW"
            # Plot LZIFU measurements
            for cc, col_x in enumerate(["log N2", "sigma_gas - sigma_*", "v_gas - v_*"]):
                # Plot full SAMI sample
                plot2dhistcontours(df=df_sami, col_x=f"{col_x} (total)" if col_x == "log N2" else f"{col_x}",
                                   col_y=f"{col_y} (total)" if col_x == "log N2" else f"{col_y}",
                                   col_z="count", log_z=True,
                                   alpha=0.5, cmap="gray_r", ax=axs_whav[cc],
                                   plot_colorbar=False)
                # Plot the S7 data
                for ii in range(3):
                    if col_z == "Number of components":
                        colz = None
                    elif col_z in df_gal.columns:
                        colz = col_z 
                    elif f"{col_z} (component 0)" in df_gal.columns:
                        colz = f"{col_z} (component {ii})"
                    plot2dscatter(df=df_gal,
                                  col_x=f"{col_x} (component {ii})",
                                  col_y=f"{col_y} (component {ii})",
                                  col_z=colz,
                                  marker=markers[ii], ax=axs_whav[cc], 
                                  cax=None,
                                  vmin=-0.5 if col_z == "log N2" else None, vmax=0.2 if col_z == "log N2" else None,
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

            # Axis limits
            axs_whav[0].set_xlim(
                [np.nanmin([vmin_fn("log N2"),
                            df_gal["log N2 (component 0)"].min(), 
                            df_gal["log N2 (component 1)"].min(), 
                            df_gal["log N2 (component 2)"].min()]) - 0.1,
                 np.nanmax([vmax_fn("log N2"), 
                            df_gal["log N2 (component 0)"].max(),
                            df_gal["log N2 (component 1)"].max(),
                            df_gal["log N2 (component 2)"].max()]) + 0.1])
            axs_whav[1].set_xlim(
                [np.nanmin([vmin_fn("sigma_gas - sigma_*"), 
                            df_gal["sigma_gas - sigma_* (component 0)"].min(), 
                            df_gal["sigma_gas - sigma_* (component 1)"].min(), 
                            df_gal["sigma_gas - sigma_* (component 2)"].min()]) - 50,
                 np.nanmax([vmax_fn("sigma_gas - sigma_*"),
                            df_gal["sigma_gas - sigma_* (component 0)"].max(),
                            df_gal["sigma_gas - sigma_* (component 1)"].max(),
                            df_gal["sigma_gas - sigma_* (component 2)"].max()]) + 50])
            axs_whav[2].set_xlim(
                [np.nanmin([vmin_fn("v_gas - v_*"),
                            df_gal["v_gas - v_* (component 0)"].min(), 
                            df_gal["v_gas - v_* (component 1)"].min(), 
                            df_gal["v_gas - v_* (component 2)"].min()]) - 50,
                 np.nanmax([vmax_fn("v_gas - v_*"),
                            df_gal["v_gas - v_* (component 0)"].max(),
                            df_gal["v_gas - v_* (component 1)"].max(),
                            df_gal["v_gas - v_* (component 2)"].max()]) + 50])
            
            # Legend
            legend_elements = [Line2D([0], [0], marker=markers[ii], 
                                      color="none", markeredgecolor="black",
                                      label=f"Component {ii + 1}",
                                      markerfacecolor=component_colours[ii] if col_z == "Number of components" else "white", markersize=5) for ii in range(3)]
            axs_bpt[-1].legend(handles=legend_elements, fontsize="x-small", loc="upper right")

            if savefigs:
                fname = fname_fn(col_z)
                fig_collage.savefig(os.path.join(fig_path, f"{gal}_LZIFU_summary_{fname}.pdf"), format="pdf", bbox_inches="tight")

        ###########################################################################
        # Scatter plot: line ratios vs. velocity dispersion
        ###########################################################################
        col_y_list = ["log N2", "log S2", "log O1", "log O3"]
        fig_line_ratios, axs = plt.subplots(nrows=len(col_y_list), ncols=2, figsize=(7, 9), sharex="col", sharey="row")
        fig_line_ratios.subplots_adjust(wspace=0, hspace=0)
        bbox = axs[0][0].get_position()
        cax = fig_line_ratios.add_axes([bbox.x0, bbox.y0 + bbox.height, 2 * bbox.width, 0.03])

        # log N2, S2, O1 vs. velocity dispersion
        # axs[0][0].text(s=f"{gal} ($i = {df_gal['Inclination i (degrees)'].unique()[0]:.2f}^\circ$)", x=0.1, y=0.9, transform=axs[0][0].transAxes)
        for rr, col_y in enumerate(col_y_list):
            for ii in range(3):
                plot2dscatter(df=df_gal, 
                              col_x=f"sigma_gas (component {ii})",
                              col_y=f"{col_y} (component {ii})",
                              col_z=f"BPT (numeric) (component {ii})",
                              marker=markers[ii], markerfacecolor=component_colours[ii], markeredgecolor="black",
                              ax=axs[rr][0], plot_colorbar=False)
                plot2dscatter(df=df_gal, 
                              col_x=f"sigma_gas - sigma_* (component {ii})",
                              col_y=f"{col_y} (component {ii})",
                              col_z=f"BPT (numeric) (component {ii})",
                              marker=markers[ii], markerfacecolor=component_colours[ii], markeredgecolor="black",
                              ax=axs[rr][1], plot_colorbar=True if ii == 2 else False, cax_orientation="horizontal",
                              cax=cax) 
                axs[rr][1].set_ylabel("")
        [ax.autoscale(axis="x", tight=True, enable=True) for ax in axs.flat]
        [ax.autoscale(axis="y", tight=True, enable=True) for ax in axs.flat]
        [ax.grid() for ax in axs.flat]
        fig_line_ratios.suptitle(f"GAMA{gal}", y=0.99)

        ###########################################################################
        # Maps showing various quantities for each component
        ###########################################################################
        col_z_list = ["HALPHA EW", "BPT (numeric)", "v_gas", "sigma_gas", "sigma_*", "sigma_gas - sigma_*"]
        fig_maps, axs = plt.subplots(nrows=len(col_z_list), ncols=3, figsize=(9.7, 18.8))
        fig_maps.subplots_adjust(wspace=0)

        for cc, col_z in enumerate(col_z_list):
            bbox = axs[cc][-1].get_position()
            cax = fig_maps.add_axes([bbox.x0 + bbox.width, bbox.y0, 0.03, bbox.height])

            # Vmin, vmax
            if col_z == "sigma_gas - sigma_*":
                vmin, vmax = (-250, +250)
            else:
                vmin, vmax = (None, None)

            for ii in range(3):
                _, ax = plot2dmap(df_gal=df_gal, bin_type="default", survey="sami", 
                                  col_z=f"{col_z} (component {ii})" if f"{col_z} (component {ii})" in df_gal else col_z,
                                  vmin=vmin, vmax=vmax,
                                  ax=axs[cc][ii], show_title=False, plot_colorbar=True if ii == 2 else False, cax=cax)
                
                # Decorations
                if ii > 0:
                    # Turn off axis labels
                    lat = plt.gca().coords[1]
                    lat.set_ticks_visible(False)
                    lat.set_ticklabel_visible(False)
                else:
                    ax.text(s=label_fn(col_z), x=0.1, y=0.9, transform=axs[cc][0].transAxes, verticalalignment="top")
                if cc < len(col_z_list) - 1:
                    lon = plt.gca().coords[0]
                    lon.set_ticks_visible(False)
                    lon.set_ticklabel_visible(False)
        for ii in range(3):
            fig_maps.get_axes()[1 + ii].set_title(f"Component {ii + 1}")
        fig_maps.suptitle(f"GAMA{gal}", y=0.92)

        # Save 
        if savefigs:
            fig_maps.savefig(os.path.join(fig_path, f"{gal}_LZIFU_maps.pdf"), format="pdf", bbox_inches="tight")
            fig_line_ratios.savefig(os.path.join(fig_path, f"{gal}_LZIFU_line_ratios_vs_sigma_gas.pdf"), format="pdf", bbox_inches="tight")

    except:
        print(f"ERROR: processing galaxy {gal} failed!")

    plt.close("all")

  
