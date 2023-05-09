import json
from matplotlib.colors import ListedColormap, to_rgba
import matplotlib.patheffects as PathEffects
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Tuple

from spaxelsleuth import config
from spaxelsleuth.utils.linefns import Kewley2001, Kewley2006, Kauffman2003, Law2021_1sigma, Law2021_3sigma

#/////////////////////////////////////////
# Load the plotting settings
plot_settings = config.settings["plotting"]

#/////////////////////////////////////////
# Custom colour list for different components
component_colours = ["#781EE5", "#FF53B4", "#FFC107"]
component_labels = ["Component 1", "Component 2", "Component 3"]

c1 = to_rgba("#4A4A4A")
c2 = to_rgba("#6EC0FF")
c3 = to_rgba("#8AE400")
c4 = to_rgba("#A722FF")
ncomponents_colours = np.vstack((c1, c2, c3, c4))

#/////////////////////////////////////////
def trim_suffix(col: str) -> Tuple[str]:
    """Trim the suffix from col & return the trimmed column name and suffix separately."""
    suffix = ""
    for s in config.settings["plotting"]["suffixes"]:
        if s in col:
            col_s = col.split(s)[0]
            suffix = col.split(col_s)[1]
            col = col_s
            break

    # If col is a metallicity or ionisation parameter then it will also contain a substring specifying the diagnostic used. Trim these as well.
    diags = ""
    if col.startswith("log(U)"):
        diags = col.split("log(U)")[1]
        col = col.split(diags)[0]
        diags = diags.replace("/", "+")
    elif col.startswith("log(O/H) + 12"):
        diags = col.split("log(O/H) + 12")[1]
        col = col.split(diags)[0]
        diags = diags.replace("/", "+")

    return col, diags + suffix


#/////////////////////////////////////////
def get_custom_cmap(cmap_str: str) -> Tuple[ListedColormap, np.ndarray, list]:
    """Returns custom colourmaps for discrete quantities."""

    if cmap_str == "bpt_cmap":
        # Custom colour map for BPT categories
        labels = [
            "Not classified", "SF", "Composite", "LINER", "Seyfert",
            "Ambiguous"
        ]
        ticks = np.arange(len(labels)) - 1
        c1 = np.array([128 / 256, 128 / 256, 128 / 256, 1])  # Not classified
        c2 = np.array([0 / 256, 0 / 256, 256 / 256, 1])  # SF
        c3 = np.array([0 / 256, 255 / 256, 188 / 256, 1])  # Composite
        c4 = np.array([256 / 256, 100 / 256, 0 / 256, 1])  # LINER
        c5 = np.array([256 / 256, 239 / 256, 0 / 256, 1])  # Seyfert
        c6 = np.array([256 / 256, 100 / 256, 256 / 256, 1])  # Ambiguous
        colours = np.vstack((c1, c2, c3, c4, c5, c6))
        cmap = ListedColormap(colours)
        cmap.set_bad(color="white", alpha=0)

    elif cmap_str == "morph_cmap":
        # Custom colour map for morphologies
        labels = [
            "Unknown", "E", "E/S0", "S0", "S0/Early-spiral", "Early-spiral",
            "Early/Late spiral", "Late spiral"
        ]
        ticks = (np.arange(len(labels)) - 1) / 2
        rdylbu = plt.cm.get_cmap("RdYlBu")
        rdylbu_colours = rdylbu(np.linspace(0, 1, len(labels)))
        rdylbu_colours[0] = [0.5, 0.5, 0.5, 1]
        cmap = ListedColormap(rdylbu_colours)
        cmap.set_bad(color="white", alpha=0)

    elif cmap_str == "law2021_cmap":
        # Custom colour map for Law+2021 kinematic classifications
        labels = [
            "Not classified", "Cold", "Intermediate", "Warm", "Ambiguous"
        ]
        ticks = (np.arange(len(labels)) - 1)
        jet = plt.cm.get_cmap("jet")
        jet_colours = jet(np.linspace(0, 1, len(labels)))
        jet_colours[0] = [0.5, 0.5, 0.5, 1]
        cmap = ListedColormap(jet_colours)
        cmap.set_bad(color="white", alpha=0)

    elif cmap_str == "ncomponents_cmap":
        # Custom colour map for number of components
        labels = ["0", "1", "2", "3"]
        ticks = np.array([0, 1, 2, 3])
        c1 = to_rgba("#4A4A4A")
        c2 = to_rgba("#6EC0FF")
        c3 = to_rgba("#8AE400")
        c4 = to_rgba("#A722FF")
        colours = np.vstack((c1, c2, c3, c4))
        cmap = ListedColormap(colours)
        cmap.set_bad(color="#b3b3b3", alpha=0)

    else:
        raise ValueError(
            f"in get_custom_cmap(): cmap_str {cmap_str} not recognised!")

    return cmap, ticks, labels


#/////////////////////////////////////////
def get_vmin(col: str) -> float:
    """Returns minimum value used for plotting quantity col."""
    col, _ = trim_suffix(col)
    if col in plot_settings:
        return plot_settings[col]["vmin"]
    else:
        return None


#/////////////////////////////////////////
def get_vmax(col: str) -> float:
    """Returns maximum value used for plotting quantity col."""
    col, _ = trim_suffix(col)
    if col in plot_settings:
        return plot_settings[col]["vmax"]
    else:
        return None


#/////////////////////////////////////////
def get_cmap(col: str):
    """Returns colourmap (plus ticks and labels for discrete quantities) used for plotting quantity col."""
    col, _ = trim_suffix(col)
    # Get the cmap_str
    if col in plot_settings:
        cmap_str = plot_settings[col]["cmap"]
    else:
        cmap_str = plot_settings["default"]["cmap"]
    # Now get the cmap
    try:
        cmap = plt.cm.get_cmap(cmap_str).copy()
        cmap.set_bad("#b3b3b3")
        return cmap
    except ValueError:
        cmap, ticks, labels = get_custom_cmap(cmap_str)
        return cmap, ticks, labels


#/////////////////////////////////////////
def get_fname(col: str) -> str:
    """Returns system-safe filename for column col."""
    col, suffix = trim_suffix(col)
    if col in plot_settings:
        # Get the filename
        fname = plot_settings[col]["fname"]
        # Add filename for suffix, if it exists
        if len(suffix) > 0:
            fname_suffix = suffix.replace("(", "").replace(")", "").replace(
                "/", "_over_").replace("*",
                                       "_star").replace(" ",
                                                        "_").replace("$", "")
            fname += "_" + fname_suffix
    else:
        fname = col.replace("(", "").replace(")", "").replace(
            "/", "_over_").replace("*", "_star").replace(" ",
                                                         "_").replace("$", "")
    return fname


#/////////////////////////////////////////
def get_label(col: str) -> str:
    """Returns LaTeX label for column col."""
    col, suffix = trim_suffix(col)
    if col in plot_settings:
        # Get the filename
        label = plot_settings[col]["label"]
    else:
        label = col
        # At least try to beautify some strings
        label = label.replace("ALPHA", r"$\alpha$")
        label = label.replace("BETA", r"$\beta$")
        label = label.replace("GAMMA", r"$\gamma$")
        label = label.replace("delta",
                              r"$\Delta$")  # always assume upper case delta
        label = label.replace("R_e", r"$R_e$")
        label = label.replace("log ", r"$\log_{10}$ ")
    
    # Add label for suffix, if it exists
    if len(suffix) > 0:
        suffix_label_dict = config.settings["plotting"]["suffix_labels"]
        if suffix in suffix_label_dict:
            label_suffix = suffix_label_dict[suffix]
        else:
            label_suffix = suffix
        label += label_suffix

    return label

#/////////////////////////////////////////
def plot_empty_BPT_diagram(colorbar=False,
                           nrows=1,
                           include_Law2021=False,
                           axs=None,
                           figsize=None):
    """
    Create an empty BPT diagram.
    Create a figure containing empty axes for the N2, S2 and O1 Baldwin, 
    Philips & Terlevich (1986) optical line ratio diagrams, with the 
    demarcation lines of Kewley et al. (2001), Kewley et al. (2006) and 
    Kauffman et al. (2003) over-plotted. Optionally, also include the 1-sigma 
    and 3-sigma kinematic demarcation lines of Law et al. (2021).
    """
    # Make axes
    if figsize is not None:
        fig = plt.figure(figsize=figsize)
    else:
        fig = plt.figure(figsize=(15, 5 * nrows))
    left = 0.1
    bottom = 0.1
    if colorbar:
        cbar_width = 0.025
    else:
        cbar_width = 0
    width = (1 - 2 * left - cbar_width) / 3.
    height = 0.8 / nrows

    axs = []
    if colorbar:
        caxs = []
    for ii in range(nrows):
        bottom = 0.1 + (nrows - ii - 1) * height
        ax_N2 = fig.add_axes([left, bottom, width, height])
        ax_S2 = fig.add_axes([left + width, bottom, width, height])
        ax_O1 = fig.add_axes([left + 2 * width, bottom, width, height])
        if colorbar:
            cax = fig.add_axes([left + 3 * width, bottom, cbar_width, height])

        # Plot the reference lines from literature
        x_vals = np.linspace(-2.5, 2.5, 100)
        ax_N2.plot(x_vals,
                   Kewley2001("log N2", x_vals),
                   "gray",
                   linestyle="--")
        ax_S2.plot(x_vals,
                   Kewley2001("log S2", x_vals),
                   "gray",
                   linestyle="--")
        ax_O1.plot(x_vals,
                   Kewley2001("log O1", x_vals),
                   "gray",
                   linestyle="--")
        ax_S2.plot(x_vals,
                   Kewley2006("log S2", x_vals),
                   "gray",
                   linestyle="-.")
        ax_O1.plot(x_vals,
                   Kewley2006("log O1", x_vals),
                   "gray",
                   linestyle="-.")
        ax_N2.plot(x_vals,
                   Kauffman2003("log N2", x_vals),
                   "gray",
                   linestyle=":")

        if include_Law2021:
            y_vals = np.copy(x_vals)
            ax_N2.plot(x_vals,
                       Law2021_1sigma("log N2", x_vals),
                       "gray",
                       linestyle="-")
            ax_S2.plot(x_vals,
                       Law2021_1sigma("log S2", x_vals),
                       "gray",
                       linestyle="-")
            ax_O1.plot(x_vals,
                       Law2021_1sigma("log O1", x_vals),
                       "gray",
                       linestyle="-")
            ax_N2.plot(Law2021_3sigma("log N2", y_vals),
                       y_vals,
                       "gray",
                       linestyle="-")
            ax_S2.plot(Law2021_3sigma("log S2", y_vals),
                       y_vals,
                       "gray",
                       linestyle="-")
            ax_O1.plot(Law2021_3sigma("log O1", y_vals),
                       y_vals,
                       "gray",
                       linestyle="-")

        # Axis limits
        ymin = -1.5
        ymax = 1.2
        ax_N2.set_ylim([ymin, ymax])
        ax_S2.set_ylim([ymin, ymax])
        ax_O1.set_ylim([ymin, ymax])
        ax_N2.set_xlim([-1.3, 0.5])
        ax_S2.set_xlim([-1.3, 0.5])
        ax_O1.set_xlim([-2.2, 0.2])

        # Add axis labels
        ax_N2.set_ylabel(r"$\log_{10}$[O III]$\lambda5007$/H$\beta$")
        if ii == nrows - 1:
            ax_N2.set_xlabel(r"$\log_{10}$[N II]$\lambda6583$/H$\alpha$")
            ax_S2.set_xlabel(
                r"$\log_{10}$[S II]$\lambda\lambda6716,6731$/H$\alpha$")
            ax_O1.set_xlabel(r"$\log_{10}$[O I]$\lambda6300$/H$\alpha$")

        ax_S2.set_yticklabels([])
        ax_O1.set_yticklabels([])

        # Add axes to lists
        axs.append(ax_N2)
        axs.append(ax_S2)
        axs.append(ax_O1)
        if colorbar:
            caxs.append(cax)

    if colorbar:
        if nrows == 1:
            return fig, axs, cax
        else:
            return fig, axs, caxs
    else:
        return fig, axs


#/////////////////////////////////////////
def plot_BPT_lines(ax,
                   col_x,
                   include_Law2021=False,
                   color="gray",
                   linewidth=1,
                   zorder=1):
    """
    Plot BPT demarcation lines of Kewley+2001, Kauffman+2003, Kewley+2006 and Law+2021) on the provided axis.
    The y-axis is assumed to be O3 and the x-axis is specified by col_x.
    """
    if col_x not in ["log N2", "log S2", "log O1"]:
        raise ValueError("col_x must be one of log N2, log S2 or log O1!")

    # Plot the demarcation lines from literature
    x_vals = np.linspace(-2.5, 2.5, 100)

    # Kewley+2001: all 3 diagrams
    ax.plot(x_vals,
            Kewley2001(col_x, x_vals),
            color=color,
            linewidth=linewidth,
            linestyle="--",
            zorder=zorder)

    # Kewley+2006: S2 and O1 only
    if col_x == "log S2" or col_x == "log O1":
        ax.plot(x_vals,
                Kewley2006(col_x, x_vals),
                color=color,
                linewidth=linewidth,
                linestyle="-.",
                zorder=zorder)

    # Kauffman+2003: log N2 only
    if col_x == "log N2":
        ax.plot(x_vals,
                Kauffman2003(col_x, x_vals),
                color=color,
                linewidth=linewidth,
                linestyle=":",
                zorder=zorder)

    if include_Law2021:
        y_vals = np.copy(x_vals)
        ax.plot(x_vals,
                Law2021_1sigma(col_x, x_vals),
                color=color,
                linewidth=linewidth,
                linestyle="-",
                zorder=zorder)
        ax.plot(Law2021_3sigma(col_x, y_vals),
                y_vals,
                color=color,
                linewidth=linewidth,
                linestyle="-",
                zorder=zorder)

    # Axis limits
    ax.set_ylim([-1.5, 1.2])
    if col_x == "log N2":
        ax.set_xlim([-1.3, 0.5])
    elif col_x == "log S2":
        ax.set_xlim([-1.3, 0.5])
    elif col_x == "log O1":
        ax.set_xlim([-2.2, 0.2])

    return


#/////////////////////////////////////////
def plot_compass(PA_deg=0,
                 flipped=True,
                 color="white",
                 bordercolor=None,
                 fontsize=10,
                 ax=None,
                 zorder=999999):
    """
    Plot a compass showing the directions of N and E.
    Display North and East-pointing arrows on a plot corresponding to the 
    position angle given by PA_deg.
    """
    PA_rad = np.deg2rad(PA_deg)
    if not ax:
        ax = plt.gca()
    w_x = np.diff(ax.get_xlim())[0]
    w_y = np.diff(ax.get_ylim())[0]
    w = min(w_x, w_y)
    l = 0.05 * w
    if flipped:
        origin_x = ax.get_xlim()[0] + 0.9 * w - l * np.sin(PA_rad)
    else:
        origin_x = ax.get_xlim()[0] + 0.1 * w - l * np.sin(PA_rad)
    origin_y = ax.get_ylim()[0] + 0.1 * w
    if np.abs(PA_deg) > 90:
        origin_y += l * np.sin(PA_rad - np.pi / 2)
    text_offset = 0.05 * w
    head_width = 0.015 * w
    head_length = 0.015 * w
    overhang = 0.1

    if not flipped:
        # N arrow
        ax.arrow(origin_x,
                 origin_y,
                 -l * np.sin(PA_rad),
                 l * np.cos(PA_rad),
                 head_width=head_width,
                 head_length=head_length,
                 overhang=overhang,
                 fc=color,
                 ec=color,
                 zorder=zorder)
        # E arrow
        ax.arrow(origin_x,
                 origin_y,
                 l * np.cos(PA_rad),
                 l * np.sin(PA_rad),
                 head_width=head_width,
                 head_length=head_length,
                 overhang=overhang,
                 fc=color,
                 ec=color,
                 zorder=zorder)
        ax.text(x=origin_x - 1.1 * (l + text_offset) * np.sin(PA_rad),
                y=origin_y + 1.1 * (l + text_offset) * np.cos(PA_rad),
                s="N",
                color=color,
                zorder=zorder,
                verticalalignment="center",
                horizontalalignment="center")
        ax.text(x=origin_x + 1.1 * (l + text_offset) * np.cos(PA_rad),
                y=origin_y + 1.1 * (l + text_offset) * np.sin(PA_rad),
                s="E",
                color=color,
                zorder=zorder,
                verticalalignment="center",
                horizontalalignment="center")

    else:
        # N arrow
        ax.arrow(origin_x,
                 origin_y,
                 l * np.sin(PA_rad),
                 l * np.cos(PA_rad),
                 head_width=head_width,
                 head_length=head_length,
                 overhang=overhang,
                 fc=color,
                 ec=color,
                 zorder=zorder)
        # E arrow
        ax.arrow(origin_x,
                 origin_y,
                 -l * np.cos(PA_rad),
                 l * np.sin(PA_rad),
                 head_width=head_width,
                 head_length=head_length,
                 overhang=overhang,
                 fc=color,
                 ec=color,
                 zorder=zorder)
        t = ax.text(x=origin_x + 1.1 * (l + text_offset) * np.sin(PA_rad),
                    y=origin_y + 1.1 * (l + text_offset) * np.cos(PA_rad),
                    s="N",
                    color=color,
                    zorder=zorder,
                    verticalalignment="center",
                    horizontalalignment="center",
                    fontsize=fontsize)
        if bordercolor is not None:
            t.set_path_effects([
                PathEffects.withStroke(linewidth=1.5, foreground=bordercolor)
            ])
        t = ax.text(x=origin_x - 1.1 * (l + text_offset) * np.cos(PA_rad),
                    y=origin_y + 1.1 * (l + text_offset) * np.sin(PA_rad),
                    s="E",
                    color=color,
                    zorder=zorder,
                    verticalalignment="center",
                    horizontalalignment="center",
                    fontsize=fontsize)
        if bordercolor is not None:
            t.set_path_effects([
                PathEffects.withStroke(linewidth=1.5, foreground=bordercolor)
            ])

        return


###############################################################################
def plot_scale_bar(as_per_px,
                   kpc_per_as,
                   l=0.5,
                   ax=None,
                   loffset=0.20,
                   boffset=0.075,
                   units="arcsec",
                   color="white",
                   fontsize=12,
                   long_dist_str=True,
                   bordercolor=None,
                   zorder=999999):
    """
    Plot a bar showing the angular scale.
    Plots a nice little bar in the lower-right-hand corner of a plot indicating
    the scale of the image in kiloparsecs corresponding to the plate scale of 
    the image in arcseconds per pixel (specified by as_per_px) and the 
    physical scale of the object in kiloparsecs per arcsecond (specified by 
    kpc_per_as).
    """
    if not ax:
        ax = plt.gca()

    w_x = np.diff(ax.get_xlim())[0]
    w_y = np.diff(ax.get_ylim())[0]
    w = min(w_x, w_y)
    line_centre_x = ax.get_xlim()[0] + loffset * w_x
    line_centre_y = ax.get_ylim()[0] + boffset * w_x
    text_offset = 0.035 * w

    # want to round l_kpc to the nearest half
    if units == "arcsec":
        l_arcsec = l
    elif units == "arcmin":
        l_arcmin = l
        l_arcsec = 60 * l_arcmin
    endpoints_x = np.array([-0.5, 0.5]) * l_arcsec / as_per_px + line_centre_x
    endpoints_y = np.array([0, 0]) + line_centre_y

    # How long is our bar?
    l_kpc = l_arcsec * kpc_per_as

    ax.plot(endpoints_x, endpoints_y, color, linewidth=5, zorder=zorder)
    if units == "arcsec":
        dist_str = f'{l_arcsec:.2f}" = {l_kpc:.2f} kpc' if long_dist_str else f'{l_kpc:.2f} kpc'
    elif units == "arcmin":
        dist_str = f"{l_arcmin:.2f}' = {l_kpc:.2f} kpc" if long_dist_str else f'{l_kpc:.2f} kpc'

    t = ax.text(x=line_centre_x,
                y=line_centre_y + text_offset,
                s=dist_str,
                size=fontsize,
                horizontalalignment="center",
                color=color,
                zorder=zorder)
    if bordercolor is not None:
        t.set_path_effects(
            [PathEffects.withStroke(linewidth=1.5, foreground=bordercolor)])

    return