import numpy as np
import copy
import pandas as pd

from matplotlib.colors import ListedColormap, to_rgba, LogNorm
import matplotlib.patheffects as PathEffects
import matplotlib.pyplot as plt

###############################################################################
# Custom colour maps for discrete quantities
###############################################################################
# Custom colour map for BPT categories
bpt_labels = ["Not classified", "SF", "Composite", "LINER", "Seyfert", "Ambiguous"]
bpt_ticks = np.arange(len(bpt_labels)) - 1
c1 = np.array([128/256,   128/256,   128/256, 1])   # Not classified
c2 = np.array([0/256,   0/256,   256/256, 1])       # SF
c3 = np.array([0/256,   255/256, 188/256, 1])       # Composite
c4 = np.array([256/256, 100/256, 0/256, 1])         # LINER
c5 = np.array([256/256, 239/256, 0/256, 1])         # Seyfert
c6 = np.array([256/256, 100/256, 256/256, 1])       # Ambiguous
bpt_colours = np.vstack((c1, c2, c3, c4, c5, c6))
bpt_cmap = ListedColormap(bpt_colours)
bpt_cmap.set_bad(color="white", alpha=0)

# Custom colour map for morphologies
morph_labels = ["Unknown", "E", "E/S0", "S0", "S0/Early-spiral", "Early-spiral", "Early/Late spiral", "Late spiral"]
morph_ticks = (np.arange(len(morph_labels)) - 1) / 2
rdylbu = plt.cm.get_cmap("RdYlBu")
rdylbu_colours = rdylbu(np.linspace(0, 1, len(morph_labels)))
rdylbu_colours[0] = [0.5, 0.5, 0.5, 1]
morph_cmap = ListedColormap(rdylbu_colours)
morph_cmap.set_bad(color="white", alpha=0)

# Custom colour map for Law+2021 kinematic classifications
law2021_labels = ["Not classified", "Cold", "Intermediate", "Warm", "Ambiguous"]
law2021_ticks = (np.arange(len(law2021_labels)) - 1)
jet = plt.cm.get_cmap("jet")
jet_colours = jet(np.linspace(0, 1, len(law2021_labels)))
jet_colours[0] = [0.5, 0.5, 0.5, 1]
law2021_cmap = ListedColormap(jet_colours)
law2021_cmap.set_bad(color="white", alpha=0)

# Custom colour map for number of components
ncomponents_labels = ["0", "1", "2", "3"]
ncomponents_ticks = [0, 1, 2, 3]
c1 = to_rgba("grey")
c2 = to_rgba("dodgerblue")
c3 = to_rgba("forestgreen")
c4 = to_rgba("purple")
ncomponents_colours = np.vstack((c1, c2, c3, c4))
ncomponents_cmap = ListedColormap(ncomponents_colours)
ncomponents_cmap.set_bad(color="white", alpha=0.0)

# SFR
sfr_cmap = copy.copy(plt.cm.get_cmap("magma"))
sfr_cmap.set_under("lightgray")

###############################################################################
# Colourmaps, min/max values and labels for each quantity
###############################################################################
cmap_dict = {
    "count": copy.copy(plt.cm.get_cmap("cubehelix")),
    "log N2": copy.copy(plt.cm.get_cmap("viridis")),
    "log O3": copy.copy(plt.cm.get_cmap("viridis")),
    "log O1": copy.copy(plt.cm.get_cmap("viridis")),
    "log S2": copy.copy(plt.cm.get_cmap("viridis")),
    "O3O2": copy.copy(plt.cm.get_cmap("viridis")),
    "log HALPHA EW": copy.copy(plt.cm.get_cmap("Spectral")),
    "log HALPHA EW (total)": copy.copy(plt.cm.get_cmap("Spectral")),
    "HALPHA EW": copy.copy(plt.cm.get_cmap("Spectral")),
    "HALPHA EW (total)": copy.copy(plt.cm.get_cmap("Spectral")),
    "log sigma_gas": copy.copy(plt.cm.get_cmap("plasma")),
    "sigma_gas": copy.copy(plt.cm.get_cmap("plasma")),
    "sigma_*": copy.copy(plt.cm.get_cmap("plasma")),
    "sigma_gas - sigma_*": copy.copy(plt.cm.get_cmap("RdYlBu_r")),
    "v_gas - v_*": copy.copy(plt.cm.get_cmap("RdYlBu_r")),
    "HALPHA S/N": copy.copy(plt.cm.get_cmap("copper")),
    "BPT (numeric)": bpt_cmap,
    "Law+2021 (numeric)": law2021_cmap,
    "radius": copy.copy(plt.cm.get_cmap("gnuplot2_r")),
    "D4000": copy.copy(plt.cm.get_cmap("pink_r")),
    "HALPHA": copy.copy(plt.cm.get_cmap("viridis")),
    "v_gas": copy.copy(plt.cm.get_cmap("coolwarm")),
    "v_*": copy.copy(plt.cm.get_cmap("coolwarm")),
    "A_V": copy.copy(plt.cm.get_cmap("afmhot_r")),
    "S2 ratio": copy.copy(plt.cm.get_cmap("cividis")),
    "O1O3": copy.copy(plt.cm.get_cmap("cividis")),
    "mstar": copy.copy(plt.cm.get_cmap("jet")),
    "g_i": copy.copy(plt.cm.get_cmap("RdYlBu_r")),
    "Morphology (numeric)": morph_cmap,
    "m_r": copy.copy(plt.cm.get_cmap("Reds")),
    "z_spec": copy.copy(plt.cm.get_cmap("plasma")),
    "log O3 ratio (1/0)": copy.copy(plt.cm.get_cmap("PiYG")),
    "log O3 ratio (2/1)": copy.copy(plt.cm.get_cmap("PiYG")),
    "log O1 ratio (1/0)": copy.copy(plt.cm.get_cmap("PiYG")),
    "log O1 ratio (2/1)": copy.copy(plt.cm.get_cmap("PiYG")),
    "log S2 ratio (1/0)": copy.copy(plt.cm.get_cmap("PiYG")),
    "log S2 ratio (2/1)": copy.copy(plt.cm.get_cmap("PiYG")),
    "log N2 ratio (1/0)": copy.copy(plt.cm.get_cmap("PiYG")),
    "log N2 ratio (2/1)": copy.copy(plt.cm.get_cmap("PiYG")),
    "sigma_gas/sigma_*": copy.copy(plt.cm.get_cmap("RdYlBu_r")),
    "N2O2": copy.copy(plt.cm.get_cmap("cividis")),
    "HALPHA EW/HALPHA EW (total)": copy.copy(plt.cm.get_cmap("jet")),
    "HALPHA EW ratio (1/0)": copy.copy(plt.cm.get_cmap("jet")),
    "HALPHA EW ratio (2/1)": copy.copy(plt.cm.get_cmap("jet")),
    "delta sigma_gas (1/0)": copy.copy(plt.cm.get_cmap("autumn")),
    "delta sigma_gas (2/1)": copy.copy(plt.cm.get_cmap("autumn")),
    "delta v_gas (1/0)": copy.copy(plt.cm.get_cmap("autumn")),
    "delta v_gas (2/1)": copy.copy(plt.cm.get_cmap("autumn")),
    "r/R_e": copy.copy(plt.cm.get_cmap("gnuplot2_r")),
    "R_e (kpc)": copy.copy(plt.cm.get_cmap("gnuplot2_r")),
    "log(M/R_e)": copy.copy(plt.cm.get_cmap("gnuplot2_r")),
    "Inclination i (degrees)": copy.copy(plt.cm.get_cmap("Spectral_r")), 
    "Bin size (square kpc)": copy.copy(plt.cm.get_cmap("gnuplot2_r")),
    "SFR": sfr_cmap,
    "SFR surface density": sfr_cmap,
    "log SFR": sfr_cmap,
    "log SFR surface density": sfr_cmap,
    "Delta HALPHA EW (0/1)": copy.copy(plt.cm.get_cmap("Spectral_r")),
    "Number of components": ncomponents_cmap,
    "HALPHA extinction correction": copy.copy(plt.cm.get_cmap("pink")),
    "v_grad": copy.copy(plt.cm.get_cmap("plasma")),
}

for key in cmap_dict.keys():
    # cmap_dict[key].set_bad("#b3b3b3")
    cmap_dict[key].set_bad("white")


vmin_dict = {
    "count": None,
    "log N2": -1.3,
    "log O3": -1.5,
    "log O1": -2.2,
    "log S2": -1.3,
    "O3O2": -2.5,
    "log HALPHA EW": -1,
    "log HALPHA EW (total)": -1,
    "HALPHA EW": 3,
    "HALPHA EW (total)": 3,
    "log sigma_gas": 1,
    "sigma_gas": 10,
    "sigma_*": 10,
    "sigma_gas - sigma_*": -300,
    "v_gas - v_*": -100,
    "HALPHA S/N": 3,
    "BPT (numeric)": -1.5,
    "Law+2021 (numeric)": -1.5,
    "radius": 0,
    "D4000": 1.0,
    "HALPHA": 0,
    "v_gas": -250,
    "v_*": -250,
    "A_V": 0,
    "S2 ratio": 0.38,
    "O1O3": -2,
    "mstar": 7.5,
    "g_i": -0.5,
    "Morphology (numeric)": -0.75,
    "m_r": -25,
    "z_spec": 0,
    "log O3 ratio (1/0)": -2,
    "log O3 ratio (2/1)": -2,
    "log O1 ratio (1/0)": -1,
    "log O1 ratio (2/1)": -1,
    "log N2 ratio (1/0)": -1,
    "log N2 ratio (2/1)": -1,
    "log S2 ratio (1/0)": -1,
    "log S2 ratio (2/1)": -1,
    "sigma_gas/sigma_*": 0,
    "N2O2": -1.5,
    "HALPHA EW/HALPHA EW (total)": 0,
    "HALPHA EW ratio (1/0)": 0,
    "HALPHA EW ratio (2/1)": 0,
    "delta sigma_gas (1/0)": 0,
    "delta sigma_gas (2/1)": 0,
    "delta v_gas (1/0)": -150,
    "delta v_gas (2/1)": -150,
    "r/R_e": 0,
    "R_e (kpc)": 0,
    "log(M/R_e)": 6,
    "Inclination i (degrees)": 0, 
    "Bin size (square kpc)": 0,
    "SFR": 0,
    "SFR surface density": 0,
    "log SFR": -5.0,
    "log SFR surface density": -4.0,
    "Delta HALPHA EW (0/1)": -1.0,
    "Number of components": -0.5,
    "HALPHA extinction correction": 1,
    "v_grad": 0,
}

vmax_dict = {
    "count": None,
    "log N2": 0.5,
    "log O3": 1.2,
    "log O1": 0.2,
    "log S2": 0.5,
    "O3O2": 0.5,
    "log HALPHA EW": 3.5,
    "log HALPHA EW (total)": 3.5,
    "HALPHA EW": 14,
    "HALPHA EW (total)": 14,
    "log sigma_gas": 3,
    "sigma_gas": 300,
    "sigma_*": 300,
    "sigma_gas - sigma_*": +300,
    "v_gas - v_*": +100,
    "HALPHA S/N": 50,
    "BPT (numeric)": 4.5,
    "Law+2021 (numeric)": 3.5,
    "radius": 10,
    "D4000": 2.2,
    "HALPHA": 1e3,  # 1.5 is good for SAMI
    "v_gas": +250,
    "v_*": +250,
    "A_V": 5,
    "S2 ratio": 1.44,
    "O1O3": 1.5,
    "mstar": 11.5,
    "g_i": 1.7,
    "Morphology (numeric)": 3.25,
    "m_r": -12.5,
    "z_spec": 0.1,
    "log O3 ratio (1/0)": +2,
    "log O3 ratio (2/1)": +2,
    "log O1 ratio (1/0)": +1,
    "log O1 ratio (2/1)": +1,
    "log N2 ratio (1/0)": +1,
    "log N2 ratio (2/1)": +1,
    "log S2 ratio (1/0)": +1,
    "log S2 ratio (2/1)": +1,
    "sigma_gas/sigma_*": 4,
    "N2O2": 0.5,
    "HALPHA EW/HALPHA EW (total)": 1,
    "HALPHA EW ratio (1/0)": 2,
    "HALPHA EW ratio (2/1)": 2,
    "delta sigma_gas (1/0)": +150,
    "delta sigma_gas (2/1)": +150,
    "delta v_gas (1/0)": +150,
    "delta v_gas (2/1)": +150,
    "r/R_e": 1,
    "R_e (kpc)": 10,
    "log(M/R_e)": 12,
    "Inclination i (degrees)": 90, 
    "Bin size (square kpc)": 0.5,
    "SFR": 0.02,
    "SFR surface density": 0.05,
    "log SFR": -1.0,
    "log SFR surface density": -0.0,
    "Delta HALPHA EW (0/1)": +2.0,
    "Number of components": +3.5,
    "HALPHA extinction correction": 5,
    "v_grad": 50,
}

label_dict = {
     "count": r"$N$", 
     "log N2": "N2",
     "log O3": "O3",
     "log O1": "O1",
     "log S2": "S2",
     "O3O2": "O3O2",
     "log HALPHA EW": r"$\log_{10} \left(W_{\rm H\alpha}\,[{\rm \AA}]\right)$",
     "log HALPHA EW (total)": r"$\log_{10} \left(W_{\rm H\alpha}\,[{\rm \AA}]\right)$ (total)",
     "HALPHA EW": r"$W_{\rm H\alpha}\,\rm (\AA)$",
     "HALPHA EW (total)": r"$W_{\rm H\alpha}\,\rm (\AA)$ (total)",
     "log sigma_gas": r"$\log_{10} \left(\sigma_{\rm gas}\,[\rm km\,s^{-1}]\right)$", 
     "sigma_gas": r"$\sigma_{\rm gas}\,\rm(km\,s^{-1})$", 
     "sigma_*": r"$\sigma_*\,\rm(km\,s^{-1})$", 
     "sigma_gas - sigma_*": r"$\sigma_{\rm gas} - \sigma_*\,\rm\left(km\,s^{-1}\right)$", 
     "v_gas - v_*": r"$v_{\rm gas} - v_*\,\rm\left(km\,s^{-1}\right)$", 
     "HALPHA S/N": r"$\rm H\alpha$ S/N",
     "BPT (numeric)": "Spectral classification",
     "Law+2021 (numeric)": "Law+2021 kinematic classification",
     "radius": "Radius (arcsec)",
     "D4000": r"$\rm D_n 4000 \, \AA$ break strength",
     "HALPHA": r"$\rm H\alpha$ flux",
     "v_gas": r"$v_{\rm gas} \,\rm (km\,s^{-1})$",
     "v_*": r"$v_* \,\rm (km\,s^{-1})$",
     "A_V": r"$A_V\,\rm (mag)$",
     "S2 ratio": r"[S II]$6716/6731$ ratio",
     "O1O3": "O1O3",
     "mstar": r"$\log_{10}(M_*\,[\rm M_\odot])$",
     "g_i": r"$g - i$ colour",
     "Morphology (numeric)": "Morphology",
     "m_r": r"$M_r$ (mag)",
     "z_spec": r"$z$",
     "log O3 ratio (1/0)": r"Component 1/component 0 $\log_{10}$ O3 ratio (dex)",
     "log O3 ratio (1/0)": r"Component 2/component 1 $\log_{10}$ O3 ratio (dex)",
     "log O1 ratio (1/0)": r"Component 1/component 0 $\log_{10}$ O1 ratio (dex)",
     "log O1 ratio (2/1)": r"Component 2/component 1 $\log_{10}$ O1 ratio (dex)",
     "log N2 ratio (1/0)": r"Component 1/component 0 $\log_{10}$ N2 ratio (dex)",
     "log N2 ratio (2/1)": r"Component 2/component 1 $\log_{10}$ N2 ratio (dex)",
     "log S2 ratio (1/0)": r"Component 1/component 0 $\log_{10}$ S2 ratio (dex)",
     "log S2 ratio (2/1)": r"Component 2/component 1 $\log_{10}$ S2 ratio (dex)",
     "sigma_gas/sigma_*": r"$\sigma_{\rm gas}/\sigma_*$",
     "N2O2": "N2O2",
     "HALPHA EW/HALPHA EW (total)": r"$\rm EW(H\alpha)/EW_{\rm tot}(H\alpha)$",
     "HALPHA EW ratio (1/0)": r"Component 1/component 0 $\rm EW(H\alpha)$ ratio",
     "HALPHA EW ratio (2/1)": r"Component 2/component 1 $\rm EW(H\alpha)$ ratio",
     "delta sigma_gas (1/0)": r"$\sigma_{\rm gas,\,1} - \sigma_{\rm gas\,0}$",
     "delta sigma_gas (2/1)": r"$\sigma_{\rm gas,\,2} - \sigma_{\rm gas\,1}$",
     "delta v_gas (1/0)": r"$v_{\rm gas,\,1} - v_{\rm gas\,0}$",
     "delta v_gas (2/1)": r"$v_{\rm gas,\,2} - v_{\rm gas\,1}$",
     "r/R_e": r"$r/R_e$",
     "R_e (kpc)": r"$R_e$ (kpc)",
     "log(M/R_e)": r"$\log_{10}(M_* / R_e \,\rm [M_\odot \, kpc^{-1}])$",
     "Inclination i (degrees)": r"Inclination $i$ (degrees)",  
     "Bin size (square kpc)": r"Bin size (kpc$^2$)",
     "SFR": r"$\rm SFR \, (M_\odot \, yr^{-1})$",
     "SFR surface density": r"$\rm \Sigma_{SFR} \, (M_\odot \, yr^{-1} \, kpc^{-2})$",
     "log SFR": r"$\log_{\rm 10} \rm (SFR \, [M_\odot \, yr^{-1}])$",
     "log SFR surface density": r"$\log_{\rm 10} \rm (\Sigma_{SFR} \, [M_\odot \, yr^{-1} \, kpc^{-2}])$",
     "Delta HALPHA EW (0/1)": r"$\log_{10} \rm EW(H\alpha)_{0} - \log_{10} \rm EW(H\alpha)_{1}$",
     "Number of components": "Number of components",
     "HALPHA extinction correction": r"H$\alpha$ extinction correction factor",
     "v_grad" : r"$v_{\rm grad}$",
}

###############################################################################
# Helper functions to return colourmaps, min/max values and labels
###############################################################################
def cmap_fn(col):
    if " (component" in col:
        col = col.split(" (component")[0]
    elif "(total)" in col:
        col = col.split(" (total)")[0]
    if col in cmap_dict.keys():
        return cmap_dict[col]
    else:
        print("WARNING: in cmap_fn(): undefined column")
        return copy.copy(plt.cm.get_cmap("jet"))

###############################################################################
def vmin_fn(col):
    if " (component" in col:
        col = col.split(" (component")[0]
    elif "(total)" in col:
        col = col.split(" (total)")[0]
    if col in cmap_dict.keys():
        return vmin_dict[col]
    else:
        print("WARNING: in vmin_fn(): undefined column")
        return None

###############################################################################
def vmax_fn(col):
    if " (component" in col:
        col = col.split(" (component")[0]
    elif "(total)" in col:
        col = col.split(" (total)")[0]
    if col in cmap_dict.keys():
        return vmax_dict[col]
    else:
        print("WARNING: in vmax_fn(): undefined column")
        return None

###############################################################################
def label_fn(col):
    if " (component" in col:
        col = col.split(" (component")[0]
    elif "(total)" in col:
        col = col.split(" (total)")[0]
    if col in cmap_dict.keys():
        return label_dict[col]
    else:
        print("WARNING: in label_fn(): undefined column")
        return col

###############################################################################
# Helper function for plotting 2D histograms
###############################################################################
def histhelper(df, col_x, col_y, col_z, nbins, ax, cmap,
                xmin, xmax, ymin, ymax, vmin, vmax,
                log_z=False, alpha=1.0):

    # Determine bin edges for the x & y-axis line ratio 
    # Messy hack to include that final bin...
    ybins = np.linspace(ymin, ymax, nbins)
    dy = np.diff(ybins)[0]
    ybins = list(ybins)
    ybins.append(ybins[-1] + dy)
    ybins = np.array(ybins)
    ycut = pd.cut(df[col_y], ybins)

    xbins = np.linspace(xmin, xmax, nbins)
    dx = np.diff(xbins)[0]
    xbins = list(xbins)
    xbins.append(xbins[-1] + dx)
    xbins = np.array(xbins)
    xcut = pd.cut(df[col_x], xbins)

    # Combine the x- and y-cuts
    cuts = pd.DataFrame({"xbin": xcut, "ybin": ycut})

    # Calculate the desired quantities for the data binned by x and y    
    gb_binned = df.join(cuts).groupby(list(cuts))
    if col_z == "count":
        df_binned = gb_binned.agg({df.columns[0]: lambda g: g.count()})
        df_binned = df_binned.rename(columns={df.columns[0]: "count"})
    else:
        df_binned = gb_binned.agg({col_z: np.nanmedian})

    # Pull out arrays to plot
    count_map = df_binned[col_z].values.reshape((nbins, nbins))

    # Plot.
    if log_z:
        m = ax.pcolormesh(xbins[:-1], ybins[:-1], count_map.T, cmap=cmap,
            edgecolors="none", vmin=vmin, vmax=vmax,
            norm=LogNorm(vmin=vmin, vmax=vmax))
    else:
        m = ax.pcolormesh(xbins[:-1], ybins[:-1], count_map.T, cmap=cmap,
            edgecolors="none", vmin=vmin, vmax=vmax)
    m.set_rasterized(True)

    # Dodgy...
    if alpha < 1:
        overlay = np.full_like(count_map.T, 1.0)
        mo = ax.pcolormesh(xbins[:-1], ybins[:-1], overlay, alpha=1 - alpha, cmap="gray", vmin=0, vmax=1)
        mo.set_rasterized(True)

    return m

###############################################################################
# Plot empty BPT diagrams
###############################################################################
def plot_empty_BPT_diagram(colorbar=False, nrows=1, include_Law2021=False):
    """
    Plot Baldwin, Philips & Terlevich (1986) optical line ratio diagrams.
    To add a colorbar:
        cbar_ax = fig.get_axes[-1]
        cb = fig.colorbar(s, cax=cbar_ax)
        cb.ax.set_ylabel(zorder_label)
    ----------------------------------------------------------------------------
    Returns:
        FIG: 
            The figure instance.
    """
    # Make axes
    fig = plt.figure(figsize=(15, 5 * nrows))
    left = 0.1
    bottom = 0.1
    if colorbar:
        cbar_width = 0.025
    else:
        cbar_width = 0
    width = (1 - 2*left - cbar_width) / 3.
    height = 0.8 / nrows

    axs = []
    caxs = []
    for ii in range(nrows):
        bottom = 0.1 + (nrows - ii - 1) * height
        ax_N2 = fig.add_axes([left,bottom,width,height])
        ax_S2 = fig.add_axes([left+width,bottom,width,height])
        ax_O1 = fig.add_axes([left+2*width,bottom,width,height])
        if colorbar:
            cax = fig.add_axes([left+3*width,bottom,cbar_width,height])

        # Plot the reference lines from literature
        x_vals = np.linspace(-2.5, 2.5, 100)
        ax_N2.plot(x_vals, Kewley2001("log N2", x_vals), "gray", linestyle="--")
        ax_S2.plot(x_vals, Kewley2001("log S2", x_vals), "gray", linestyle="--")
        ax_O1.plot(x_vals, Kewley2001("log O1", x_vals), "gray", linestyle="--")
        ax_S2.plot(x_vals, Kewley2006("log S2", x_vals), "gray", linestyle="-.")
        ax_O1.plot(x_vals, Kewley2006("log O1", x_vals), "gray", linestyle="-.")
        ax_N2.plot(x_vals, Kauffman2003("log N2", x_vals), "gray", linestyle=":")

        if include_Law2021:
            y_vals = np.copy(x_vals)
            ax_N2.plot(x_vals, Law2021_1sigma("log N2", x_vals), "gray", linestyle="-")
            ax_S2.plot(x_vals, Law2021_1sigma("log S2", x_vals), "gray", linestyle="-")
            ax_O1.plot(x_vals, Law2021_1sigma("log O1", x_vals), "gray", linestyle="-")
            ax_N2.plot(Law2021_3sigma("log N2", y_vals), y_vals, "gray", linestyle="-")
            ax_S2.plot(Law2021_3sigma("log S2", y_vals), y_vals, "gray", linestyle="-")
            ax_O1.plot(Law2021_3sigma("log O1", y_vals), y_vals, "gray", linestyle="-")

        # Axis limits
        ymin = -1.5
        ymax = 1.2
        ax_N2.set_ylim([ymin, ymax])
        ax_S2.set_ylim([ymin, ymax])
        ax_O1.set_ylim([ymin, ymax])
        ax_N2.set_xlim([-1.3,0.5])
        ax_S2.set_xlim([-1.3,0.5])
        ax_O1.set_xlim([-2.2,0.2])

        # Add axis labels
        ax_N2.set_ylabel(r"$\log_{10}$[O III]$\lambda5007$/H$\beta$")
        if ii == nrows - 1:
            ax_N2.set_xlabel(r"$\log_{10}$[N II]$\lambda6583$/H$\alpha$")
            ax_S2.set_xlabel(r"$\log_{10}$[S II]$\lambda\lambda6716,6731$/H$\alpha$")
            ax_O1.set_xlabel(r"$\log_{10}$[O I]$\lambda6300$/H$\alpha$")

        ax_S2.set_yticklabels([])
        ax_O1.set_yticklabels([])

        # Add axes to lists
        axs.append(ax_N2)
        axs.append(ax_S2)
        axs.append(ax_O1)
        caxs.append(cax)

    fig.show()

    if colorbar:
        if nrows == 1:
            return fig, axs, cax
        else:
            return fig, axs, caxs
    else:
        return fig, axs

###############################################################################
# Compass & scale bar functions for 2D map plots
###############################################################################
def plot_compass(PA_deg=0, flipped=True,
                 color="white",
                 bordercolor=None,
                 fontsize=10,
                 ax=None,
                 zorder=999999):
    # Display North and East-pointing arrows on a plot.
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
        ax.arrow(origin_x, origin_y,
                 - l * np.sin(PA_rad),
                 l * np.cos(PA_rad),
                 head_width=head_width, head_length=head_length, overhang=overhang,
                 fc=color, ec=color, zorder=zorder)
        # E arrow
        ax.arrow(origin_x, origin_y,
                 l * np.cos(PA_rad),
                 l * np.sin(PA_rad),
                 head_width=head_width, head_length=head_length, overhang=overhang,
                 fc=color, ec=color, zorder=zorder)
        ax.text(x=origin_x - 1.1 * (l + text_offset) * np.sin(PA_rad),
                y=origin_y + 1.1 * (l + text_offset) * np.cos(PA_rad),
                s="N", color=color, zorder=zorder, verticalalignment="center", horizontalalignment="center")
        ax.text(x=origin_x + 1.1 * (l + text_offset) * np.cos(PA_rad),
                y=origin_y + 1.1 * (l + text_offset) * np.sin(PA_rad),
                s="E", color=color, zorder=zorder, verticalalignment="center", horizontalalignment="center")

    else:
        # N arrow
        ax.arrow(origin_x, origin_y,
                 l * np.sin(PA_rad),
                 l * np.cos(PA_rad),
                 head_width=head_width, head_length=head_length, overhang=overhang,
                 fc=color, ec=color, zorder=zorder)
        # E arrow
        ax.arrow(origin_x, origin_y,
                 - l * np.cos(PA_rad),
                 l * np.sin(PA_rad),
                 head_width=head_width, head_length=head_length, overhang=overhang,
                 fc=color, ec=color, zorder=zorder)
        t = ax.text(x=origin_x + 1.1 * (l + text_offset) * np.sin(PA_rad),
                    y=origin_y + 1.1 * (l + text_offset) * np.cos(PA_rad),
                    s="N", color=color, zorder=zorder, verticalalignment="center", horizontalalignment="center",
                    fontsize=fontsize)
        if bordercolor is not None:
            t.set_path_effects([PathEffects.withStroke(
                linewidth=1.5, foreground=bordercolor)])
        t = ax.text(x=origin_x - 1.1 * (l + text_offset) * np.cos(PA_rad),
                    y=origin_y + 1.1 * (l + text_offset) * np.sin(PA_rad),
                    s="E", color=color, zorder=zorder, verticalalignment="center", horizontalalignment="center",
                    fontsize=fontsize)
        if bordercolor is not None:
            t.set_path_effects([PathEffects.withStroke(
                linewidth=1.5, foreground=bordercolor)])

        return

###############################################################################
def plot_scale_bar(as_per_px, kpc_per_as,
                   l=0.5,
                   ax=None,
                   loffset=0.20,
                   boffset=0.075,
                   units="arcsec",
                   color="white",
                   fontsize=12,
                   bordercolor=None,
                   zorder=999999):
    """
    Plots a nice little bar in the lower-right-hand corner of a plot indicating
    the scale of the image in kiloparsecs.
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
        dist_str = f"{l_arcsec:.2f} = {l_kpc:.2f} kpc"
    elif units == "arcmin":
        dist_str = f"{l_arcmin:.2f} = {l_kpc:.2f} kpc"
        
    t = ax.text(
        x=line_centre_x,
        y=line_centre_y + text_offset,
        s=dist_str,
        size=fontsize,
        horizontalalignment="center",
        color=color,
        zorder=zorder)
    if bordercolor is not None:
        t.set_path_effects([PathEffects.withStroke(
            linewidth=1.5, foreground=bordercolor)])

    return


