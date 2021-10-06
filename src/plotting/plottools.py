import numpy as np
import copy
from matplotlib.colors import ListedColormap, to_rgba

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
    cmap_dict[key].set_bad("#b3b3b3")

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

