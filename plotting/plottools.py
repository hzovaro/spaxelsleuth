import numpy as np
import copy
import pandas as pd

from spaxelsleuth.utils.linefns import Kewley2001, Kewley2006, Kauffman2003, Law2021_1sigma, Law2021_3sigma

from matplotlib.colors import ListedColormap, to_rgba, LogNorm
import matplotlib.patheffects as PathEffects
import matplotlib.pyplot as plt
from matplotlib import cm

from IPython.core.debugger import Tracer

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

# Custom colour map for BPT categories
whav_labels = [
    "Unknown",
    "HOLMES",    
    "Mixing + HOLMES + no wind",
    "Mixing + HOLMES + wind",
    "Mixing + no wind",
    "Mixing + wind",
    "AGN + HOLMES + no wind",
    "AGN + HOLMES + wind",
    "AGN + no wind",
    "AGN + wind",
    "SF + HOLMES + no wind",
    "SF + HOLMES + wind",
    "SF + no wind",
    "SF + wind",
]
whav_ticks = np.arange(len(whav_labels)) - 1
Spectral = plt.cm.get_cmap("jet_r")
whav_colors = Spectral(np.linspace(0, 1, len(whav_labels)))
whav_colors[0] = [0.5, 0.5, 0.5, 1]  # Unknown
whav_colors[12] = np.array(to_rgba("#004cff"))  # SF + no wind 
whav_colors[11] = np.array(to_rgba("#00ffff"))  # SF + HOLMES + wind
whav_colors[10] = np.array(to_rgba("#00bfff"))  # SF + HOLMES + no wind
whav_colors[9] = np.array(to_rgba("#ff00ff"))  # AGN + wind
whav_colors[8] = np.array(to_rgba("#ffa7ff"))  # AGN + no wind
whav_colors[7] = np.array(to_rgba("#13cc00"))  # AGN + HOLMES + wind
whav_colors[6] = np.array(to_rgba("#49ff36"))  # AGN + HOLMES + no wind
whav_colors[5] = np.array(to_rgba("#ffff00"))  # Mixing + wind

whav_cmap = ListedColormap(whav_colors)
whav_cmap.set_bad(color="white", alpha=0)

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
c1 = to_rgba("#4A4A4A")
c2 = to_rgba("#6EC0FF")
c3 = to_rgba("#8AE400")
c4 = to_rgba("#A722FF")
ncomponents_colours = np.vstack((c1, c2, c3, c4))
ncomponents_cmap = ListedColormap(ncomponents_colours)
ncomponents_cmap.set_bad(color="white", alpha=0.0)

# Custom colour list for different components
component_colours = ["#781EE5", "#FF53B4", "#FFC107"]
component_labels = ["Component 1", "Component 2", "Component 3"]

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
    "log HALPHA EW": copy.copy(plt.cm.get_cmap("Spectral")),
    "log HALPHA EW (total)": copy.copy(plt.cm.get_cmap("Spectral")),
    "HALPHA EW": copy.copy(plt.cm.get_cmap("Spectral")),
    "HALPHA EW (total)": copy.copy(plt.cm.get_cmap("Spectral")),
    "log sigma_gas": copy.copy(plt.cm.get_cmap("plasma")),
    "sigma_gas": copy.copy(plt.cm.get_cmap("plasma")),
    "sigma_*": copy.copy(plt.cm.get_cmap("plasma")),
    "sigma_gas - sigma_*": copy.copy(plt.cm.get_cmap("RdYlBu_r")),
    "sigma_gas^2 - sigma_*^2": copy.copy(plt.cm.get_cmap("RdYlBu_r")),
    "v_gas - v_*": copy.copy(plt.cm.get_cmap("RdYlBu_r")),
    "HALPHA S/N": copy.copy(plt.cm.get_cmap("copper")),
    "BPT (numeric)": bpt_cmap,
    "WHAV* (numeric)": whav_cmap,
    "Law+2021 (numeric)": law2021_cmap,
    "radius": copy.copy(plt.cm.get_cmap("gnuplot2_r")),
    "D4000": copy.copy(plt.cm.get_cmap("pink_r")),
    "HALPHA": copy.copy(plt.cm.get_cmap("viridis")),
    "HALPHA luminosity": copy.copy(plt.cm.get_cmap("viridis")),
    "HALPHA continuum luminosity": copy.copy(plt.cm.get_cmap("viridis")),
    "log HALPHA luminosity": copy.copy(plt.cm.get_cmap("viridis")),
    "log HALPHA continuum luminosity": copy.copy(plt.cm.get_cmap("viridis")),
    "v_gas": copy.copy(plt.cm.get_cmap("coolwarm")),
    "v_*": copy.copy(plt.cm.get_cmap("coolwarm")),
    "A_V": copy.copy(plt.cm.get_cmap("afmhot_r")),
    "S2 ratio": copy.copy(plt.cm.get_cmap("cividis")),
    "log S2 ratio": copy.copy(plt.cm.get_cmap("cividis")),
    "O1O3": copy.copy(plt.cm.get_cmap("cividis")),
    "mstar": copy.copy(plt.cm.get_cmap("jet")),
    "g_i": copy.copy(plt.cm.get_cmap("RdYlBu_r")),
    "Morphology (numeric)": morph_cmap,
    "m_r": copy.copy(plt.cm.get_cmap("Reds")),
    "z_spec": copy.copy(plt.cm.get_cmap("plasma")),
    "delta log O3 (2/1)": copy.copy(plt.cm.get_cmap("PiYG")),
    "delta log O3 (3/2)": copy.copy(plt.cm.get_cmap("PiYG")),
    "delta log O1 (2/1)": copy.copy(plt.cm.get_cmap("PiYG")),
    "delta log O1 (3/2)": copy.copy(plt.cm.get_cmap("PiYG")),
    "delta log N2 (2/1)": copy.copy(plt.cm.get_cmap("PiYG")),
    "delta log N2 (3/2)": copy.copy(plt.cm.get_cmap("PiYG")),
    "delta log S2 (2/1)": copy.copy(plt.cm.get_cmap("PiYG")),
    "delta log S2 (3/2)": copy.copy(plt.cm.get_cmap("PiYG")),
    "sigma_gas/sigma_*": copy.copy(plt.cm.get_cmap("RdYlBu_r")),
    "log(U)": copy.copy(plt.cm.get_cmap("cubehelix")), 
    "log(O/H) + 12": copy.copy(plt.cm.get_cmap("cividis")),
    "N2O2": copy.copy(plt.cm.get_cmap("cividis")),
    "R23": copy.copy(plt.cm.get_cmap("cividis")),
    "N2S2": copy.copy(plt.cm.get_cmap("cividis")),
    "O3O2": copy.copy(plt.cm.get_cmap("cividis")),
    "HALPHA EW/HALPHA EW (total)": copy.copy(plt.cm.get_cmap("jet")),
    "HALPHA EW ratio (2/1)": copy.copy(plt.cm.get_cmap("jet")),
    "HALPHA EW ratio (3/2)": copy.copy(plt.cm.get_cmap("jet")),
    "delta sigma_gas (2/1)": copy.copy(plt.cm.get_cmap("autumn")),
    "delta sigma_gas (3/2)": copy.copy(plt.cm.get_cmap("autumn")),
    "delta v_gas (2/1)": copy.copy(plt.cm.get_cmap("autumn")),
    "delta v_gas (3/2)": copy.copy(plt.cm.get_cmap("autumn")),
    "r/R_e": copy.copy(plt.cm.get_cmap("gnuplot2_r")),
    "R_e (kpc)": copy.copy(plt.cm.get_cmap("gnuplot2_r")),
    "log(M/R_e)": copy.copy(plt.cm.get_cmap("gnuplot2_r")),
    "Inclination i (degrees)": copy.copy(plt.cm.get_cmap("Spectral_r")), 
    "Bin size (square kpc)": copy.copy(plt.cm.get_cmap("gnuplot2_r")),
    "kpc per arcsec": copy.copy(plt.cm.get_cmap("gnuplot2_r")),
    "SFR": sfr_cmap,
    "SFR surface density": sfr_cmap,
    "log SFR": sfr_cmap,
    "log SFR surface density": sfr_cmap,
    "Delta HALPHA EW (1/2)": copy.copy(plt.cm.get_cmap("Spectral_r")),
    "Number of components": ncomponents_cmap,
    "HALPHA extinction correction": copy.copy(plt.cm.get_cmap("pink")),
    "v_grad": copy.copy(plt.cm.get_cmap("plasma")),
    "n_e (cm^-3)": copy.copy(plt.cm.get_cmap("cividis")),
    "log n_e (cm^-3)": copy.copy(plt.cm.get_cmap("cividis")),
}

for key in cmap_dict.keys():
    # cmap_dict[key].set_bad("#b3b3b3")
    cmap_dict[key].set_bad("white")


vmin_dict = {
    "count": None,
    "log N2": -1.5,
    "log O3": -1.5,
    "log O1": -2.2,
    "log S2": -1.3,
    "log HALPHA EW": -1,
    "log HALPHA EW (total)": -1,
    "HALPHA EW": 3,
    "HALPHA EW (total)": 3,
    "log sigma_gas": 1,
    "sigma_gas": 10,
    "sigma_*": 10,
    "sigma_gas - sigma_*": -300,
    "sigma_gas^2 - sigma_*^2": -2e5,
    "v_gas - v_*": -200,  # CHANGE BACK TO 600
    "HALPHA S/N": 3,
    "BPT (numeric)": -1.5,
    "WHAV* (numeric)": -1.5,
    "Law+2021 (numeric)": -1.5,
    "radius": 0,
    "D4000": 0.5,
    "HALPHA": 0,
    "HALPHA luminosity": 1e37,
    "HALPHA continuum luminosity": 1e35,
    "log HALPHA luminosity": 37,
    "log HALPHA continuum luminosity": 35,
    "v_gas": -200,
    "v_*": -250,
    "A_V": 0,
    "S2 ratio": 0.38,
    "log S2 ratio": -0.45,
    "O1O3": -2,
    "mstar": 7.5,
    "g_i": -0.5,
    "Morphology (numeric)": -0.75,
    "m_r": -25,
    "z_spec": 0,
    "delta log O3 (2/1)": -1.0,
    "delta log O3 (3/2)": -1.0,
    "delta log O1 (2/1)": -0.5,
    "delta log O1 (3/2)": -0.5,
    "delta log N2 (2/1)": -0.5,
    "delta log N2 (3/2)": -0.5,
    "delta log S2 (2/1)": -0.5,
    "delta log S2 (3/2)": -0.5,
    "sigma_gas/sigma_*": 0,
    "log(U)": -3.5, 
    "log(O/H) + 12": 7.5,
    "N2O2": -1.5,
    "R23": -0.3,
    "N2S2": -1.3,
    "O3O2": -1.5,
    "HALPHA EW/HALPHA EW (total)": 0,
    "HALPHA EW ratio (2/1)": 0,
    "HALPHA EW ratio (3/2)": 0,
    "delta sigma_gas (2/1)": 0,
    "delta sigma_gas (3/2)": 0,
    "delta v_gas (2/1)": -150,
    "delta v_gas (3/2)": -150,
    "r/R_e": 0,
    "R_e (kpc)": 0,
    "log(M/R_e)": 6,
    "Inclination i (degrees)": 0, 
    "Bin size (square kpc)": 0,
    "kpc per arcsec": 0,
    "SFR": 0,
    "SFR surface density": 0,
    "log SFR": -5.0,
    "log SFR surface density": -4.0,
    "Delta HALPHA EW (1/2)": -1.0,
    "Number of components": -0.5,
    "HALPHA extinction correction": 1,
    "v_grad": 0,
    "n_e (cm^-3)": 40,
    "log n_e (cm^-3)": np.log10(40),
}

vmax_dict = {
    "count": None,
    "log N2": 0.5,
    "log O3": 1.2,
    "log O1": 0.2,
    "log S2": 0.5,
    "log HALPHA EW": 3.5,
    "log HALPHA EW (total)": 3.5,
    "HALPHA EW": 14,
    "HALPHA EW (total)": 14,
    "log sigma_gas": 3,
    "sigma_gas": 300,  # CHANGE BACK TO 300
    "sigma_*": 300,
    "sigma_gas - sigma_*": 600,  # CHANGE BACK TO 600
    "sigma_gas^2 - sigma_*^2": +2e5,
    "v_gas - v_*": +600,  # CHANGE BACK TO 600
    "HALPHA S/N": 50,
    "BPT (numeric)": 4.5,
    "WHAV* (numeric)": 12.5,
    "Law+2021 (numeric)": 3.5,
    "radius": 10,
    "D4000": 2.2,
    "HALPHA": 1e3,  # 1.5 is good for SAMI
    "HALPHA luminosity": 1e42,
    "HALPHA continuum luminosity": 1e41,
    "log HALPHA luminosity": 42,
    "log HALPHA continuum luminosity": 41,
    "v_gas": +200,
    "v_*": +250,
    "A_V": 5,
    "S2 ratio": 1.44,
    "log S2 ratio": +0.45,
    "O1O3": 1.5,
    "mstar": 11.5,
    "g_i": 1.7,
    "Morphology (numeric)": 3.25,
    "m_r": -12.5,
    "z_spec": 0.1,
    "delta log O3 (2/1)": +1.0,
    "delta log O3 (3/2)": +1.0,
    "delta log O1 (2/1)": +0.5,
    "delta log O1 (3/2)": +0.5,
    "delta log N2 (2/1)": +0.5,
    "delta log N2 (3/2)": +0.5,
    "delta log S2 (2/1)": +0.5,
    "delta log S2 (3/2)": +0.5,
    "sigma_gas/sigma_*": 4,
    "log(U)": -2.5, 
    "log(O/H) + 12": 9.5,
    "N2O2": 0.5,
    "R23": 1.6,
    "N2S2": 0.9,
    "O3O2": 1.6,
    "HALPHA EW/HALPHA EW (total)": 1,
    "HALPHA EW ratio (2/1)": 2,
    "HALPHA EW ratio (3/2)": 2,
    "delta sigma_gas (2/1)": +150,
    "delta sigma_gas (3/2)": +150,
    "delta v_gas (2/1)": +150,
    "delta v_gas (3/2)": +150,
    "r/R_e": 2,
    "R_e (kpc)": 10,
    "log(M/R_e)": 12,
    "Inclination i (degrees)": 90, 
    "Bin size (square kpc)": 0.5,
    "kpc per arcsec": 2,
    "SFR": 0.02,
    "SFR surface density": 0.05,
    "log SFR": -1.0,
    "log SFR surface density": 0.5,
    "Delta HALPHA EW (1/2)": +2.0,
    "Number of components": +3.5,
    "HALPHA extinction correction": 5,
    "v_grad": 50,
    "n_e (cm^-3)": 1e4,
    "log n_e (cm^-3)": 4,
}

label_dict = {
     "count": r"$N$", 
     "log N2": r"$\log_{10}$([N II]6583/H$\alpha$)",
     "log O3": r"$\log_{10}$([O III]5007/H$\beta$)",
     "log O1": r"$\log_{10}$([O I]6300/H$\alpha$)",
     "log S2": r"$\log_{10}$([S II]6716,31/H$\alpha$)",
     "log HALPHA EW": r"$\log_{10} \left(W_{\rm H\alpha}\,[{\rm \AA}]\right)$",
     "log HALPHA EW (total)": r"$\log_{10} \left(W_{\rm H\alpha}\,[{\rm \AA}]\right)$ (total)",
     "HALPHA EW": r"$W_{\rm H\alpha}\,\rm (\AA)$",
     "HALPHA EW (total)": r"$W_{\rm H\alpha}\,\rm (\AA)$ (total)",
     "log sigma_gas": r"$\log_{10} \left(\sigma_{\rm gas}\,[\rm km\,s^{-1}]\right)$", 
     "sigma_gas": r"$\sigma_{\rm gas}\,\rm(km\,s^{-1})$", 
     "sigma_*": r"$\sigma_*\,\rm(km\,s^{-1})$", 
     "sigma_gas - sigma_*": r"$\sigma_{\rm gas} - \sigma_*\,\rm\left(km\,s^{-1}\right)$", 
     "sigma_gas^2 - sigma_*^2": r"$\sigma_{\rm gas}^2 - \sigma_*^2\,\rm\left(km^2\,s^{-2}\right)$", 
     "v_gas - v_*": r"$v_{\rm gas} - v_*\,\rm\left(km\,s^{-1}\right)$", 
     "HALPHA S/N": r"$\rm H\alpha$ S/N",
     "BPT (numeric)": "Spectral classification",
     "WHAV* (numeric)": "WHAV* classification",
     "Law+2021 (numeric)": "Law+2021 kinematic classification",
     "radius": "Radius (arcsec)",
     "D4000": r"$\rm D_n 4000 \, \AA$ break strength",
     "HALPHA": r"$\rm H\alpha$ flux",
     "HALPHA luminosity": r"$L(\rm H\alpha) \, \rm (erg\,s^{-1}\,kpc^{-2})$",
     "HALPHA continuum luminosity": r"$F(C_{\rm H\alpha}) \, \rm (erg\,s^{-1}\,Å^{-1}\,kpc^{-2})$",
     "log HALPHA luminosity": r"$\log_{10} \left(L(\rm H\alpha) \, \rm [erg\,s^{-1}\,kpc^{-2}]\right)$",
     "log HALPHA continuum luminosity": r"$\log_{10} \left(F(C_{\rm H\alpha}) \, \rm [erg\,s^{-1}\,Å^{-1}\,kpc^{-2}]\right)$",
     "v_gas": r"$v_{\rm gas} \,\rm (km\,s^{-1})$",
     "v_*": r"$v_* \,\rm (km\,s^{-1})$",
     "A_V": r"$A_V\,\rm (mag)$",
     "S2 ratio": r"[S II]$6716/6731$ ratio",
     "log S2 ratio": r"$\log_{10}$ [S II]$6716/6731$ ratio",
     "O1O3": "O1O3",
     "mstar": r"$\log_{10}(M_*\,[\rm M_\odot])$",
     "g_i": r"$g - i$ colour",
     "Morphology (numeric)": "Morphology",
     "m_r": r"$M_r$ (mag)",
     "z_spec": r"$z$",
     "delta log O3 (2/1)": r"$\Delta$ log O3 (2/1)",
     "delta log O3 (3/2)": r"$\Delta$ log O3 (3/2)",
     "delta log O1 (2/1)": r"$\Delta$ log O1 (2/1)",
     "delta log O1 (3/2)": r"$\Delta$ log O1 (3/2)",
     "delta log N2 (2/1)": r"$\Delta$ log N2 (2/1)",
     "delta log N2 (3/2)": r"$\Delta$ log N2 (3/2)",
     "delta log S2 (2/1)": r"$\Delta$ log S2 (2/1)",
     "delta log S2 (3/2)": r"$\Delta$ log S2 (3/2)",
     "sigma_gas/sigma_*": r"$\sigma_{\rm gas}/\sigma_*$",
     "log(U)": r"$\log(U)$", 
     "log(O/H) + 12": r"$\log{\rm (O/H)} + 12$",
     "N2O2": "N2O2",
     "R23": "R23",
     "N2S2": "N2S2",
     "O3O2": "O3O2",
     "HALPHA EW/HALPHA EW (total)": r"$\rm EW(H\alpha)/EW_{\rm tot}(H\alpha)$",
     "HALPHA EW ratio (2/1)": r"Component 2/component 1 $\rm EW(H\alpha)$ ratio",
     "HALPHA EW ratio (3/2)": r"Component 3/component 2 $\rm EW(H\alpha)$ ratio",
     "delta sigma_gas (2/1)": r"$\sigma_{\rm gas,\,2} - \sigma_{\rm gas,\,1}$",
     "delta sigma_gas (3/2)": r"$\sigma_{\rm gas,\,3} - \sigma_{\rm gas,\,2}$",
     "delta sigma_gas (3/1)": r"$\sigma_{\rm gas,\,3} - \sigma_{\rm gas,\,1}$",
     "delta v_gas (2/1)": r"$v_{\rm gas,\,2} - v_{\rm gas,\,1}$",
     "delta v_gas (3/2)": r"$v_{\rm gas,\,3} - v_{\rm gas,\,2}$",
     "delta v_gas (3/1)": r"$v_{\rm gas,\,3} - v_{\rm gas,\,1}$",
     "r/R_e": r"$r/R_e$",
     "R_e (kpc)": r"$R_e$ (kpc)",
     "log(M/R_e)": r"$\log_{10}(M_* / R_e \,\rm [M_\odot \, kpc^{-1}])$",
     "Inclination i (degrees)": r"Inclination $i$ (degrees)",  
     "Bin size (square kpc)": r"Bin size (kpc$^2$)",
     "kpc per arcsec": "kpc per arcsec",
     "SFR": r"$\rm SFR \, (M_\odot \, yr^{-1})$",
     "SFR surface density": r"$\rm \Sigma_{SFR} \, (M_\odot \, yr^{-1} \, kpc^{-2})$",
     "log SFR": r"$\log_{\rm 10} \rm (SFR \, [M_\odot \, yr^{-1}])$",
     "log SFR surface density": r"$\log_{\rm 10} \rm (\Sigma_{SFR} \, [M_\odot \, yr^{-1} \, kpc^{-2}])$",
     "Delta HALPHA EW (1/2)": r"$\log_{10} \rm EW(H\alpha)_{0} - \log_{10} \rm EW(H\alpha)_{1}$",
     "Number of components": "Number of components",
     "HALPHA extinction correction": r"H$\alpha$ extinction correction factor",
     "v_grad" : r"$v_{\rm grad}$",
     "n_e (cm^-3)": r"$n_e \,\rm (cm^{-3})$",
     "log n_e (cm^-3)": r"$\log_{10} n_e \,\rm (cm^{-3})$",
}

fname_dict = {
     "count": "count",
     "log N2": "logN2",
     "log O3": "logO3",
     "log O1": "logO1",
     "log S2": "logS2",
     "log HALPHA EW": "logHaEW",
     "log HALPHA EW (total)": "logHaEWtot",
     "HALPHA EW": "HaEW",
     "HALPHA EW (total)": "HaEwtot",
     "log sigma_gas": "log_sigma_gas",
     "sigma_gas": "sigma_gas",
     "sigma_*": "sigma_star",
     "sigma_gas - sigma_*": "sigma_gas-sigma_star",
     "sigma_gas^2 - sigma_*^2": "sigma_gas2-sigma_star2",
     "v_gas - v_*": "v_gas-v_star",
     "HALPHA S/N": "HaSNR",
     "BPT (numeric)": "BPT",
     "WHAV* (numeric)": "WHAV",
     "Law+2021 (numeric)": "Law2021",
     "radius": "radius",
     "D4000": "D4000",
     "HALPHA": "HALPHA",
     "HALPHA luminosity": "HALPHA_lum_per_kpc2",
     "HALPHA continuum luminosity": "HALPHA_cont_lum_per_kpc2",
     "log HALPHA luminosity": "log_HALPHA_lum_per_kpc2",
     "log HALPHA continuum luminosity": "log_HALPHA_cont_lum_per_kpc2",
     "v_gas": "v_gas",
     "v_*": "v_star",
     "A_V": "A_V",
     "S2 ratio": "S2ratio",
     "log S2 ratio": "logS2ratio",
     "O1O3": "O1O3",
     "mstar": "mstar",
     "g_i": "g_i",
     "Morphology (numeric)": "morphology",
     "m_r": "m_r",
     "z_spec": "z_spec",
     "delta log O3 (2/1)": "deltalogO3_21",
     "delta log O3 (3/2)": "deltalogO3_32",
     "delta log O1 (2/1)": "deltalogO1_21",
     "delta log O1 (3/2)": "deltalogO1_32",
     "delta log N2 (2/1)": "deltalogN2_21",
     "delta log N2 (3/2)": "deltalogN2_32",
     "delta log S2 (2/1)": "deltalogS2_21",
     "delta log S2 (3/2)": "deltalogS2_32",
     "sigma_gas/sigma_*": "sigma_gas_over_sigma_star",
     "log(U)": "logU", 
     "log(O/H) + 12": "logOH12",
     "N2O2": "N2O2",
     "R23": "R23",
     "N2S2": "N2S2",
     "O3O2": "O3O2",
     "HALPHA EW/HALPHA EW (total)": "HaEW_over_HaEWtot",
     "HALPHA EW ratio (2/1)": "HaEwRatio_21",
     "HALPHA EW ratio (3/2)": "HaEwRatio_32",
     "delta sigma_gas (2/1)": "delta_sigma_gas_21",
     "delta sigma_gas (3/2)": "delta_sigma_gas_32",
     "delta v_gas (2/1)": "delta_v_gas_21",
     "delta v_gas (3/2)": "delta_v_gas_32",
     "r/R_e": "r_over_Re",
     "R_e (kpc)": "Re_kpc",
     "log(M/R_e)": "log_M_over_Re",
     "Inclination i (degrees)": "inclination",
     "Bin size (square kpc)": "bin_size",
     "kpc per arcsec": "kpc_per_arcsec",
     "SFR": "SFR",
     "SFR surface density": "SFR_surface_density",
     "log SFR": "logSFR",
     "log SFR surface density": "logSFR_surface_density",
     "Delta HALPHA EW (1/2)": "Delta_HaEW_21",
     "Number of components": "ncomponents",
     "HALPHA extinction correction": "Ha_ext_corr",
     "v_grad" : "v_grad" ,
     "n_e (cm^-3)": "ne",
     "log n_e (cm^-3)": "logne",
}

###############################################################################
# Helper functions to return colourmaps, min/max values and labels
###############################################################################
def cmap_fn(col):
    """
    Helper function to return the colourmap corresponding to column col. 
    """
    if " (component" in col:
        col = col.split(" (component")[0]
    elif "(total)" in col:
        col = col.split(" (total)")[0]
    if col in cmap_dict.keys():
        return cmap_dict[col]
    # Special case for log(U) and log(O/H) + 12, since the column names will 
    # include the diagnostic(s) used
    elif col.startswith("log(U)"):
        return cmap_dict["log(U)"]
    elif col.startswith("log(O/H) + 12"):
        return cmap_dict["log(O/H) + 12"]
    else:
        print("WARNING: in cmap_fn(): undefined column")
        return copy.copy(plt.cm.get_cmap("jet"))

###############################################################################
def vmin_fn(col):
    """
    Helper function to return the minimum range of parameter corresponding to
    column col. Useful in plotting. 
    """
    if " (component" in col:
        col = col.split(" (component")[0]
    elif "(total)" in col:
        col = col.split(" (total)")[0]
    if col in vmin_dict.keys():
        return vmin_dict[col]
    # Special case for log(U) and log(O/H) + 12, since the column names will 
    # include the diagnostic(s) used
    elif col.startswith("log(U)"):
        return vmin_dict["log(U)"]
    elif col.startswith("log(O/H) + 12"):
        return vmin_dict["log(O/H) + 12"]
    else:
        print("WARNING: in vmin_fn(): undefined column")
        return None

###############################################################################
def vmax_fn(col):
    """
    Helper function to return the maximum range of parameter corresponding to
    column col. Useful in plotting. 
    """
    if " (component" in col:
        col = col.split(" (component")[0]
    elif "(total)" in col:
        col = col.split(" (total)")[0]
    if col in vmax_dict.keys():
        return vmax_dict[col]
    # Special case for log(U) and log(O/H) + 12, since the column names will 
    # include the diagnostic(s) used
    elif col.startswith("log(U)"):
        return vmax_dict["log(U)"]
    elif col.startswith("log(O/H) + 12"):
        return vmax_dict["log(O/H) + 12"]
    else:
        print("WARNING: in vmax_fn(): undefined column")
        return None

###############################################################################
def label_fn(col):
    """
    Helper function to return a pretty LaTeX label corresponding to column
    col to use for plotting axis labels, titles, etc. 
    """
    if " (component" in col:
        col = col.split(" (component")[0]
    elif "(total)" in col:
        col = col.split(" (total)")[0]
    if col in label_dict.keys():
        return label_dict[col]
    # Special case for log(U) and log(O/H) + 12, since the column names will 
    # include the diagnostic(s) used
    elif col.startswith("log(U)"):
        diags = col.split("log(U) ")[1].split(" ")[0]
        return label_dict["log(U)"] + " " + diags
    elif col.startswith("log(O/H) + 12"):
        diags = col.split("log(O/H) + 12 ")[1].split(" ")[0]
        return label_dict["log(O/H) + 12"] + " " + diags
    else:
        print("WARNING: in label_fn(): undefined column")
        return col

###############################################################################
def fname_fn(col):
    """
    Helper function to return a system-safe filename corresponding to column
    col to use for saving figures and other data to file. 
    """
    if " (component" in col:
        col = col.split(" (component")[0]
    elif "(total)" in col:
        col = col.split(" (total)")[0]
    if col in fname_dict.keys():
        return fname_dict[col]
    # Special case for log(U) and log(O/H) + 12, since the column names will 
    # include the diagnostic(s) used
    elif col.startswith("log(U)"):
        diags = col.split("log(U) ")[1].split(" ")[0]
        return fname_dict["log(U)"] + "_" + diags.replace("(", "").replace(")", "").replace("/", "_")
    elif col.startswith("log(O/H) + 12"):
        diags = col.split("log(O/H) + 12 ")[1].split(" ")[0]
        return fname_dict["log(O/H) + 12"] + "_" + diags.replace("(", "").replace(")", "").replace("/", "_")
    else:
        print("WARNING: in fname_fn(): undefined column")
        # Remove bad characters 
        col = col.replace("(", "_").replace(")", "_").replace("/", "_over_").replace("*", "_star").replace(" ", "_")
        return col

###############################################################################
# Helper function for computing the mode in a data set 
###############################################################################
def mode(data):
    vals, counts = np.unique(data, return_counts=True)
    idx = np.nanargmax(counts)
    return vals[idx]

###############################################################################
# Helper function for plotting 2D histograms
###############################################################################
def histhelper(df, col_x, col_y, col_z, nbins, ax, cmap,
                xmin, xmax, ymin, ymax, vmin, vmax,
                log_z=False, alpha=1.0):
    """
    Helper function used to plot 2D histograms. This function is used in 
    plotgalaxies.plo2dhist() which is in turn used in 
    plotgalaxies.plot2dhistcontours().
    """
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

    # Function for colouring the histogram: if it's a continuous property, e.g.
    # SFR, then use the median. If it's a discrete quantity, e.g. BPT category,
    # then use the mode (= most frequent number in a data set). This will 
    # help to avoid the issue in which np.nanmedian returns a non-integer value.
    if col_z.startswith("BPT") or col_z.startswith("Morphology") or col_z.startswith("WHAV*"):
        func = mode
    else:
        func = np.nanmedian

    # Calculate the desired quantities for the data binned by x and y    
    gb_binned = df.join(cuts).groupby(list(cuts))
    if col_z == "count":
        df_binned = gb_binned.agg({df.columns[0]: lambda g: g.count()})
        df_binned = df_binned.rename(columns={df.columns[0]: "count"})
    else:
        df_binned = gb_binned.agg({col_z: func})

    # Pull out arrays to plot
    count_map = df_binned[col_z].values.reshape((nbins, nbins))

    # Plot.
    if log_z:
        m = ax.pcolormesh(xbins[:-1], ybins[:-1], count_map.T, cmap=cmap,
            edgecolors="none", vmin=vmin, vmax=vmax, shading="auto",
            norm=LogNorm(vmin=vmin, vmax=vmax))
    else:
        m = ax.pcolormesh(xbins[:-1], ybins[:-1], count_map.T, cmap=cmap,
            edgecolors="none", vmin=vmin, vmax=vmax, shading="auto")
    m.set_rasterized(True)

    # Dodgy...
    if alpha < 1:
        overlay = np.full_like(count_map.T, 1.0)
        mo = ax.pcolormesh(xbins[:-1], ybins[:-1], overlay, alpha=1 - alpha, cmap="gray", vmin=0, vmax=1, shading="auto")
        mo.set_rasterized(True)

    return m

###############################################################################
# Plot empty BPT diagrams
###############################################################################
def plot_empty_BPT_diagram(colorbar=False, nrows=1, include_Law2021=False,
                           axs=None, figsize=None):
    """
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
    width = (1 - 2*left - cbar_width) / 3.
    height = 0.8 / nrows

    axs = []
    if colorbar:
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
        if colorbar:
            caxs.append(cax)

    if colorbar:
        if nrows == 1:
            return fig, axs, cax
        else:
            return fig, axs, caxs
    else:
        return fig, axs

###############################################################################
# Convenience functions for plotting lines over BPT diagrams
###############################################################################
def plot_BPT_lines(ax, col_x, include_Law2021=False,
                   color="gray", linewidth=1, zorder=1):
    """
    Over-plot demarcation lines of Kewley et al. (2001), Kauffman et al. 
    (2003), Kewley et al. (2006) and Law et al. (2021) on the provided axis
    where the y-axis is O3 and the x-axis is specified by col_x.
    """
    assert col_x in ["log N2", "log S2", "log O1"],\
        "col_x must be one of log N2, log S2 or log O1!"

    # Plot the demarcation lines from literature
    x_vals = np.linspace(-2.5, 2.5, 100)
    
    # Kewley+2001: all 3 diagrams
    ax.plot(x_vals, Kewley2001(col_x, x_vals), color=color, linewidth=linewidth, linestyle="--", zorder=zorder)
    
    # Kewley+2006: S2 and O1 only
    if col_x == "log S2" or col_x == "log O1":
        ax.plot(x_vals, Kewley2006(col_x, x_vals), color=color, linewidth=linewidth, linestyle="-.", zorder=zorder)
    
    # Kauffman+2003: log N2 only
    if col_x == "log N2":
        ax.plot(x_vals, Kauffman2003(col_x, x_vals), color=color, linewidth=linewidth, linestyle=":", zorder=zorder)

    if include_Law2021:
        y_vals = np.copy(x_vals)
        ax.plot(x_vals, Law2021_1sigma(col_x, x_vals), color=color, linewidth=linewidth, linestyle="-", zorder=zorder)
        ax.plot(Law2021_3sigma(col_x, y_vals), y_vals, color=color, linewidth=linewidth, linestyle="-", zorder=zorder)

    # Axis limits
    ax.set_ylim([-1.5, 1.2])
    if col_x == "log N2":
        ax.set_xlim([-1.3,0.5])
    elif col_x == "log S2":
        ax.set_xlim([-1.3,0.5])
    elif col_x == "log O1":
        ax.set_xlim([-2.2,0.2])

    return

###############################################################################
# Compass & scale bar functions for 2D map plots
###############################################################################
def plot_compass(PA_deg=0, flipped=True,
                 color="white",
                 bordercolor=None,
                 fontsize=10,
                 ax=None,
                 zorder=999999):
    """
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
        dist_str = f'{l_arcsec:.2f}" = {l_kpc:.2f} kpc'
    elif units == "arcmin":
        dist_str = f"{l_arcmin:.2f}' = {l_kpc:.2f} kpc"
        
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


