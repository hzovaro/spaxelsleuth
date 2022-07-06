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

###########################################################################
# Options
fig_path = "/priv/meggs3/u5708159/SAMI/figs/paper/"
plt.close("all")

###########################################################################
# Load the data
###########################################################################
# Load the ubinned data 
df = load_sami_galaxies(ncomponents="recom",
                        bin_type="default",
                        eline_SNR_min=5,
                        correct_extinction=True,
                        debug=True)

###########################################################################
# Assertion checks
###########################################################################
# CHECK: stellar kinematics have been masked out
cond_bad_stekin = df["Bad stellar kinematics"]
for col in [c for c in df.columns if "*" in c]:
    assert all(df.loc[cond_bad_stekin, col].isna())

# CHECK: sigma_gas S/N has worked 
for ii in range(3):
    assert all(df.loc[df[f"sigma_obs S/N (component {ii})"] < 3, f"Low sigma_gas S/N flag (component {ii})"])
    cond_bad_sigma = df[f"Low sigma_gas S/N flag (component {ii})"]
    for col in [c for c in df.columns if "sigma_gas" in c and f"component {ii}" in c and "flag" not in c]:
        assert all(df.loc[cond_bad_sigma, col].isna())

# CHECK: line amplitudes 
for ii in range(3):
    assert all(df.loc[df[f"HALPHA A (component {ii})"] < 3 * df[f"HALPHA continuum std. dev."], f"Low amplitude flag - HALPHA (component {ii})"])
    cond_low_amp = df[f"Low amplitude flag - HALPHA (component {ii})"]
    for col in [c for c in df.columns if f"component {ii}" in c and "flag" not in c]:
        assert all(df.loc[cond_low_amp, col].isna())

# CHECK: flux S/N
for ii in range(3):
    assert all(df.loc[df[f"Low flux S/N flag - HALPHA (component {ii})"], f"HALPHA (component {ii})"].isna())
for eline in ["HALPHA", "HBETA", "NII6583", "OIII5007", "SII6716", "SII6731", "OII3726+OII3729", "OI6300"]:
    assert all(df.loc[df[f"Low flux S/N flag - {eline} (total)"], f"{eline} (total)"].isna())

# CHECK: BPT categories 
for eline in ["HALPHA", "HBETA", "NII6583", "OIII5007"]:
    cond_has_no_line = df[f"{eline} (total)"].isna()
    assert all(df.loc[cond_has_no_line, "BPT (total)"] == "Not classified")

###########################################################################
# Load the data (1-component fit)
###########################################################################
# Load the ubinned data 
df = load_sami_galaxies(ncomponents="1",
                        bin_type="default",
                        eline_SNR_min=5,
                        correct_extinction=True,
                        debug=True)

###########################################################################
# Assertion checks
###########################################################################
# CHECK: stellar kinematics have been masked out
cond_bad_stekin = df["Bad stellar kinematics"]
for col in [c for c in df.columns if "*" in c]:
    assert all(df.loc[cond_bad_stekin, col].isna())

# CHECK: sigma_gas S/N has worked 
for ii in range(1):
    assert all(df.loc[df[f"sigma_obs S/N (component {ii})"] < 3, f"Low sigma_gas S/N flag (component {ii})"])
    cond_bad_sigma = df[f"Low sigma_gas S/N flag (component {ii})"]
    for col in [c for c in df.columns if "sigma_gas" in c and f"component {ii}" in c and "flag" not in c]:
        assert all(df.loc[cond_bad_sigma, col].isna())

# CHECK: line amplitudes 
for ii in range(1):
    assert all(df.loc[df[f"HALPHA A (component {ii})"] < 3 * df[f"HALPHA continuum std. dev."], f"Low amplitude flag - HALPHA (component {ii})"])
    cond_low_amp = df[f"Low amplitude flag - HALPHA (component {ii})"]
    for col in [c for c in df.columns if f"component {ii}" in c and "flag" not in c]:
        assert all(df.loc[cond_low_amp, col].isna())

# CHECK: flux S/N
for ii in range(1):
    assert all(df.loc[df[f"Low flux S/N flag - HALPHA (component {ii})"], f"HALPHA (component {ii})"].isna())
for eline in ["HALPHA", "HBETA", "NII6583", "OIII5007", "SII6716", "SII6731", "OII3726+OII3729", "OI6300"]:
    assert all(df.loc[df[f"Low flux S/N flag - {eline} (total)"], f"{eline} (total)"].isna())

# CHECK: BPT categories 
for eline in ["HALPHA", "HBETA", "NII6583", "OIII5007"]:
    cond_has_no_line = df[f"{eline} (total)"].isna()
    assert all(df.loc[cond_has_no_line, "BPT (total)"] == "Not classified")




