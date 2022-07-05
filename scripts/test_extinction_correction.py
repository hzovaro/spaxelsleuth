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
ncomponents = "recom"   # Options: "1" or "recom"
eline_SNR_min = 5       # Minimum S/N of emission lines to accept
plt.close("all")

###########################################################################
# Load the data
###########################################################################
# Load the ubinned data 
df_extcorr = load_sami_galaxies(ncomponents=ncomponents,
                        bin_type="default",
                        eline_SNR_min=eline_SNR_min, 
                        vgrad_cut=False,
                        line_amplitude_SNR_cut=True,
                        correct_extinction=True,
                        sigma_gas_SNR_cut=True,
                        debug=True)

df_noextcorr = load_sami_galaxies(ncomponents=ncomponents,
                        bin_type="default",
                        eline_SNR_min=eline_SNR_min, 
                        vgrad_cut=False,
                        line_amplitude_SNR_cut=True,
                        correct_extinction=False,
                        sigma_gas_SNR_cut=True,
                        debug=True)

###########################################################################
# Plot to check that it's worked
###########################################################################
fig, ax = plt.subplots(nrows=1, ncols=1)
m = ax.scatter(df_noextcorr["OIII5007 (total)"], df_extcorr["OIII5007 (total)"], c=df_extcorr["A_V (total)"], cmap="inferno")
plt.colorbar(mappable=m, ax=ax)
ax.plot([0, 10], [0, 10], "k")
ax.set_xlabel("Halpha (no extinction correction)")
ax.set_ylabel("Halpha (extinction correction)")

###########################################################################
# BPT before/after 
###########################################################################
fig, axs_bpt = plot_empty_BPT_diagram(nrows=3)
axs_bpt = [axs_bpt[:3], axs_bpt[3:6], axs_bpt[6:]]
col_y = "log O3"
for cc, col_x in enumerate(["log N2", "log S2", "log O1"]):
    # Add BPT functions
    plot_BPT_lines(ax=axs_bpt[0][cc], col_x=col_x)    
    plot_BPT_lines(ax=axs_bpt[1][cc], col_x=col_x)    
    plot_BPT_lines(ax=axs_bpt[2][cc], col_x=col_x)    

    # Plot measurements for this galaxy
    plot2dscatter(df=df_noextcorr,
                  col_x=f"{col_x} (total)",
                  col_y=f"{col_y} (total)",
                  col_z="HALPHA (total)",
                  marker="o", ax=axs_bpt[0][cc], 
                  cax=None, vmin=0, vmax=10,
                  markersize=20, 
                  markerfacecolor=None, 
                  markeredgecolor="black",
                  plot_colorbar=False)
    

    # Plot measurements for this galaxy
    plot2dscatter(df=df_extcorr,
                  col_x=f"{col_x} (total)",
                  col_y=f"{col_y} (total)",
                  col_z="HALPHA (total)",
                  marker="o", ax=axs_bpt[1][cc], 
                  cax=None, vmin=0, vmax=10,
                  markersize=20, 
                  markerfacecolor=None, 
                  markeredgecolor="black",
                  plot_colorbar=False)
    

    # Plot measurements for this galaxy
    plot2dscatter(df=df_extcorr,
                  col_x=f"{col_x} (total)",
                  col_y=f"{col_y} (total)",
                  col_z="A_V (total)",
                  marker="o", ax=axs_bpt[2][cc], 
                  cax=None, vmin=0, vmax=3,
                  markersize=20, 
                  markerfacecolor=None, 
                  markeredgecolor="black",
                  plot_colorbar=False)

axs_bpt[0][0].text(s="Before extinction correction (HALPHA flux)",
                       x=0.05, y=0.95,
                       transform=axs_bpt[0][0].transAxes)
axs_bpt[1][0].text(s="After extinction correction (HALPHA flux)",
                       x=0.05, y=0.95,
                       transform=axs_bpt[1][0].transAxes)
axs_bpt[2][0].text(s="After extinction correction (A_V)",
                   x=0.05, y=0.95,
                   transform=axs_bpt[2][0].transAxes)

###########################################################################
# Assertion tests
###########################################################################
# TEST: make sure weird stuff hasn't happened with the indices
assert len([c for c in df_extcorr["HALPHA (total)"].index.values if c not in df_noextcorr["HALPHA (total)"].index.values]) == 0

# TEST: all HALPHA fluxes in the extinction-corrected DataFrame are greater than 
# or equal to those in the non-extinction-corrected DataFrame
eline_list = ["HALPHA", "HBETA", "NII6583", "OI6300", "OII3726+OII3729", "OIII5007", "SII6716", "SII6731"]
for eline in eline_list:
    assert np.all(df_extcorr[f"{eline} (total)"].dropna() >= df_noextcorr[f"{eline} (total)"].dropna())
    assert np.all(df_extcorr[f"{eline} error (total)"].dropna() >= df_noextcorr[f"{eline} error (total)"].dropna())

# TEST: check no negative A_V's
assert not np.any(df_extcorr["A_V (total)"] < 0)
assert not np.any(df_extcorr["A_V error (total)"] < 0)

# TEST: check no nonzero A_V's in rows where S/N in HALPHA or HBETA are less than 5
cond_low_SN = df_extcorr["HALPHA S/N (total)"] < 5
cond_low_SN |= df_extcorr["HBETA S/N (total)"] < 5
assert np.all(df_extcorr.loc[cond_low_SN, "A_V (total)"].isna())
assert np.all(df_extcorr.loc[cond_low_SN, "A_V error (total)"].isna())

###########################################################################
# Check how long it takes to extinction-correct the full sample
###########################################################################
Tracer()()
plt.close("all")
df_extcorr = load_sami_galaxies(ncomponents=ncomponents,
                        bin_type="default",
                        eline_SNR_min=eline_SNR_min, 
                        vgrad_cut=False,
                        line_amplitude_SNR_cut=True,
                        correct_extinction=True,
                        sigma_gas_SNR_cut=True,
                        debug=False)

df_noextcorr = load_sami_galaxies(ncomponents=ncomponents,
                        bin_type="default",
                        eline_SNR_min=eline_SNR_min, 
                        vgrad_cut=False,
                        line_amplitude_SNR_cut=True,
                        correct_extinction=False,
                        sigma_gas_SNR_cut=True,
                        debug=False)

###########################################################################
# BPT before/after 
###########################################################################
fig, axs_bpt = plot_empty_BPT_diagram(nrows=3)
axs_bpt = [axs_bpt[:3], axs_bpt[3:6], axs_bpt[6:]]
col_y = "log O3"
for cc, col_x in enumerate(["log N2", "log S2", "log O1"]):
    # Add BPT functions
    plot_BPT_lines(ax=axs_bpt[0][cc], col_x=col_x)    
    plot_BPT_lines(ax=axs_bpt[1][cc], col_x=col_x)    
    plot_BPT_lines(ax=axs_bpt[2][cc], col_x=col_x)    

    # Plot measurements for this galaxy
    plot2dhistcontours(df=df_noextcorr,
                  col_x=f"{col_x} (total)",
                  col_y=f"{col_y} (total)",
                  col_z="HALPHA (total)",
                  ax=axs_bpt[0][cc], 
                  cax=None, vmin=0, vmax=10,
                  plot_colorbar=False)
    

    # Plot measurements for this galaxy
    plot2dhistcontours(df=df_extcorr,
                  col_x=f"{col_x} (total)",
                  col_y=f"{col_y} (total)",
                  col_z="HALPHA (total)",
                  ax=axs_bpt[1][cc], 
                  cax=None, vmin=0, vmax=10,
                  plot_colorbar=False)
    

    # Plot measurements for this galaxy
    plot2dhistcontours(df=df_extcorr,
                  col_x=f"{col_x} (total)",
                  col_y=f"{col_y} (total)",
                  col_z="A_V (total)",
                  ax=axs_bpt[2][cc], 
                  cax=None, vmin=0, vmax=3,
                  plot_colorbar=False)

axs_bpt[0][0].text(s="Before extinction correction (HALPHA flux)",
                       x=0.05, y=0.95,
                       transform=axs_bpt[0][0].transAxes)
axs_bpt[1][0].text(s="After extinction correction (HALPHA flux)",
                       x=0.05, y=0.95,
                       transform=axs_bpt[1][0].transAxes)
axs_bpt[2][0].text(s="After extinction correction (A_V)",
                   x=0.05, y=0.95,
                   transform=axs_bpt[2][0].transAxes)

