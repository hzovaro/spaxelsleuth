# Imports
import sys
import os 
import numpy as np
import pandas as pd

from spaxelsleuth.loaddata.sami import load_sami_galaxies
from spaxelsleuth.plotting.plottools import plot_empty_BPT_diagram, plot_BPT_lines
from spaxelsleuth.plotting.plotgalaxies import plot2dscatter, plot2dhistcontours

from IPython.core.debugger import Tracer

import matplotlib
from matplotlib import rc, rcParams
import matplotlib.pyplot as plt

plt.ion()
plt.close("all")

###########################################################################
# Options
ncomponents, bin_type, eline_SNR_min = [sys.argv[1], sys.argv[2], int(sys.argv[3])]

###########################################################################
# Load the data
###########################################################################
# Load the ubinned data 
df_extcorr = load_sami_galaxies(ncomponents=ncomponents,
                        bin_type=bin_type,
                        eline_SNR_min=eline_SNR_min, 
                        correct_extinction=True,
                        debug=True)

df_noextcorr = load_sami_galaxies(ncomponents=ncomponents,
                        bin_type=bin_type,
                        eline_SNR_min=eline_SNR_min, 
                        correct_extinction=False,
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

