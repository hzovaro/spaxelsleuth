# Imports
import sys
import os 
import numpy as np
import pandas as pd

from spaxelsleuth.loaddata.sami import load_sami_df
from spaxelsleuth.utils import linefns, metallicity
from spaxelsleuth.plotting.plotgalaxies import plot2dhistcontours, plot2dscatter, plot2dcontours
from spaxelsleuth.plotting.plottools import plot_empty_BPT_diagram, plot_BPT_lines

from IPython.core.debugger import Tracer

import matplotlib.pyplot as plt
plt.ion()
plt.close("all")

###########################################################################
# Options
ncomponents, bin_type, eline_SNR_min = [sys.argv[1], sys.argv[2], int(sys.argv[3])]

###########################################################################
# Check metallicity: constant log(U)
###########################################################################
df = load_sami_df(ncomponents=ncomponents,
                        bin_type=bin_type,
                        eline_SNR_min=eline_SNR_min,
                        correct_extinction=True,
                        debug=True)

fig, axs = plt.subplots(ncols=2)
# Non-iterative
axs[0].errorbar(x=df["log(O/H) + 12 (N2O2) (total)"].values,
            xerr=[df["log(O/H) + 12 error (lower) (N2O2) (total)"].values, 
                  df["log(O/H) + 12 error (upper) (N2O2) (total)"].values],
            y=df["log(O/H) + 12 (Dopita+2016) (total)"].values,
            yerr=[df["log(O/H) + 12 error (lower) (Dopita+2016) (total)"].values, 
                  df["log(O/H) + 12 error (upper) (Dopita+2016) (total)"].values],
            ls="none", mec="none", ecolor="k", zorder=9999, linewidth=0.5)
axs[0].scatter(x=df["log(O/H) + 12 (N2O2) (total)"].values,
           y=df["log(O/H) + 12 (Dopita+2016) (total)"].values,
           s=10, edgecolors="k", facecolor="w", zorder=10000)
axs[0].plot([7.5, 10], [7.5, 10], "k--")
axs[0].set_xlabel("log(O/H + 12 - N2O2")
axs[0].set_ylabel("log(O/H + 12 - Dopita+2016")

# Iterative
axs[1].errorbar(x=df["log(O/H) + 12 (N2O2/O3O2) (total)"].values,
            xerr=[df["log(O/H) + 12 error (lower) (N2O2/O3O2) (total)"].values, 
                  df["log(O/H) + 12 error (upper) (N2O2/O3O2) (total)"].values],
            y=df["log(O/H) + 12 (Dopita+2016/O3O2) (total)"].values,
            yerr=[df["log(O/H) + 12 error (lower) (Dopita+2016/O3O2) (total)"].values, 
                  df["log(O/H) + 12 error (upper) (Dopita+2016/O3O2) (total)"].values],
            ls="none", mec="none", ecolor="k", zorder=9999, linewidth=0.5)
m = axs[1].scatter(x=df["log(O/H) + 12 (N2O2/O3O2) (total)"].values,
           y=df["log(O/H) + 12 (Dopita+2016/O3O2) (total)"].values,
           c=df["log(U) (N2O2/O3O2) (total)"].values,
           cmap="cubehelix", vmin=-3.5, vmax=-1.5,
           s=10, edgecolors="k", facecolor="w", zorder=10000)
axs[1].plot([7.5, 10], [7.5, 10], "k--")
axs[1].set_xlabel("log(O/H + 12 - N2O2/O3O2")
axs[1].set_ylabel("log(O/H + 12 - Dopita+2016/O3O2")
plt.colorbar(mappable=m, ax=axs[1])



"""
###########################################################################
# Check metallicity: constant log(U)
###########################################################################
df = load_sami_df(ncomponents=ncomponents,
                        bin_type=bin_type,
                        eline_SNR_min=eline_SNR_min,
                        correct_extinction=True,
                        debug=True)

# Remove metallicity columns so we can re-calculate them
cols_to_remove = [c for c in df.columns if "log(U)" in c or "log(O/H)" in c]
df = df.drop(columns=cols_to_remove)

df = metallicity.metallicity_fn(df, compute_errors=True, 
                                met_diagnostic="N2O2", logU=-3.0, 
                                niters=1000,
                                s=" (total)")

# Plot the diagnostics
fig, ax = plt.subplots(nrows=1, ncols=1)
R_vals = np.linspace(-2.0, 1.9, 100)
for logU in [-4., -3., -2., -1.]:
    logOH12_vals = metallicity.get_metallicity("N2O2", R_vals, logU)
    ax.plot(logOH12_vals, R_vals, label=f"log(U) = {logU:.1f}")
ax.set_xlabel("log(O/H) + 12")
ax.set_ylabel("log(N2O2)")
ax.set_xlim([7.5, 9.3])
ax.set_ylim([-2, 4])
ax.grid()
ax.legend()

# Plot the data on top
cond_has_met = ~df["N2O2 (total)"].isna()
df_has_met = df[cond_has_met]
ax.errorbar(x=df_has_met["log(O/H) + 12 (N2O2) (total)"].values,
            xerr=[df_has_met["log(O/H) + 12 error (lower) (N2O2) (total)"].values, 
                  df_has_met["log(O/H) + 12 error (upper) (N2O2) (total)"].values],
            y=df_has_met["N2O2 (total)"], 
            ls="none", mec="none", ecolor="k", zorder=9999, linewidth=0.5)
ax.scatter(x=df_has_met["log(O/H) + 12 (N2O2) (total)"], 
           y=df_has_met["N2O2 (total)"], 
           c=df_has_met["log(U) (O3O2) (total)"],
           s=20, edgecolors="k", facecolor="w", zorder=9999)

###########################################################################
# Check metallicity: iterative method
###########################################################################
df_iter = load_sami_df(ncomponents=ncomponents,
                        bin_type=bin_type,
                        eline_SNR_min=eline_SNR_min,
                        correct_extinction=True,
                        debug=True)

# Remove metallicity columns so we can re-calculate them
cols_to_remove = [c for c in df_iter.columns if "log(U)" in c or "log(O/H)" in c]
df_iter = df_iter.drop(columns=cols_to_remove)

df_iter = metallicity.iter_metallicity_fn(df_iter, compute_errors=True, 
                                     met_diagnostic="N2O2", ion_diagnostic="O3O2", 
                                     niters=20,
                                     s=" (total)")

# Plot
fig, axs = plt.subplots(nrows=1, ncols=2)
for logU in [-4., -3., -2., -1.]:
    logOH12_vals = metallicity.get_metallicity("N2O2", R_vals, logU)
    axs[0].plot(logOH12_vals, R_vals, label=f"log(U) = {logU:.1f}")
m = axs[0].scatter(x=df_iter["log(O/H) + 12 (N2O2) (total)"],
               y=df_iter["N2O2 (total)"], 
               c=df_iter["log(U) (O3O2) (total)"])
axs[0].set_xlim([7.5, 9.3])
axs[0].set_ylim([-2, 4])
axs[0].legend()
plt.colorbar(mappable=m, ax=axs[0])

R_vals = np.linspace(-2, 2, 100)
for logOH12 in np.linspace(7.6, 9.3, 5):
    logU_vals = metallicity.get_ionisation_parameter("O3O2", R_vals, logOH12)
    axs[1].plot(logU_vals, R_vals, label=f"log(O/H) + 12 = {logOH12:.1f}")
m = axs[1].scatter(x=df_iter["log(U) (O3O2) (total)"],
               y=df_iter["O3O2 (total)"], 
               c=df_iter["log(U) (O3O2) (total)"])
plt.colorbar(mappable=m, ax=axs[1])

axs[1].legend()
axs[1].set_xlim([-4, -1.8])
axs[1].set_ylim([-2, 2])

# Check...
cond_has_met = ~df_iter["N2O2 (total)"].isna()
cond_has_met &= ~df_iter["O3O2 (total)"].isna()
df_has_met = df_iter[cond_has_met]

###########################################################################
# Check how long it takes to compute metallicities in the full sample
###########################################################################
plt.close("all")
df = load_sami_df(ncomponents="1",
                        bin_type="default",
                        eline_SNR_min=eline_SNR_min,
                        correct_extinction=True,
                        debug=False)

fig, axs_bpt = plot_empty_BPT_diagram(nrows=1)
col_y = "log O3"
for cc, col_x in enumerate(["log N2", "log S2", "log O1"]):
    # Add BPT functions
    plot_BPT_lines(ax=axs_bpt[cc], col_x=col_x)    

    # Plot measurements for this galaxy
    plot2dhistcontours(df=df,
                  col_x=f"{col_x} (total)",
                  col_y=f"{col_y} (total)",
                  col_z="log(O/H) + 12 (Dopita+2016) (total)",
                  ax=axs_bpt[cc], 
                  cax=None,
                  plot_colorbar=True if cc==2 else False)

"""

###########################################################################
# Assertion checks
###########################################################################
# TEST: check no spaxels with non-SF BPT classifications have metallicities
met_diagnostic = "N2O2"
cond_no_met = df["BPT (total)"] != "SF"
assert all(df.loc[cond_no_met, f"log(O/H) + 12 ({met_diagnostic}) (total)"].isna())

# TEST: check no spaxels with no S/N in relevant emission lines have metallicities 
cond_low_SN = df["OII3726+OII3729 S/N (total)"] < eline_SNR_min
cond_low_SN |= df["NII6583 S/N (total)"] < eline_SNR_min
assert all(df.loc[cond_low_SN, f"log(O/H) + 12 ({met_diagnostic}) (total)"].isna())

