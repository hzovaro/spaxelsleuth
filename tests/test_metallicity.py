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

from spaxelsleuth.loaddata import linefns, metallicity

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
df = load_sami_galaxies(ncomponents="1",
                        bin_type="default",
                        eline_SNR_min=eline_SNR_min,
                        correct_extinction=True,
                        debug=True)

###########################################################################
# Check metallicity function
###########################################################################
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

###########################################################################
# Check metallicity computation (fixed log(U))
###########################################################################
cond_has_met = ~df["N2O2 (total)"].isna()
df_has_met = df[cond_has_met]

# Plot
ax.scatter(x=df_has_met["log(O/H) + 12 (N2O2) (total)"],
           y=df_has_met["N2O2 (total)"], s=5, c="k")
sys.exit()
###########################################################################
# Check metallicity computation (iterative method)
###########################################################################
met_diagnostic = "N2O2"
ion_diagnostic = "O3O2"
df_with_met_iter = metallicity.iter_metallicity_fn(df, 
                                                   met_diagnostic, 
                                                   ion_diagnostic, 
                                                   s=" (total)")

# Plot
fig, axs = plt.subplots(nrows=1, ncols=2)
for logU in [-4., -3., -2., -1.]:
    logOH12_vals = metallicity.get_metallicity("N2O2", R_vals, logU)
    axs[0].plot(logOH12_vals, R_vals, label=f"log(U) = {logU:.1f}")
m = axs[0].scatter(x=df_with_met_iter["log(O/H) + 12 (N2O2) (total)"],
               y=df_with_met_iter["N2O2 (total)"], 
               c=df_with_met_iter["log U (O3O2) (total)"])
axs[0].set_xlim([7.5, 9.3])
axs[0].set_ylim([-2, 4])
axs[0].legend()
plt.colorbar(mappable=m, ax=axs[0])

R_vals = np.linspace(-2, 2, 100)
for logOH12 in np.linspace(7.6, 9.3, 5):
    logU_vals = metallicity.get_ionisation_parameter("O3O2", R_vals, logOH12)
    axs[1].plot(logU_vals, R_vals, label=f"log(O/H) + 12 = {logOH12:.1f}")
axs[1].scatter(x=df_with_met_iter["log U (O3O2) (total)"],
               y=df_with_met_iter["O3O2 (total)"], 
               c=df_with_met_iter["log(O/H) + 12 (N2O2) (total)"])
axs[1].legend()
axs[1].set_xlim([-4, -1.8])
axs[1].set_ylim([-2, 2])

# Check...
cond_has_met = ~df_with_met_iter["N2O2 (total)"].isna()
cond_has_met &= ~df_with_met_iter["O3O2 (total)"].isna()
df_has_met = df_with_met_iter[cond_has_met]

###########################################################################
# Check how long it takes to compute metallicities in the full sample
###########################################################################
plt.close("all")
df = load_sami_galaxies(ncomponents=ncomponents,
                        bin_type="default",
                        eline_SNR_min=eline_SNR_min, 
                        vgrad_cut=False,
                        line_amplitude_SNR_cut=True,
                        correct_extinction=True,
                        sigma_gas_SNR_cut=True,
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

###########################################################################
# Assertion checks
###########################################################################
# TEST: check no spaxels with non-SF BPT classifications have metallicities
cond_no_met = df["BPT (total)"] != "SF"
assert all(df.loc[cond_no_met, f"log(O/H) + 12 ({met_diagnostic}) (total)"].isna())

# TEST: check no spaxels with no S/N in relevant emission lines have metallicities 
cond_low_SN = df["OII3726+OII3729 S/N (total)"] < eline_SNR_min
cond_low_SN |= df["NII6583 S/N (total)"] < eline_SNR_min
assert all(df.loc[cond_low_SN, f"log(O/H) + 12 ({met_diagnostic}) (total)"].isna())


