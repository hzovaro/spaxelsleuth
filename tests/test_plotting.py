# Imports
import sys
import os 
import numpy as np
import pandas as pd

from spaxelsleuth.loaddata.sami import load_sami_df
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
assert "SAMI_DIR" in os.environ, "Environment variable SAMI_DIR is not defined!"
sami_data_path = os.environ["SAMI_DIR"]
assert "SAMI_DATACUBE_DIR" in os.environ, "Environment variable SAMI_DATACUBE_DIR is not defined!"
sami_datacube_path = os.environ["SAMI_DATACUBE_DIR"]

###########################################################################
# Options
ncomponents, bin_type, eline_SNR_min = [sys.argv[1], sys.argv[2], int(sys.argv[3])]

###########################################################################
# Load the data
###########################################################################
# Load the ubinned data 
df = load_sami_df(ncomponents=ncomponents,
                  bin_type=bin_type,
                  eline_SNR_min=eline_SNR_min,
                  correct_extinction=True,
                  debug=False)

###########################################################################
# Checking plotting functions work
###########################################################################
#//////////////////////////////////////////////////////////////////////////
# SDSS image 
plot_sdss_image(df[df["ID"] == df["ID"].values[0]])

#//////////////////////////////////////////////////////////////////////////
# 2D scatter, 2D hist 
fig, axs_bpt = plot_empty_BPT_diagram(nrows=1)
col_y = "log O3"
for cc, col_x in enumerate(["log N2", "log S2", "log O1"]):
    # Add BPT functions
    plot_BPT_lines(ax=axs_bpt[cc], col_x=col_x)    

    # Plot histograms showing distribution for whole sample
    plot2dhistcontours(df=df,
                  col_x=f"{col_x} (total)",
                  col_y=f"{col_y} (total)",
                  col_z="HALPHA EW (total)",
                  ax=axs_bpt[cc],
                  cax=None,
                  plot_colorbar=True if cc==2 else False)

    # Overlay measurements for a specific galaxy
    plot2dscatter(df=df[df["ID"] == df["ID"].values[0]],
                  col_x=f"{col_x} (total)",
                  col_y=f"{col_y} (total)",
                  ax=axs_bpt[cc], 
                  cax=None, markerfacecolor="green")

#//////////////////////////////////////////////////////////////////////////
# 2D map plot
plot2dmap(df_gal=df[df["ID"] == df["ID"].values[0]],
          col_z="BPT (numeric) (total)", bin_type=bin_type, survey="sami")

