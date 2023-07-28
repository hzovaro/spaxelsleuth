# Imports
import sys
import os 
import numpy as np
import pandas as pd

from spaxelsleuth import load_user_config, load_default_config
load_default_config()
load_user_config("/home/u5708159/.spaxelsleuthconfig.json")

from spaxelsleuth.loaddata.sami import load_sami_df
from spaxelsleuth.plotting.plot2dmap import plot2dmap

import matplotlib
from matplotlib import rc, rcParams
import matplotlib.pyplot as plt

from IPython.core.debugger import Tracer

rc("text", usetex=False)
rc("font",**{"family": "serif", "size": 11})
rcParams["savefig.bbox"] = "tight"
rcParams["savefig.format"] = "pdf"
plt.ion()
plt.close("all")

###########################################################################
# Options
ncomponents, bin_type, eline_SNR_min = ("recom", "default", 5)
gal = 572402

###########################################################################
# Load the data
###########################################################################
# Load the ubinned data 
df = load_sami_df(ncomponents=ncomponents,
                  bin_type=bin_type,
                  eline_SNR_min=eline_SNR_min,
                  correct_extinction=True,
                  debug=True)

#//////////////////////////////////////////////////////////////////////////
# 2D map plot
plot2dmap(df=df, gal=gal, col_z="v_gas (component 1)")
plot2dmap(df=df, gal=gal, col_z="BPT (numeric) (total)")
plot2dmap(df=df, gal=gal, col_z="Number of components")

###########################################################################
# Checking plotting functions work
###########################################################################
#//////////////////////////////////////////////////////////////////////////
# SDSS image 
# plot_sdss_image(df[df["ID"] == df["ID"].values[0]])

# #//////////////////////////////////////////////////////////////////////////
# # 2D scatter, 2D hist 
# fig, axs_bpt = plot_empty_BPT_diagram(nrows=1)
# col_y = "log O3"
# for cc, col_x in enumerate(["log N2", "log S2", "log O1"]):
#     # Add BPT functions
#     plot_BPT_lines(ax=axs_bpt[cc], col_x=col_x)    

#     # Plot histograms showing distribution for whole sample
#     plot2dhistcontours(df=df,
#                   col_x=f"{col_x} (total)",
#                   col_y=f"{col_y} (total)",
#                   col_z="HALPHA EW (total)",
#                   ax=axs_bpt[cc],
#                   cax=None,
#                   plot_colorbar=True if cc==2 else False)

#     # Overlay measurements for a specific galaxy
#     plot2dscatter(df=df[df["ID"] == df["ID"].values[0]],
#                   col_x=f"{col_x} (total)",
#                   col_y=f"{col_y} (total)",
#                   ax=axs_bpt[cc], 
#                   cax=None, markerfacecolor="green")

