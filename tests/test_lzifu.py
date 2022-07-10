# Test all functions.

import numpy as np

from loaddata.lzifu import load_lzifu_galaxies
from plotting.sdssimg import plot_sdss_image
from plotting.plotgalaxies import plot2dscatter, plot2dhist, plot2dcontours, plot2dhistcontours
from plotting.plottools import label_fn, bpt_labels, vmin_fn, vmax_fn, label_fn, component_labels
from plotting.plot2dmap import plot2dmap

import seaborn as sns

import matplotlib.pyplot as plt
plt.close("all")
plt.ion()

from IPython.core.debugger import Tracer

##############################################################################
# Load a dataset
##############################################################################
ncomponents = "recom"
bin_type = "default"
eline_SNR_min = 5

gal = 572402
df_gal = load_lzifu_galaxies(gal=gal, 
                            ncomponents=ncomponents, 
                            bin_type=bin_type,
                            eline_SNR_min=eline_SNR_min,
                            correct_extinction=True)

##############################################################################
# Test: SDSS image
##############################################################################
plot_sdss_image(df_gal)

##############################################################################
# Test: 2D scatter
##############################################################################
fig, ax = plt.subplots(nrows=1, ncols=1)
bbox = ax.get_position()
cax = fig.add_axes([bbox.x0 + bbox.width, bbox.y0, 0.05, bbox.height])
plot2dscatter(df_gal, col_x="log N2 (total)", col_y="log O3 (total)",
              col_z="log sigma_gas (component 0)", ax=ax, cax=cax)

# Test without providing axes
fig, ax = plt.subplots(nrows=1, ncols=1)
plot2dscatter(df_gal, col_x="log N2 (total)", col_y="log O3 (total)",
              col_z="log sigma_gas (component 0)", ax=ax)

##############################################################################
# Test: 2D histogram & 2D contours
##############################################################################
fig, ax = plt.subplots(nrows=1, ncols=1)
plot2dhist(df_gal, col_x="log N2 (total)", col_y="log O3 (total)",
           col_z="BPT (numeric) (total)", ax=ax, nbins=30)

plot2dcontours(df_gal, col_x="log N2 (total)", col_y="log O3 (total)",
              ax=ax, nbins=30)

##############################################################################
# Test: 2D contours
##############################################################################
fig, ax = plt.subplots(nrows=1, ncols=1)
plot2dhist(df_gal, col_x="log N2 (total)", col_y="log O3 (total)",
           col_z="log sigma_gas (component 0)", ax=ax, nbins=30)

##############################################################################
# Test: 2D histogram + contours
##############################################################################
plot2dhistcontours(df_gal, col_x="log sigma_gas (component 0)", col_y="log HALPHA EW (component 0)",
                   col_z="count", log_z=True)

##############################################################################
# Test: 2D map plots
##############################################################################
plot2dmap(df_gal, survey="sami", bin_type=bin_type, col_z="HALPHA (total)")


