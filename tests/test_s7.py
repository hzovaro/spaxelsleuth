# Imports
import sys

from spaxelsleuth.loaddata.s7 import make_s7_metadata_df, make_s7_df, load_s7_df

from spaxelsleuth.plotting.sdssimg import plot_sdss_image
from spaxelsleuth.plotting.plotgalaxies import plot2dscatter, plot2dhist, plot2dcontours, plot2dhistcontours
from spaxelsleuth.plotting.plottools import label_fn, bpt_labels, vmin_fn, vmax_fn, label_fn, component_labels
from spaxelsleuth.plotting.plot2dmap import plot2dmap

import matplotlib.pyplot as plt
plt.close("all")
plt.ion()

###########################################################################
# Options
ncomponents, bin_type, eline_SNR_min = [sys.argv[1], sys.argv[2], int(sys.argv[3])]

###########################################################################
# Create the metadata DataFrame
###########################################################################
make_s7_metadata_df()

###########################################################################
# Create the DataFrame
###########################################################################
make_s7_df(ncomponents=ncomponents,
           bin_type=bin_type,
           eline_SNR_min=eline_SNR_min, 
           debug=False)

##############################################################################
# Load a dataset
##############################################################################
df = load_s7_df(ncomponents=ncomponents, 
                  bin_type=bin_type,
                  eline_SNR_min=eline_SNR_min,
                  correct_extinction=True,
                  debug=True)

##############################################################################
# Test: 2D scatter
##############################################################################
fig, ax = plt.subplots(nrows=1, ncols=1)
bbox = ax.get_position()
cax = fig.add_axes([bbox.x0 + bbox.width, bbox.y0, 0.05, bbox.height])
plot2dscatter(df, col_x="log N2 (total)", col_y="log O3 (total)",
              col_z="log sigma_gas (component 0)", ax=ax, cax=cax)

# Test without providing axes
fig, ax = plt.subplots(nrows=1, ncols=1)
plot2dscatter(df, col_x="log N2 (total)", col_y="log O3 (total)",
              col_z="log sigma_gas (component 0)", ax=ax)

##############################################################################
# Test: 2D histogram & 2D contours
##############################################################################
fig, ax = plt.subplots(nrows=1, ncols=1)
plot2dhist(df, col_x="log N2 (total)", col_y="log O3 (total)",
           col_z="BPT (numeric) (total)", ax=ax, nbins=30)

plot2dcontours(df, col_x="log N2 (total)", col_y="log O3 (total)",
              ax=ax, nbins=30)

##############################################################################
# Test: 2D contours
##############################################################################
fig, ax = plt.subplots(nrows=1, ncols=1)
plot2dhist(df, col_x="log N2 (total)", col_y="log O3 (total)",
           col_z="log sigma_gas (component 0)", ax=ax, nbins=30)

##############################################################################
# Test: 2D histogram + contours
##############################################################################
plot2dhistcontours(df, col_x="log sigma_gas (component 0)", col_y="log HALPHA EW (component 0)",
                   col_z="count", log_z=True)

##############################################################################
# Test: 2D map plots
##############################################################################
gal = df["catid"].values[0]
plot2dmap(df[df["catid"] == gal], survey="s7", bin_type=bin_type, col_z="HALPHA (total)", 
         vmax=None)

