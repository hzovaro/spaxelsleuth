# Test all functions.

import numpy as np

from loaddata.sami import load_sami_galaxies
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
vgrad_cut = False
stekin_cut = True

df = load_sami_galaxies(ncomponents="recom", bin_type="default",
                        eline_SNR_min=eline_SNR_min, vgrad_cut=vgrad_cut,
                        debug=True)

gal = df.catid.unique()[1]
df_gal = df[df.catid == gal]

##############################################################################
# Testing seaborn
##############################################################################

col_x = "sigma_gas - sigma_* (component 0)"
col_y = "log HALPHA EW (component 0)"

cond = ~np.isnan(df[col_x]) & ~np.isnan(df[col_y])
x = df.loc[cond, col_x].values
y = df.loc[cond, col_y].values

plt.figure(); sns.kdeplot(data=x); plt.xlabel(label_fn(col_x))
plt.figure(); sns.kdeplot(data=y); plt.xlabel(label_fn(col_y))

plt.figure()
sns.kdeplot(x, y, shade=True)
plt.scatter(x=df[col_x], y=df[col_y], s=1)

Tracer()()


fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(20, 5))
fig.subplots_adjust(wspace=0)

col_x = "sigma_gas - sigma_*"
col_y = "log HALPHA EW"

for bpt in bpt_labels:
    df_bpt = df[df["BPT (total)"] == bpt]
    for ii in range(3):
        cond = ~np.isnan(df_bpt[f"{col_x} (component {ii})"]) & ~np.isnan(df_bpt[f"{col_y} (component {ii})"])
        data_x = df_bpt.loc[cond, f"{col_x} (component {ii})"]
        data_y = df_bpt.loc[cond, f"{col_y} (component {ii})"]
        if len(data_x) > 1 and len(data_y) > 1:
            sns.kdeplot(data=data_x, data2=data_y, 
                        clip=((-300, +300), (-1.0, 3.0)),
                        ax=axs[ii], legend=True)




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
plot2dscatter(df, col_x="log N2 (total)", col_y="log O3 (total)",
              col_z="log sigma_gas (component 0)", ax=ax, cax=cax)

# Test without providing axes
fig, ax = plt.subplots(nrows=1, ncols=1)
plot2dscatter(df_gal, col_x="log N2 (total)", col_y="log O3 (total)",
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
plot2dmap(df_gal, bin_type=bin_type, col_z="HALPHA (total)")



