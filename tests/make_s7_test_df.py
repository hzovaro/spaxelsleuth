from spaxelsleuth import load_user_config
load_user_config("/home/u5708159/.spaxelsleuthconfig.json")
from spaxelsleuth.loaddata.s7_2 import make_s7_metadata_df, make_s7_df, load_s7_metadata_df, load_s7_df

# Create the metadata DataFrame
make_s7_metadata_df()
df_metadata = load_s7_metadata_df()

# TODO: merge metadata DataFrame onto the final one
make_s7_df(gals=["NGC1068", "MARK573",], metallicity_diagnostics=["N2Ha_PP04",], eline_SNR_min=3, nthreads=1)

# Load 
df = load_s7_df(correct_extinction=True, eline_SNR_min=3)

# Histograms showing the distribution in velocity dispersion
import matplotlib.pyplot as plt
from astropy.visualization import hist
fig, ax = plt.subplots(nrows=1, ncols=1)
for nn in range(1, 4):
    hist(df[f"sigma_gas (component {nn})"].values, bins="scott", ax=ax, range=(0, 500), density=True, histtype="step", label=f"Component {nn}")
ax.legend()
ax.set_xlabel(r"$\sigma_{\rm gas}$")
ax.set_ylabel(r"$N$ (normalised)")

# 2D maps
from spaxelsleuth.plotting.plot2dmap import plot2dmap
plot2dmap(df=df, gal="MARK573", col_z="HALPHA (total)")
plot2dmap(df=df, gal="MARK573", col_z="OII3726 (total)")
plot2dmap(df=df, gal="MARK573", col_z="HALPHA EW (total)")
plot2dmap(df=df, gal="MARK573", col_z="BPT (numeric) (total)")
plot2dmap(df=df, gal="MARK573", col_z="D4000")

# Plot a 2D histogram showing the distribution of SAMI spaxels in the WHAN diagram
from spaxelsleuth.plotting.plotgalaxies import plot2dhistcontours, plot2dscatter
_ = plot2dhistcontours(df=df,
              col_x=f"log N2 (total)",
              col_y=f"log HALPHA EW (total)",
              col_z="count", log_z=True,
              plot_colorbar=True)
