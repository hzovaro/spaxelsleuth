import os

from spaxelsleuth import load_user_config
load_user_config("/home/u5708159/.spaxelsleuthconfig.json")
from spaxelsleuth.loaddata.lzifu import make_lzifu_df, load_lzifu_df
from spaxelsleuth.config import settings

# gals = [g.rstrip("_R.fits.gz") for g in os.listdir(data_cube_path) if g.endswith("_R.fits.gz")]
gals = [901028857804452]
kwargs = {
    "eline_SNR_min": 1,
    "sigma_gas_SNR_min": 1,
    "line_flux_SNR_cut": False,
    "missing_fluxes_cut": False,
    "line_amplitude_SNR_cut": False,
    "flux_fraction_cut": False,
    "sigma_gas_SNR_cut": False,
    "vgrad_cut": False,
    "stekin_cut": False,
    "correct_extinction": False,
    "bin_type": "default",
}
# make_lzifu_df(gals=gals, ncomponents=1, sigma_inst_kms=29.6, nthreads_max=1, df_fname="test_lzifu.hd5", **kwargs)

df = load_lzifu_df(ncomponents=1, bin_type="default", df_fname="test_lzifu.hd5")

# Check that it's worked 
import matplotlib.pyplot as plt
import pandas as pd
from spaxelsleuth.plotting.plot2dmap import plot2dmap
plt.ion()

plot2dmap(df, gal=int(gals[0]), col_z="HALPHA (total)")
plot2dmap(df, gal=int(gals[0]), col_z="D4000")

