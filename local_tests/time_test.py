"""How long would it take to compute "simple" quantities at runtime rather than saving them to disk?"""

from time import time
import pandas as pd

from spaxelsleuth import load_user_config
load_user_config("/home/u5708159/.spaxelsleuthconfig.json")
from spaxelsleuth.config import settings
from spaxelsleuth.loaddata.sami import load_sami_df, make_sami_df
from spaxelsleuth.utils import continuum, linefns, misc, metallicity

# Make DEBUG DataFrame to see how much disk space is used by columns added below
# make_sami_df(ncomponents="recom", bin_type="default", eline_SNR_min=5, correct_extinction=False, debug=True, nthreads=10)

# Load the DataFrame (manually, because we don't want to add non-numeric columns)
bin_type = "default"
ncomponents = "recom"
eline_SNR_min = 5
eline_ANR_min = 3


# How long does it take to load?
t = time()
df = pd.read_hdf(settings["sami"]["output_path"] + f"sami_{bin_type}_{ncomponents}-comp_extcorr_minSNR={eline_SNR_min}_minANR={eline_ANR_min}.hd5",
                 key=f"{bin_type}{ncomponents}comp")
dt = time() - t
print(f"Total time elapsed to read original DataFrame: {dt:.2f} seconds")
# 15.94 seconds

# # Time how long it takes to compute stuff
# About ~30 seconds for the full DataFrame
# t = time()
# ncomponents_max = 3
# df = continuum.compute_continuum_luminosity(df)
# df = linefns.compute_eline_luminosity(df, ncomponents_max, eline_list=["HALPHA"])
# df = linefns.compute_FWHM(df, ncomponents_max)
# df = misc.compute_gas_stellar_offsets(df, ncomponents_max)
# df = misc.compute_log_columns(df, ncomponents_max)
# df = misc.compute_component_offsets(df, ncomponents_max)
# dt = time() - t
# print(f"Total time elapsed: {dt:.2f} seconds")

# These are columns that get added by the functions above
new_cols = [
    "HALPHA continuum luminosity",
    "HALPHA continuum luminosity error",
    "HALPHA luminosity (total)",
    "HALPHA luminosity error (total)",
    "HALPHA luminosity (component 1)",
    "HALPHA luminosity error (component 1)",
    "HALPHA luminosity (component 2)",
    "HALPHA luminosity error (component 2)",
    "HALPHA luminosity (component 3)",
    "HALPHA luminosity error (component 3)",
    "FWHM_gas (component 1)",
    "FWHM_gas error (component 1)",
    "FWHM_gas (component 2)",
    "FWHM_gas error (component 2)",
    "FWHM_gas (component 3)",
    "FWHM_gas error (component 3)",
    "v_gas - v_* (component 1)",
    "v_gas - v_* error (component 1)",
    "sigma_gas - sigma_* (component 1)",
    "sigma_gas^2 - sigma_*^2 (component 1)",
    "sigma_gas/sigma_* (component 1)",
    "sigma_gas - sigma_* error (component 1)",
    "sigma_gas^2 - sigma_*^2 error (component 1)",
    "sigma_gas/sigma_* error (component 1)",
    "v_gas - v_* (component 2)",
    "v_gas - v_* error (component 2)",
    "sigma_gas - sigma_* (component 2)",
    "sigma_gas^2 - sigma_*^2 (component 2)",
    "sigma_gas/sigma_* (component 2)",
    "sigma_gas - sigma_* error (component 2)",
    "sigma_gas^2 - sigma_*^2 error (component 2)",
    "sigma_gas/sigma_* error (component 2)",
    "v_gas - v_* (component 3)",
    "v_gas - v_* error (component 3)",
    "sigma_gas - sigma_* (component 3)",
    "sigma_gas^2 - sigma_*^2 (component 3)",
    "sigma_gas/sigma_* (component 3)",
    "sigma_gas - sigma_* error (component 3)",
    "sigma_gas^2 - sigma_*^2 error (component 3)",
    "sigma_gas/sigma_* error (component 3)",
    "log HALPHA luminosity (total)",
    "log HALPHA luminosity error (lower) (total)",
    "log HALPHA luminosity error (upper) (total)",
    "log HALPHA luminosity (component 1)",
    "log HALPHA luminosity error (lower) (component 1)",
    "log HALPHA luminosity error (upper) (component 1)",
    "log HALPHA luminosity (component 2)",
    "log HALPHA luminosity error (lower) (component 2)",
    "log HALPHA luminosity error (upper) (component 2)",
    "log HALPHA luminosity (component 3)",
    "log HALPHA luminosity error (lower) (component 3)",
    "log HALPHA luminosity error (upper) (component 3)",
    "log HALPHA EW (total)",
    "log HALPHA EW error (lower) (total)",
    "log HALPHA EW error (upper) (total)",
    "log HALPHA EW (component 1)",
    "log HALPHA EW error (lower) (component 1)",
    "log HALPHA EW error (upper) (component 1)",
    "log HALPHA EW (component 2)",
    "log HALPHA EW error (lower) (component 2)",
    "log HALPHA EW error (upper) (component 2)",
    "log HALPHA EW (component 3)",
    "log HALPHA EW error (lower) (component 3)",
    "log HALPHA EW error (upper) (component 3)",
    "log sigma_gas (component 1)",
    "log sigma_gas error (lower) (component 1)",
    "log sigma_gas error (upper) (component 1)",
    "log sigma_gas (component 2)",
    "log sigma_gas error (lower) (component 2)",
    "log sigma_gas error (upper) (component 2)",
    "log sigma_gas (component 3)",
    "log sigma_gas error (lower) (component 3)",
    "log sigma_gas error (upper) (component 3)",
    "log S2 ratio (total)",
    "log S2 ratio error (lower) (total)",
    "log S2 ratio error (upper) (total)",
    "log SFR (total)",
    "log SFR error (lower) (total)",
    "log SFR error (upper) (total)",
    "log SFR (component 1)",
    "log SFR error (lower) (component 1)",
    "log SFR error (upper) (component 1)",
    "log SFR (component 2)",
    "log SFR error (lower) (component 2)",
    "log SFR error (upper) (component 2)",
    "log SFR (component 3)",
    "log SFR error (lower) (component 3)",
    "log SFR error (upper) (component 3)",
    "log SFR surface density (total)",
    "log SFR surface density error (lower) (total)",
    "log SFR surface density error (upper) (total)",
    "log SFR surface density (component 1)",
    "log SFR surface density error (lower) (component 1)",
    "log SFR surface density error (upper) (component 1)",
    "log SFR surface density (component 2)",
    "log SFR surface density error (lower) (component 2)",
    "log SFR surface density error (upper) (component 2)",
    "log SFR surface density (component 3)",
    "log SFR surface density error (lower) (component 3)",
    "log SFR surface density error (upper) (component 3)",
    "delta sigma_gas (2/1)",
    "delta sigma_gas error (2/1)",
    "delta v_gas (2/1)",
    "delta v_gas error (2/1)",
    "HALPHA EW ratio (2/1)",
    "HALPHA EW ratio error (2/1)",
    "Delta HALPHA EW (2/1)",
    "delta sigma_gas (3/1)",
    "delta sigma_gas error (3/1)",
    "delta v_gas (3/1)",
    "delta v_gas error (3/1)",
    "HALPHA EW ratio (3/1)",
    "HALPHA EW ratio error (3/1)",
    "Delta HALPHA EW (3/1)",
    "delta sigma_gas (3/2)",
    "delta sigma_gas error (3/2)",
    "delta v_gas (3/2)",
    "delta v_gas error (3/2)",
    "HALPHA EW ratio (3/2)",
    "HALPHA EW ratio error (3/2)",
    "Delta HALPHA EW (3/2)",
    "HALPHA EW/HALPHA EW (total) (component 1)",
    "HALPHA EW/HALPHA EW (total) (component 2)",
    "HALPHA EW/HALPHA EW (total) (component 3)",
]

print("Saving test DataFrames to file...")
# Disk space: 2.4 GB
df_simple_cols = df[new_cols].copy()
df_simple_cols.to_hdf(settings["sami"]["output_path"] + "cols_simple_test.hdf", key="simple")

# Disk space: 5.7 GB
df_complex_cols = df[[c for c in df if c not in new_cols]].copy()
df_complex_cols.to_hdf(settings["sami"]["output_path"] + "cols_complex_test.hdf", key="complex")

# How long does it take to load?
t = time()
df_complex_cols = pd.read_hdf(settings["sami"]["output_path"] + "cols_complex_test.hdf", key="complex")
dt = time() - t
print(f"Total time elapsed to read smaller DataFrame: {dt:.2f} seconds")
# 15.75 seconds - not much time saved!! 

# Now, see how much disk space is occupied by calc

# Compare to full DataFrame: 7.5 GB
