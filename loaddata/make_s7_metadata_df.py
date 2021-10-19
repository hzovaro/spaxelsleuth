import os
import numpy as np
import pandas as pd
from astropy.io import fits
from cosmocalc import get_dist

from astropy import units as u
from astropy.coordinates import SkyCoord

from IPython.core.debugger import Tracer

###############################################################################
# Paths
s7_data_path = "/priv/meggs3/u5708159/S7/"

###############################################################################
# Filenames
df_metadata_fname = "s7_metadata.csv"
df_fname = "s7_metadata.hd5"

###############################################################################
# READ IN THE METADATA
###############################################################################
df_metadata = pd.read_csv(os.path.join(s7_data_path, df_metadata_fname), skiprows=58)
gals = df_metadata["S7_Name"].values

###############################################################################
# Convert object coordinates to degrees
###############################################################################
coords = SkyCoord(df_metadata["RA_hms"], df_metadata["Dec_sxgsml"],
             unit=(u.hourangle, u.deg))
df_metadata["ra_obj"] = coords.ra.deg
df_metadata["dec_obj"] = coords.dec.deg

###############################################################################
# Rename columns
###############################################################################
rename_dict = {
    "S7_Name": "catid",
    "HL_inclination": "Inclination i (degrees)",
    "HL_Re": "R_e (arcsec)",
    "HL_Re_err": "R_e error (arcsec)",
    "NED_ax_ratio": "b/a",
    "NED_ax_ratio_err": "b/a error",
    "HL_PA": "pa",
    "S7_best_WiFeS_PA": "WiFeS PA",
    "S7_Mstar": "mstar",
    "S7_Mstar_err": "mstar error",
    "S7_Sy1_subtraction?": "Sy1 subtraction?",
    "S7_mosaic?": "Mosaic?",
    "S7_BPT_classification": "BPT (global)",
    "S7_z": "z_spec",
    "S7_nucleus_index_x": "x0 (pixels)",
    "S7_nucleus_index_y": "y0 (pixels)"
}
df_metadata = df_metadata.rename(columns=rename_dict)
df_metadata = df_metadata.set_index(df_metadata["catid"])

# Get rid of unneeded columns
good_cols = [rename_dict[k] for k in rename_dict.keys()] + ["ra_obj", "dec_obj"]
df_metadata = df_metadata[good_cols]

###############################################################################
# Add angular scale info
###############################################################################
for gal in gals:
    D_A_Mpc, D_L_Mpc = get_dist(z=df_metadata.loc[gal, "z_spec"])
    df_metadata.loc[gal, "D_A (Mpc)"] = D_A_Mpc
    df_metadata.loc[gal, "D_L (Mpc)"] = D_L_Mpc
df_metadata["kpc per arcsec"] = df_metadata["D_A (Mpc)"] * 1e3 * np.pi / 180.0 / 3600.0

###############################################################################
# Define a "Good?" column
###############################################################################
df_metadata["Sy1 subtraction?"] = [True if x == "Y" else False for x in df_metadata["Sy1 subtraction?"].values]
df_metadata["Good?"] = ~df_metadata["Sy1 subtraction?"].values

###############################################################################
# Save to file
###############################################################################
df_metadata.to_hdf(os.path.join(s7_data_path, df_fname), key="metadata")
