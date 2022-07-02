import os
import numpy as np
import pandas as pd
from astropy.io import fits
from cosmocalc import get_dist

from IPython.core.debugger import Tracer

"""
This script is used to create a DataFrame containing "metadata", including
stellar masses, spectroscopic redshifts, morphologies and other information
for each galaxy in SAMI. In addition to the provided values in the input
catalogues, the angular scale (in kpc per arcsecond) and inclination are 
computed for each galaxy.

This script must be run before make_df_sami.py, as the resulting DataFrame
is used there.

The information used here is from the catalogues are available at 
https://datacentral.org.au/. 

The DataFrame is saved to "SAMI_DIR/sami_dr3_metadata.hd5".

"""

###############################################################################
# Paths
sami_data_path = os.environ["SAMI_DIR"]
assert "SAMI_DIR" in os.environ, "Environment variable SAMI_DIR is not defined!"

###############################################################################
# Filenames
df_fname = f"sami_dr3_metadata.hd5"

# .csv files containing metadata 
# These correspond to the tables of the same name at 
# https://datacentral.org.au/ 
gama_metadata_fname = "sami_InputCatGAMADR3.csv"
cluster_metadata_fname = "sami_InputCatClustersDR3.csv"
filler_metadata_fname = "sami_InputCatFiller.csv"
morphologies_fname = "sami_VisualMorphologyDR3.csv"
flag_metadata_fname = "sami_CubeObs.csv"

###############################################################################
# Read in galaxy metadata
###############################################################################
df_metadata_gama = pd.read_csv(os.path.join("../data/", gama_metadata_fname))  # ALL possible GAMA targets
df_metadata_cluster = pd.read_csv(os.path.join("../data/", cluster_metadata_fname))  # ALL possible cluster targets
df_metadata_filler = pd.read_csv(os.path.join("../data/", filler_metadata_fname))  # ALL possible filler targets
df_metadata = pd.concat([df_metadata_gama, df_metadata_cluster, df_metadata_filler]).drop(["Unnamed: 0"], axis=1)

gal_ids_metadata = list(np.sort(list(df_metadata["catid"])))

###############################################################################
# Append morphology data
###############################################################################
df_morphologies = pd.read_csv(os.path.join("../data/", morphologies_fname)).drop(["Unnamed: 0"], axis=1)
df_morphologies = df_morphologies.rename(columns={"type": "Morphology (numeric)"})

# Morphologies (numeric) - merge "?" and "no agreement" into a single category.
df_morphologies.loc[df_morphologies["Morphology (numeric)"] == 5.0, "Morphology (numeric)"] = -0.5
df_morphologies.loc[df_morphologies["Morphology (numeric)"] == -9.0, "Morphology (numeric)"] = -0.5
df_morphologies.loc[df_morphologies["Morphology (numeric)"] == np.nan, "Morphology (numeric)"] = -0.5

# Key: Morphological Type
morph_dict = {
    "0.0": "E",
    "0.5": "E/S0",
    "1.0": "S0",
    "1.5": "S0/Early-spiral",
    "2.0": "Early-spiral",
    "2.5": "Early/Late spiral",
    "3.0": "Late spiral",
    "5.0": "?",
    "-9.0": "no agreement",
    "-0.5": "Unknown"
}
df_morphologies["Morphology"] = [morph_dict[str(m)] for m in df_morphologies["Morphology (numeric)"]]

# merge with metadata, but do NOT include the morphology column as it 
# causes all data to be cast to "object" type which is extremely slow!!!
df_metadata = df_metadata.merge(df_morphologies[["catid", "Morphology (numeric)", "Morphology"]], on="catid")

###############################################################################
# Read in flag metadata
###############################################################################
df_flags = pd.read_csv(os.path.join("../data/", flag_metadata_fname)).drop(["Unnamed: 0"], axis=1)
df_flags = df_flags.astype({col: "int64" for col in df_flags.columns if col.startswith("warn")})
df_flags = df_flags.astype({"isbest": bool})

# Get rid of rows failing the following data quality criteria
cond = df_flags["isbest"] == True
cond &= df_flags["warnstar"] == 0
# cond &= df_flags["warnz"] == 0
cond &= df_flags["warnmult"] < 2  # multiple objects overlapping with galaxy area
# cond &= df_flags["warnare"] == 0  # absent Re aperture spectra
cond &= df_flags["warnfcal"] == 0  # flux calibration issues
cond &= df_flags["warnfcbr"] == 0  # flux calibration issues
cond &= df_flags["warnskyb"] == 0  # bad sky subtraction residuals
cond &= df_flags["warnskyr"] == 0  # bad sky subtraction residuals
cond &= df_flags["warnre"] == 0  # significant difference between standard & MGE Re
df_flags_cut = df_flags[cond]

for gal in df_flags_cut.catid:
    if df_flags_cut[df_flags_cut.catid == gal].shape[0] > 1:
        # If there are two "best" observations, drop the second one.
        drop_idxs = df_flags_cut.index[df_flags_cut.catid == gal][1:]
        df_flags_cut = df_flags_cut.drop(drop_idxs)

if df_flags_cut.shape[0] != len(df_flags_cut.catid.unique()):
    Tracer()() 

# Convert to int
df_metadata["catid"] = df_metadata["catid"].astype(int) 
df_flags_cut["catid"] = df_flags_cut["catid"].astype(int)
gal_ids_dq_cut = list(df_flags_cut["catid"])

# Remove 9008500001 since it's a duplicate!
gal_ids_dq_cut.pop(gal_ids_dq_cut.index(9008500001))

# Add DQ cut column
df_metadata["Good?"] = False
df_metadata.loc[df_metadata["catid"].isin(gal_ids_dq_cut), "Good?"] = True

# Reset index
df_metadata = df_metadata.set_index(df_metadata["catid"])

###############################################################################
# Add angular scale info
###############################################################################
for gal in gal_ids_dq_cut:
    D_A_Mpc, D_L_Mpc = get_dist(z=df_metadata.loc[gal, "z_spec"])
    df_metadata.loc[gal, "D_A (Mpc)"] = D_A_Mpc
    df_metadata.loc[gal, "D_L (Mpc)"] = D_L_Mpc
df_metadata["kpc per arcsec"] = df_metadata["D_A (Mpc)"] * 1e3 * np.pi / 180.0 / 3600.0
df_metadata["R_e (kpc)"] = df_metadata["r_e"] * df_metadata["kpc per arcsec"]
df_metadata["log(M/R_e)"] = df_metadata["mstar"] - np.log10(df_metadata["R_e (kpc)"])

###############################################################################
# Save to file
###############################################################################
df_metadata.to_hdf(os.path.join(sami_data_path, df_fname), key="metadata")
