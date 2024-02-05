"""
Export FITS files for the Hector busy week.
"""
import sys

from spaxelsleuth import load_user_config
try:
    load_user_config("/Users/u5708159/Desktop/spaxelsleuth_test/.myconfig.json")
except FileNotFoundError:
    load_user_config("/home/u5708159/.spaxelsleuthconfig.json")
from spaxelsleuth.config import settings

from spaxelsleuth.plotting.plot2dmap import plot2dmap
from spaxelsleuth.utils.exportfits import export_fits

# List of columns to export 
cols_to_store_no_suffixes = []

# Emission line kinematics
cols_to_store_no_suffixes += [
    "Number of components",
    "Number of components (original)",
    "v_gas",
    "v_gas error",
    "sigma_gas",
    "sigma_gas error",
    "v_grad",
    "Missing components flag",
    "Missing v_gas flag",
    "Missing sigma_gas flag",
    "Low sigma_gas S/N flag",
    "Beam smearing flag",
]

# Emission line data products 
eline_list = settings["hector"]["eline_list"]
cols_to_store_no_suffixes += eline_list
cols_to_store_no_suffixes += [f"{eline} error" for eline in eline_list]
cols_to_store_no_suffixes += [f"{eline} S/N" for eline in eline_list]
cols_to_store_no_suffixes += [f"Low flux S/N flag - {eline}" for eline in eline_list]
cols_to_store_no_suffixes += [f"Low amplitude flag - {eline}" for eline in eline_list]
cols_to_store_no_suffixes += [f"Missing flux flag - {eline}" for eline in eline_list]
cols_to_store_no_suffixes += [
    "HALPHA A/N (measured)",
    "HALPHA EW",
    "HALPHA EW error",
]
cols_to_store_no_suffixes += [
    "Balmer decrement",
    "Balmer decrement error",
    "A_V",
    "A_V error",
    "log N2",
    "log N2 error (lower)",
    "log N2 error (upper)",
    "log O1",
    "log O1 error (lower)",
    "log O1 error (upper)",
    "log S2",
    "log S2 error (lower)",
    "log S2 error (upper)",
    "log O3",
    "log O3 error (lower)",
    "log O3 error (upper)",
    "BPT (numeric)",
    "[SII] ratio error",
    "log [SII] ratio",
    "log [SII] ratio error (lower)",
    "log [SII] ratio error (upper)",
    "SFR",
    "SFR error",
    # NOTE: the below must be in the DataFrame
    "log(O/H) + 12 (N2Ha_K19/O3O2_K19)",
    "log(O/H) + 12 (N2Ha_K19/O3O2_K19) error (lower)",
    "log(O/H) + 12 (N2Ha_K19/O3O2_K19) error (upper)",
    "log(U) (N2Ha_K19/O3O2_K19)",
    "log(U) (N2Ha_K19/O3O2_K19) error (lower)",
    "log(U) (N2Ha_K19/O3O2_K19) error (upper)",
]

# Stellar kinematics 
cols_to_store_no_suffixes += [
    "v_*",
    "v_* error",
    "sigma_*",
    "sigma_* error",
    "chi2 (ppxf)",
    "Median continuum S/N (ppxf)",
    "Bad stellar kinematics",
    "Missing v_* flag",
    "Missing sigma_* flag",
]

# Continuum data products 
cols_to_store_no_suffixes += [
    "D4000",
    "D4000 error",
    "HALPHA continuum",
    "HALPHA continuum std. dev.",
    "HALPHA continuum error",
    "B-band continuum",
    "B-band continuum std. dev.",
    "B-band continuum error",
    "Median spectral value (blue)",
    "Median spectral value (red)",
]

export_fits(survey="hector",
            cols_to_store_no_suffixes=cols_to_store_no_suffixes,
            ncomponents="rec", 
            eline_SNR_min=5, 
            eline_ANR_min=3, 
            correct_extinction=True,
            df_fname_tag="nocuts",
            fname_suffix="nocuts"
)

export_fits(survey="hector",
            cols_to_store_no_suffixes=cols_to_store_no_suffixes,
            ncomponents="rec", 
            eline_SNR_min=5, 
            eline_ANR_min=3, 
            correct_extinction=True,
            df_fname_tag="cuts",
            fname_suffix="cuts"
)