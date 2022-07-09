# Imports
import sys
import os 
import numpy as np
import pandas as pd

from spaxelsleuth.loaddata.sami import load_sami_galaxies

from IPython.core.debugger import Tracer

###########################################################################
# Paths
assert "SAMI_DIR" in os.environ, "Environment variable SAMI_DIR is not defined!"
sami_data_path = os.environ["SAMI_DIR"]

###########################################################################
# Options
ncomponents, bin_type, eline_SNR_min = [sys.argv[1], sys.argv[2], int(sys.argv[3])]

###########################################################################
# Load the data
###########################################################################
# Load the ubinned data 
df = load_sami_galaxies(ncomponents=ncomponents,
                        bin_type=bin_type,
                        eline_SNR_min=eline_SNR_min,
                        correct_extinction=True,
                        debug=True)

###########################################################################
# Assertion checks
###########################################################################
# CHECK: stellar kinematics have been masked out
cond_bad_stekin = df["Bad stellar kinematics"]
for col in [c for c in df.columns if "*" in c]:
    assert all(df.loc[cond_bad_stekin, col].isna())

# CHECK: sigma_gas S/N has worked 
for ii in range(3 if ncomponents == "recom" else 1):
    assert all(df.loc[df[f"sigma_obs S/N (component {ii})"] < 3, f"Low sigma_gas S/N flag (component {ii})"])
    cond_bad_sigma = df[f"Low sigma_gas S/N flag (component {ii})"]
    for col in [c for c in df.columns if "sigma_gas" in c and f"component {ii}" in c and "flag" not in c]:
        assert all(df.loc[cond_bad_sigma, col].isna())

# CHECK: line amplitudes 
for ii in range(3 if ncomponents == "recom" else 1):
    assert all(df.loc[df[f"HALPHA A (component {ii})"] < 3 * df[f"HALPHA continuum std. dev."], f"Low amplitude flag - HALPHA (component {ii})"])
    cond_low_amp = df[f"Low amplitude flag - HALPHA (component {ii})"]
    for col in [c for c in df.columns if f"component {ii}" in c and "flag" not in c]:
        assert all(df.loc[cond_low_amp, col].isna())

# CHECK: flux S/N
for ii in range(3 if ncomponents == "recom" else 1):
    assert all(df.loc[df[f"Low flux S/N flag - HALPHA (component {ii})"], f"HALPHA (component {ii})"].isna())
for eline in ["HALPHA", "HBETA", "NII6583", "OIII5007", "SII6716", "SII6731", "OII3726+OII3729", "OI6300"]:
    assert all(df.loc[df[f"Low flux S/N flag - {eline} (total)"], f"{eline} (total)"].isna())

# CHECK: BPT categories 
for eline in ["HALPHA", "HBETA", "NII6583", "OIII5007"]:
    cond_has_no_line = df[f"{eline} (total)"].isna()
    assert all(df.loc[cond_has_no_line, "BPT (total)"] == "Not classified")

# CHECK: "Number of components" is NaN, 0, 1, 2 or 3
if ncomponents == "recom":
    assert np.all(df.loc[~df["Number of components"].isna(), "Number of components"].unique() == [0, 1, 2, 3])
elif ncomponents == "1":
    assert np.all(df.loc[~df["Number of components"].isna(), "Number of components"].unique() == [0, 1])

# CHECK: all spaxels in which the original number of components does NOT match the "high-quality" number of components have the flag set 
cond_missing_components = df["Missing components flag"]
assert all(df.loc[cond_missing_components, "Number of components (original)"] != df.loc[cond_missing_components, "Number of components"])

# CHECK: all spaxels with original number of compnents == 0 have zero emission line fluxes 
for eline in ["HALPHA", "HBETA", "NII6583", "OI6300", "OII3726+OII3729", "OIII5007", "SII6716", "SII6731"]:
    assert not any(df.loc[df["Number of components (original)"] == 0, f"{eline} (total)"] > 0)
    assert not any(df.loc[df["Number of components (original)"] == 0, f"{eline} error (total)"] > 0)
    assert np.all(df.loc[df["Number of components (original)"] == 0, f"{eline} (total)"].isna() | (df.loc[df["Number of components (original)"] == 0, f"{eline} (total)"] == 0))
    assert np.all(df.loc[df["Number of components (original)"] == 0, f"{eline} error (total)"].isna() | (df.loc[df["Number of components (original)"] == 0, f"{eline} error (total)"] == 0))

# CHECK: all spaxels with Number of components == 0 but Number of components (original) == 1 have either sigma_gas or HALPHA masked out
cond = df["Number of components"] == 0
cond &= df["Number of components (original)"] == 1
assert all(df.loc[cond, "sigma_gas (component 0)"].isna() | df.loc[cond, "HALPHA (component 0)"].isna())

# CHECK: all fluxes (and EWs) that are NaN have NaN errors
for eline in ["HALPHA", "HBETA", "NII6583", "OI6300", "OII3726+OII3729", "OIII5007", "SII6716", "SII6731"]:
    assert all(df.loc[df[f"{eline} (total)"].isna(), f"{eline} error (total)"].isna())
for ii in range(3 if ncomponents == "recom" else 1):
    assert all(df.loc[df[f"HALPHA (component {ii})"].isna(), f"HALPHA error (component {ii})"].isna())
    assert all(df.loc[df[f"HALPHA EW (component {ii})"].isna(), f"HALPHA EW error (component {ii})"].isna())
    assert all(df.loc[df[f"log HALPHA EW (component {ii})"].isna(), f"log HALPHA EW error (upper) (component {ii})"].isna())
    assert all(df.loc[df[f"log HALPHA EW (component {ii})"].isna(), f"log HALPHA EW error (lower) (component {ii})"].isna())

# CHECK: all HALPHA-derived columns in spaxels with 0 components are NaN
col = "HALPHA EW"
assert np.all(np.isnan(df.loc[df["Number of components (original)"] == 0, f"{col} (total)"]))
assert np.all(np.isnan(df.loc[df["Number of components (original)"] == 0, f"{col} error (total)"]))
for ii in range(3 if ncomponents == "recom" else 1):
    assert np.all(np.isnan(df.loc[df["Number of components (original)"] == 0, f"{col} (component {ii})"]))
    assert np.all(np.isnan(df.loc[df["Number of components (original)"] == 0, f"{col} error (component {ii})"]))

# CHECK: all HALPHA-derived columns in spaxels with 0 components are NaN
col = "log HALPHA EW"
assert np.all(np.isnan(df.loc[df["Number of components (original)"] == 0, f"{col} (total)"]))
assert np.all(np.isnan(df.loc[df["Number of components (original)"] == 0, f"{col} error (upper) (total)"]))
assert np.all(np.isnan(df.loc[df["Number of components (original)"] == 0, f"{col} error (lower) (total)"]))
for ii in range(3 if ncomponents == "recom" else 1):
    assert np.all(np.isnan(df.loc[df["Number of components (original)"] == 0, f"{col} (component {ii})"]))
    assert np.all(np.isnan(df.loc[df["Number of components (original)"] == 0, f"{col} error (upper) (component {ii})"]))
    assert np.all(np.isnan(df.loc[df["Number of components (original)"] == 0, f"{col} error (lower) (component {ii})"]))

# CHECK: all kinematic quantities in spaxels with 0 original components are NaN
for col in ["sigma_gas", "v_gas"]:
    for ii in range(3 if ncomponents == "recom" else 1):
        assert np.all(np.isnan(df.loc[df["Number of components (original)"] == 0, f"{col} (component {ii})"]))
        assert np.all(np.isnan(df.loc[df["Number of components (original)"] == 0, f"{col} error (component {ii})"]))

# CHECK: all emission line fluxes with S/N < SNR_min are NaN
for eline in ["HALPHA", "HBETA", "NII6583", "OI6300", "OII3726+OII3729", "OIII5007", "SII6716", "SII6731"]:
    assert np.all(np.isnan(df.loc[df[f"{eline} S/N (total)"] < eline_SNR_min, f"{eline} (total)"]))
    assert np.all(np.isnan(df.loc[df[f"{eline} S/N (total)"] < eline_SNR_min, f"{eline} error (total)"]))

# CHECK: all Halpha components below S/N limit are NaN
for ii in range(3 if ncomponents == "recom" else 1):
    assert np.all(np.isnan(df.loc[df[f"HALPHA S/N (component {ii})"] < eline_SNR_min, f"HALPHA (component {ii})"]))
    assert np.all(np.isnan(df.loc[df[f"HALPHA S/N (component {ii})"] < eline_SNR_min, f"HALPHA error (component {ii})"]))
    assert np.all(np.isnan(df.loc[df[f"HALPHA S/N (component {ii})"] < eline_SNR_min, f"HALPHA EW (component {ii})"]))
    assert np.all(np.isnan(df.loc[df[f"HALPHA S/N (component {ii})"] < eline_SNR_min, f"HALPHA EW error (component {ii})"]))
    assert np.all(np.isnan(df.loc[df[f"HALPHA S/N (component {ii})"] < eline_SNR_min, f"log HALPHA EW (component {ii})"]))
    assert np.all(np.isnan(df.loc[df[f"HALPHA S/N (component {ii})"] < eline_SNR_min, f"log HALPHA EW error (upper) (component {ii})"]))
    assert np.all(np.isnan(df.loc[df[f"HALPHA S/N (component {ii})"] < eline_SNR_min, f"log HALPHA EW error (lower) (component {ii})"]))

# CHECK: all sigma_gas components with S/N < S/N target are NaN
for ii in range(3 if ncomponents == "recom" else 1):
    assert np.all(np.isnan(df.loc[df[f"sigma_obs S/N (component {ii})"] < df[f"sigma_obs target S/N (component {ii})"], f"sigma_gas (component {ii})"]))
    assert np.all(np.isnan(df.loc[df[f"sigma_obs S/N (component {ii})"] < df[f"sigma_obs target S/N (component {ii})"], f"sigma_gas error (component {ii})"]))

# CHECK: stellar kinematics
assert np.all(df.loc[~np.isnan(df["sigma_*"]), "sigma_*"] > 35)
assert np.all(df.loc[~np.isnan(df["v_* error"]), "v_* error"] < 30)
assert np.all(df.loc[~np.isnan(df["sigma_* error"]), "sigma_* error"] < df.loc[~np.isnan(df["sigma_* error"]), "sigma_*"] * 0.1 + 25)

# CHECK: SFR/SFR surface density columns exist 
for col in ["SFR (total)", "SFR (component 0)", "SFR surface density (total)", "SFR surface density (component 0)"]:
    assert f"{col}" in df.columns
    assert f"log {col}" in df.columns
    assert all(df.loc[df["HALPHA (total)"].isna(), f"{col}"].isna())
    assert all(df.loc[df["HALPHA (total)"].isna(), f"log {col}"].isna())

# CHECK: number of components has been set properly
assert all(~df[df["Number of components"] > df["Number of components (original)"]])

# CHECK: low S/N components have been masked out 
for eline in ["HALPHA"]:
    for ii in range(3 if ncomponents == "recom" else 1):
        assert all(df.loc[df[f"Low flux S/N flag - {eline} (component {ii})"], f"{eline} (component {ii})"].isna())
        assert all(df.loc[df[f"Low flux S/N flag - {eline} (component {ii})"], f"{eline} error (component {ii})"].isna())
        if eline == "HALPHA":
            assert all(df.loc[df[f"Low flux S/N flag - {eline} (component {ii})"], f"v_gas (component {ii})"].isna())
            assert all(df.loc[df[f"Low flux S/N flag - {eline} (component {ii})"], f"v_gas error (component {ii})"].isna())
            assert all(df.loc[df[f"Low flux S/N flag - {eline} (component {ii})"], f"sigma_gas (component {ii})"].isna())
            assert all(df.loc[df[f"Low flux S/N flag - {eline} (component {ii})"], f"sigma_gas error (component {ii})"].isna())

# CHECK: has HALPHA EW been NaNd out propertly?
for ii in range(3 if ncomponents == "recom" else 1):
    assert all(df.loc[df[f"HALPHA (component {ii})"].isna(), f"HALPHA EW (component {ii})"].isna())
    assert all(df.loc[df[f"HALPHA (component {ii})"].isna(), f"HALPHA EW error (component {ii})"].isna())
    assert all(df.loc[df[f"HALPHA continuum"].isna() | (df[f"HALPHA continuum"] <= 0), f"HALPHA EW (component {ii})"].isna())
    assert all(df.loc[df[f"HALPHA continuum"].isna() | (df[f"HALPHA continuum"] <= 0), f"HALPHA EW error (component {ii})"].isna())
assert all(df.loc[df[f"HALPHA (total)"].isna(), f"HALPHA EW (total)"].isna())
assert all(df.loc[df[f"HALPHA continuum"].isna() | (df[f"HALPHA continuum"] <= 0), f"HALPHA EW (total)"].isna())
assert all(df.loc[df[f"HALPHA continuum"].isna() | (df[f"HALPHA continuum"] <= 0), f"HALPHA EW error (total)"].isna())
