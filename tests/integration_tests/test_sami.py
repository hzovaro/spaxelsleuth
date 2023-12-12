import numpy as np
from pathlib import Path
import pkgutil
import sys

from spaxelsleuth import load_user_config, configure_logger

# load_user_config(Path(__file__.rpartition("/")[0]) / "test_config.json")
load_user_config("test_config.json")
configure_logger(level="DEBUG")
from spaxelsleuth.loaddata.sami import make_sami_metadata_df, make_sami_df, load_sami_df

import logging
logger = logging.getLogger(__name__)

def test_make_sami_metadata_df():
    # Create the metadata DataFrame
    make_sami_metadata_df(recompute_continuum_SNRs=True, nthreads=10)

def test_sami():
    NCOMPONENTS="recom", 
    BIN_TYPE="default", 
    ELINE_SNR_MIN=5, 
    ELINE_ANR_MIN=3, 
    NTHREADS=10, 
    DEBUG=False, 

    kwargs = {
        "ncomponents": NCOMPONENTS,
        "bin_type": BIN_TYPE,
        "eline_SNR_min": ELINE_SNR_MIN,
        "eline_ANR_min": ELINE_ANR_MIN,
        "debug": DEBUG,
    }

    # Create the DataFrame
    make_sami_df(**kwargs, correct_extinction=True, nthreads=NTHREADS)  
    make_sami_df(**kwargs, correct_extinction=False, nthreads=NTHREADS)  
    
    # Load the DataFrame
    df = load_sami_df(**kwargs, correct_extinction=True)
    df_noextcorr = load_sami_df(**kwargs, correct_extinction=False)

    #//////////////////////////////////////////////////////////////////////////////
    # Run assertion tests
    # CHECK: SFR/SFR surface density columns exist 
    for col in [c for c in df.columns if "SFR" in c and "error" not in c]:
        assert f"{col}" in df.columns
        assert all(df.loc[df["HALPHA (total)"].isna(), f"{col}"].isna())

    # CHECK: BPT categories 
    for eline in ["HALPHA", "HBETA", "NII6583", "OIII5007"]:
        cond_has_no_line = df[f"{eline} (total)"].isna()
        assert all(df.loc[cond_has_no_line, "BPT (total)"] == "Not classified")

    # CHECK: "Number of components" is NaN, 0, 1, 2 or 3
    if NCOMPONENTS == "recom":
        assert np.all(df.loc[~df["Number of components"].isna(), "Number of components"].unique() == [0, 1, 2, 3])
    elif NCOMPONENTS == "1":
        assert np.all(df.loc[~df["Number of components"].isna(), "Number of components"].unique() == [0, 1])

    # CHECK: all spaxels with original number of compnents == 0 have zero emission line fluxes 
    for eline in ["HALPHA", "HBETA", "NII6583", "OI6300", "OII3726+OII3729", "OIII5007", "SII6716", "SII6731"]:
        assert not any(df.loc[df["Number of components (original)"] == 0, f"{eline} (total)"] > 0)
        assert not any(df.loc[df["Number of components (original)"] == 0, f"{eline} error (total)"] > 0)
        assert np.all(df.loc[df["Number of components (original)"] == 0, f"{eline} (total)"].isna() | (df.loc[df["Number of components (original)"] == 0, f"{eline} (total)"] == 0))
        assert np.all(df.loc[df["Number of components (original)"] == 0, f"{eline} error (total)"].isna() | (df.loc[df["Number of components (original)"] == 0, f"{eline} error (total)"] == 0))

    #//////////////////////////////////////////////////////////////////////////////
    # DATA QUALITY AND S/N CUT TESTS
    # CHECK: stellar kinematics have been masked out
    cond_bad_stekin = df["Bad stellar kinematics"]
    for col in [c for c in df.columns if "v_*" in c or "sigma_*" in c]:
        assert all(df.loc[cond_bad_stekin, col].isna())

    # CHECK: sigma_gas S/N cut
    for nn in range(3 if NCOMPONENTS == "recom" else 1):
        assert all(df.loc[df[f"sigma_obs S/N (component {nn + 1})"] < df["sigma_gas_SNR_min"], f"Low sigma_gas S/N flag (component {nn + 1})"])
        cond_bad_sigma = df[f"Low sigma_gas S/N flag (component {nn + 1})"]
        for col in [c for c in df.columns if "sigma_gas" in c and f"component {nn + 1}" in c and "flag" not in c]:
            assert all(df.loc[cond_bad_sigma, col].isna())

    # CHECK: line amplitude cut 
    for nn in range(3 if NCOMPONENTS == "recom" else 1):
        assert all(df.loc[df[f"HALPHA A (component {nn + 1})"] < df["eline_ANR_min"] * df[f"HALPHA continuum std. dev."], f"Low amplitude flag - HALPHA (component {nn + 1})"])
        cond_low_amp = df[f"Low amplitude flag - HALPHA (component {nn + 1})"]
        for col in [c for c in df.columns if f"component {nn + 1}" in c and "flag" not in c]:
            assert all(df.loc[cond_low_amp, col].isna())

    # CHECK: flux S/N cut
    for nn in range(3 if NCOMPONENTS == "recom" else 1):
        assert all(df.loc[df[f"Low flux S/N flag - HALPHA (component {nn + 1})"], f"HALPHA (component {nn + 1})"].isna())
    for eline in ["HALPHA", "HBETA", "NII6583", "OIII5007", "SII6716", "SII6731", "OII3726+OII3729", "OI6300"]:
        assert all(df.loc[df[f"Low flux S/N flag - {eline} (total)"], f"{eline} (total)"].isna())

    # CHECK: all spaxels in which the original number of components does NOT match the "high-quality" number of components have the flag set 
    cond_missing_components = df["Missing components flag"]
    assert all(df.loc[cond_missing_components, "Number of components (original)"] != df.loc[cond_missing_components, "Number of components"])

    # CHECK: all spaxels with Number of components == 0 but Number of components (original) == 1 have either sigma_gas or HALPHA masked out
    cond = df["Number of components"] == 0
    cond &= df["Number of components (original)"] == 1
    assert all(df.loc[cond, "sigma_gas (component 1)"].isna() | df.loc[cond, "HALPHA (component 1)"].isna())

    # CHECK: all fluxes (and EWs) that are NaN have NaN errors
    for eline in ["HALPHA", "HBETA", "NII6583", "OI6300", "OII3726+OII3729", "OIII5007", "SII6716", "SII6731"]:
        assert all(df.loc[df[f"{eline} (total)"].isna(), f"{eline} error (total)"].isna())
    for nn in range(3 if NCOMPONENTS == "recom" else 1):
        assert all(df.loc[df[f"HALPHA (component {nn + 1})"].isna(), f"HALPHA error (component {nn + 1})"].isna())
        assert all(df.loc[df[f"HALPHA EW (component {nn + 1})"].isna(), f"HALPHA EW error (component {nn + 1})"].isna())
        assert all(df.loc[df[f"log HALPHA EW (component {nn + 1})"].isna(), f"log HALPHA EW error (upper) (component {nn + 1})"].isna())
        assert all(df.loc[df[f"log HALPHA EW (component {nn + 1})"].isna(), f"log HALPHA EW error (lower) (component {nn + 1})"].isna())

    # CHECK: all HALPHA-derived columns in spaxels with 0 components are NaN
    col = "HALPHA EW"
    assert np.all(np.isnan(df.loc[df["Number of components (original)"] == 0, f"{col} (total)"]))
    assert np.all(np.isnan(df.loc[df["Number of components (original)"] == 0, f"{col} error (total)"]))
    for nn in range(3 if NCOMPONENTS == "recom" else 1):
        assert np.all(np.isnan(df.loc[df["Number of components (original)"] == 0, f"{col} (component {nn + 1})"]))
        assert np.all(np.isnan(df.loc[df["Number of components (original)"] == 0, f"{col} error (component {nn + 1})"]))

    # CHECK: all HALPHA-derived columns in spaxels with 0 components are NaN
    col = "log HALPHA EW"
    assert np.all(np.isnan(df.loc[df["Number of components (original)"] == 0, f"{col} (total)"]))
    assert np.all(np.isnan(df.loc[df["Number of components (original)"] == 0, f"{col} error (upper) (total)"]))
    assert np.all(np.isnan(df.loc[df["Number of components (original)"] == 0, f"{col} error (lower) (total)"]))
    for nn in range(3 if NCOMPONENTS == "recom" else 1):
        assert np.all(np.isnan(df.loc[df["Number of components (original)"] == 0, f"{col} (component {nn + 1})"]))
        assert np.all(np.isnan(df.loc[df["Number of components (original)"] == 0, f"{col} error (upper) (component {nn + 1})"]))
        assert np.all(np.isnan(df.loc[df["Number of components (original)"] == 0, f"{col} error (lower) (component {nn + 1})"]))

    # CHECK: all kinematic quantities in spaxels with 0 original components are NaN
    for col in ["sigma_gas", "v_gas"]:
        for nn in range(3 if NCOMPONENTS == "recom" else 1):
            assert np.all(np.isnan(df.loc[df["Number of components (original)"] == 0, f"{col} (component {nn + 1})"]))
            assert np.all(np.isnan(df.loc[df["Number of components (original)"] == 0, f"{col} error (component {nn + 1})"]))

    # CHECK: all emission line fluxes with S/N < SNR_min are NaN
    for eline in ["HALPHA", "HBETA", "NII6583", "OI6300", "OII3726+OII3729", "OIII5007", "SII6716", "SII6731"]:
        assert np.all(np.isnan(df.loc[df[f"{eline} S/N (total)"] < ELINE_SNR_MIN, f"{eline} (total)"]))
        assert np.all(np.isnan(df.loc[df[f"{eline} S/N (total)"] < ELINE_SNR_MIN, f"{eline} error (total)"]))

    # CHECK: all Halpha components below S/N limit are NaN
    for nn in range(3 if NCOMPONENTS == "recom" else 1):
        assert np.all(np.isnan(df.loc[df[f"HALPHA S/N (component {nn + 1})"] < ELINE_SNR_MIN, f"HALPHA (component {nn + 1})"]))
        assert np.all(np.isnan(df.loc[df[f"HALPHA S/N (component {nn + 1})"] < ELINE_SNR_MIN, f"HALPHA error (component {nn + 1})"]))
        assert np.all(np.isnan(df.loc[df[f"HALPHA S/N (component {nn + 1})"] < ELINE_SNR_MIN, f"HALPHA EW (component {nn + 1})"]))
        assert np.all(np.isnan(df.loc[df[f"HALPHA S/N (component {nn + 1})"] < ELINE_SNR_MIN, f"HALPHA EW error (component {nn + 1})"]))
        assert np.all(np.isnan(df.loc[df[f"HALPHA S/N (component {nn + 1})"] < ELINE_SNR_MIN, f"log HALPHA EW (component {nn + 1})"]))
        assert np.all(np.isnan(df.loc[df[f"HALPHA S/N (component {nn + 1})"] < ELINE_SNR_MIN, f"log HALPHA EW error (upper) (component {nn + 1})"]))
        assert np.all(np.isnan(df.loc[df[f"HALPHA S/N (component {nn + 1})"] < ELINE_SNR_MIN, f"log HALPHA EW error (lower) (component {nn + 1})"]))

    # CHECK: all sigma_gas components with S/N < S/N target are NaN
    for nn in range(3 if NCOMPONENTS == "recom" else 1):
        assert np.all(np.isnan(df.loc[df[f"sigma_obs S/N (component {nn + 1})"] < df[f"sigma_obs target S/N (component {nn + 1})"], f"sigma_gas (component {nn + 1})"]))
        assert np.all(np.isnan(df.loc[df[f"sigma_obs S/N (component {nn + 1})"] < df[f"sigma_obs target S/N (component {nn + 1})"], f"sigma_gas error (component {nn + 1})"]))

    # CHECK: stellar kinematics have been correctly masked out
    assert np.all(df.loc[~np.isnan(df["sigma_*"]), "sigma_*"] > 35)
    assert np.all(df.loc[~np.isnan(df["v_* error"]), "v_* error"] < 30)
    assert np.all(df.loc[~np.isnan(df["sigma_* error"]), "sigma_* error"] < df.loc[~np.isnan(df["sigma_* error"]), "sigma_*"] * 0.1 + 25)

    # CHECK: number of components has been set properly
    assert all(~df[df["Number of components"] > df["Number of components (original)"]])

    # CHECK: low S/N components have been masked out 
    for eline in ["HALPHA"]:
        for nn in range(3 if NCOMPONENTS == "recom" else 1):
            assert all(df.loc[df[f"Low flux S/N flag - {eline} (component {nn + 1})"], f"{eline} (component {nn + 1})"].isna())
            assert all(df.loc[df[f"Low flux S/N flag - {eline} (component {nn + 1})"], f"{eline} error (component {nn + 1})"].isna())
            if eline == "HALPHA":
                assert all(df.loc[df[f"Low flux S/N flag - {eline} (component {nn + 1})"], f"v_gas (component {nn + 1})"].isna())
                assert all(df.loc[df[f"Low flux S/N flag - {eline} (component {nn + 1})"], f"v_gas error (component {nn + 1})"].isna())
                assert all(df.loc[df[f"Low flux S/N flag - {eline} (component {nn + 1})"], f"sigma_gas (component {nn + 1})"].isna())
                assert all(df.loc[df[f"Low flux S/N flag - {eline} (component {nn + 1})"], f"sigma_gas error (component {nn + 1})"].isna())

    # CHECK: has HALPHA EW been NaNd out propertly?
    for nn in range(3 if NCOMPONENTS == "recom" else 1):
        assert all(df.loc[df[f"HALPHA (component {nn + 1})"].isna(), f"HALPHA EW (component {nn + 1})"].isna())
        assert all(df.loc[df[f"HALPHA (component {nn + 1})"].isna(), f"HALPHA EW error (component {nn + 1})"].isna())
        assert all(df.loc[df[f"HALPHA continuum"].isna() | (df[f"HALPHA continuum"] <= 0), f"HALPHA EW (component {nn + 1})"].isna())
        assert all(df.loc[df[f"HALPHA continuum"].isna() | (df[f"HALPHA continuum"] <= 0), f"HALPHA EW error (component {nn + 1})"].isna())
    assert all(df.loc[df[f"HALPHA (total)"].isna(), f"HALPHA EW (total)"].isna())
    assert all(df.loc[df[f"HALPHA continuum"].isna() | (df[f"HALPHA continuum"] <= 0), f"HALPHA EW (total)"].isna())
    assert all(df.loc[df[f"HALPHA continuum"].isna() | (df[f"HALPHA continuum"] <= 0), f"HALPHA EW error (total)"].isna())

    #//////////////////////////////////////////////////////////////////////////////
    # SFR TESTS
    # CHECK: check no spaxels with non-SF BPT classifications have SFRs or SFR surface densities
    cond_no_met = df["BPT (total)"] != "SF"
    sfr_cols = [c for c in df.columns if "SFR" in c]
    for met_col in sfr_cols:
        assert all(df.loc[cond_no_met, sfr_cols].isna())

    #//////////////////////////////////////////////////////////////////////////////
    # METALLICITY CALCULATION TESTS
    # CHECK: check no spaxels with non-SF BPT classifications have metallicities
    cond_no_met = df["BPT (total)"] != "SF"
    met_cols = [c for c in df.columns if "log(O/H) + 12" in c or "log(U)" in c]
    for met_col in met_cols:
        assert all(df.loc[cond_no_met, met_cols].isna())

    # NOTE: need to change below to use updated column names for metallicity calculations
    # # CHECK: check no spaxels with no S/N in relevant emission lines have metallicities 
    # cond_low_SN = df["OII3726+OII3729 S/N (total)"] < eline_SNR_min
    # cond_low_SN |= df["NII6583 S/N (total)"] < eline_SNR_min
    # assert all(df.loc[cond_low_SN, f"log(O/H) + 12 ({met_diagnostic}) (total)"].isna())

    # CHECK: rows with NaN in any required emission lines have NaN metallicities and ionisation parameters. 
    from spaxelsleuth.utils.metallicity import line_list_dict
    for met_diagnostic in line_list_dict.keys():
        for line in [l for l in line_list_dict[met_diagnostic] if f"{l} (total)" in df.columns]:
            cond_isnan = np.isnan(df[f"{line} (total)"])
            cols = [c for c in df.columns if met_diagnostic in c]
            for c in cols:
                assert all(df.loc[cond_isnan, c].isna())
            
    # CHECK: all rows with NaN metallicities also have NaN log(U).
    for c in [c for c in df.columns if "log(O/H) + 12" in c and "error" not in c]:
        diagnostic_str = c.split("log(O/H) + 12 (")[1].split(")")[0]
        cond_nan_logOH12 = df[c].isna()
        if f"log(U) ({diagnostic_str}) (total)" in df.columns:
            assert all(df.loc[cond_nan_logOH12, f"log(U) ({diagnostic_str}) (total)"].isna())
            # Also check the converse 
            cond_finite_logU = ~df[f"log(U) ({diagnostic_str}) (total)"].isna()
            assert all(~df.loc[cond_finite_logU, f"log(O/H) + 12 ({diagnostic_str}) (total)"].isna())

    #//////////////////////////////////////////////////////////////////////////////
    # EXTINCTION CORRECTION TESTS 
    # CHECK: make sure weird stuff hasn't happened with the indices
    assert len([c for c in df["HALPHA (total)"].index.values if c not in df_noextcorr["HALPHA (total)"].index.values]) == 0

    # CHECK: all HALPHA fluxes in the extinction-corrected DataFrame are greater than 
    # or equal to those in the non-extinction-corrected DataFrame
    eline_list = ["HALPHA", "HBETA", "NII6583", "OI6300", "OII3726+OII3729", "OIII5007", "SII6716", "SII6731"]
    for eline in eline_list:
        assert np.all(df[f"{eline} (total)"].dropna() >= df_noextcorr[f"{eline} (total)"].dropna())
        assert np.all(df[f"{eline} error (total)"].dropna() >= df_noextcorr[f"{eline} error (total)"].dropna())
    for nn in range(3 if NCOMPONENTS == "recom" else 1):
        assert np.all(df[f"HALPHA (component {nn + 1})"].dropna() >= df_noextcorr[f"HALPHA (component {nn + 1})"].dropna())
        assert np.all(df[f"HALPHA error (component {nn + 1})"].dropna() >= df_noextcorr[f"HALPHA error (component {nn + 1})"].dropna())

    # CHECK: check no negative A_V's
    assert not np.any(df["A_V (total)"] < 0)
    assert not np.any(df["A_V error (total)"] < 0)

    # CHECK: check no nonzero A_V's in rows where S/N in HALPHA or HBETA are less than 5
    cond_low_SN = df["HALPHA S/N (total)"] < 5
    cond_low_SN |= df["HBETA S/N (total)"] < 5
    assert np.all(df.loc[cond_low_SN, "A_V (total)"].isna())
    assert np.all(df.loc[cond_low_SN, "A_V error (total)"].isna())

    #//////////////////////////////////////////////////////////////////////////////
    logger.info("All assertion tests passed!")

