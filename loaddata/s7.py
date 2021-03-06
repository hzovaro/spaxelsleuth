import os
import pandas as pd
import numpy as np
from scipy import constants

from spaxelsleuth.loaddata import dqcut, linefns

from IPython.core.debugger import Tracer

"""
In this file:
- a function or class to load a dataframe containing the entire SAMI sample.

"""
s7_data_path = "/priv/meggs3/u5708159/S7/"
sami_datacube_path = "/priv/myrtle1/sami/sami_data/Final_SAMI_data/cube/sami/dr3/"

def load_s7_galaxies(eline_SNR_min, eline_list=["HALPHA", "HBETA", "NII6583", "OI6300", "OII3726", "OII3729", "OIII5007", "SII6716", "SII6731"],
                     sigma_gas_SNR_cut=True, sigma_gas_SNR_min=3,
                     line_amplitude_SNR_cut=True,
                     vgrad_cut=False, correct_extinction=False,
                     stekin_cut=True):

    #######################################################################
    # INPUT CHECKING
    #######################################################################
    assert eline_SNR_min >= 0, "eline_SNR_min must be positive!"
    assert sigma_gas_SNR_min >= 0, "sigma_gas_SNR_min must be positive!"
    
    df_fname = f"s7_spaxels.hd5"
    assert os.path.exists(os.path.join(s7_data_path, df_fname)),\
        f"File {os.path.join(s7_data_path, df_fname)} does does not exist!"

    #######################################################################
    # LOAD THE DATAFRAME FROM MEMORY
    #######################################################################    
    df = pd.read_hdf(os.path.join(s7_data_path, df_fname))

    ######################################################################
    # DQ and S/N CUTS
    ######################################################################
    # For WiFes
    FWHM_inst_A = 0.9 # Based on skyline at 6950 A
    sigma_inst_A = FWHM_inst_A / ( 2 * np.sqrt( 2 * np.log(2) ))
    sigma_inst_km_s = sigma_inst_A * constants.c / 1e3 / 6562.8  # Defined at Halpha
    print("WARNING: in load_s7_galaxies: estimating instrumental dispersion from my own WiFeS observations - may not be consistent with assumed value in LZIFU!")

    df = dqcut.dqcut(df=df, ncomponents=3,
                  eline_SNR_min=eline_SNR_min, eline_list=eline_list,
                  sigma_gas_SNR_cut=sigma_gas_SNR_cut,
                  sigma_gas_SNR_min=sigma_gas_SNR_min,
                  sigma_inst_kms=sigma_inst_km_s,
                  vgrad_cut=vgrad_cut,
                  line_amplitude_SNR_cut=line_amplitude_SNR_cut,
                  stekin_cut=stekin_cut)
    
    ######################################################################
    # EVALUATE ADDITIONAL COLUMNS - log quantites, etc.
    ######################################################################
    df = dqcut.compute_extra_columns(df, ncomponents=3)

    ######################################################################
    # EVALUATE LINE RATIOS & SPECTRAL CLASSIFICATIONS
    ######################################################################
    df = linefns.ratio_fn(df, s=f" (total)")
    df = linefns.bpt_fn(df, s=f" (total)")
    df = linefns.law2021_fn(df, s=f" (total)")
    for ii in range(3):
        df = linefns.ratio_fn(df, s=f" (component {ii})")
        df = linefns.bpt_fn(df, s=f" (component {ii})")
        df = linefns.law2021_fn(df, s=f" (component {ii})")

    ######################################################################
    # WHAV* classification
    ######################################################################
    df = linefns.whav_fn(df, ncomponents=3)

    ######################################################################
    # CORRECT HALPHA FLUX AND EW FOR EXTINCTION
    # We use the same A_V for all 3 components because the Hbeta fluxes
    # for individual components aren't always reliable.
    ######################################################################
    if correct_extinction:
        cond_SF = df["BPT (total)"] == "SF"
        df_SF = df[cond_SF]
        df_not_SF = df[~cond_SF]

        print("WARNING: in load_s7_galaxies: correcting for extinction!")
        # Use the provided extinction correction map to correct Halpha 
        # fluxes & EWs
        for rr in rainge(df_SF.shape[0]):
            for eline in ["HALPHA", "HBETA", "NII6583", "OI6300", "OII3726", "OII3729", "OIII5007", "SII6716", "SII6731"]:
                A = extinction.fm07(wave=np.array([eline_lambdas_A[eline]]), 
                                    a_v=df.iloc[rr]["A_V (total)"])[0] if ~np.isnan(df.iloc[rr]["A_V (total)"]) else 0
                if A > 0:
                    corr_factor = 10**(0.4 * A)

                    # Total fluxes
                    df.iloc[rr][f"{eline} (total)"] *= corr_factor
                    df.iloc[rr][f"{eline} error (total)"] *= corr_factor
                    if eline == "HALPHA":
                        df.iloc[rr]["HALPHA (total)"] *= corr_factor
                        df.iloc[rr]["HALPHA error (total)"] *= corr_factor
                        df.iloc[rr]["HALPHA EW (total)"] *= corr_factor
                        df.iloc[rr]["HALPHA EW error (total)"] *= corr_factor                
                        df.iloc[rr]["log HALPHA EW (total)"] += np.log10(corr_factor)
                        # Re-compute log errors
                        df.iloc[rr]["log HALPHA EW error (lower) (total)"] = df.iloc[rr]["log HALPHA EW (total)"] - np.log10(df.iloc[rr]["HALPHA EW (total)"] - df.iloc[rr]["HALPHA EW error (total)"])
                        df.iloc[rr]["log HALPHA EW error (upper) (total)"] = np.log10(df.iloc[rr]["HALPHA EW (total)"] + df.iloc[rr]["HALPHA EW error (total)"]) -  df.iloc[rr]["log HALPHA EW (total)"]

                    # Individual components
                    for ii in range(3):
                        df.iloc[rr][f"{eline} (component {ii})"] *= corr_factor
                        df.iloc[rr][f"{eline} error (component {ii})"] *= corr_factor
                        if eline == "HALPHA":
                            df.iloc[rr][f"HALPHA (component {ii})"] *= corr_factor
                            df.iloc[rr][f"HALPHA error (component {ii})"] *= corr_factor
                            df.iloc[rr][f"HALPHA EW (component {ii})"] *= corr_factor
                            df.iloc[rr][f"HALPHA EW error (component {ii})"] *= corr_factor
                            df.iloc[rr][f"log HALPHA EW (component {ii})"] += np.log10(corr_factor)
                            # Re-compute log errors
                            df.iloc[rr][f"log HALPHA EW error (lower) (component {ii})"] = df.iloc[rr][f"log HALPHA EW (component {ii})"] - np.log10(df.iloc[rr][f"HALPHA EW (component {ii})"] - df.iloc[rr][f"HALPHA EW error (component {ii})"])
                            df.iloc[rr][f"log HALPHA EW error (upper) (component {ii})"] = np.log10(df.iloc[rr][f"HALPHA EW (component {ii})"] + df.iloc[rr][f"HALPHA EW error (component {ii})"]) -  df.iloc[rr][f"log HALPHA EW (component {ii})"]
    else:
        print("WARNING: in load_s7_galaxies: NOT correcting for extinction!")

    return df

###############################################################################
if __name__ == "__main__":

    import numpy as np
    import matplotlib.pyplot as plt
    plt.close("all")
    plt.ion()

    ncomponents = "recom"
    bin_type = "default"
    eline_SNR_min = 5
    vgrad_cut = False
    stekin_cut = True

    df = load_s7_galaxies(eline_SNR_min=eline_SNR_min, vgrad_cut=vgrad_cut)
    df_nocut = load_s7_galaxies(eline_SNR_min=0, vgrad_cut=False)

    # CHECK: proper S/N cuts have been made
    for eline in ["HALPHA", "HBETA", "OIII5007", "OI6300", "NII6583", "SII6716", "SII6731"]:
        plt.figure()
        plt.scatter(x=df_nocut[f"{eline} (total)"], y=df_nocut[f"{eline} S/N (total)"], c="grey", alpha=0.1, s=5)
        plt.scatter(x=df[f"{eline} (total)"], y=df[f"{eline} S/N (total)"], c="k", s=5)
        plt.ylabel(f"{eline} (total) S/N")
        plt.axhline(eline_SNR_min, color="r")
        plt.xlabel(f"{eline} (total) flux")
    for ii in range(3):
        plt.figure()
        plt.scatter(x=df_nocut[f"HALPHA (component {ii})"], y=df_nocut[f"HALPHA S/N (component {ii})"], c="grey", alpha=0.1, s=5)
        plt.scatter(x=df[f"HALPHA (component {ii})"], y=df[f"HALPHA S/N (component {ii})"], c="k", s=5)
        plt.ylabel(f"HALPHA (component {ii}) S/N")
        plt.axhline(eline_SNR_min, color="r")
        plt.xlabel(f"HALPHA (component {ii}) flux")
    for ii in range(3):
        plt.figure()
        plt.scatter(x=df_nocut[f"HALPHA EW (component {ii})"], y=df_nocut[f"HALPHA S/N (component {ii})"], c="grey", alpha=0.1, s=5)
        plt.scatter(x=df[f"HALPHA EW (component {ii})"], y=df[f"HALPHA S/N (component {ii})"], c="k", s=5)
        plt.ylabel(f"HALPHA (component {ii}) S/N")
        plt.axhline(eline_SNR_min, color="r")
        plt.xlabel(f"HALPHA EW (component {ii})")

    # CHECK: gas kinematics
    for ii in range(3):
        plt.figure()
        plt.scatter(x=df_nocut[f"sigma_obs S/N (component {ii})"], y=df_nocut[f"sigma_obs target S/N (component {ii})"], c="grey", alpha=0.1, s=5)
        plt.scatter(x=df[f"sigma_obs S/N (component {ii})"], y=df[f"sigma_obs target S/N (component {ii})"], c="k", s=5)
        plt.ylabel(f"sigma_obs target S/N (component {ii})")
        plt.axhline(3, color="b")
        plt.xlabel(f"sigma_obs S/N (component {ii})")

    # CHECK: "Number of components" is 0, 1, 2 or 3
    assert np.all(df["Number of components"].unique() == [0, 1, 2, 3])

    # CHECK: all emission line fluxes in spaxels with 0 components are NaN
    for eline in ["HALPHA", "HBETA", "NII6583", "OI6300", "OII3726", "OII3729", "OIII5007", "SII6716", "SII6731"]:
        assert np.all(np.isnan(df.loc[df["Number of components"] == 0, f"{eline} (total)"]))
        assert np.all(np.isnan(df.loc[df["Number of components"] == 0, f"{eline} error (total)"]))

    # CHECK: all emission line columns in spaxels with 0 components are NaN
    for eline in ["HALPHA", "HBETA", "NII6583", "OI6300", "OII3726", "OII3729", "OIII5007", "SII6716", "SII6731"]:
        assert np.all(np.isnan(df.loc[df["Number of components"] == 0, f"{eline} (total)"]))
        assert np.all(np.isnan(df.loc[df["Number of components"] == 0, f"{eline} error (total)"]))
        if eline != "OII3726" and eline != "OII3729":
            for ii in range(3):
                assert np.all(np.isnan(df.loc[df["Number of components"] == 0, f"{eline} (component {ii})"]))
                assert np.all(np.isnan(df.loc[df["Number of components"] == 0, f"{eline} error (component {ii})"]))

    # CHECK: all HALPHA-derived columns in spaxels with 0 components are NaN
    col = "HALPHA EW"
    assert np.all(np.isnan(df.loc[df["Number of components"] == 0, f"{col} (total)"]))
    assert np.all(np.isnan(df.loc[df["Number of components"] == 0, f"{col} error (total)"]))
    for ii in range(3):
        assert np.all(np.isnan(df.loc[df["Number of components"] == 0, f"{col} (component {ii})"]))
        assert np.all(np.isnan(df.loc[df["Number of components"] == 0, f"{col} error (component {ii})"]))

    # CHECK: all HALPHA-derived columns in spaxels with 0 components are NaN
    col = "log HALPHA EW"
    assert np.all(np.isnan(df.loc[df["Number of components"] == 0, f"{col} (total)"]))
    assert np.all(np.isnan(df.loc[df["Number of components"] == 0, f"{col} error (upper) (total)"]))
    assert np.all(np.isnan(df.loc[df["Number of components"] == 0, f"{col} error (lower) (total)"]))
    for ii in range(3):
        assert np.all(np.isnan(df.loc[df["Number of components"] == 0, f"{col} (component {ii})"]))
        assert np.all(np.isnan(df.loc[df["Number of components"] == 0, f"{col} error (upper) (component {ii})"]))
        assert np.all(np.isnan(df.loc[df["Number of components"] == 0, f"{col} error (lower) (component {ii})"]))

    # CHECK: all kinematic quantities in spaxels with 0 components are NaN
    for col in ["sigma_gas", "v_gas"]:
        for ii in range(3):
            assert np.all(np.isnan(df.loc[df["Number of components"] == 0, f"{col} (component {ii})"]))
            assert np.all(np.isnan(df.loc[df["Number of components"] == 0, f"{col} error (component {ii})"]))

    # CHECK: all emission line fluxes with S/N < SNR_min are NaN
    for eline in ["HALPHA", "HBETA", "NII6583", "OI6300", "OII3726", "OII3729", "OIII5007", "SII6716", "SII6731"]:
        assert np.all(np.isnan(df.loc[df[f"{eline} S/N (total)"] < eline_SNR_min, f"{eline} (total)"]))

    # CHECK: all Halpha components below S/N limit are NaN
    for ii in range(3):
        assert np.all(np.isnan(df.loc[df[f"HALPHA S/N (component {ii})"] < eline_SNR_min, f"HALPHA (component {ii})"]))
        assert np.all(np.isnan(df.loc[df[f"HALPHA S/N (component {ii})"] < eline_SNR_min, f"HALPHA error (component {ii})"]))
        assert np.all(np.isnan(df.loc[df[f"HALPHA S/N (component {ii})"] < eline_SNR_min, f"HALPHA EW (component {ii})"]))
        assert np.all(np.isnan(df.loc[df[f"HALPHA S/N (component {ii})"] < eline_SNR_min, f"HALPHA EW error (component {ii})"]))
        assert np.all(np.isnan(df.loc[df[f"HALPHA S/N (component {ii})"] < eline_SNR_min, f"log HALPHA EW (component {ii})"]))
        assert np.all(np.isnan(df.loc[df[f"HALPHA S/N (component {ii})"] < eline_SNR_min, f"log HALPHA EW error (upper) (component {ii})"]))
        assert np.all(np.isnan(df.loc[df[f"HALPHA S/N (component {ii})"] < eline_SNR_min, f"log HALPHA EW error (lower) (component {ii})"]))

    # CHECK: all sigma_gas components with S/N < S/N target are NaN
    for ii in range(3):
        assert np.all(np.isnan(df.loc[df[f"sigma_obs S/N (component {ii})"] < df[f"sigma_obs target S/N (component {ii})"], f"sigma_gas (component {ii})"]))
        assert np.all(np.isnan(df.loc[df[f"sigma_obs S/N (component {ii})"] < df[f"sigma_obs target S/N (component {ii})"], f"sigma_gas error (component {ii})"]))

    # CHECK: all sigma_gas components that don't meet the v_grad requirement are NaN
    if vgrad_cut:
        for ii in range(3):
            assert np.all(np.isnan(df.loc[df[f"v_grad (component {ii})"] > 2 * df[f"sigma_gas (component {ii})"], f"sigma_gas (component {ii})"]))
            assert np.all(np.isnan(df.loc[df[f"v_grad (component {ii})"] > 2 * df[f"sigma_gas (component {ii})"], f"sigma_gas error (component {ii})"]))

    # CHECK: stellar kinematics
    if stekin_cut:
        assert np.all(df.loc[~np.isnan(df["sigma_*"]), "sigma_*"] > 35)
        assert np.all(df.loc[~np.isnan(df["v_* error"]), "v_* error"] < 30)
        assert np.all(df.loc[~np.isnan(df["sigma_* error"]), "sigma_* error"] < df.loc[~np.isnan(df["sigma_* error"]), "sigma_*"] * 0.1 + 25)

    # CHECK: no "orphan" components
    assert df[np.isnan(df["HALPHA (component 0)"]) & ~np.isnan(df["HALPHA (component 1)"])].shape[0] == 0
    assert df[np.isnan(df["HALPHA (component 0)"]) & ~np.isnan(df["HALPHA (component 2)"])].shape[0] == 0
    assert df[np.isnan(df["HALPHA (component 1)"]) & ~np.isnan(df["HALPHA (component 2)"])].shape[0] == 0
    assert df[np.isnan(df["sigma_gas (component 0)"]) & ~np.isnan(df["sigma_gas (component 1)"])].shape[0] == 0
    assert df[np.isnan(df["sigma_gas (component 0)"]) & ~np.isnan(df["sigma_gas (component 2)"])].shape[0] == 0
    assert df[np.isnan(df["sigma_gas (component 1)"]) & ~np.isnan(df["sigma_gas (component 2)"])].shape[0] == 0
    