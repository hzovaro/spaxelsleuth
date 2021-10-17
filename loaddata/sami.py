import os
import pandas as pd
import numpy as np

from spaxelsleuth.loaddata import dqcut, linefns

from IPython.core.debugger import Tracer

"""
In this file:
- a function or class to load a dataframe containing the entire SAMI sample.

"""
sami_data_path = "/priv/meggs3/u5708159/SAMI/sami_dr3/"
sami_datacube_path = "/priv/myrtle1/sami/sami_data/Final_SAMI_data/cube/sami/dr3/"

def load_sami_galaxies(ncomponents, bin_type,
                       eline_SNR_min, SNR_linelist=["HALPHA", "HBETA", "NII6583", "OI6300", "OII3726+OII3729", "OIII5007", "SII6716", "SII6731"],
                       sigma_gas_SNR_cut=True, sigma_gas_SNR_min=3,
                       vgrad_cut=False, correct_extinction=False,
                       debug=False):

    #######################################################################
    # INPUT CHECKING
    #######################################################################
    assert (ncomponents == "recom") | (ncomponents == "1"), "ncomponents must be 'recom' or '1'!!"
    assert (bin_type == "default") | (bin_type == "adaptive"), "bin_type must be 'default' or 'adaptive'!!"
    assert eline_SNR_min >= 0, "eline_SNR_min must be positive!"
    assert sigma_gas_SNR_min >= 0, "sigma_gas_SNR_min must be positive!"
    
    if not debug:
        df_fname = f"sami_{bin_type}_{ncomponents}-comp.hd5"
    else:
        df_fname = f"sami_{bin_type}_{ncomponents}-comp_mini.hd5"
    assert os.path.exists(os.path.join(sami_data_path, df_fname)),\
        f"File {os.path.join(sami_data_path, df_fname)} does does not exist!"

    #######################################################################
    # LOAD THE DATAFRAME FROM MEMORY
    #######################################################################    
    df = pd.read_hdf(os.path.join(sami_data_path, df_fname))

    ######################################################################
    # DQ and S/N CUTS
    ######################################################################
    df = dqcut.dqcut(df=df, ncomponents=3 if ncomponents == "recom" else 1,
                  eline_SNR_min=eline_SNR_min, SNR_linelist=SNR_linelist,
                  sigma_gas_SNR_cut=sigma_gas_SNR_cut,
                  sigma_gas_SNR_min=sigma_gas_SNR_min,
                  sigma_inst_kms=29.6,
                  vgrad_cut=vgrad_cut,
                  stekin_cut=True)
    
    ######################################################################
    # EVALUATE ADDITIONAL COLUMNS - log quantites, etc.
    ######################################################################
    df = dqcut.compute_extra_columns(df, ncomponents=3 if ncomponents=="recom" else 1)

    ######################################################################
    # EVALUATE LINE RATIOS & SPECTRAL CLASSIFICATIONS
    ######################################################################
    df = linefns.ratio_fn(df, s=f" (total)")
    df = linefns.bpt_fn(df, s=f" (total)")
    df = linefns.law2021_fn(df, s=f" (total)")

    ######################################################################
    # CORRECT HALPHA FLUX AND EW FOR EXTINCTION
    # Note that we only have enough information from the provided extinction
    # maps to correct the Halpha line. We therefore apply this correction 
    # AFTER we compute line ratios, etc.
    ######################################################################
    if correct_extinction:
        print("WARNING: correcting Halpha and HALPHA EW for extinction!")
        # Use the provided extinction correction map to correct Halpha 
        # fluxes & EWs
        df["HALPHA (total)"] *= df["HALPHA extinction correction"]
        df["HALPHA error (total)"] *= df["HALPHA extinction correction"]
        df["HALPHA EW (total)"] *= df["HALPHA extinction correction"]
        df["HALPHA EW error (total)"] *= df["HALPHA extinction correction"]
        df["log HALPHA EW (total)"] += np.log10(df["HALPHA extinction correction"])
        # Re-compute log errors
        df["log HALPHA EW error (lower) (total)"] = df["log HALPHA EW (total)"] - np.log10(df["HALPHA EW (total)"] - df["HALPHA EW error (total)"])
        df["log HALPHA EW error (upper) (total)"] = np.log10(df["HALPHA EW (total)"] + df["HALPHA EW error (total)"]) -  df["log HALPHA EW (total)"]

        for component in range(3 if ncomponents == "recom" else 1):
            df[f"HALPHA (component {component})"] *= df["HALPHA extinction correction"]
            df[f"HALPHA error (component {component})"] *= df["HALPHA extinction correction"]
            df[f"HALPHA EW (component {component})"] *= df["HALPHA extinction correction"]
            df[f"HALPHA EW error (component {component})"] *= df["HALPHA extinction correction"]
            df[f"log HALPHA EW (component {component})"] += np.log10(df["HALPHA extinction correction"])
            # Re-compute log errors
            df[f"log HALPHA EW error (lower) (component {component})"] = df[f"log HALPHA EW (component {component})"] - np.log10(df[f"HALPHA EW (component {component})"] - df[f"HALPHA EW error (component {component})"])
            df[f"log HALPHA EW error (upper) (component {component})"] = np.log10(df[f"HALPHA EW (component {component})"] + df[f"HALPHA EW error (component {component})"]) -  df[f"log HALPHA EW (component {component})"]
    else:
        print("WARNING: NOT correcting Halpha and HALPHA EW for extinction!")
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

    df = load_sami_galaxies(ncomponents="recom", bin_type="default",
                            eline_SNR_min=eline_SNR_min, vgrad_cut=vgrad_cut)
    df_nocut = load_sami_galaxies(ncomponents="recom", bin_type="default",
                                  eline_SNR_min=0, vgrad_cut=False)

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
    for eline in ["HALPHA", "HBETA", "NII6583", "OI6300", "OII3726+OII3729", "OIII5007", "SII6716", "SII6731"]:
        assert np.all(np.isnan(df.loc[df["Number of components"] == 0, f"{eline} (total)"]))
        assert np.all(np.isnan(df.loc[df["Number of components"] == 0, f"{eline} error (total)"]))

    # CHECK: all HALPHA-derived columns in spaxels with 0 components are NaN
    for col in ["HALPHA EW", "log HALPHA EW"]:
        assert np.all(np.isnan(df.loc[df["Number of components"] == 0, f"{col} (total)"]))
        assert np.all(np.isnan(df.loc[df["Number of components"] == 0, f"{col} error (total)"]))
        for ii in range(3):
            assert np.all(np.isnan(df.loc[df["Number of components"] == 0, f"{col} (component {ii})"]))
            assert np.all(np.isnan(df.loc[df["Number of components"] == 0, f"{col} error (component {ii})"]))

    # CHECK: all kinematic quantities in spaxels with 0 components are NaN
    for col in ["sigma_gas", "v_gas"]:
        for ii in range(3):
            assert np.all(np.isnan(df.loc[df["Number of components"] == 0, f"{col} (component {ii})"]))
            assert np.all(np.isnan(df.loc[df["Number of components"] == 0, f"{col} error (component {ii})"]))

    # CHECK: all emission line fluxes with S/N < SNR_min are NaN
    for eline in ["HALPHA", "HBETA", "NII6583", "OI6300", "OII3726+OII3729", "OIII5007", "SII6716", "SII6731"]:
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
    