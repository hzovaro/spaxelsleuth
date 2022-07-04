import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import extinction

from spaxelsleuth.loaddata import dqcut, linefns

from IPython.core.debugger import Tracer

"""
This script contains the function load_sami_galaxies() which is used to 
load the Pandas DataFrame containing spaxel-by-spaxel information for all 
SAMI galaxies that was created in make_df_sami.py.

After the DataFrame is loaded, a number of S/N and data quality cuts are 
optionally made based on emission line S/N, the quality of the stellar 
kinematics fit, and other quantities. 

Other quantities, such as BPT categories, are computed for each spaxel.

The function returns the DataFrame with added columns containing these 
additional quantities, and with rows/cells that do not meet data quality 
or S/N requirements either droppped or replaced with NaNs.

"""
###############################################################################
# Paths
sami_data_path = os.environ["SAMI_DIR"]
assert "SAMI_DIR" in os.environ, "Environment variable SAMI_DIR is not defined!"
sami_datacube_path = os.environ["SAMI_DATACUBE_DIR"]
assert "SAMI_DATACUBE_DIR" in os.environ, "Environment variable SAMI_DATACUBE_DIR is not defined!"


###############################################################################
def load_sami_galaxies(ncomponents, bin_type,
                       eline_SNR_min, eline_list=["HALPHA", "HBETA", "NII6583", "OI6300", "OII3726+OII3729", "OIII5007", "SII6716", "SII6731"],
                       sigma_gas_SNR_cut=True, sigma_gas_SNR_min=3,
                       line_amplitude_SNR_cut=True,
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
        df_fname = f"sami_{bin_type}_{ncomponents}-comp_DEBUG.hd5"
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
                  eline_SNR_min=eline_SNR_min, eline_list=eline_list,
                  sigma_gas_SNR_cut=sigma_gas_SNR_cut,
                  sigma_gas_SNR_min=sigma_gas_SNR_min,
                  sigma_inst_kms=29.6,
                  vgrad_cut=vgrad_cut,
                  line_amplitude_SNR_cut=line_amplitude_SNR_cut,
                  stekin_cut=True)
    
    # ISSUE: there are spaxels which do not have any HALPHA flux due to our 
    # S/N requirement that still have defined SFRs. We need to NaN these out. 
    sfr_cols = [s for s in df.columns if "SFR" in s]
    df.loc[df["Number of components"] == 0, sfr_cols] = np.nan

    ######################################################################
    # EXTINCTION CORRECTION
    # Correct emission line fluxes (but not EWs!)
    # NOTE: extinction.fm07 assumes R_V = 3.1 so do not change R_V from 
    # this value!!!
    ######################################################################
    if correct_extinction:
        print("WARNING: in load_sami_galaxies: correcting emission line fluxes (but not EWs) for extinction!")
        """
        for cc in ["total"]:
            # Compute A_V in each spaxel
            df[f"Balmer decrement ({cc})"] = df[f"HALPHA ({cc})"] / df[f"HBETA ({cc})"]
            df[f"Balmer decrement error ({cc})"] =\
                df[f"Balmer decrement ({cc})"] * \
                np.sqrt( (df[f"HALPHA error ({cc})"] / df[f"HALPHA ({cc})"])**2 +\
                         (df[f"HBETA error ({cc})"] / df[f"HBETA ({cc})"])**2 )

            # Compute E(B-V)
            E_ba = 2.5 * (np.log10(df[f"Balmer decrement ({cc})"])) - 2.5 * np.log10(2.86)
            E_ba_err = 2.5 / np.log(10) * df[f"Balmer decrement error ({cc})"] / df[f"Balmer decrement ({cc})"]

            # Calculate ( A(Ha) - A(Hb) ) / E(B-V) from extinction curve
            R_V = 3.1
            wave_1_A = np.array([linefns.eline_lambdas_A["HALPHA"]])
            wave_2_A = np.array([linefns.eline_lambdas_A["HBETA"]])
            E_ba_over_E_BV = float(extinction.fm07(wave_2_A, a_v=1.0) - extinction.fm07(wave_1_A, a_v=1.0) ) /  1.0 * R_V

            # Calculate E(B-V)
            E_BV = 1 / E_ba_over_E_BV * E_ba
            E_BV_err = 1 / E_ba_over_E_BV * E_ba_err

            # Calculate A(V)
            df[f"A_V ({cc})"] = R_V * E_BV
            df[f"A_V error ({cc})"] = R_V * E_BV_err

            # DQ cut: non-physical Balmer decrement (set A_V = 0)
            cond_negative_A_V = df[f"A_V ({cc})"] < 0
            df.loc[cond_negative_A_V, f"A_V ({cc})"] = 0
            df.loc[cond_negative_A_V, f"A_V error ({cc})"] = 0

            # DQ cut: low S/N HALPHA and HBETA fluxes (set A_V = NaN)
            cond_bad_A_V = df[f"HALPHA S/N ({cc})"] < 5
            cond_bad_A_V |= df[f"HBETA S/N ({cc})"] < 5
            df.loc[cond_bad_A_V, f"A_V ({cc})"] = np.nan
            df.loc[cond_bad_A_V, f"A_V error ({cc})"] = np.nan

            # Correct emission line fluxes in cells where A_V > 0
            # idxs_to_correct = df[(df[f"A_V ({cc})"] > 0) & (~df[f"A_V ({cc})"].isna())].index.values
            for rr in tqdm(df.index.values):
                if df.loc[rr, f"A_V ({cc})"] > 0:
                    for eline in eline_list:
                        lambda_A = linefns.eline_lambdas_A[eline]
                        A_line = extinction.fm07(wave=np.array([lambda_A]), 
                                                 a_v=df.loc[rr, f"A_V ({cc})"], 
                                                 unit="aa")[0]
                        
                        # Apply correction
                        # print("Before correction: %.6f" % df.iloc[rr][f"{eline} ({cc})"])
                        tmp = df.loc[rr, f"{eline} ({cc})"].copy()
                        df.loc[rr, f"{eline} ({cc})"] *= 10**(0.4 * A_line)
                        df.loc[rr, f"{eline} error ({cc})"] *= 10**(0.4 * A_line)
                        # Check that the extinction correction has actually been applied
                        try:
                            if ~np.isnan(tmp):
                                assert df.loc[rr, f"{eline} ({cc})"] >= tmp
                        except AssertionError:
                            Tracer()()

                        # print("After correction: %.6f" % df.iloc[rr][f"{eline} ({cc})"])
        """
        df = linefns.extinction_corr_fn(df, eline_list=eline_list, 
                                        reddening_curve="fm07", 
                                        balmer_SNR_min=5,
                                        s=f" (total)")
        df["Corrected for extinction?"] = True
    else:
        print("WARNING: in load_sami_galaxies: NOT correcting emission line fluxes for extinction!")
        df["Corrected for extinction?"] = False

    ######################################################################
    # EVALUATE LINE RATIOS & SPECTRAL CLASSIFICATIONS
    ######################################################################
    df = linefns.ratio_fn(df, s=f" (total)")
    df = linefns.bpt_fn(df, s=f" (total)")
    df = linefns.law2021_fn(df, s=f" (total)")

    ######################################################################
    # EVALUATE ADDITIONAL COLUMNS - log quantites, etc.
    ######################################################################
    df = dqcut.compute_extra_columns(df, ncomponents=3 if ncomponents=="recom" else 1)

    ######################################################################
    # WHAV* classification
    ######################################################################
    df = linefns.whav_fn(df, ncomponents=3 if ncomponents=="recom" else 1)

    return df.sort_index()

###############################################################################
if __name__ == "__main__":

    import sys
    import numpy as np
    import matplotlib.pyplot as plt
    plt.close("all")
    plt.ion()

    ncomponents = sys.argv[1]
    bin_type = "default"
    eline_SNR_min = 5
    vgrad_cut = False
    stekin_cut = True

    df = load_sami_galaxies(ncomponents=ncomponents, bin_type="default",
                            eline_SNR_min=eline_SNR_min, vgrad_cut=vgrad_cut, 
                            debug=True)
    df_nocut = load_sami_galaxies(ncomponents=ncomponents, bin_type="default",
                                  eline_SNR_min=0, vgrad_cut=False, 
                                  debug=True)

    ######################################################################
    # RUN ASSERTION CHECKS
    ######################################################################
    # CHECK: "Number of components" is 0, 1, 2 or 3
    if ncomponents == "recom":
        assert np.all(df["Number of components"].unique() == [0, 1, 2, 3])
    elif ncomponents == "1":
        assert np.all(df["Number of components"].unique() == [0, 1])

    # CHECK: all emission line fluxes in spaxels with 0 components are NaN
    for eline in ["HALPHA", "HBETA", "NII6583", "OI6300", "OII3726+OII3729", "OIII5007", "SII6716", "SII6731"]:
        assert np.all(np.isnan(df.loc[df["Number of components"] == 0, f"{eline} (total)"]))
        assert np.all(np.isnan(df.loc[df["Number of components"] == 0, f"{eline} error (total)"]))

    # CHECK: all HALPHA-derived columns in spaxels with 0 components are NaN
    col = "HALPHA EW"
    assert np.all(np.isnan(df.loc[df["Number of components"] == 0, f"{col} (total)"]))
    assert np.all(np.isnan(df.loc[df["Number of components"] == 0, f"{col} error (total)"]))
    for ii in range(3 if ncomponents == "recom" else 1):
        assert np.all(np.isnan(df.loc[df["Number of components"] == 0, f"{col} (component {ii})"]))
        assert np.all(np.isnan(df.loc[df["Number of components"] == 0, f"{col} error (component {ii})"]))

    # CHECK: all HALPHA-derived columns in spaxels with 0 components are NaN
    col = "log HALPHA EW"
    assert np.all(np.isnan(df.loc[df["Number of components"] == 0, f"{col} (total)"]))
    assert np.all(np.isnan(df.loc[df["Number of components"] == 0, f"{col} error (upper) (total)"]))
    assert np.all(np.isnan(df.loc[df["Number of components"] == 0, f"{col} error (lower) (total)"]))
    for ii in range(3 if ncomponents == "recom" else 1):
        assert np.all(np.isnan(df.loc[df["Number of components"] == 0, f"{col} (component {ii})"]))
        assert np.all(np.isnan(df.loc[df["Number of components"] == 0, f"{col} error (upper) (component {ii})"]))
        assert np.all(np.isnan(df.loc[df["Number of components"] == 0, f"{col} error (lower) (component {ii})"]))

    # CHECK: all kinematic quantities in spaxels with 0 components are NaN
    for col in ["sigma_gas", "v_gas"]:
        for ii in range(3 if ncomponents == "recom" else 1):
            assert np.all(np.isnan(df.loc[df["Number of components"] == 0, f"{col} (component {ii})"]))
            assert np.all(np.isnan(df.loc[df["Number of components"] == 0, f"{col} error (component {ii})"]))

    # CHECK: all emission line fluxes with S/N < SNR_min are NaN
    for eline in ["HALPHA", "HBETA", "NII6583", "OI6300", "OII3726+OII3729", "OIII5007", "SII6716", "SII6731"]:
        assert np.all(np.isnan(df.loc[df[f"{eline} S/N (total)"] < eline_SNR_min, f"{eline} (total)"]))

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

    # CHECK: all sigma_gas components that don't meet the v_grad requirement are NaN
    if vgrad_cut:
        for ii in range(3 if ncomponents == "recom" else 1):
            assert np.all(np.isnan(df.loc[df[f"v_grad (component {ii})"] > 2 * df[f"sigma_gas (component {ii})"], f"sigma_gas (component {ii})"]))
            assert np.all(np.isnan(df.loc[df[f"v_grad (component {ii})"] > 2 * df[f"sigma_gas (component {ii})"], f"sigma_gas error (component {ii})"]))

    # CHECK: stellar kinematics
    assert np.all(df.loc[~np.isnan(df["sigma_*"]), "sigma_*"] > 35)
    assert np.all(df.loc[~np.isnan(df["v_* error"]), "v_* error"] < 30)
    assert np.all(df.loc[~np.isnan(df["sigma_* error"]), "sigma_* error"] < df.loc[~np.isnan(df["sigma_* error"]), "sigma_*"] * 0.1 + 25)

    # CHECK: no "orphan" components
    if ncomponents == "recom":
        assert df[np.isnan(df["HALPHA (component 0)"]) & ~np.isnan(df["HALPHA (component 1)"])].shape[0] == 0
        assert df[np.isnan(df["HALPHA (component 0)"]) & ~np.isnan(df["HALPHA (component 2)"])].shape[0] == 0
        assert df[np.isnan(df["HALPHA (component 1)"]) & ~np.isnan(df["HALPHA (component 2)"])].shape[0] == 0
        assert df[np.isnan(df["sigma_gas (component 0)"]) & ~np.isnan(df["sigma_gas (component 1)"])].shape[0] == 0
        assert df[np.isnan(df["sigma_gas (component 0)"]) & ~np.isnan(df["sigma_gas (component 2)"])].shape[0] == 0
        assert df[np.isnan(df["sigma_gas (component 1)"]) & ~np.isnan(df["sigma_gas (component 2)"])].shape[0] == 0

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
            assert all(df.loc[df[f"Low S/N component - {eline} (component {ii})"], f"{eline} (component {ii})"].isna())
            assert all(df.loc[df[f"Low S/N component - {eline} (component {ii})"], f"{eline} error (component {ii})"].isna())
            if eline == "HALPHA":
                assert all(df.loc[df[f"Low S/N component - {eline} (component {ii})"], f"v_gas (component {ii})"].isna())
                assert all(df.loc[df[f"Low S/N component - {eline} (component {ii})"], f"v_gas error (component {ii})"].isna())
                assert all(df.loc[df[f"Low S/N component - {eline} (component {ii})"], f"sigma_gas (component {ii})"].isna())
                assert all(df.loc[df[f"Low S/N component - {eline} (component {ii})"], f"sigma_gas error (component {ii})"].isna())

    # CHECK: has HALPHA EW been NaNd out propertly?
    for ii in range(3 if ncomponents == "recom" else 1):
        assert all(df.loc[df[f"HALPHA (component {ii})"].isna(), f"HALPHA EW (component {ii})"].isna())
        assert all(df.loc[df[f"HALPHA (component {ii})"].isna(), f"HALPHA EW error (component {ii})"].isna())
        assert all(df.loc[df[f"HALPHA continuum"].isna() | (df[f"HALPHA continuum"] <= 0), f"HALPHA EW (component {ii})"].isna())
        assert all(df.loc[df[f"HALPHA continuum"].isna() | (df[f"HALPHA continuum"] <= 0), f"HALPHA EW error (component {ii})"].isna())
    assert all(df.loc[df[f"HALPHA (total)"].isna(), f"HALPHA EW (total)"].isna())
    assert all(df.loc[df[f"HALPHA continuum"].isna() | (df[f"HALPHA continuum"] <= 0), f"HALPHA EW (total)"].isna())
    assert all(df.loc[df[f"HALPHA continuum"].isna() | (df[f"HALPHA continuum"] <= 0), f"HALPHA EW error (total)"].isna())

    # CHECK: proper S/N cuts have been made
    for eline in ["HALPHA", "HBETA", "OIII5007", "OI6300", "NII6583", "SII6716", "SII6731"]:
        plt.figure()
        plt.scatter(x=df_nocut[f"{eline} (total)"], y=df_nocut[f"{eline} S/N (total)"], c="grey", alpha=0.1, s=5)
        plt.scatter(x=df[f"{eline} (total)"], y=df[f"{eline} S/N (total)"], c="k", s=5)
        plt.ylabel(f"{eline} (total) S/N")
        plt.axhline(eline_SNR_min, color="r")
        plt.xlabel(f"{eline} (total) flux")
    for ii in range(3 if ncomponents == "recom" else 1):
        plt.figure()
        plt.scatter(x=df_nocut[f"HALPHA (component {ii})"], y=df_nocut[f"HALPHA S/N (component {ii})"], c="grey", alpha=0.1, s=5)
        plt.scatter(x=df[f"HALPHA (component {ii})"], y=df[f"HALPHA S/N (component {ii})"], c="k", s=5)
        plt.ylabel(f"HALPHA (component {ii}) S/N")
        plt.axhline(eline_SNR_min, color="r")
        plt.xlabel(f"HALPHA (component {ii}) flux")
    for ii in range(3 if ncomponents == "recom" else 1):
        plt.figure()
        plt.scatter(x=df_nocut[f"HALPHA EW (component {ii})"], y=df_nocut[f"HALPHA S/N (component {ii})"], c="grey", alpha=0.1, s=5)
        plt.scatter(x=df[f"HALPHA EW (component {ii})"], y=df[f"HALPHA S/N (component {ii})"], c="k", s=5)
        plt.ylabel(f"HALPHA (component {ii}) S/N")
        plt.axhline(eline_SNR_min, color="r")
        plt.xlabel(f"HALPHA EW (component {ii})")

    # CHECK: gas kinematics
    for ii in range(3 if ncomponents == "recom" else 1):
        plt.figure()
        plt.scatter(x=df_nocut[f"sigma_obs S/N (component {ii})"], y=df_nocut[f"sigma_obs target S/N (component {ii})"], c="grey", alpha=0.1, s=5)
        plt.scatter(x=df[f"sigma_obs S/N (component {ii})"], y=df[f"sigma_obs target S/N (component {ii})"], c="k", s=5)
        plt.ylabel(f"sigma_obs target S/N (component {ii})")
        plt.axhline(3, color="b")
        plt.xlabel(f"sigma_obs S/N (component {ii})")

    # CHECK: calculation of HALPHA luminosity per kpc2
    # These two should have an approx. 1:1 relationship.
    plt.figure()
    m = plt.scatter(df["log SFR surface density (total)"], np.log10(df["HALPHA luminosity (total)"] * 5.5e-42), c=df["BPT (numeric) (total)"])
    plt.xlabel("log SFR surface density (total)")
    plt.ylabel("log(HALPHA luminosity (total) * 5.5e-42)")
    plt.colorbar(m)
    plt.plot([-10,1], [-10,1], "black")
    plt.xlim([-5, 1])
