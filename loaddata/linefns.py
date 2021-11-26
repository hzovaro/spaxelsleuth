import numpy as np

from IPython.core.debugger import Tracer
###############################################################################
# Reference lines from literature
def Kewley2001(ratio_x, ratio_x_vals, log=True):
    ratio_x_vals = np.copy(ratio_x_vals)
    if not log:
        ratio_y_vals = np.log10(ratio_y_vals)
    """
    Maximum SB limit
    """
    assert ratio_x in ["log N2", "log S2", "log O1"], "x must be one of log N2, log S2 or log O1!"
    if ratio_x == "log N2":
        ratio_x_vals[ratio_x_vals > 0.47] = np.nan
        return 0.61 / (ratio_x_vals - 0.47) + 1.19
    elif ratio_x == "log S2":
        ratio_x_vals[ratio_x_vals > 0.32] = np.nan
        return 0.72 / (ratio_x_vals - 0.32) + 1.30
    elif ratio_x == "log O1":
        ratio_x_vals[ratio_x_vals > -0.59] = np.nan
        return 0.73 / (ratio_x_vals + 0.59) + 1.33

def Kauffman2003(ratio_x, ratio_x_vals, log=True):
    ratio_x_vals = np.copy(ratio_x_vals)
    if not log:
        ratio_y_vals = np.log10(ratio_y_vals)
    """
    Empirical equivalent of Kewley2001
    """
    assert ratio_x in ["log N2"], "x must be log N2!"
    if ratio_x == "log N2":
        ratio_x_vals[ratio_x_vals > 0.05] = np.nan
        return 0.61 / (ratio_x_vals - 0.05) + 1.3

def Kewley2006(ratio_x, ratio_x_vals, log=True):
    ratio_x_vals = np.copy(ratio_x_vals)
    if not log:
        ratio_y_vals = np.log10(ratio_y_vals)
    """
    Spearating LINERs from Seyferts
    """
    assert ratio_x in ["log S2", "log O1"], "x must be one of log N2, log S2 or log O1!"
    if ratio_x == "log S2":
        ratio_x_vals[ratio_x_vals < -0.3143200520185163] = np.nan
        return 1.89 * ratio_x_vals + 0.76
    elif ratio_x == "log O1":
        ratio_x_vals[ratio_x_vals < -1.1259] = np.nan
        return 1.18 * ratio_x_vals + 1.30

def Law2021_1sigma(ratio_x, ratio_x_vals, log=True):
    ratio_x_vals = np.copy(ratio_x_vals)
    if not log:
        ratio_y_vals = np.log10(ratio_y_vals)

    assert ratio_x in ["log N2", "log S2", "log O1"], "x must be one of log N2, log S2 or log O1!"
    if ratio_x == "log N2":
        ratio_x_vals[ratio_x_vals > -0.032] = np.nan
        return 0.359 / (ratio_x_vals + 0.032) + 1.083
    elif ratio_x == "log S2":
        ratio_x_vals[ratio_x_vals > 0.198] = np.nan
        return 0.410 / (ratio_x_vals - 0.198) + 1.164
    elif ratio_x == "log O1":
        ratio_x_vals[ratio_x_vals > -0.360] = np.nan
        return 0.612 / (ratio_x_vals + 0.360) + 1.179

def Law2021_3sigma(ratio_x, ratio_y_vals, log=True):
    ratio_y_vals = np.copy(ratio_y_vals)
    if not log:
        ratio_y_vals = np.log10(ratio_y_vals)

    assert ratio_x in ["log N2", "log S2", "log O1"], "x must be one of log N2, log S2 or log O1!"
    if ratio_x == "log N2":
        ratio_y_vals[ratio_y_vals < -0.61] = np.nan
        return -0.479 * ratio_y_vals**4 - 0.594 * ratio_y_vals**3 - 0.542 * ratio_y_vals**2 - 0.056 * ratio_y_vals - 0.143
    elif ratio_x == "log S2":
        ratio_y_vals[ratio_y_vals < -0.80] = np.nan
        return -0.943 * ratio_y_vals**4 - 0.450 * ratio_y_vals**3 + 0.408 * ratio_y_vals**2 - 0.610 * ratio_y_vals - 0.025
    elif ratio_x == "log O1":
        ratio_y_vals[ratio_y_vals > 0.65] = np.nan
        return 18.664 * ratio_y_vals**4 - 36.343 * ratio_y_vals**3 + 22.238 * ratio_y_vals**2 - 6.134 * ratio_y_vals - 0.283

###############################################################################
def Proxauf2014(R):
    """
    Electron density computation from eqn. 3 Proxauf+2014.
    The calculation assumes an electron temperature of 10^4 K.
    Input: 
        R = F([SII]6716) / F([SII]6731)
    """
    log_n_e = 0.0543 * np.tan(-3.0553 * R + 2.8506)\
             + 6.98 - 10.6905 * R\
             + 9.9186 * R**2 - 3.5442 * R**3
    n_e = 10**log_n_e

    # High & low density limits
    if type(R) == "float":
        if n_e < 40:
            return 40
        elif n_e > 1e4:
            return 1e4
    else:
        lolim_mask = n_e < 40
        uplim_mask = n_e > 1e4
        n_e[lolim_mask] = 40
        n_e[uplim_mask] = 1e4

    return n_e

###############################################################################
def bpt_fn(df, s=None):
    """
    Make new columns in the given DataFrame corresponding to their BPT 
    classification.
    """
    # Remove suffixes on columns
    if s is not None:
        df_old = df
        suffix_cols = [c for c in df.columns if c.endswith(s)]
        suffix_removed_cols = [c.split(s)[0] for c in suffix_cols]
        df = df_old.rename(columns=dict(zip(suffix_cols, suffix_removed_cols)))

    old_cols = df.columns
    if np.all(c in df.columns for c in ["log O3", "log N2", "log S2"]):

        """
            Categories:
            If SF(N2):
                if SF(S2): SF
                if NOT SF(S2): ambiguous 
            If NOT SF(N2):
                if SF(S2): ambiguous
                if NOT SF(S2): 
                    if COMP(N2):
                        if COMP(S2): ambiguous
                        if NOT COMP(S2): ambit

            Categories:
            if N2 or O3 or S2 is Nan: NOT CLASSIFIED (-1)
            Then:
                if SF(N2) and SF(S2): SF (0)
                elif COMP(N2) and COMP(S2): COMP (1)
                elif LINER(N2) and LINER(S2): LINER (2)
                elif SEYFERT(N2) and SEYFERT(N2): SEYFERT (3)
                else: AMBIGUOUS (4)

    
        """
        # Re-define this here, because the definition in grid_utils.py
        # excludes log S2 values < -0.314 in its classification of LINERs & 
        # Seyferts
        def Kewley2006(ratio_x, ratio_x_vals, log=True):
            ratio_x_vals = np.copy(ratio_x_vals)
            if not log:
                ratio_y_vals = np.log10(ratio_y_vals)
            """
            Spearating LINERs from Seyferts
            """
            assert ratio_x in ["log S2", "log O1"], "x must be one of log N2, log S2 or log O1!"
            if ratio_x == "log S2":
                return 1.89 * ratio_x_vals + 0.76
            elif ratio_x == "log O1":
                ratio_x_vals[ratio_x_vals < -1.1259] = np.nan
                return 1.18 * ratio_x_vals + 1.30

        # Not classified 
        cond_not_classified  = np.isnan(df["log O3"])
        cond_not_classified |= np.isnan(df["log N2"])
        cond_not_classified |= np.isnan(df["log S2"])
        df_not_classified = df[cond_not_classified]
        if not df_not_classified.empty:
            df_not_classified.loc[:, "BPT"] = "Not classified"
            df_not_classified.loc[:, "BPT (numeric)"] = -1

        # Everything that can be classified
        df_classified = df[~cond_not_classified]
        if not df_classified.empty:
            df_classified.loc[:, "BPT"] = "Ambiguous"

        # SF
        cond_SF  = df_classified["log O3"] < Kauffman2003("log N2", df_classified["log N2"])
        cond_SF &= df_classified["log O3"] < Kewley2001("log S2", df_classified["log S2"])
        df_SF = df_classified[cond_SF]
        if not df_SF.empty:
            df_SF.loc[:, "BPT"] = "SF"
            df_SF.loc[:, "BPT (numeric)"] = 0
        df_classified = df_classified[~cond_SF]

        # Composite
        cond_Comp  = df_classified["log O3"] >= Kauffman2003("log N2", df_classified["log N2"])
        cond_Comp &= df_classified["log O3"] <  Kewley2001("log N2", df_classified["log N2"])
        cond_Comp &= df_classified["log O3"] <  Kewley2001("log S2", df_classified["log S2"])
        df_Comp = df_classified[cond_Comp]
        if not df_Comp.empty:
            df_Comp.loc[:, "BPT"] = "Composite"
            df_Comp.loc[:, "BPT (numeric)"] = 1
        df_classified = df_classified[~cond_Comp]

        # LINER
        cond_LINER  = df_classified["log O3"] >= Kewley2001("log N2", df_classified["log N2"])
        cond_LINER &= df_classified["log O3"] >= Kewley2001("log S2", df_classified["log S2"])
        cond_LINER &= df_classified["log O3"] < Kewley2006("log S2", df_classified["log S2"])
        df_LINER = df_classified[cond_LINER]
        if not df_LINER.empty:
            df_LINER.loc[:, "BPT"] = "LINER"
            df_LINER.loc[:, "BPT (numeric)"] = 2
        df_classified = df_classified[~cond_LINER]

        # Seyfert
        cond_Seyfert  = df_classified["log O3"] >= Kewley2001("log N2", df_classified["log N2"])
        cond_Seyfert &= df_classified["log O3"] >= Kewley2001("log S2", df_classified["log S2"])
        cond_Seyfert &= df_classified["log O3"] >= Kewley2006("log S2", df_classified["log S2"])
        df_Seyfert = df_classified[cond_Seyfert]
        if not df_Seyfert.empty:
            df_Seyfert.loc[:, "BPT"] = "Seyfert"
            df_Seyfert.loc[:, "BPT (numeric)"] = 3

        # Ambiguous
        df_ambiguous = df_classified[~cond_Seyfert]
        if not df_ambiguous.empty:
            df_ambiguous.loc[:, "BPT"] = "Ambiguous"
            df_ambiguous.loc[:, "BPT (numeric)"] = 4

        # Smoosh them back together
        df = df_not_classified.append([df_SF, df_Comp, df_LINER, df_Seyfert, df_ambiguous])
        df = df.astype({"BPT (numeric)": float})
    else:
        df.loc[:, "BPT"] = "Not classified"
        df.loc[:, "BPT (numeric)"] = -1
    
    # Rename columns
    if s is not None:
        # Get list of new columns that have been added
        added_cols = [c for c in df.columns if c not in old_cols]
        suffix_added_cols = [f"{c}{s}" for c in added_cols]
        # Rename the new columns
        df = df.rename(columns=dict(zip(added_cols, suffix_added_cols)))
        # Replace the suffix in the column names
        df = df.rename(columns=dict(zip(suffix_removed_cols, suffix_cols)))

    return df

###############################################################################
def law2021_fn(df, s=None):
    """
    Make new columns in the given DataFrame corresponding to their kinematic 
    classification from Law+2021.
    """
    # Remove suffixes on columns
    if s is not None:
        df_old = df
        suffix_cols = [c for c in df.columns if c.endswith(s)]
        suffix_removed_cols = [c.split(s)[0] for c in suffix_cols]
        df = df_old.rename(columns=dict(zip(suffix_cols, suffix_removed_cols)))

    old_cols = df.columns

    if np.all(c in df.columns for c in ["log O3", "log N2", "log S2"]):
        """
        Categories: cold, intermediate, warm
        """
        # Not classified 
        cond_not_classified  = np.isnan(df["log O3"])
        cond_not_classified |= np.isnan(df["log N2"])
        cond_not_classified |= np.isnan(df["log S2"])
        df_not_classified = df[cond_not_classified]
        if not df_not_classified.empty:
            df_not_classified.loc[:, "Law+2021"] = "Not classified"
            df_not_classified.loc[:, "Law+2021 (numeric)"] = "-1"

        # Everything that can be classified
        df_classified = df[~cond_not_classified]

        # cold
        cond_cold  = df_classified["log O3"] < Law2021_1sigma("log N2", df_classified["log N2"])
        cond_cold &= df_classified["log O3"] < Law2021_1sigma("log S2", df_classified["log S2"])
        # cond_cold &= df_classified["log N2"] < -0.032
        # cond_cold &= df_classified["log S2"] < 0.198
        df_cold = df_classified[cond_cold]
        if not df_cold.empty:
            df_cold.loc[:, "Law+2021"] = "Cold"
            df_cold.loc[:, "Law+2021 (numeric)"] = 0
        df_classified = df_classified[~cond_cold]

        # intermediate
        cond_intermediate  = df_classified["log O3"] >= Law2021_1sigma("log N2", df_classified["log N2"])
        cond_intermediate &= df_classified["log O3"] >= Law2021_1sigma("log S2", df_classified["log S2"])
        cond_intermediate &= df_classified["log N2"] < Law2021_3sigma("log N2", df_classified["log O3"])
        cond_intermediate &= df_classified["log S2"] < Law2021_3sigma("log S2", df_classified["log O3"])
        cond_intermediate &= df_classified["log O3"] > -0.61
        # cond_intermediate &= df_classified["log O3"] < -0.61
        df_intermediate = df_classified[cond_intermediate]
        if not df_intermediate.empty:
            df_intermediate.loc[:, "Law+2021"] = "Intermediate"
            df_intermediate.loc[:, "Law+2021 (numeric)"] = 1
        df_classified = df_classified[~cond_intermediate]

        # warm
        # cond_warm  = df_classified["log O3"] >= Law2021_1sigma("log N2", df_classified["log N2"])
        # cond_warm &= df_classified["log O3"] >= Law2021_1sigma("log S2", df_classified["log S2"])
        cond_warm = df_classified["log N2"] >= Law2021_3sigma("log N2", df_classified["log O3"])
        cond_warm &= df_classified["log S2"] >= Law2021_3sigma("log S2", df_classified["log O3"])
        # cond_warm |= df_classified["log N2"] > -0.032
        df_warm = df_classified[cond_warm]
        if not df_warm.empty:
            df_warm.loc[:, "Law+2021"] = "Warm"
            df_warm.loc[:, "Law+2021 (numeric)"] = 2

        # Ambiguous
        df_ambiguous = df_classified[~cond_warm]
        if not df_ambiguous.empty:
            df_ambiguous.loc[:, "Law+2021"] = "Ambiguous"
            df_ambiguous.loc[:, "Law+2021 (numeric)"] = 3

        # Smoosh them back together
        df = df_not_classified.append([df_cold, df_intermediate, df_warm, df_ambiguous])
        df = df.astype({"Law+2021 (numeric)": float})


    # Rename columns
    if s is not None:
        # Get list of new columns that have been added
        added_cols = [c for c in df.columns if c not in old_cols]
        suffix_added_cols = [f"{c}{s}" for c in added_cols]
        # Rename the new columns
        df = df.rename(columns=dict(zip(added_cols, suffix_added_cols)))
        # Replace the suffix in the column names
        df = df.rename(columns=dict(zip(suffix_removed_cols, suffix_cols)))

    return df

###############################################################################
def ratio_fn(df, s=None):
    """
    Make new columns in the given DataFrame corresponding to certain line ratios.
    """
    # Remove suffixes on columns
    if s is not None:
        df_old = df
        suffix_cols = [c for c in df.columns if c.endswith(s)]
        suffix_removed_cols = [c.split(s)[0] for c in suffix_cols]
        df = df_old.rename(columns=dict(zip(suffix_cols, suffix_removed_cols)))

    def in_df(col_list):
        for col in col_list:
            if col not in df.columns:
                return False
        return True

    def not_in_df(col_list):
        for col in col_list:
            if col in df.columns:
                return False
        return True

    # First, add doublets if they are in the dataframe.
    old_cols = df.columns
    if in_df(["OII3726", "OII3729"]) and not_in_df(["OII3726+OII3729"]):
        df["OII3726+OII3729"] = df["OII3726"] + df["OII3729"]
    if in_df(["SII6716", "SII6731"]) and not_in_df(["SII6716+SII6731"]):
        df["SII6716+SII6731"] = df["SII6716"] + df["SII6731"]
    # Errors
    if in_df(["OII3726 error", "OII3729 error"]) and not_in_df(["OII3726+OII3729 error"]):
        df["OII3726+OII3729 error"] = np.sqrt(df["OII3726 error"]**2 + df["OII3729 error"]**2)
    if in_df(["SII6716 error", "SII6731 error"]) and not_in_df(["SII6716+SII6731 error"]):
        df["SII6716+SII6731 error"] = np.sqrt(df["SII6716 error"]**2 + df["SII6731 error"]**2)

    # Then, split doublets if the line ratios are fixed.
    if in_df(["OIII4959+OIII5007"]) and not_in_df(["OIII4959", "OIII5007"]):
        df["OIII4959"] = df["OIII4959+OIII5007"] / (1 + 2.94)
        df["OIII5007"] = df["OIII4959+OIII5007"] / (1 + 1 / 2.94)
    if in_df(["NII6548+NII6583"]) and not_in_df(["NII6548", "NII6583"]):
        df["NII6548"] = df["NII6548+NII6583"] / (1 + 3.06)
        df["NII6583"] = df["NII6548+NII6583"] / (1 + 1 / 3.06)
    if in_df(["SIII9069+SIII9531"]) and not_in_df(["SIII9069", "SIII9531"]):
        df["SIII9069"] = df["SIII9069+SIII9531"] / (1 + 2.47)
        df["SIII9531"] = df["SIII9069+SIII9531"] / (1 + 1 / 2.47)
    # Errors
    if in_df(["OIII4959+OIII5007 error"]) and not_in_df(["OIII4959 error", "OIII5007 error"]):
        df["OIII4959 error"] = df["OIII4959+OIII5007 error"] / (1 + 2.94)
        df["OIII5007 error"] = df["OIII4959+OIII5007 error"] / (1 + 1 / 2.94)
    if in_df(["NII6548+NII6583 error"]) and not_in_df(["NII6548 error", "NII6583 error"]):
        df["NII6548 error"] = df["NII6548+NII6583 error"] / (1 + 3.06)
        df["NII6583 error"] = df["NII6548+NII6583 error"] / (1 + 1 / 3.06)
    if in_df(["SIII9069+SIII9531 error"]) and not_in_df(["SIII9069 error", "SIII9531 error"]):
        df["SIII9069 error"] = df["SIII9069+SIII9531 error"] / (1 + 2.47)
        df["SIII9531 error"] = df["SIII9069+SIII9531 error"] / (1 + 1 / 2.47)

    # Then, add in the second line if only one line is present.
    if in_df(["NII6583"]) and not_in_df(["NII6548"]):
        df["NII6548"] = df["NII6583"] / 3.06 
    elif in_df(["NII6548"]) and not_in_df(["NII6583"]):
        df["NII6583"] = df["NII6548"] * 3.06

    if in_df(["OIII5007"]) and not_in_df(["OIII4959"]):
        df["OIII4959"] = df["OIII5007"] / 2.94
    elif in_df(["OIII4959"]) and not_in_df(["OIII5007"]):
        df["OIII5007"] = df["OIII4959"] * 2.94

    if in_df(["SIII9069"]) and not_in_df(["SIII9531"]):
        df["SIII9531"] = df["SIII9069"] * 2.947
    elif in_df(["SIII9531"]) and not_in_df(["SIII9069"]):
        df["SIII9069"] = df["SIII9531"] / 2.947

    # Errors
    if in_df(["NII6583 error"]) and not_in_df(["NII6548 error"]):
        df["NII6548 error"] = df["NII6583 error"] / 3.06 
    elif in_df(["NII6548 error"]) and not_in_df(["NII6583 error"]):
        df["NII6583 error"] = df["NII6548 error"] * 3.06

    if in_df(["OIII5007 error"]) and not_in_df(["OIII4959 error"]):
        df["OIII4959 error"] = df["OIII5007 error"] / 2.94
    elif in_df(["OIII4959 error"]) and not_in_df(["OIII5007 error"]):
        df["OIII5007 error"] = df["OIII4959 error"] * 2.94

    if in_df(["SIII9069 error"]) and not_in_df(["SIII9531 error"]):
        df["SIII9531 error"] = df["SIII9069 error"] * 2.947
    elif in_df(["SIII9531 error"]) and not_in_df(["SIII9069 error"]):
        df["SIII9069 error"] = df["SIII9531 error"] / 2.947

    # Add the doublets
    if in_df(["NII6548", "NII6583"]) and not_in_df(["NII6548+NII6583"]):
        df["NII6548+NII6583"] = df["NII6548"] + df["NII6583"]
    if in_df(["OIII4959", "OIII5007"]) and not_in_df(["OIII4959+OIII5007"]):
        df["OIII4959+OIII5007"] = df["OIII4959"] + df["OIII5007"]
    if in_df(["SIII9069", "SIII9531"]) and not_in_df(["SIII9069+SIII9531"]):
        df["SIII9069+SIII9531"] = df["SIII9069"] + df["SIII9531"]
    # Errors
    if in_df(["NII6548 error", "NII6583 error"]) and not_in_df(["NII6548+NII6583 error"]):
        df["NII6548+NII6583 error"] = np.sqrt(df["NII6548 error"]**2 + df["NII6583 error"]**2)
    if in_df(["OIII4959 error", "OIII5007 error"]) and not_in_df(["OIII4959+OIII5007 error"]):
        df["OIII4959+OIII5007 error"] = np.sqrt(df["OIII4959 error"]**2 + df["OIII5007 error"]**2)
    if in_df(["SIII9069 error", "SIII9531 error"]) and not_in_df(["SIII9069+SIII9531 error"]):
        df["SIII9069+SIII9531 error"] = np.sqrt(df["SIII9069 error"]**2 + df["SIII9531 error"]**2)

    # Used for metallicity, ionisation parameter diagnostics
    if in_df(["NII6583", "OII3726+OII3729"]):
        df["N2O2"] = np.log10((df["NII6583"]) / (df["OII3726+OII3729"]))
    if in_df(["NII6583", "SII6716+SII6731"]):
        df["N2S2"] = np.log10((df["NII6583"]) / (df["SII6716+SII6731"]))
    if in_df(["OIII5007", "HBETA", "NII6583", "HALPHA"]):
        df["O3N2"] = np.log10((df["OIII5007"] / df["HBETA"]) / (df["NII6583"] / df["HALPHA"]))
    if in_df(["OIII4959+OIII5007", "OII3726+OII3729", "HBETA"]):
        df["R23"] = np.log10((df["OIII4959+OIII5007"] + df["OII3726+OII3729"]) / (df["HBETA"]))
    if in_df(["SII6716+SII6731", "SIII9069", "SIII9531", "HALPHA"]):
        df["S23"] = np.log10((df["SII6716+SII6731"] + df["SIII9069"] + df["SIII9531"]) / df["HALPHA"])
    if in_df(["SIII9069+SIII9531", "OIII4959+OIII5007"]):
        df["S3O3"] = np.log10((df["SIII9069+SIII9531"]) / (df["OIII4959+OIII5007"]))
    if in_df(["OIII5007", "OII3726", "OII3729"]):
        df["O3O2"] = np.log10((df["OIII5007"]) / (df["OII3726"] + df["OII3729"]))
    if in_df(["OIII5007", "OII3726"]):
        df["O2O3"] = np.log10(df["OII3726"] / df["OIII5007"])  # fig. 13 of Allen+1999
    if in_df(["OIII5007", "OI6300"]):
        df["O1O3"] = np.log10(df["OI6300"] / df["OIII5007"])  # fig. 15 of Allen+1999
    if in_df(["OI6300", "OII3726", "OII3729"]):
        df["O3O2"] = np.log10((df["OIII5007"]) / (df["OII3726"] + df["OII3729"]))
    if in_df(["SIII9069+SIII9531", "SII6716+SII6731"]):
        df["S32"] = np.log10((df["SIII9069+SIII9531"]) / (df["SII6716+SII6731"]))
    if in_df(["SIII9069+SIII9531", "HALPHA"]):
        df["S3"] = np.log10((df["SIII9069+SIII9531"]) / (df["HALPHA"]))
    if in_df(["NeV3426", "NeIII3869"]):
        df["Ne53"] = np.log10(df["NeV3426"] / df["NeIII3869"])

    # Standard BPT axes
    if in_df(["NII6583", "HALPHA"]):
        df["log N2"] = np.log10(df["NII6583"] / df["HALPHA"])
        df["N2"] = df["NII6583"] / df["HALPHA"]

    if in_df(["OI6300", "HALPHA"]):
        df["log O1"] = np.log10(df["OI6300"] / df["HALPHA"])
        df["O1"] = df["OI6300"] / df["HALPHA"]

    if in_df(["SII6716+SII6731", "HALPHA"]):
        df["log S2"] = np.log10((df["SII6716+SII6731"]) / df["HALPHA"])
        df["S2"] = (df["SII6716+SII6731"]) / df["HALPHA"]

    if in_df(["OIII5007", "HBETA"]):
        df["log O3"] = np.log10(df["OIII5007"] / df["HBETA"])
        df["O3"] = df["OIII5007"] / df["HBETA"]

    if in_df(["HeII4686", "HBETA"]):
        df["He2"] = df["HeII4686"] / df["HBETA"]
        df["log He2"] = np.log10(df["He2"])

    if in_df(["SII6716", "SII6731"]):
        df["S2 ratio"] = df["SII6716"] / df["SII6731"] 

    # ERRORS for standard BPT axes
    if in_df(["NII6583 error", "HALPHA error"]):
        df["N2 error"] = df["N2"] * np.sqrt((df["NII6583 error"] / df["NII6583"])**2 + (df["HALPHA error"] / df["HALPHA"])**2)
        df["log N2 error (lower)"] = df["log N2"] - np.log10(df["N2"] - df["N2 error"])
        df["log N2 error (upper)"] = np.log10(df["N2"] + df["N2 error"]) -  df["log N2"]

    if in_df(["OI6300 error", "HALPHA error"]):
        df["O1 error"] = df["O1"] * np.sqrt((df["OI6300 error"] / df["OI6300"])**2 + (df["HALPHA error"] / df["HALPHA"])**2)
        df["log O1 error (lower)"] = df["log O1"] - np.log10(df["O1"] - df["O1 error"])
        df["log O1 error (upper)"] = np.log10(df["O1"] + df["O1 error"]) -  df["log O1"]

    if in_df(["SII6716+SII6731 error", "HALPHA error"]):
        df["S2 error"] = df["S2"] * np.sqrt((df["SII6716+SII6731 error"] / df["SII6716+SII6731"])**2 + (df["HALPHA error"] / df["HALPHA"])**2)
        df["log S2 error (lower)"] = df["log S2"] - np.log10(df["S2"] - df["S2 error"])
        df["log S2 error (upper)"] = np.log10(df["S2"] + df["S2 error"]) -  df["log S2"]

    if in_df(["OIII5007 error", "HBETA error"]):
        df["O3 error"] = df["O3"] * np.sqrt((df["OIII5007 error"] / df["OIII5007"])**2 + (df["HBETA error"] / df["HBETA"])**2)
        df["log O3 error (lower)"] = df["log O3"] - np.log10(df["O3"] - df["O3 error"])
        df["log O3 error (upper)"] = np.log10(df["O3"] + df["O3 error"]) -  df["log O3"]
    
    if in_df(["SII6716 error", "SII6731 error"]):
        df["S2 ratio error"] = df["S2 ratio"] * np.sqrt((df["SII6716 error"] / df["SII6716"])**2 + (df["SII6731 error"] / df["SII6731"])**2)

    # Rename columns
    if s is not None:
        # Get list of new columns that have been added
        added_cols = [c for c in df.columns if c not in old_cols]
        suffix_added_cols = [f"{c}{s}" for c in added_cols]
        # Rename the new columns
        df = df.rename(columns=dict(zip(added_cols, suffix_added_cols)))
        # Replace the suffix in the column names
        df = df.rename(columns=dict(zip(suffix_removed_cols, suffix_cols)))

    return df

