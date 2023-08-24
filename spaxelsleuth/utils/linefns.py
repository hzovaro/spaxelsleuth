import numpy as np
import pandas as pd

import logging
logger = logging.getLogger(__name__)

bpt_dict = {
     "0.0": "SF",
     "1.0": "Composite",
     "2.0": "LINER",
     "3.0": "Seyfert",
     "4.0": "Ambiguous",
     "-1.0": "Not classified",
}
def bpt_num_to_str(s):
    return [bpt_dict[str(a)] for a in s]

######################################################################
def compute_eline_luminosity(df, ncomponents_max, eline_list):
    """Compute emission line luminosities."""
    # Line luminosity: units of erg s^-1 kpc^-2
    if all([col in df for col in ["D_L (Mpc)", "Bin size (square kpc)"]]):
        for eline in eline_list:
            if all([col in df for col in [f"{eline} (total)", f"{eline} error (total)"]]):
                df[f"{eline} luminosity (total)"] = df[f"{eline} (total)"] * 1e-16 * 4 * np.pi * (df["D_L (Mpc)"] * 1e6 * 3.086e18)**2 * 1 / df["Bin size (square kpc)"]
                df[f"{eline} luminosity error (total)"] = df[f"{eline} error (total)"] * 1e-16 * 4 * np.pi * (df["D_L (Mpc)"] * 1e6 * 3.086e18)**2 * 1 / df["Bin size (square kpc)"]
            for nn in range(ncomponents_max):
                if all([col in df for col in [f"{eline} (component {nn + 1})", f"{eline} error (component {nn + 1})"]]):
                    df[f"{eline} luminosity (component {nn + 1})"] = df[f"{eline} (component {nn + 1})"] * 1e-16 * 4 * np.pi * (df["D_L (Mpc)"] * 1e6 * 3.086e18)**2 * 1 / df["Bin size (square kpc)"]
                    df[f"{eline} luminosity error (component {nn + 1})"] = df[f"{eline} error (component {nn + 1})"] * 1e-16 * 4 * np.pi * (df["D_L (Mpc)"] * 1e6 * 3.086e18)**2 * 1 / df["Bin size (square kpc)"]

    return df

######################################################################
def compute_FWHM(df, ncomponents_max):
    """Compute the Full-Width at Half Maximum from the velocity dispersion."""
    for nn in range(ncomponents_max):
        if f"sigma_gas (component {nn + 1})" in df:
            df[f"FWHM_gas (component {nn + 1})"] = df[f"sigma_gas (component {nn + 1})"] * 2 * np.sqrt(2 * np.log(2))
        if f"sigma_gas error (component {nn + 1})" in df:
            df[f"FWHM_gas error (component {nn + 1})"] = df[f"sigma_gas error (component {nn + 1})"] * 2 * np.sqrt(2 * np.log(2))

    return df

###############################################################################
# Reference lines from literature
def Kewley2001(ratio_x, ratio_x_vals, log=True):
    """
    Returns the "maximum starburst" demarcation lines of Kewley et al. (2001)
    for the N2, S2 and O1 BPT diagrams.

    INPUTS
    ---------------------------------------------------------------------------
    ratio_x         str
        x-axis ratio. Must be one of log N2, log S2 or log O1.

    ratio_x_vals    Numpy array
        x-axis values for which to compute the demarcation line.

    log             bool
        If True, return the log of the computed O3 values.

    OUTPUTS
    ---------------------------------------------------------------------------
    O3 (or log O3) ratio values corresponding to the demarcation line and the 
    input x-axis ratio and values.

    """
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
    """
    Returns the empirically derived star-forming/AGN demarcation line of 
    Kauffman et al. (2003) for the N2 BPT diagram.

    INPUTS
    ---------------------------------------------------------------------------
    ratio_x         str
        x-axis ratio. Must be one of log N2, log S2 or log O1.

    ratio_x_vals    Numpy array
        x-axis values for which to compute the demarcation line.

    log             bool
        If True, return the log of the computed O3 values.

    OUTPUTS
    ---------------------------------------------------------------------------
    O3 (or log O3) ratio values corresponding to the demarcation line and the 
    input x-axis ratio and values.

    """
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
    """
    Returns the Seyfert/LINER separation lines of Kewley et al. (2006) for the 
    S2 and O1 BPT diagrams.

    INPUTS
    ---------------------------------------------------------------------------
    ratio_x         str
        x-axis ratio. Must be one of log N2, log S2 or log O1.

    ratio_x_vals    Numpy array
        x-axis values for which to compute the demarcation line.

    log             bool
        If True, return the log of the computed O3 values.

    OUTPUTS
    ---------------------------------------------------------------------------
    O3 (or log O3) ratio values corresponding to the demarcation line and the 
    input x-axis ratio and values.

    """
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
    """
    Returns the 1-sigma demarcation line based on the kinematic classification
    scheme of Law et al. (2021) for the N2, S2 and O1 BPT diagrams.

    INPUTS
    ---------------------------------------------------------------------------
    ratio_x         str
        x-axis ratio. Must be one of log N2, log S2 or log O1.

    ratio_x_vals    Numpy array
        x-axis values for which to compute the demarcation line.

    log             bool
        If True, return the log of the computed O3 values.

    OUTPUTS
    ---------------------------------------------------------------------------
    O3 (or log O3) ratio values corresponding to the demarcation line and the 
    input x-axis ratio and values.

    """
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
    """
    Returns the 3-sigma demarcation line based on the kinematic classification
    scheme of Law et al. (2021) for the N2, S2 and O1 BPT diagrams.

    INPUTS
    ---------------------------------------------------------------------------
    ratio_x         str
        x-axis ratio. Must be one of log N2, log S2 or log O1.

    ratio_x_vals    Numpy array
        x-axis values for which to compute the demarcation line.

    log             bool
        If True, return the log of the computed O3 values.

    OUTPUTS
    ---------------------------------------------------------------------------
    O3 (or log O3) ratio values corresponding to the demarcation line and the 
    input x-axis ratio and values.

    """
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
    Electron density computation from eqn. 3 of Proxauf (2014), which assumes 
    an electron temperature of 10^4 K.

    INPUT
    --------------------------------------------------------------------------
    R       float or Numpy array
        Flux ratio F([SII]6716) / F([SII]6731).

    OUTPUT
    --------------------------------------------------------------------------
    n_e (in cm^-3) computed using eqn. 3 of Proxauf (2014).

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

    INPUTS
    --------------------------------------------------------------------------
    df:     pandas DataFrame
        DataFrame in which to compute 

    s:      str 
        Column suffix to trim before carrying out computation - e.g. if 
        you want to compute metallicities for "total" fluxes, and the 
        columns of the DataFrame look like 

            "HALPHA (total)", "HALPHA error (total)", etc.,

        then setting s=" (total)" will mean that this function "sees"

            "HALPHA", "HALPHA error".

        Useful for running this function on different emission line 
        components. The suffix is added back to the columns (and appended
        to any new columns that are added) before being returned. For 
        example, using the above example, the new added columns will be 

            "BPT (numeric) (total)"

    OUTPUTS
    -----------------------------------------------------------------------
    The original DataFrame with the following columns added:

        BPT (numeric)   int
            Integer code corresponding to the BPT classification. 

    PREREQUISITES 
    -----------------------------------------------------------------------
    BPT classification requires the emission line ratios log O3, log N2 and 
    log S2 to be defined. These can be computed using ratio_fn().

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
            df_not_classified.loc[:, "BPT (numeric)"] = -1

        # Everything that can be classified
        df_classified = df[~cond_not_classified]
        if not df_classified.empty:
            #TODO why is this line here???
            df_classified.loc[:, "BPT (numeric)"] = 4

        # SF
        cond_SF  = df_classified["log O3"] < Kauffman2003("log N2", df_classified["log N2"])
        cond_SF &= df_classified["log O3"] < Kewley2001("log S2", df_classified["log S2"])
        df_SF = df_classified[cond_SF]
        if not df_SF.empty:
            df_SF.loc[:, "BPT (numeric)"] = 0
        df_classified = df_classified[~cond_SF]

        # Composite
        cond_Comp  = df_classified["log O3"] >= Kauffman2003("log N2", df_classified["log N2"])
        cond_Comp &= df_classified["log O3"] <  Kewley2001("log N2", df_classified["log N2"])
        cond_Comp &= df_classified["log O3"] <  Kewley2001("log S2", df_classified["log S2"])
        df_Comp = df_classified[cond_Comp]
        if not df_Comp.empty:
            df_Comp.loc[:, "BPT (numeric)"] = 1
        df_classified = df_classified[~cond_Comp]

        # LINER
        cond_LINER  = df_classified["log O3"] >= Kewley2001("log N2", df_classified["log N2"])
        cond_LINER &= df_classified["log O3"] >= Kewley2001("log S2", df_classified["log S2"])
        cond_LINER &= df_classified["log O3"] < Kewley2006("log S2", df_classified["log S2"])
        df_LINER = df_classified[cond_LINER]
        if not df_LINER.empty:
            df_LINER.loc[:, "BPT (numeric)"] = 2
        df_classified = df_classified[~cond_LINER]

        # Seyfert
        cond_Seyfert  = df_classified["log O3"] >= Kewley2001("log N2", df_classified["log N2"])
        cond_Seyfert &= df_classified["log O3"] >= Kewley2001("log S2", df_classified["log S2"])
        cond_Seyfert &= df_classified["log O3"] >= Kewley2006("log S2", df_classified["log S2"])
        df_Seyfert = df_classified[cond_Seyfert]
        if not df_Seyfert.empty:
            df_Seyfert.loc[:, "BPT (numeric)"] = 3

        # Ambiguous
        df_ambiguous = df_classified[~cond_Seyfert]
        if not df_ambiguous.empty:
            df_ambiguous.loc[:, "BPT (numeric)"] = 4

        # Smoosh them back together
        df = pd.concat([df_not_classified, df_SF, df_Comp, df_LINER, df_Seyfert, df_ambiguous])
        df = df.astype({"BPT (numeric)": float})
    else:
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
def whav_fn(df, ncomponents):
    """
    Make new columns in the given DataFrame corresponding to the WHAV* 
    classification of each spaxel.
    """
    df["WHAV*"] = "Unknown"  # Initialise everything to "unknown"
    df["WHAN"] = "Unknown"  # Initialise everything to "unknown"

    #///////////////////////////////////////////////////////////////////////////////
    # Step 1: filter out evolved stars
    cond = df["HALPHA EW (total)"] <= 3
    df.loc[cond, "WHAV*"] = "HOLMES"
    df.loc[cond, "WHAN"] = "HOLMES"
    cond_remainder = df["WHAV*"] == "Unknown"

    #///////////////////////////////////////////////////////////////////////////////
    # Step 2: use the N2 ratio to divide into SF, mixed and AGN/evolved stars/shocks
    # Because we used the TOTAL N2 ratio in each spaxel to determine these boundaries, these categories are representative of the DOMINANT ionisation mechanism in each spaxel.
    cond_SF = cond_remainder & (df["log N2 (total)"] < -0.35)
    cond_Mixing = cond_remainder & (df["log N2 (total)"] >= -0.35) & (df["log N2 (total)"] < -0.20)
    cond_AGN = cond_remainder & (df["log N2 (total)"] >= -0.20)
    cond_no_N2 = cond_remainder & (np.isnan(df["log N2 (total)"]))

    df.loc[cond_SF, "WHAN"] = "SF" 
    df.loc[cond_Mixing, "WHAN"] = "Mixing"
    df.loc[cond_AGN, "WHAN"] = "AGN/HOLMES/shocks"
    df.loc[cond_no_N2, "WHAN"] = "No N2"

    #///////////////////////////////////////////////////////////////////////////////
    # For convenience: mark components as possible HOLMES 
    # Question: how confident can we be that these are ALWAYS HOLMES? how common are components from e.g. LLAGN?
    for nn in range(ncomponents):
        cond_possible_HOLMES = cond_AGN & (df[f"HALPHA EW (component {nn + 1})"] < 3) & (df[f"sigma_gas - sigma_* (component {nn + 1})"] < 0)
        df.loc[cond_possible_HOLMES, f"Possible HOLMES (component {nn + 1})"] = True
        df.loc[~cond_possible_HOLMES, f"Possible HOLMES (component {nn + 1})"] = False
        
    #///////////////////////////////////////////////////////////////////////////////
    # For convenience: mark components as being kinematically disturbed (by 3sigma)
    for nn in range(ncomponents):
        cond_kinematically_disturbed = df[f"sigma_gas - sigma_* (component {nn + 1})"] - 3 * df[f"sigma_gas - sigma_* error (component {nn + 1})"] > 0
        df.loc[cond_kinematically_disturbed, f"Kinematically disturbed (component {nn + 1})"] = True
        df.loc[~cond_kinematically_disturbed, f"Kinematically disturbed (component {nn + 1})"] = False
    if ncomponents == 3:
        df["Number of kinematically disturbed components"] = df["Kinematically disturbed (component 1)"].astype(int) + df["Kinematically disturbed (component 2)"].astype(int) + df["Kinematically disturbed (component 3)"].astype(int)
    else:
        df["Number of kinematically disturbed components"] = df["Kinematically disturbed (component 1)"].astype(int)

    # Test 
    for nn in range(ncomponents):
        assert not df.loc[np.isnan(df[f"sigma_gas - sigma_* (component {nn + 1})"]), f"Kinematically disturbed (component {nn + 1})"].any(),\
            f"There are rows where sigma_gas - sigma_* (component {nn + 1}) == NaN but 'Kinematically disturbed (component {nn + 1})' == True!"

    for nn in range(ncomponents):
        assert not df.loc[np.isnan(df[f"log HALPHA EW (component {nn + 1})"]), f"Possible HOLMES (component {nn + 1})"].any(),\
            f"There are rows where log HALPHA EW (component {nn + 1}) == NaN but 'Possible HOLMES (component {nn + 1})' == True!"
        assert not df.loc[np.isnan(df[f"sigma_gas - sigma_* (component {nn + 1})"]), f"Possible HOLMES (component {nn + 1})"].any(),\
            f"There are rows where sigma_gas - sigma_* (component {nn + 1}) == NaN but 'Possible HOLMES (component {nn + 1})' == True!"
        assert not df.loc[df[f"log HALPHA EW (component {nn + 1})"] > 3, f"Possible HOLMES (component {nn + 1})"].any(),\
            f"There are rows where log HALPHA EW (component {nn + 1}) > 3 but 'Possible HOLMES (component {nn + 1})' == True!"
        
        assert df[df[f"Possible HOLMES (component {nn + 1})"] & df[f"Kinematically disturbed (component {nn + 1})"]].shape[0] == 0,\
            f"There are rows where both 'Possible HOLMES (component {nn + 1})' and 'Kinematically disturbed (component {nn + 1})' are true!"

    if ncomponents == 3:
        # ///////////////////////////////////////////////////////////////////////////////
        # SF-like spaxels
        #///////////////////////////////////////////////////////////////////////////////
        # Wind: number of components > 1, AND EITHER delta sigma of 1 or 2 is > 0
        # Note: may want to also add if ncomponents == 1 but delta_sigma >> 0. 
        # How many SF (either classified via BPT or N2) spaxels are there like this, though? Just checked - only ~0.1% have dsigma > 0 by 3sigma, so probably don't worry 
        cond_SF_no_wind = cond_SF & ~(df["Kinematically disturbed (component 1)"] | df["Kinematically disturbed (component 2)"] | df["Kinematically disturbed (component 3)"])
        df.loc[cond_SF_no_wind, "WHAV*"] = "SF + no wind"

        cond_SF_wind = cond_SF & (df["Kinematically disturbed (component 1)"] | df["Kinematically disturbed (component 2)"] | df["Kinematically disturbed (component 3)"])
        df.loc[cond_SF_wind, "WHAV*"] = "SF + wind"

        # SF + HOLMES 
        cond_SF_no_wind_HOLMES = cond_SF_no_wind & (df["Number of components"] >= 2) & (df["Possible HOLMES (component 1)"] | df["Possible HOLMES (component 2)"] | df["Possible HOLMES (component 3)"])
        df.loc[cond_SF_no_wind_HOLMES, "WHAV*"] = "SF + HOLMES + no wind"

        cond_SF_wind_HOLMES = cond_SF_wind & (df["Number of components"] >= 2) & (df["Possible HOLMES (component 1)"] | df["Possible HOLMES (component 2)"] | df["Possible HOLMES (component 3)"])
        df.loc[cond_SF_wind_HOLMES, "WHAV*"] = "SF + HOLMES + wind"


        # Note: what to do about low-metallicity AGN? e.g., ones that are classified as ambiguous that have log N2 < -0.35 so get lumped in with SF?

        #///////////////////////////////////////////////////////////////////////////////
        # Mixing-like spaxels
        #///////////////////////////////////////////////////////////////////////////////
        # wind/no wind
        # Note: <1% of composite/mixing-like spaxels have ncomponents == 1 but delta_sigma >> 0 by 3sigma
        cond_Mixing_no_wind = cond_Mixing & ~(df["Kinematically disturbed (component 1)"] | df["Kinematically disturbed (component 2)"] | df["Kinematically disturbed (component 3)"])
        df.loc[cond_Mixing_no_wind, "WHAV*"] = "Mixing + no wind"

        cond_Mixing_wind = cond_Mixing & (df["Kinematically disturbed (component 1)"] | df["Kinematically disturbed (component 2)"] | df["Kinematically disturbed (component 3)"])
        df.loc[cond_Mixing_wind, "WHAV*"] = "Mixing + wind"

        # Mixing + HOLMES 
        cond_Mixing_no_wind_HOLMES = cond_Mixing_no_wind & (df["Number of components"] >= 2) & (df["Possible HOLMES (component 1)"] | df["Possible HOLMES (component 2)"] | df["Possible HOLMES (component 3)"])
        df.loc[cond_Mixing_no_wind_HOLMES, "WHAV*"] = "Mixing + HOLMES + no wind"

        # Mixing + HOLMES + wind
        cond_Mixing_wind_HOLMES = cond_Mixing_wind & (df["Number of components"] >= 2) & (df["Possible HOLMES (component 1)"] | df["Possible HOLMES (component 2)"] | df["Possible HOLMES (component 3)"])
        df.loc[cond_Mixing_wind_HOLMES, "WHAV*"] = "Mixing + HOLMES + wind"

        #///////////////////////////////////////////////////////////////////////////////
        # AGN-like spaxels
        #///////////////////////////////////////////////////////////////////////////////
        # If there is 1 component and its EW is > 0, then it's an AGN. Note that Seyfert-like components have a range of EWs, so we can't really split between LLAGN and Seyferts here - really need [OIII] for that.
        cond_AGN_no_wind = cond_AGN & (df["Number of components"] == 1) & (df["HALPHA EW (component 1)"] > 3) & ~df["Kinematically disturbed (component 1)"]
        df.loc[cond_AGN_no_wind, "WHAV*"] = "AGN only"

        # AGN + wind
        cond_AGN_nowind = cond_AGN & ~(df["Kinematically disturbed (component 1)"] | df["Kinematically disturbed (component 2)"] | (df["Kinematically disturbed (component 3)"] ))
        df.loc[cond_AGN_nowind, "WHAV*"] = "AGN + no wind"

        cond_AGN_wind = cond_AGN & (df["Kinematically disturbed (component 1)"] | df["Kinematically disturbed (component 2)"] | (df["Kinematically disturbed (component 3)"] ))
        df.loc[cond_AGN_wind, "WHAV*"] = "AGN + wind"

        # If there are multiple components and at least one of them is in the HOLMES regime, then classify it as HOLMES + AGN. 
        cond_AGN_nowind_HOLMES = cond_AGN_nowind & (df["Number of components"] >= 2) & (df["Possible HOLMES (component 1)"] | df["Possible HOLMES (component 2)"] | df["Possible HOLMES (component 3)"])
        df.loc[cond_AGN_nowind_HOLMES, "WHAV*"] = "AGN + HOLMES + no wind"

        cond_AGN_wind_HOLMES = cond_AGN_wind & (df["Number of components"] >= 2) & (df["Possible HOLMES (component 1)"] | df["Possible HOLMES (component 2)"] | df["Possible HOLMES (component 3)"])
        df.loc[cond_AGN_wind_HOLMES, "WHAV*"] = "AGN + HOLMES + wind"

        #///////////////////////////////////////////////////////////////////////////////
        # Numerical labels
        #///////////////////////////////////////////////////////////////////////////////
        num_dict = {
            "Unknown": -1,
            "HOLMES": 0,
            "Mixing + HOLMES + no wind": 1,
            "Mixing + HOLMES + wind": 2,
            "Mixing + no wind": 3,
            "Mixing + wind": 4,    
            "AGN + HOLMES + no wind": 5,
            "AGN + HOLMES + wind": 6,
            "AGN + no wind": 7,
            "AGN + wind": 8,
            "SF + HOLMES + no wind": 9,
            "SF + HOLMES + wind": 10,
            "SF + no wind": 11,
            "SF + wind": 12
        }
        cats = list(num_dict.keys())
        for cat in cats:
            df.loc[df["WHAV*"] == cat, "WHAV* (numeric)"] = num_dict[cat]
    else:
        logger.warn("not computing WHAV* categories because ncomponents is not 3")

    return df

###############################################################################
def law2021_fn(df, s=None):
    """
    Make new columns in the given DataFrame corresponding to their kinematic 
    classification from Law et al. (2021).

    INPUTS
    --------------------------------------------------------------------------
    df:     pandas DataFrame
        DataFrame in which to compute 

    s:      str 
        Column suffix to trim before carrying out computation - e.g. if 
        you want to compute metallicities for "total" fluxes, and the 
        columns of the DataFrame look like 

            "HALPHA (total)", "HALPHA error (total)", etc.,

        then setting s=" (total)" will mean that this function "sees"

            "HALPHA", "HALPHA error".

        Useful for running this function on different emission line 
        components. The suffix is added back to the columns (and appended
        to any new columns that are added) before being returned. For 
        example, using the above example, the new added columns will be 

            "Law+2021 (total)", "Law+2021 (numeric) (total)"

    OUTPUTS
    -----------------------------------------------------------------------
    The original DataFrame with the following columns added:

        Law+2021             str
            Law et al. (2021) classification.

        Law+2021 (numeric)   int
            Integer code corresponding to the Law et al. classification. Useful 
            in plotting.

    PREREQUISITES 
    -----------------------------------------------------------------------
    Law et al. (2021) classification requires the emission line ratios log O3, 
    log N2 and log S2 to be defined. These can be computed using ratio_fn().
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
        df = pd.concat([df_not_classified, df_cold, df_intermediate, df_warm, df_ambiguous])
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
    Given an input DataFrame containing emission line fluxes, computes emission 
    line ratios, e.g. N2 and S2, and adds additional columns containing the 
    sums of emission line doublets - e.g., if the DataFrame contains OIII5007
    but not OIII4959, the OIII4959 flux is computed based on the OIII5007 flux
    using the expected ratio predicted by QM and added to the DataFrame. 
    The column OIII4959+OIII5007 will also be added (and similar for other 
    emission lines, e.g. SIII9531 and SIII9051, NII6548 and NII6583).

    INPUTS
    --------------------------------------------------------------------------
    df:     pandas DataFrame
        DataFrame in which to compute 

    s:      str 
        Column suffix to trim before carrying out computation - e.g. if 
        you want to compute metallicities for "total" fluxes, and the 
        columns of the DataFrame look like 

            "HALPHA (total)", "HALPHA error (total)", etc.,

        then setting s=" (total)" will mean that this function "sees"

            "HALPHA", "HALPHA error".

        Useful for running this function on different emission line 
        components. The suffix is added back to the columns (and appended
        to any new columns that are added) before being returned. For 
        example, using the above example, the new added columns will be 

            "log N2 (total)", "log N2 error (lower) (total)", 
            "log N2 error (upper) (total)"

    OUTPUTS
    -----------------------------------------------------------------------
    The original DataFrame with new columns added.

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
    if in_df(["SII6716+SII6731", "SIII9069+SIII9531", "HALPHA"]):
        df["S23"] = np.log10((df["SII6716+SII6731"] + df["SIII9069+SIII9531"]) / df["HALPHA"])
    if in_df(["SIII9069+SIII9531", "OIII4959+OIII5007"]):
        df["S3O3"] = np.log10((df["SIII9069+SIII9531"]) / (df["OIII4959+OIII5007"]))
    if in_df(["OIII5007", "OII3726", "OII3729"]):
        df["O3O2"] = np.log10((df["OIII5007"]) / (df["OII3726"] + df["OII3729"]))
    elif in_df(["OIII5007", "OII3726+OII3729"]):
        df["O3O2"] = np.log10((df["OIII5007"]) / (df["OII3726+OII3729"]))
    if in_df(["OIII5007", "OII3726"]):
        df["O2O3"] = np.log10(df["OII3726"] / df["OIII5007"])  # fig. 13 of Allen+1999
    if in_df(["OIII5007", "OI6300"]):
        df["O1O3"] = np.log10(df["OI6300"] / df["OIII5007"])  # fig. 15 of Allen+1999
    if in_df(["OIII5007", "OII3726+OII3729"]):
        df["O3O2"] = np.log10((df["OIII5007"]) / (df["OII3726+OII3729"]))
    if in_df(["SIII9069+SIII9531", "SII6716+SII6731"]):
        df["S32"] = np.log10((df["SIII9069+SIII9531"]) / (df["SII6716+SII6731"]))
    if in_df(["SIII9069+SIII9531", "HALPHA"]):
        df["S3"] = np.log10((df["SIII9069+SIII9531"]) / (df["HALPHA"]))
    if in_df(["NeV3426", "NeIII3869"]):
        df["Ne53"] = np.log10(df["NeV3426"] / df["NeIII3869"])
    
    # For the Dopita+2016 metallicity diagnostic, which uses the ratio
    #   y = log[Nii]/[Sii] + 0.264 log[Nii]/Hα
    # (see eqn. 13 of Kewley+2019)
    if in_df(["NII6583", "SII6716+SII6731", "HALPHA"]):
        df["Dopita+2016"] = np.log10(df["NII6583"] / df["SII6716+SII6731"]) +\
                            0.264 * np.log10(df["NII6583"] / df["HALPHA"])

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

###############################################################################
def sfr_fn(df, s=f" (total)"):
    """Compute the SFR from the Halpha luminosity using the relation of Calzetti 2013."""
    # Remove suffixes on columns
    if s is not None:
        df_old = df
        suffix_cols = [c for c in df.columns if c.endswith(s)]
        suffix_removed_cols = [c.split(s)[0] for c in suffix_cols]
        df = df_old.rename(columns=dict(zip(suffix_cols, suffix_removed_cols)))
    old_cols = df.columns
    
    # Check whether SFR measurements already exist in the DataFrame
    sfr_cols = [col for col in df if "SFR" in col]
    if len(sfr_cols) > 0:
        logger.warn(f"the following columns already exist in the DataFrame: {', '.join(sfr_cols)}")

    # Use the Calzetti relation to calculate the SFR
    if "HALPHA luminosity" in df and "BPT (numeric)" in df:
        cond_SF = df["BPT (numeric)"] != 0
        df.loc[cond_SF, "SFR"] = df.loc[cond_SF, "HALPHA luminosity"] * 5.5e-42  # Taken from Calzetti (2013); assumes stellar mass range 0.1–100 M⊙, τ ≥6 Myr, Te=104 k, ne=100 cm−3
        df.loc[cond_SF, "SFR error"] = df.loc[cond_SF, "HALPHA luminosity error"] * 5.5e-42  

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
def compute_SFR(df, ncomponents_max):
    """Comptue the SFR from the Halpha luminosity in each kinematic component."""
    df = sfr_fn(df, s=f" (total)")
    for nn in range(ncomponents_max):
        df = sfr_fn(df, s=f" (component {nn + 1})")
    return df