import numpy as np
import warnings

import logging
logger = logging.getLogger(__name__)

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
def morph_num_to_str(s):
    return [morph_dict[str(a)] for a in s]

###############################################################################
def in_dataframe(df, cols) -> bool:
    """Returns True if all colums in cols are present in DataFrame df."""
    if type(cols) == list:
        return all([c in df for c in cols])
    elif type(cols) == str:
        return cols in df
    else:
        raise ValueError("cols must be str or list of str!")

######################################################################
# Compute offsets between gas & stellar kinematics
def compute_gas_stellar_offsets(df, ncomponents_max):    
    logger.debug("computing kinematic gas/stellar offsets...")
    df = df.copy()  # To suppress "PerformanceWarning: DataFrame is highly fragmented." warning
    if "v_*" in df and "sigma_*" in df:
        for nn in range(ncomponents_max):
            #//////////////////////////////////////////////////////////////////////
            # Velocity offsets
            if f"v_gas (component {nn + 1})" in df:
                df[f"v_gas - v_* (component {nn + 1})"] = df[f"v_gas (component {nn + 1})"] - df["v_*"]
            if f"v_gas error (component {nn + 1})" in df:
                df[f"v_gas - v_* error (component {nn + 1})"] = np.sqrt(df[f"v_gas error (component {nn + 1})"]**2 + df["v_* error"]**2)

            #//////////////////////////////////////////////////////////////////////
            # Velocity dispersion offsets
            if f"sigma_gas (component {nn + 1})" in df:
                df[f"sigma_gas - sigma_* (component {nn + 1})"] = df[f"sigma_gas (component {nn + 1})"] - df["sigma_*"]
                df[f"sigma_gas^2 - sigma_*^2 (component {nn + 1})"] = df[f"sigma_gas (component {nn + 1})"]**2 - df["sigma_*"]**2
                df[f"sigma_gas/sigma_* (component {nn + 1})"] = df[f"sigma_gas (component {nn + 1})"] / df["sigma_*"]

            if f"sigma_gas error (component {nn + 1})" in df:
                df[f"sigma_gas - sigma_* error (component {nn + 1})"] = np.sqrt(df[f"sigma_gas error (component {nn + 1})"]**2 + df["sigma_* error"]**2)
                df[f"sigma_gas^2 - sigma_*^2 error (component {nn + 1})"] = 2 * np.sqrt(df[f"sigma_gas (component {nn + 1})"]**2 * df[f"sigma_gas error (component {nn + 1})"]**2 +\
                                                                                df["sigma_*"]**2 * df["sigma_* error"]**2)
                df[f"sigma_gas/sigma_* error (component {nn + 1})"] =\
                    df[f"sigma_gas/sigma_* (component {nn + 1})"] *\
                    np.sqrt((df[f"sigma_gas error (component {nn + 1})"] / df[f"sigma_gas (component {nn + 1})"])**2 +\
                            (df["sigma_* error"] / df["sigma_*"])**2)
        
    return df

######################################################################
# Compute differences in Halpha EW, sigma_gas between different components
def compute_component_offsets(df, ncomponents_max):
    logger.debug("computing kinematic component offsets...")
    df = df.copy()  # To suppress "PerformanceWarning: DataFrame is highly fragmented." warning

    component_combinations = []
    for ii in range(ncomponents_max):
        for jj in range(ii):
            component_combinations.append([ii + 1, jj + 1])
    
    for nn_2, nn_1 in component_combinations:

        # Difference between gas velocity dispersion between components
        if all([col in df for col in [f"sigma_gas (component {nn_1})", f"sigma_gas (component {nn_2})"]]):
            df[f"delta sigma_gas ({nn_2}/{nn_1})"] = df[f"sigma_gas (component {nn_2})"] - df[f"sigma_gas (component {nn_1})"]
        
        # Error in the difference between gas velocity dispersion between components   
        if all([col in df for col in [f"sigma_gas error (component {nn_1})", f"sigma_gas error (component {nn_2})"]]):
            df[f"delta sigma_gas error ({nn_2}/{nn_1})"] = np.sqrt(df[f"sigma_gas error (component {nn_2})"]**2 +\
                                                                df[f"sigma_gas error (component {nn_1})"]**2)

        # DIfference between gas velocity between components
        if all([col in df for col in [f"v_gas (component {nn_1})", f"v_gas (component {nn_2})"]]):     
            df[f"delta v_gas ({nn_2}/{nn_1})"] = df[f"v_gas (component {nn_2})"] - df[f"v_gas (component {nn_1})"]
        if all([col in df for col in [f"v_gas error (component {nn_2})", f"v_gas error (component {nn_1})"]]):  
            df[f"delta v_gas error ({nn_2}/{nn_1})"] = np.sqrt(df[f"v_gas error (component {nn_2})"]**2 +\
                                                            df[f"v_gas error (component {nn_1})"]**2)
        
        # Ratio of HALPHA EWs between components   
        if all([col in df for col in [f"HALPHA EW (component {nn_1})", f"HALPHA EW (component {nn_2})"]]):     
            df[f"HALPHA EW ratio ({nn_2}/{nn_1})"] = df[f"HALPHA EW (component {nn_2})"] / df[f"HALPHA EW (component {nn_1})"]
        if all([col in df for col in [f"HALPHA EW error (component {nn_1})", f"HALPHA EW error (component {nn_2})"]]):     
            df[f"HALPHA EW ratio error ({nn_2}/{nn_1})"] = df[f"HALPHA EW ratio ({nn_2}/{nn_1})"] *\
                np.sqrt((df[f"HALPHA EW error (component {nn_2})"] / df[f"HALPHA EW (component {nn_2})"])**2 +\
                        (df[f"HALPHA EW error (component {nn_1})"] / df[f"HALPHA EW (component {nn_1})"])**2)

        # Ratio of HALPHA EWs between components (log)
        if all([col in df for col in [f"log HALPHA EW (component {nn_2})", f"log HALPHA EW (component {nn_1})"]]):     
            df[f"Delta HALPHA EW ({nn_2}/{nn_1})"] = df[f"log HALPHA EW (component {nn_2})"] - df[f"log HALPHA EW (component {nn_1})"]

        # Forbidden line ratios:
        for col in ["log O3", "log N2", "log S2", "log O1"]:
            if f"{col} (component {nn_1})" in df and f"{col} (component {nn_2})" in df:
                df[f"delta {col} ({nn_2}/{nn_1})"] = df[f"{col} (component {nn_2})"] - df[f"{col} (component {nn_1})"]
            if f"{col} error (component {nn_2})" in df and f"{col} error (component {nn_1})" in df:
                df[f"delta {col} ({nn_2}/{nn_1}) error"] = np.sqrt(df[f"{col} error (component {nn_2})"]**2 + df[f"{col} error (component {nn_1})"]**2)

    # Fractional of total Halpha EW in each component
    for nn in range(ncomponents_max):
        if all([col in df.columns for col in [f"HALPHA EW (component {nn + 1})", f"HALPHA EW (total)"]]):
            df[f"HALPHA EW/HALPHA EW (total) (component {nn + 1})"] = df[f"HALPHA EW (component {nn + 1})"] / df[f"HALPHA EW (total)"]

    return df

######################################################################
# Compute log quantities + errors for Halpha EW, sigma_gas and SFRs
def compute_log_columns(df, ncomponents_max):
    logger.debug("computing logs...")
    df = df.copy()  # To suppress "PerformanceWarning: DataFrame is highly fragmented." warning

    # Halpha flux and EW for individual components
    with warnings.catch_warnings():
        warnings.filterwarnings(action="ignore", category=RuntimeWarning, message="invalid value encountered in log10")
        for col in ["HALPHA luminosity", "HALPHA continuum", "HALPHA EW", "sigma_gas", "S2 ratio"]:
            for s in ["(total)"] + [f"(component {nn})" for nn in range(1, ncomponents_max + 1)]:
                # Compute log quantities for total 
                if f"{col} {s}" in df:
                    df[f"log {col} {s}"] = np.log10(df[f"{col} {s}"])
                if f"{col} error {s}" in df:
                    df[f"log {col} error (lower) {s}"] = df[f"log {col} {s}"] - np.log10(df[f"{col} {s}"] - df[f"{col} error {s}"])
                    df[f"log {col} error (upper) {s}"] = np.log10(df[f"{col} {s}"] + df[f"{col} error {s}"]) -  df[f"log {col} {s}"]

        # Compute log quantities for total SFR
        for col in ["SFR", "SFR surface density", "sSFR"]:
            for s in ["(total)"] + [f"(component {nn})" for nn in range(1, ncomponents_max + 1)]:
                if f"{col} {s}" in df:
                    cond = ~np.isnan(df[f"{col} {s}"])
                    cond &= df[f"{col} {s}"] > 0
                    df.loc[cond, f"log {col} {s}"] = np.log10(df.loc[cond, f"{col} {s}"])
                    if f"{col} error {s}" in df:
                        df.loc[cond, f"log {col} error (lower) {s}"] = df.loc[cond, f"log {col} {s}"] - np.log10(df.loc[cond, f"{col} {s}"] - df.loc[cond, f"{col} error {s}"])
                        df.loc[cond, f"log {col} error (upper) {s}"] = np.log10(df.loc[cond, f"{col} {s}"] + df.loc[cond, f"{col} error {s}"]) -  df.loc[cond, f"log {col} {s}"]
                
    return df

