"""
File:       metallicity.py
Author:     Henry Zovaro
Email:      henry.zovaro@anu.edu.au

DESCRIPTION
------------------------------------------------------------------------------
A collection of functions used to compute metallicities and ionisation 
parameters using strong-line diagnostics.

The following functions are included:

    met_line_ratio_fn():
        Helper function to quickly compute emission line ratios for metallicity 
        diagnostics.     
    
    ion_line_ratio_fn():
        Helper function to quickly compute emission line ratios for ionisation
        parameter diagnostics. 

    get_metallicity():
        Estimate the metallicity given an ionisation parameter using
        emission line ratios via the diagnostics given in Kewley (2019)
        (i.e., the ARAA paper)

    get_ionisation_parameter():
        Estimate the ionisation parameter given the metallicity using
        emission line ratios via the diagnostics given in Kewley (2019)
        (i.e., the ARAA paper)

    iter_met_helper_fn():
        Helper function used in iter_metallicity_fn().

    iter_metallicity_fn():
        Use the iterative method described in Kewley & Dopita (2002) to estimate 
        the metallicity and ionisation parameter using emission line ratios via 
        the diagnostics given in Kewley (2019).

    met_helper_fn():
        Helper function used in metallicity_fn().

    metallicity_fn():
        Estimate the metallicity assuming a fixed ionisation parameter.

------------------------------------------------------------------------------
Copyright (C) 2022 Henry Zovaro 
"""
################################################################################
# Imports
import numpy as np
import extinction
import pandas as pd
import multiprocessing
from tqdm import tqdm

from spaxelsleuth.utils.linefns import ratio_fn

from IPython.core.debugger import Tracer

################################################################################
# Coefficients used in metallicity and ionisation parameter diagnostics
# These have all been taken from the relevant tables in Kewley (2019), except
# for Dopita et al. (2016).

# Valid for log(P/k) = 5.0
ion_coeffs = {
    "O3O2" : {
        "A" : 13.768,
        "B" : 9.4940,
        "C" : -4.3223,
        "D" : -2.3531,
        "E" : -0.5769,
        "F" : 0.2794,
        "G" : 0.1574,
        "H" : 0.0890,
        "I" : 0.0311,
        "J" : 0.0000,
        "RMS ERR" : 1.35,   # PERCENT
        "Zmin" : 7.63,
        "Zmax" : 8.93,
        "Umin" : -3.98,
        "Umax" : -2.98,
    },
}

# Valid for log(P/k) = 5.0 and -3.98 < log(U) < -1.98
met_coeffs = {
    "N2"     : {
        "A" : 10.526,
        "B" : 1.9958,
        "C" : -0.6741,
        "D" : 0.2892,
        "E" : 0.5712,
        "F" : -0.6597,
        "G" : 0.0101,
        "H" : 0.0800,
        "I" : 0.0782,
        "J" : -0.0982,
        "RMS ERR" : 0.67,   # PERCENT
        "Zmin" : 7.63,
        "Zmax" : 8.53,
        },
    "S2"     : {
    # Notes: strong dependence on U. Can be avoided by using S23. 
    # Shouldn"t be used unless ionisation parameter can be determined to an 
    # accuracy of better than 0.05 dex in log(U).
        "A" : 23.370,
        "B" : 11.700,
        "C" : 7.2562,
        "D" : 4.3320,
        "E" : 3.1564,
        "F" : 1.0361,
        "G" : 0.4315,
        "H" : 0.6576,
        "I" : 0.3319,
        "J" : 0.0336,
        "RMS ERR" : 0.92,   # PERCENT
        "Zmin" : 7.63,
        "Zmax" : 8.53,
        },
    "N2S2"  : {
        "A" : 5.8892,
        "B" : 3.1688,
        "C" : -3.5991,
        "D" : 1.6394,
        "E" : -2.3939,
        "F" : -1.6764,
        "G" : 0.4455,
        "H" : -0.9302,
        "I" : -0.0966,
        "J" : -0.2490,
        "RMS ERR" : 1.19,   # PERCENT
        "Zmin" : 7.63,
        "Zmax" : 8.53,
        },
    "S23"   : {
        "A" : 11.033,
        "B" : 0.9907,
        "C" : 1.5789,
        "D" : 0.4233,
        "E" : -3.1663,
        "F" : 0.3666,
        "G" : 0.0654,
        "H" : -0.2146,
        "I" : -1.7045,
        "J" : 0.0316,
        "RMS ERR" : 0.55,   # PERCENT
        "Zmin" : 7.63,
        "Zmax" : 8.53,
        },
    "O3N2"  : { 
    # Notes: strong dependence on U. Not recommended to use. 
    # [NII]/Ha is a better alternative when few lines are available.
        "A" : 10.312,
        "B" : -1.6575,
        "C" : 2.2525,
        "D" : -1.3594,
        "E" : 0.4764,
        "F" : 1.1730,
        "G" : -0.2968,
        "H" : 0.1974,
        "I" : -0.0544,
        "J" : 0.1891,
        "RMS ERR" : 2.97,   # PERCENT
        "Zmin" : 8.23,
        "Zmax" : 8.93,
        },
    "O2S2"  : {
    # Notes: less sensitive to U than [SII]/Halpha, but N2O2 is better if available
        "A" : 12.489,
        "B" : -3.2646,
        "C" : 3.2581,
        "D" : -2.0544,
        "E" : 0.5282,
        "F" : 1.0730,
        "G" : -0.3445,
        "H" : 0.2130,
        "I" : -0.3047,
        "J" : 0.1209,
        "RMS ERR" : 2.52,   # PERCENT
        "Zmin" : 8.23,
        "Zmax" : 9.23,
    },
    "OIIHB"     : {
        "A" : 6.2084,
        "B" : -4.0513,
        "C" : -1.4847,
        "D" : -1.9125,
        "E" : -1.0071,
        "F" : -0.1275,
        "G" : -0.2471,
        "H" : -0.1872,
        "I" : -0.1052,
        "J" : 0.0173,
        "RMS ERR" : 0.02,   # PERCENT
        "Zmin" : 8.53,
        "Zmax" : 9.23,
    },
    "OIIIHB"    : {
        "A" : 12.489,
        "B" : -3.2646,
        "C" : 3.2581,
        "D" : -2.0544,
        "E" : 0.5282,
        "F" : 1.0730,
        "G" : -0.3445,
        "H" : 0.2130,
        "I" : -0.3047,
        "J" : 0.1209,
        "RMS ERR" : 2.52,   # PERCENT
        "Zmin" : 8.23,
        "Zmax" : 8.93,
    },
    "N2O2"  : {
    # Notes: "By far the most reliable" diagnostic in the optical. 
    # No dependence on U and only marginally sensitive to pressure.
    # Also relatively insensitive to DIG and AGN radiation.
        "A" :   9.4772,
        "B" :   1.1797,
        "C" :   0.5085,
        "D" :   0.6879,
        "E" :   0.2807,
        "F" :   0.1612,
        "G" :   0.1187,
        "H" :   0.1200,
        "I" :   0.2293,
        "J" :   0.0164,
        "RMS ERR" : 2.65,   # PERCENT
        "Zmin" : 7.63,
        "Zmax" : 9.23,
    },
    "R23"   : {
    # Notes: two-valued and sensitive to ISM pressure at metallicities 
    # exceeding ~8.5. A pressure diagnostic should be used in conjunction!
        "A" : 9.7757,
        "B" : -0.5059,
        "C" : 0.9707,
        "D" : -0.1744,
        "E" : -0.0255,
        "F" : 0.3838,
        "G" : -0.0378,
        "H" : 0.0806,
        "I" : -0.0852,
        "J" : 0.0462,
        "RMS ERR" : 2.11,   # PERCENT
        "Zmin" : 8.23,
        "Zmax" : 8.93,
    },  
    # Notes: these limits guesstimated from fig. 3 of this paper
    "Dopita+2016": {
        "Zmin" : 7.5, 
        "Zmax" : 9.4
    }
}

################################################################################
def met_line_ratio_fn(met_diagnostic, df_row):
    """
    Helper function to quickly compute emission line ratios for metallicity 
    diagnostics. 

    We use this instead of linefns.ratio_fn() because that function expects
    a full DataFrame, and won't work if the input is a single row from a 
    DataFrame (i.e. a Series object)
    """
    if met_diagnostic == "N2O2":
        df_row["N2O2"] = np.log10((df_row["NII6583"]) / (df_row["OII3726+OII3729"]))
    elif met_diagnostic == "R23":
        df_row["R23"] = np.log10((df_row["OIII4959+OIII5007"] + df_row["OII3726+OII3729"]) / (df_row["HBETA"]))
    elif met_diagnostic == "N2S2":
        df_row["N2S2"] = np.log10((df_row["NII6583"]) / (df_row["SII6716+SII6731"]))
    elif met_diagnostic == "Dopita+2016":
        df_row["Dopita+2016"] = np.log10(df_row["NII6583"] / df_row["SII6716+SII6731"]) + 0.264 * np.log10(df_row["NII6583"] / df_row["HALPHA"])
    else:
        raise ValueError(f"In loaddata.metallicity.met_line_ratio_fn(): Metallicity diagnostic {met_diagnostic} not supported!")

    return df_row

def ion_line_ratio_fn(ion_diagnostic, df_row):
    """
    Helper function to quickly compute emission line ratios for ionisation
    parameter diagnostics. 

    We use this instead of linefns.ratio_fn() because that function expects
    a full DataFrame, and won't work if the input is a single row from a 
    DataFrame (i.e. a Series object)
    """
    if ion_diagnostic == "O3O2":
        df_row["O3O2"] = np.log10((df_row["OIII5007"]) / (df_row["OII3726+OII3729"]))
    else:
        raise ValueError(f"In loaddata.metallicity.ion_line_ratio_fn(): ionisation parameter diagnostic {ion_diagnostic} not supported!")

    return df_row

################################################################################
def get_metallicity(met_diagnostic, logR_met, logU):
    """
    Estimate the metallicity given an ionisation parameter using
    emission line ratios via the diagnostics given in Lisa"s ARAA paper.

    INPUTS
    ------------------------------------------------------------------------
    met_diagnostic: str
        For example,
            "R23" : ([OIII] + [OII])/Hbeta
            "N2O2" : [NII]/[OII]
            "O2S2" : [OII]/[SII]

    logR_met:       float  
        Log of the ratio corresponding to the metallicity diagnostic.       

    logU:           float
        log base 10 of the assumed ionisation parameter.

    OUTPUTS
    ------------------------------------------------------------------------
    log(O/H) + 12 corresponding to the chosen diagnostic, ionisation 
    parameter and emission line ratio.

    """
    # Make sure met_diagnostic is in keys
    assert met_diagnostic in met_coeffs.keys(), "metallicity diagnostic " + met_diagnostic + " has not been implemented!"    
    
    # Make sure logU is a float
    assert isinstance(logU, float), "log(U) must be a float!"

    # Metallicity
    # x = log(R)
    # y = log(U)
    if met_diagnostic == "Dopita+2016":
        return 8.77 + logR_met + 0.45 * (logR_met + 0.5)**5
    else:
        logOH12_func = lambda x, y : \
              met_coeffs[met_diagnostic]["A"] \
            + met_coeffs[met_diagnostic]["B"] * x \
            + met_coeffs[met_diagnostic]["C"] * y \
            + met_coeffs[met_diagnostic]["D"] * x * y \
            + met_coeffs[met_diagnostic]["E"] * x**2 \
            + met_coeffs[met_diagnostic]["F"] * y**2 \
            + met_coeffs[met_diagnostic]["G"] * x * y**2 \
            + met_coeffs[met_diagnostic]["H"] * y * x**2 \
            + met_coeffs[met_diagnostic]["I"] * x**3 \
            + met_coeffs[met_diagnostic]["J"] * y**3
        return logOH12_func(x=logR_met, y=logU)
    

################################################################################
def get_ionisation_parameter(ion_diagnostic, logR_ion, logOH12):
    """
    Estimate the ionisation parameter given a metallicity using
    emission line ratios via the diagnostics given in Lisa"s ARAA paper.
    
    INPUTS
    ------------------------------------------------------------------------
    ion_diagnostic:   string
        Must be "O3O2" (= [OIII]/[OII])

    logR_ion:            float
        Log of the emission line ratio corresponding to the ionisation 
        parameter diagnostic.

    logOH12:        float
        Assumed metallicity (log(O/H) + 12).
    
    OUTPUTS
    ------------------------------------------------------------------------
    log(U) corresponding to the chosen diagnostic, metallicity and emission 
    line ratio.

    """
    # Make sure ion_diagnostic is in keys
    assert ion_diagnostic in ion_coeffs.keys(), "ionisation parameter diagnostic " + ion_diagnostic + " has not been implemented!"    

    # Make sure logOH12 is a float
    assert isinstance(logOH12, float), "logOH12 must be a float!"

    # Ionisation parameter
    # x = log(R)
    # y = log(O/H) + 12
    logU_func = lambda x, y : \
          ion_coeffs[ion_diagnostic]["A"]   \
        + ion_coeffs[ion_diagnostic]["B"] * x     \
        + ion_coeffs[ion_diagnostic]["C"] * y     \
        + ion_coeffs[ion_diagnostic]["D"] * x * y   \
        + ion_coeffs[ion_diagnostic]["E"] * x**2  \
        + ion_coeffs[ion_diagnostic]["F"] * y**2  \
        + ion_coeffs[ion_diagnostic]["G"] * x * y**2    \
        + ion_coeffs[ion_diagnostic]["H"] * y * x**2    \
        + ion_coeffs[ion_diagnostic]["I"] * x**3  \
        + ion_coeffs[ion_diagnostic]["J"] * y**3  \

    return logU_func(x=logR_ion, y=logOH12) 


################################################################################
def iter_met_helper_fn(args):
    """
    Function used to parallelise metallicity/ionisation parameter computation in
    iter_metallicity_fn().
    """
    rr, df, met_diagnostic, ion_diagnostic, niters = args
    df_row = df.loc[rr]
    
    # Compute metallicity
    # Arrays to store the metallicity and ionisation parameter estimated
    # in each iteration
    logOH12_vals = np.full(niters,np.nan)
    logU_vals = np.full(niters,np.nan)

    # Emission lines used in metallicity calculation
    if met_diagnostic == "N2O2":
        eline_list = ["NII6583", "OII3726+OII3729"]
    elif met_diagnostic == "R23":
        eline_list = ["OIII4959+OIII5007", "OII3726+OII3729", "HBETA"]
    elif met_diagnostic == "N2S2":
        eline_list = ["NII6583", "SII6716+SII6731"]
    else:
        raise ValueError(f"In loaddata.metallicity.iter_met_helper_fn(): Metallicity diagnostic {met_diagnostic} not supported!")

    # Emission lines used in ionisation parameter calculation
    if ion_diagnostic == "O3O2":
        eline_list += ["OIII5007", "OII3726+OII3729"]
    else:
        raise ValueError(f"In loaddata.metallicity.iter_met_helper_fn(): Ionisation parameter diagnostic {ion_diagnostic} not supported!")

    # Check that the emission lines are in the DataFrame
    for eline in eline_list:
        assert eline in df_row, f"{eline} not found in df!"

    for nn in range(niters):
        # Make a copy of the row
        df_tmp = df_row.copy()
        
        # Add random error 
        for eline in eline_list:
            df_tmp[eline] += np.random.normal(scale=df_tmp[f"{eline} error"])

        # Re-compute emission line ratios
        df_tmp = met_line_ratio_fn(met_diagnostic, df_tmp)
        df_tmp = ion_line_ratio_fn(ion_diagnostic, df_tmp)

        # Starting guesses
        logOH12 = 8.0
        logOH12_old = 0
        logU = -3.0
        logU_old = 0

        # Use an iterative method to determine the metallicity and
        # ionisation parameter
        iters = 0
        max_iters = 1e3
        while np.abs((logOH12 - logOH12_old) / logOH12) > 0.001 and np.abs((logU - logU_old) / logU) > 0.001:
            if iters >= max_iters:
                break
            logU_old = logU
            logOH12_old = logOH12
            logU = get_ionisation_parameter(ion_diagnostic, df_tmp[met_diagnostic], logOH12)
            logOH12 = get_metallicity(met_diagnostic, df_tmp[met_diagnostic], logU)
            iters += 1

        # If the chosen diagnostic is not one of the default ones, then assume its valid range is infinite
        if met_diagnostic in met_coeffs:
            Zmin = met_coeffs[met_diagnostic]["Zmin"]
            Zmax = met_coeffs[met_diagnostic]["Zmax"]
        else:
            Zmin = -np.inf
            Zmax = np.inf
        if logOH12 >= Zmin and logOH12 <= Zmax and ~np.isnan(logU) and ~np.isnan(logOH12) and iters < max_iters:
            logOH12_vals[nn] = logOH12
            logU_vals[nn] = logU

    # Add to DataFrame
    df_row[f"log(O/H) + 12 ({met_diagnostic})"] = np.nanmean(logOH12_vals)
    df_row[f"log(U) ({ion_diagnostic})"] = np.nanmean(logU_vals)
    df_row[f"log(O/H) + 12 error (lower) ({met_diagnostic})"] = np.quantile(logOH12_vals, q=0.16)
    df_row[f"log(O/H) + 12 error (upper) ({met_diagnostic})"] = np.quantile(logOH12_vals, q=0.84)
    df_row[f"log(U) error (lower) ({ion_diagnostic})"] = np.quantile(logU_vals, q=0.16)
    df_row[f"log(U) error (upper) ({ion_diagnostic})"] = np.quantile(logU_vals, q=0.84)

    return df_row

################################################################################
def iter_metallicity_fn(df, met_diagnostic, ion_diagnostic, 
                        compute_errors=True, niters=100, nthreads=20,
                        s=None):
    """
    Use the iterative method described in Kewley & Dopita (2002) to estimate 
    the metallicity and ionisation parameter using strong-line diagnostics.

    INPUTS
    -----------------------------------------------------------------------
    df:                 str
        Pandas DataFrame containing emission line fluxes.

    met_diagnostic:     str
        Strong-line metallicity diagnostic to use. Must be one of 
        "N2O2", "R23", "O3N2", or "Dopita+2016".

    ion_diagnostic:     str
        Strong-line ionisation parameter diagnostic to use. Must be "O3O2" for 
        now.

    niters:             int 
        Number of MC iterations. 1000 is recommended.

    compute_errors:      bool
        If True, estimate 1-sigma errors on log(O/H) + 12 and log(U) using a 
        Monte Carlo approach, in which the 1-sigma uncertainties on the 
        emission line fluxes are used generate a distribution in log(O/H) + 12 
        values, the mean and standard deviation of which are used to 
        evaluate the metallicity and corresponding uncertainty.

    nthreads:           int
        Number of threads to use when computing errors.

    s:                  str 
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

            "log(O/H) + 12 (N2O2) (total)", "log(O/H) + 12 error (N2O2) (total)"

    OUTPUTS
    -----------------------------------------------------------------------
    The original DataFrame with the following columns added:

        log(O/H) + 12 (<met_diagnostic>)        float
            Metallicity corresponding to the diagnostic chosen in each
            spaxel or component.

        log(O/H) + 12 error (lower) (<met_diagnostic>)  float
            Corresponding 16th percentile in the distribution of log(O/H) + 12
            values computed in the MC simulation, if compute_errors is True.
        
        log(O/H) + 12 error (upper) (<met_diagnostic>)  float
            Corresponding 84th percentile in the distribution of log(O/H) + 12
            values computed in the MC simulation, if compute_errors is True.

        log(U) (<ion_diagnostic>)               float
            Ionisation parameter corresponding to the diagnostic chosen in each
            spaxel or component.

        log(U) error (lower) (<met_diagnostic>)  float
            Corresponding 16th percentile in the distribution of log(U)
            values computed in the MC simulation, if compute_errors is True.
        
        log(U) error (upper) (<met_diagnostic>)  float
            Corresponding 84th percentile in the distribution of log(U)
            values computed in the MC simulation, if compute_errors is True.
        
    """
    #//////////////////////////////////////////////////////////////////////////
    # Input checking
    #//////////////////////////////////////////////////////////////////////////
    # Check valid metallicity/ionisation parameter diagnostic
    assert met_diagnostic in ["N2O2", "R23", "O3N2"],\
        "met_diagnostic must be N2O2, R23 or O3N2!"
    assert ion_diagnostic in ["O3O2"],\
        "ion_diagnostic must be O3O2!"

    # Add new columns
    df[f"log(O/H) + 12 ({met_diagnostic})" + s] = np.nan
    df[f"log(U) ({ion_diagnostic})" + s] = np.nan
    df[f"log(O/H) + 12 error (lower) ({met_diagnostic})" + s] = np.nan
    df[f"log(O/H) + 12 error (upper) ({met_diagnostic})" + s] = np.nan
    df[f"log(U) error (lower) ({ion_diagnostic})" + s] = np.nan
    df[f"log(U) error (upper) ({ion_diagnostic})" + s] = np.nan

    # Deal with case where 
    if not compute_errors:
        niters = 1

    #//////////////////////////////////////////////////////////////////////////
    # Remove suffixes on columns
    #//////////////////////////////////////////////////////////////////////////
    if s is not None:
        df_old = df
        suffix_cols = [c for c in df.columns if c.endswith(s)]
        suffix_removed_cols = [c.split(s)[0] for c in suffix_cols]
        df = df_old.rename(columns=dict(zip(suffix_cols, suffix_removed_cols)))
    old_cols = df.columns

    #//////////////////////////////////////////////////////////////////////////
    # DQ cuts
    #//////////////////////////////////////////////////////////////////////////
    # Compute the emission line ratios, in case they haven't been computed yet
    if met_diagnostic not in df:
        df = met_line_ratio_fn(met_diagnostic, df)
    if ion_diagnostic not in df:
        df = ion_line_ratio_fn(ion_diagnostic, df)

    # SF spaxels only
    cond_met = df["BPT"] == "SF"

    # Not NaN in both diagnostics
    cond_met &= ~df[met_diagnostic].isna()
    cond_met &= ~df[ion_diagnostic].isna()
    
    # Split into 2 DataFrames
    df_met = df[cond_met]
    df_nomet = df[~cond_met]

    #//////////////////////////////////////////////////////////////////////////
    # Compute metallicities
    #//////////////////////////////////////////////////////////////////////////
    # Turn off "settingwithcopy" warning because it pops up in a_v_fn,
    # even though we ARE changing the values properly.
    pd.options.mode.chained_assignment = None

    # Multithreading 
    args_list = [[rr, df_met, met_diagnostic, ion_diagnostic, niters] for rr in df_met.index.values]
    print(f"In linefns.metallicity.iter_metallicity_fn(): Multithreading metallicity computation across {nthreads} threads...")
    pool = multiprocessing.Pool(nthreads)
    res_list = pool.map(iter_met_helper_fn, args_list)
    pool.close()
    pool.join()

    # Turn warning back on 
    pd.options.mode.chained_assignment = "warn"

    # Check results 
    df_results_met = pd.concat(res_list, axis=1).T

    # Cast back to previous data types
    for col in df.columns:
        df_results_met[col] = df_results_met[col].astype(df[col].dtype)
    df_met = df_results_met

    # Merge back with original DataFrame
    df = pd.concat([df_nomet, df_met])

    #//////////////////////////////////////////////////////////////////////////
    # Rename columns
    #//////////////////////////////////////////////////////////////////////////
    if s is not None:
        # Get list of new columns that have been added
        added_cols = [c for c in df.columns if c not in old_cols]
        suffix_added_cols = [f"{c}{s}" for c in added_cols]
        # Rename the new columns
        df = df.rename(columns=dict(zip(added_cols, suffix_added_cols)))
        # Replace the suffix in the column names
        df = df.rename(columns=dict(zip(suffix_removed_cols, suffix_cols)))

    return df

################################################################################
def met_helper_fn(args):
    """
    Function used to parallelise metallicity parameter computation in
    metallicity_fn().
    """
    rr, df, met_diagnostic, logU, niters = args
    df_row = df.loc[rr]
    
    # Compute metallicity
    # Arrays to store the metallicity and ionisation parameter estimated
    # in each iteration
    logOH12_vals = np.full(niters,np.nan)

    # Emission lines used in metallicity calculation
    if met_diagnostic == "N2O2":
        eline_list = ["NII6583", "OII3726+OII3729"]
    elif met_diagnostic == "R23":
        eline_list = ["OIII4959+OIII5007", "OII3726+OII3729", "HBETA"]
    elif met_diagnostic == "N2S2":
        eline_list = ["NII6583", "SII6716+SII6731"]
    else:
        raise ValueError(f"In loaddata.metallicity.iter_met_helper_fn(): Metallicity diagnostic {met_diagnostic} not supported!")

    # Check that the emission lines are in the DataFrame
    for eline in eline_list:
        assert eline in df_row, f"{eline} not found in df!"

    for nn in range(niters):
        # Make a copy of the row
        df_tmp = df_row.copy()
        
        # Add random error 
        for eline in eline_list:
            df_tmp[eline] += np.random.normal(scale=df_tmp[f"{eline} error"])

        # Re-compute emission line ratios
        df_tmp = met_line_ratio_fn(met_diagnostic, df_tmp)

        # Compute corresponding metallicity
        logOH12_vals[nn] = get_metallicity(met_diagnostic, df_tmp[met_diagnostic], logU)

    # Add to DataFrame
    df_row[f"log(O/H) + 12 ({met_diagnostic})"] = np.nanmean(logOH12_vals)
    df_row[f"log(U) (const.)"] = logU
    df_row[f"log(O/H) + 12 error (lower) ({met_diagnostic})"] = np.quantile(logOH12_vals, q=0.16)
    df_row[f"log(O/H) + 12 error (upper) ({met_diagnostic})"] = np.quantile(logOH12_vals, q=0.84)

    return df_row

################################################################################
def metallicity_fn(df, met_diagnostic, logU=-3.0, 
                   compute_errors=False, niters=1000, nthreads=20, 
                   s=None):
    """
    Estimate the metallicity using a strong-line diagnostic assuming a 
    fixed ionisation parameter.

    INPUTS
    -----------------------------------------------------------------------
    df:                 str
        Pandas DataFrame containing emission line fluxes.

    met_diagnostic:     str
        Strong-line metallicity diagnostic to use. Must be one of 
        "N2O2", "R23", "O3N2", or "Dopita+2016".

    logU:               float
        Log Ionisation parameter (assumed to be fixed).

    compute_errors:      bool
        If True, estimate 1-sigma errors on log(O/H) + 12 using a Monte
        Carlo approach, in which the 1-sigma uncertainties on the emission
        line fluxes are used generate a distribution in log(O/H) + 12 
        values, the mean and standard deviation of which are used to 
        evaluate the metallicity and corresponding uncertainty.

    niters:             int 
        Number of MC iterations. 1000 is recommended.

    nthreads:           int
        Number of threads to use when computing errors 

    s:                  str 
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

            "log(O/H) + 12 (N2O2) (total)", "log(O/H) + 12 error (N2O2) (total)"

    OUTPUTS
    -----------------------------------------------------------------------
    The original DataFrame with the following columns added:

        log(O/H) + 12 (<met_diagnostic>)
            Metallicity corresponding to the diagnostic chosen in each
            spaxel or component.

        log(O/H) + 12 error (lower) (<met_diagnostic>)  float
            Corresponding 16th percentile in the distribution of log(O/H) + 12
            values computed in the MC simulation, if compute_errors is True.
        
        log(O/H) + 12 error (upper) (<met_diagnostic>)  float
            Corresponding 84th percentile in the distribution of log(O/H) + 12
            values computed in the MC simulation, if compute_errors is True.

        log(U) (const.)
            Ionisation parameter assumed.

    """

    #//////////////////////////////////////////////////////////////////////////
    # Input checking
    #//////////////////////////////////////////////////////////////////////////
    # Check valid metallicity diagnostic
    assert met_diagnostic in ["N2O2", "R23", "O3N2", "Dopita+2016"],\
        "met_diagnostic must be N2O2, R23, O3N2 or Dopita+2016!"

    # Add new columns
    df[f"log(O/H) + 12 ({met_diagnostic})" + s] = np.nan
    df[f"log(U) (const.)" + s] = np.nan
    df[f"log(O/H) + 12 error (lower) ({met_diagnostic})" + s] = np.nan
    df[f"log(O/H) + 12 error (upper) ({met_diagnostic})" + s] = np.nan

    #//////////////////////////////////////////////////////////////////////////
    # Remove suffixes on columns
    #//////////////////////////////////////////////////////////////////////////
    if s is not None:
        df_old = df
        suffix_cols = [c for c in df.columns if c.endswith(s)]
        suffix_removed_cols = [c.split(s)[0] for c in suffix_cols]
        df = df_old.rename(columns=dict(zip(suffix_cols, suffix_removed_cols)))
    old_cols = df.columns

    #//////////////////////////////////////////////////////////////////////////
    # DQ cuts
    #//////////////////////////////////////////////////////////////////////////
    # Compute the emission line ratio, in case it hasn't been defined yet
    if met_diagnostic not in df:
        df = met_line_ratio_fn(met_diagnostic, df)

    # SF spaxels only
    cond_met = df["BPT"] == "SF"

    # Not NaN in the diagnostic
    cond_met &= ~df[met_diagnostic].isna()

    # Split into 2 DataFrames
    df_met = df[cond_met]
    df_nomet = df[~cond_met]

    # Store log(U)
    pd.options.mode.chained_assignment = None
    df_met[f"log(U) (const.)"] = logU
    pd.options.mode.chained_assignment = "warn"

    #//////////////////////////////////////////////////////////////////////////
    # Compute metallicities
    #//////////////////////////////////////////////////////////////////////////
    # Turn off "settingwithcopy" warning because it pops up here,
    # even though we ARE changing the values properly.
    pd.options.mode.chained_assignment = None
    if compute_errors:

        # Multithreading 
        args_list = [[rr, df_met, met_diagnostic, logU, niters] for rr in df_met.index.values]
        print(f"In linefns.metallicity.metallicity_fn(): Multithreading metallicity computation across {nthreads} threads...")
        pool = multiprocessing.Pool(nthreads)
        res_list = pool.map(met_helper_fn, args_list)
        pool.close()
        pool.join()

        # Check results 
        df_results_met = pd.concat(res_list, axis=1).T

        # Cast back to previous data types
        for col in df.columns:
            df_results_met[col] = df_results_met[col].astype(df[col].dtype)
        df_met = df_results_met
    
    else:
        # Compute metallicity based on line ratios only.
        df_met[f"log(O/H) + 12 ({met_diagnostic})"] = get_metallicity(met_diagnostic, df_met[met_diagnostic], logU)
        df_met[f"log(O/H) + 12 error (upper) ({met_diagnostic})"] = 0.0
        df_met[f"log(O/H) + 12 error (lower) ({met_diagnostic})"] = 0.0

    # Turn warning back on 
    pd.options.mode.chained_assignment = "warn"

    # Merge back with original DataFrame
    df = pd.concat([df_nomet, df_met])

    #//////////////////////////////////////////////////////////////////////////
    # Rename columns
    #//////////////////////////////////////////////////////////////////////////
    if s is not None:
        # Get list of new columns that have been added
        added_cols = [c for c in df.columns if c not in old_cols]
        suffix_added_cols = [f"{c}{s}" for c in added_cols]
        # Rename the new columns
        df = df.rename(columns=dict(zip(added_cols, suffix_added_cols)))
        # Replace the suffix in the column names
        df = df.rename(columns=dict(zip(suffix_removed_cols, suffix_cols)))

    return df
