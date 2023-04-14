"""
File:       extcorr.py
Author:     Henry Zovaro
Email:      henry.zovaro@anu.edu.au

DESCRIPTION
------------------------------------------------------------------------------
Functions for correcting emission line fluxes for extinction.

The following functions are included:

extcorr_helper_fn()
    A helper function used for multithreading the extinction correction 
    computation.

extinction_corr_fn()
    Given a DataFrame, compute A_V based on the Balmer decrement and correct 
    the emission line fluxes (and errors) for extinction.

------------------------------------------------------------------------------
Copyright (C) 2022 Henry Zovaro
"""
###############################################################################
import numpy as np
import extinction
import pandas as pd
import multiprocessing

from IPython.core.debugger import Tracer

###############################################################################
# Emission line wavelengths 
eline_lambdas_A = {
            "NeV3347" : 3346.79,
            "OIII3429" : 3429.0,
            "OII3726" : 3726.032,
            "OII3729" : 3728.815,
            "OII3726+OII3729": 3726.032,  # For the doublet, assume the blue wavelength 
            "NEIII3869" : 3869.060,
            "HeI3889" : 3889.0,
            "HEPSILON" : 3970.072,
            "HDELTA" : 4101.734, 
            "HGAMMA" : 4340.464, 
            "HEI4471" : 4471.479,
            "OIII4363" : 4363.210, 
            "HBETA" : 4861.325, 
            "OIII4959" : 4958.911, 
            "OIII5007" : 5006.843, 
            "HEI5876" : 5875.624, 
            "OI6300" : 6300.304, 
            "SIII6312" : 6312.060,
            "OI6364" : 6363.776,
            "NII6548" : 6548.04, 
            "HALPHA" : 6562.800, 
            "NII6583" : 6583.460,
            "SII6716" : 6716.440, 
            "SII6731" : 6730.810,
            "SIII9069": 9068.600,
            "SIII9531": 9531.100,
            # Xtra lines - values from LZIFU linelist (not sure if consistent with wavelengths above)
            "FEVII3586": 3586.32,
            "HETA": 3835.39,
            "HZETA": 3889.06,
            "NEIII3967": 3967.41,
            "SII4069": 4068.60,
            "SII4076": 4076.35,
            "HEI4472": 4471.50,
            "HEII4686": 4685.74,
            "ARIV4711": 4711.26,
            "ARIV4740": 4740.12,
            "FEVII4893": 4893.37,
            "NI5198": 5197.90,
            "NI5200": 5200.26,
            "FEII5273": 5273.36,
            "FEXIV5303": 5302.86,
            "ARX5534": 5534.01,
            "FEVII5721": 5720.71,
            "NII5755": 5754.59,
            "FEVII6087": 6086.97,
            "FEX6375": 6374.53,
            "HEI6678": 6678.14,
            "ARV6435": 6435.12,
}

################################################################################
def extcorr_helper_fn(args):
    """
    Function used to parallelise extinction correction computation in
    extinction_corr_fn().
    """
    rr, df, eline_list, a_v_col_name, ext_fn = args 
    df_row = df.loc[rr]
    if df_row[a_v_col_name] > 0:
        for eline in eline_list:
            lambda_A = eline_lambdas_A[eline]
            A_line = ext_fn(wave=np.array([lambda_A]), 
                            a_v=df_row[a_v_col_name], 
                            unit="aa")[0]
            
            # Apply correction
            df_row[f"{eline}"] *= 10**(0.4 * A_line)
            df_row[f"{eline} error"] *= 10**(0.4 * A_line)

    return df_row

################################################################################
def compute_A_V(df, 
                reddening_curve="fm07", R_V=3.1, 
                balmer_SNR_min=5,
                balmer_decrement_intrinsic=2.86,
                s=None):
    """
    Compute extinction in the V-band (A_V) using the Balmer decrement.

    INPUTS
    ------------------------------------------------------------------------
    df:                 Pandas DataFrame
        DataFrame containing emission line fluxes. Must contain columns
        "HALPHA" and "HBETA" where the column names may have a suffix 
        defined by s.

    reddening_curve:    str
        Reddening curve to assume. Defaults to Fitzpatrick & Massa (2007).

    R_V:                float
        R_V to assume (in magnitudes). Ignored if Fitzpatrick & Massa (2007) is 
        used, which implicitly assumes R_V = 3.1.

    balmer_SNR_min:     float
        Minimum SNR of the HALPHA and HBETA fluxes to accept in order to compute 
        A_V.

    balmer_decrement_intrinsic:   float
        Intrinsic Ha/Hb ratio to assume. 


    RETURNS
    ------------------------------------------------------------------------

    """
    #//////////////////////////////////////////////////////////////////////////
    # Remove suffixes on columns
    #//////////////////////////////////////////////////////////////////////////
    if s is not None:
        df_old = df
        suffix_cols = [c for c in df.columns if c.endswith(s)]
        suffix_removed_cols = [c.split(s)[0] for c in suffix_cols]
        df = df_old.rename(columns=dict(zip(suffix_cols, suffix_removed_cols)))
    old_cols = df.columns

    # Check that HALPHA and HBETA are in the DataFrame 
    if ("HALPHA" not in df) or ("HBETA" not in df):
        print(f"In extcorr.extinction_corr_fn(): WARNING: HALPHA and/or HBETA are not in the DataFrame, so extinction cannot be calculated.")
    else:
        # Determine which reddening curve to use
        if reddening_curve.lower() == "fm07":
            ext_fn = extinction.fm07
            if R_V != 3.1:
                print(f"In extcorr.extinction_corr_fn(): WARNING: R_V is fixed at 3.1 in the FM07 reddening curve. Ignoring supplied R_V value of {R_V:.2f}...")
        elif reddening_curve.lower() == "ccm89":
            ext_fn = extinction.ccm89
        elif reddening_curve.lower() == "calzetti00":
            ext_fn = extinction.calzetti00
            if R_V != 4.05:
                print(f"In extcorr.extinction_corr_fn(): WARNING: R_V should be set to 4.05 for the calzetti00 reddening curve. Using supplied R_V value of {R_V:.2f}...")
        else:  
            raise ValueError("For now, 'reddening_curve' must be one of 'fm07', 'ccm89' or 'calzetti00'!")

        #//////////////////////////////////////////////////////////////////////////
        # Compute A_V in each spaxel
        #//////////////////////////////////////////////////////////////////////////
        # Compute E(B-V)
        df[f"Balmer decrement"] = df[f"HALPHA"] / df[f"HBETA"]
        E_ba = 2.5 * (np.log10(df[f"Balmer decrement"])) - 2.5 * np.log10(balmer_decrement_intrinsic)
            
        # Calculate ( A(Ha) - A(Hb) ) / E(B-V) from extinction curve
        wave_1_A = np.array([eline_lambdas_A["HALPHA"]])
        wave_2_A = np.array([eline_lambdas_A["HBETA"]])
        E_ba_over_E_BV = float(ext_fn(wave_2_A, a_v=1.0) - ext_fn(wave_1_A, a_v=1.0) ) /  1.0 * R_V

        # Calculate E(B-V)
        E_BV = 1 / E_ba_over_E_BV * E_ba
        
        # Calculate A(V)
        df[f"A_V"] = R_V * E_BV

        #//////////////////////////////////////////////////////////////////////////
        # DQ cuts
        #//////////////////////////////////////////////////////////////////////////
        # non-physical Balmer decrement (set A_V = 0)
        cond_negative_A_V = df[f"A_V"] < 0
        df.loc[cond_negative_A_V, f"A_V"] = 0

        # low S/N HALPHA and HBETA fluxes (set A_V = NaN)
        cond_bad_A_V = df[f"HALPHA S/N"] < balmer_SNR_min
        cond_bad_A_V |= df[f"HBETA S/N"] < balmer_SNR_min
        df.loc[cond_bad_A_V, f"A_V"] = np.nan
        
        #//////////////////////////////////////////////////////////////////////////
        # Compute corresponding errors, if HALPHA and HBETA errors are present
        #//////////////////////////////////////////////////////////////////////////
        if ("HALPHA error" in df) and ("HBETA error") in df:
            df[f"Balmer decrement error"] =\
                df[f"Balmer decrement"] * \
                np.sqrt( (df[f"HALPHA error"] / df[f"HALPHA"])**2 +\
                         (df[f"HBETA error"]  / df[f"HBETA"])**2 )
            
            E_ba_err = 2.5 / np.log(10) * df[f"Balmer decrement error"] / df[f"Balmer decrement"]
            E_BV_err = 1 / E_ba_over_E_BV * E_ba_err
            df[f"A_V error"] = R_V * E_BV_err

            # DQ cuts
            df.loc[cond_negative_A_V, f"A_V error"] = 0
            df.loc[cond_bad_A_V, f"A_V error"] = np.nan

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
def apply_extinction_correction(df, eline_list, a_v_col_name, 
                                reddening_curve="fm07",
                                nthreads=20,
                                s=None):
    
    """
    Correct emission line fluxes (and errors) for extinction.

    INPUTS
    ------------------------------------------------------------------------
    eline_list:         list of str
        Emission line fluxes to correct.

    a_v_col_name:       str
        Column in the DataFrame containing A_V values, e.g. "A_V (total)". 

        This is useful if you would like to e.g. apply the extinction correction
        to components 1, 2 and 3 using the A_V calculated from the total 
        emission line fluxes.

    nthreads:           int
        Number of threads on which to concurrently compute extinction correction
        factors. 

    RETURNS
    ------------------------------------------------------------------------
    The input DataFrame with new columns added containing the Balmer decrement,
    A_V, and corresponding errors.

    """
    #//////////////////////////////////////////////////////////////////////////
    # Remove suffixes on columns
    #//////////////////////////////////////////////////////////////////////////
    if s is not None:
        df_old = df
        suffix_cols = [c for c in df.columns if c.endswith(s) and c != a_v_col_name]
        suffix_removed_cols = [c.split(s)[0] for c in suffix_cols]
        df = df_old.rename(columns=dict(zip(suffix_cols, suffix_removed_cols)))
    old_cols = df.columns

    # Check that A_V is in the DataFrame
    assert a_v_col_name in df.columns,\
        f"Column containing A_V values {a_v_col_name} not found in DataFrame!"

    # Determine which reddening curve to use
    if reddening_curve.lower() == "fm07":
        ext_fn = extinction.fm07
    elif reddening_curve.lower() == "ccm89":
        ext_fn = extinction.ccm89
    elif reddening_curve.lower() == "calzetti00":
        ext_fn = extinction.calzetti00
    else:  
        raise ValueError("For now, 'reddening_curve' must be one of 'fm07', 'ccm89' or 'calzetti00'!")

    #//////////////////////////////////////////////////////////////////////////
    # Correct emission line fluxes in cells where A_V > 0
    #//////////////////////////////////////////////////////////////////////////
    # Split into 2 DataFrames
    cond_extcorr = df[a_v_col_name] > 0
    df_extcorr = df[cond_extcorr]
    df_noextcorr = df[~cond_extcorr]
    if df_extcorr.shape[0] == 0:
        print(f"In extcorr.extinction_corr_fn(): no cells found with A_V > 0 - not applying correction!")
        if s is not None:
            # Get list of new columns that have been added
            added_cols = [c for c in df.columns if c not in old_cols]
            suffix_added_cols = [f"{c}{s}" for c in added_cols]
            # Rename the new columns
            df = df.rename(columns=dict(zip(added_cols, suffix_added_cols)))
            # Replace the suffix in the column names
            df = df.rename(columns=dict(zip(suffix_removed_cols, suffix_cols)))
        return df

    # Turn off "settingwithcopy" warning because it pops up in extcorr_helper_fn,
    # even though we ARE changing the values properly.
    pd.options.mode.chained_assignment = None

    # Multithreading 
    args_list = [[rr, df_extcorr, eline_list, a_v_col_name, ext_fn] for rr in df_extcorr.index.values]
    print(f"In extcorr.extinction_corr_fn(): Multithreading A_V computation across {nthreads} threads...")
    pool = multiprocessing.Pool(nthreads)
    res_list = pool.map(extcorr_helper_fn, args_list)
    pool.close()
    pool.join()
    
    # Turn warning back on 
    pd.options.mode.chained_assignment = "warn"

    # Check results 
    df_results_extcorr = pd.concat(res_list, axis=1).T

    # Cast back to previous data types
    for col in df.columns:
        df_results_extcorr[col] = df_results_extcorr[col].astype(df[col].dtype)
    df_extcorr = df_results_extcorr

    # Merge back with original DataFrame
    df = pd.concat([df_noextcorr, df_extcorr])

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

