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
            "SIII9531": 9531.100
}

################################################################################
def extcorr_helper_fn(args):
    """
    Function used to parallelise extinction correction computation in
    extinction_corr_fn().
    """
    rr, df, eline_list, ext_fn = args 
    df_row = df.loc[rr]
    if df_row[f"A_V"] > 0:
        for eline in eline_list:
            lambda_A = eline_lambdas_A[eline]
            A_line = ext_fn(wave=np.array([lambda_A]), 
                            a_v=df_row[f"A_V"], 
                            unit="aa")[0]
            
            # Apply correction
            df_row[f"{eline}"] *= 10**(0.4 * A_line)
            df_row[f"{eline} error"] *= 10**(0.4 * A_line)
    # print(f"Finished processing row {rr}")
    return df_row

################################################################################
def extinction_corr_fn(df, eline_list,
                       reddening_curve="fm07", R_V=3.1, 
                       balmer_SNR_min=5,
                       nthreads=20,
                       s=None):
    """
    Correct emission line fluxes (and errors) for extinction.
    ------------------------------------------------------------------------
    eline_list:         list of str
        Emission line fluxes to correct.

    reddening_curve:    str
        Reddening curve to assume. Defaults to Fitzpatrick & Massa (2007).

    R_V:                float
        R_V to assume (in magnitudes). Ignored if Fitzpatrick & Massa (2007) is 
        used, which implicitly assumes R_V = 3.1.

    balmer_SNR_min:     float
        Minimum SNR of the HALPHA and HBETA fluxes to accept in order to compute 
        A_V.

    nthreads:           int
        Number of threads on which to concurrently compute extinction correction
        factors. 

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
    df[f"Balmer decrement"] = df[f"HALPHA"] / df[f"HBETA"]
    df[f"Balmer decrement error"] =\
        df[f"Balmer decrement"] * \
        np.sqrt( (df[f"HALPHA error"] / df[f"HALPHA"])**2 +\
                 (df[f"HBETA error"] / df[f"HBETA"])**2 )

    # Compute E(B-V)
    E_ba = 2.5 * (np.log10(df[f"Balmer decrement"])) - 2.5 * np.log10(2.86)
    E_ba_err = 2.5 / np.log(10) * df[f"Balmer decrement error"] / df[f"Balmer decrement"]

    # Calculate ( A(Ha) - A(Hb) ) / E(B-V) from extinction curve
    R_V = 3.1
    wave_1_A = np.array([eline_lambdas_A["HALPHA"]])
    wave_2_A = np.array([eline_lambdas_A["HBETA"]])
    E_ba_over_E_BV = float(ext_fn(wave_2_A, a_v=1.0) - ext_fn(wave_1_A, a_v=1.0) ) /  1.0 * R_V

    # Calculate E(B-V)
    E_BV = 1 / E_ba_over_E_BV * E_ba
    E_BV_err = 1 / E_ba_over_E_BV * E_ba_err

    # Calculate A(V)
    df[f"A_V"] = R_V * E_BV
    df[f"A_V error"] = R_V * E_BV_err

    #//////////////////////////////////////////////////////////////////////////
    # DQ cuts
    #//////////////////////////////////////////////////////////////////////////
    # non-physical Balmer decrement (set A_V = 0)
    cond_negative_A_V = df[f"A_V"] < 0
    df.loc[cond_negative_A_V, f"A_V"] = 0
    df.loc[cond_negative_A_V, f"A_V error"] = 0

    # low S/N HALPHA and HBETA fluxes (set A_V = NaN)
    cond_bad_A_V = df[f"HALPHA S/N"] < balmer_SNR_min
    cond_bad_A_V |= df[f"HBETA S/N"] < balmer_SNR_min
    df.loc[cond_bad_A_V, f"A_V"] = np.nan
    df.loc[cond_bad_A_V, f"A_V error"] = np.nan

    # Split into 2 DataFrames
    cond_extcorr = df["A_V"] > 0
    df_extcorr = df[cond_extcorr]
    df_noextcorr = df[~cond_extcorr]

    #//////////////////////////////////////////////////////////////////////////
    # Correct emission line fluxes in cells where A_V > 0
    #//////////////////////////////////////////////////////////////////////////
    # Turn off "settingwithcopy" warning because it pops up in extcorr_helper_fn,
    # even though we ARE changing the values properly.
    pd.options.mode.chained_assignment = None

    # Multithreading 
    args_list = [[rr, df_extcorr, eline_list, ext_fn] for rr in df_extcorr.index.values]
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

