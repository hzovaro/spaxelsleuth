"""
Test all of the different metallicity diagnostics in spaxelsleuth.utils.metallicity
accessed through the function get_metallicity(). 

For each diagnostic, given a range of emission line ratios, plot the ratio
as a function of the output log(O/H) + 12 (as a function of log(U) where
appropriate) and compare the plot to the corresponding figure in the paper 
from which the diagnostic was derived. 


Structure of metallicity.py:
    
    metallicity_fn() - compute metallicities assuming fixed log(U) w/ or w/o errors 

        --> met_helper_fn() - computes metallicity + errors for a SINGLE dataframe row 

            --> met_line_ratio_fn() 

            --> get_metallicity() - inputs: met_diagnostic (str) + line ratio. 

                --> want to change this to take a DataFrame so that it can 
                    compute all the required line ratios, etc., because some of
                    the new diagnostics require several line ratios! 

    iter_metallicity_fn() - compute iterative metallicities & ionisation parameters w/ or w/o errors 

        --> iter_met_helper_fn() - computes metallicity & ionisation parameter (+ errors optionally) for a SINGLE dataframe row 


1) non-iterative metallicity diagnostics (i.e. ones with no log(U) dependence)


2) iterative metallicity diagnostics 

"""
###############################################################################
import numpy as np

import matplotlib.pyplot as plt
plt.ion()
plt.close("all")


###############################################################################
# NEW FORMAT FOR get_metallicity()
def get_metallicity(met_diagnostic, df_row, logU=None):
    """
    TODO: write docstring
    TODO: check output
    TODO: doublets: e.g. [OII] and [OIII] - what if OII3726+3729 is present but not the lines individually?


    NOTE: it's assumed that the input will have had column suffixes removed 
    prior to being passed to this function.

    INPUTS:
    ---------------------------------------------------------------------------
        met_diagnostic      str
            Metallicity diagnostic.

        df_row              pandas DataFrame
            

    RETURNS:
    ---------------------------------------------------------------------------
        logOH12             float OR numpy.array 
            log(O/H) + 12. If the input DataFrame is a single row, then a 
            float is returned. Otherwise, an array of log(O/H) + 12 values 
            is returned (corresponding to each row).

            
    """
    # Valid for log(P/k) = 5.0 and -3.98 < log(U) < -1.98
    met_coeffs = {
        "N2Ha_K19"     : {
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
        "S2Ha_K19"     : {
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
        "N2S2_K19"  : {
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
        "S23_K19"   : {
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
        "O3N2_K19"  : { 
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
        "O2S2_K19"  : {
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
        "O2Hb_K19"     : {
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
        "N2O2_K19"  : {
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
        "R23_K19"   : {
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
    }

    # TODO: check validity of met_diagnostic. 

    #//////////////////////////////////////////////////////////////////////////
    # Dict containing the line lists for each
    line_list_dict = {
        # Kewley (2019) diagnostics - these require log(U)
        "N2Ha_K19": ["NII6583", "HALPHA"],
        "S2Ha_K19": ["SII6716", "SII6731", "HALPHA"],
        "N2S2_K19": ["NII6583", "SII6716", "SII6731"],
        "S23_K19": ["SII6716", "SII6731", "SIII9069", "SIII9531", "HALPHA"],
        "O3N2_K19": ["OIII5007", "HBETA", "NII6583", "HALPHA"],
        "O2S2_K19": ["OII3726", "OII3729", "SII6716", "SII6731"],
        "O2Hb_K19": ["OII3726", "OII3729", "HBETA"],
        "N2O2_K19": ["NII6583", "OII3726", "OII3729"],
        "R23_K19": ["OIII4959", "OIII5007", "OII3726", "OII3729", "HBETA"],

        # Others 
        "N2Ha_PP04": ["NII6583", "HALPHA"],
        "N2Ha_M13": ["NII6583", "HALPHA"],
        "O3N2_PP04": ["OIII5007", "HBETA", "HALPHA", "NII6583"],
        "O3N2_M13": ["OIII5007", "HBETA", "HALPHA", "NII6583"],
        "R23_KK04": ["NII6583", "OII3726", "OII3729", "HBETA", "OIII5007", "OIII4959", "OII3726", "OII3729"],
        "R23_C17": ["OII3726", "OIII4959", "OIII5007", "HBETA"],
        "N2S2Ha_D16": ["NII6583", "SII6716", "SII6731", "HALPHA"],
        "N2O2_KD02": ["NII6583", "OII3726", "OII3729"],
        "Rcal_PG16": ["OII3726", "OII3729", "HBETA", "NII6583", "NII6548", "OIII5007", "OIII4959"],
        "Scal_PG16": ["HBETA", "NII6583", "NII6548", "OIII5007", "OIII4959", "SII6716", "SII6731"],
        "ONS_P10": [],
        "ON_P10": [],
    }

    # For the given met_diagnostic, check that the emission lines exist in the DataFrame 
    for line in line_list_dict[met_diagnostic]:
        assert line in df.columns,\
            f"Metallicity diagnostic {met_diagnostic} requires {line} which was not found in the DataFrame!"

    #//////////////////////////////////////////////////////////////////////////
    # K19 diagnostics
    if met_diagnostic.endswith("K19"):
        assert logU is not None,\
            f"Metallicity diagnostic {met_diagnostic} requires log(U) to be specified!"
        
        # Compute the line ratio
        if met_diagnostic == "N2Ha_K19":
            logR = df["NII6583"] / df["HALPHA"]
        if met_diagnostic == "S2Ha_K19":
            logR = (df["SII6716"] + df["SII6731"]) / df["HALPHA"]
        if met_diagnostic == "N2S2_K19":
            logR = df["NII6583"] / (df["SII6716"] + df["SII6731"])
        if met_diagnostic == "S23_K19": 
            logR = (df["SII6716"] + df["SII6731"] + df["SIII9069"] + df["SIII9531"]) / df["HALPHA"]
        if met_diagnostic == "O3N2_K19":
            logR = (df["OIII5007"] / df["HBETA"]) / (df["NII6583"] / df["HALPHA"])
        if met_diagnostic == "O2S2_K19":
            logR = (df["OII3726"] + df["OII3729"]) / (df["SII6716"] + df["SII6731"])
        if met_diagnostic == "O2Hb_K19":
            logR = (df["OII3726"] + df["OII3729"]) / df["HBETA"]
        if met_diagnostic == "N2O2_K19":
            logR = df["NII6583"] / (df["OII3726"] + df["OII3729"])
        if met_diagnostic == "R23_K19": 
            logR = (df["OIII4959"] + df["OIII5007"] + df["OII3726"] + df["OII3729"]) / df["HBETA"]

        # Compute metallicity
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
        logOH12 = logOH12_func(x=logR, y=logU)

        # What about limits?
        good_pts = logOH12 > met_coeffs[met_diagnostic]["Zmin"]
        good_pts &= logOH12 < met_coeffs[met_diagnostic]["Zmax"]
        logOH12[~good_pts] = np.nan

    #//////////////////////////////////////////////////////////////////////////
    elif met_diagnostic == "R23_KK04":
        # R23 - Kobulnicky & Kewley (2004)
        logOH12 = -999


    elif met_diagnostic == "R23_C17":
        # R23 - Curti et al. (2017)
        logOH12 = -999


    elif met_diagnostic == "N2O2_KD02":
        # N2O2 - Kewley & Dopita (2002)
        # Only reliable above Z > 0.5Zsun (log(O/H) + 12 > 8.6)
        logR = np.log10(df["NII6583"] / (df["OII3726"] + df["OII3729"]))
        logOH12 = np.log10(1.54020 + 1.26602 * logR + 0.167977 * logR**2 ) + 8.93
        good_pts = (logOH12 > 8.6) & (logOH12 < 9.4)  # upper limit eyeballed from their fig. 3
        logOH12[~good_pts] = np.nan

    elif met_diagnostic == "N2Ha_PP04":
        # N2Ha - Pettini & Pagel (2004)
        logR = np.log10(df["NII6583"] / df["HALPHA"])
        logOH12 = 9.37 + 2.03 * logR + 1.26 * logR**2 + 0.32 * logR**3
        good_pts = (-2.5 < logR) & (logR < -0.3)
        logOH12[~good_pts] = np.nan

    elif met_diagnostic == "N2Ha_M13":
        # N2Ha - Marino (2013)
        logR = np.log10(df["NII6583"] / df["HALPHA"])
        logOH12 = 8.743 + 0.462 * logR
        good_pts = (-1.6 < logR) & (logR < -0.2)
        logOH12[~good_pts] = np.nan
    
    if met_diagnostic == "O3N2_PP04":
        # O3N2 - Pettini & Pagel (2004)
        logR = np.log10((df["OIII5007"] / df["HBETA"]) / (df["NII6583"] / df["HALPHA"]))
        logOH12 = 8.73 - 0.32 * logR
        good_pts = (-1 < logR) & (logR < 1.9)
        logOH12[~good_pts] = np.nan

    elif met_diagnostic == "O3N2_M13":
        # O3N2 - Marino (2013)
        logR = np.log10((df["OIII5007"] / df["HBETA"]) / (df["NII6583"] / df["HALPHA"]))
        logOH12 = 8.533 - 0.214 * logR
        good_pts = (-1.1 < logR) & (logR < 1.7)
        logOH12[~good_pts] = np.nan
    
    if met_diagnostic == "N2S2Ha_D16":
        # N2S2Ha - Dopita et al. (2016)
        logR = np.log10(df["NII6583"] / df["SII6716+SII6731"]) + 0.264 * np.log10(df["NII6583"] / df["HALPHA"])
        logOH12 = 8.77 + logR + 0.45 * (logR + 0.3)**5
        good_pts = (-1.1 < logR) & (logR < 0.5)
        logOH12[~good_pts] = np.nan

    elif met_diagnostic == "ON_P10":
        # ON - Pilyugin et al. (2016)
        logOH12 = -999

    elif met_diagnostic == "ONS_P10":
        # ONS - Pilyugin et al. (2016)
        logOH12 = -999

    elif met_diagnostic == "Rcal_PG16":
        # Rcal - Pilyugin & Grebel (2016)
        logO32 = (df["OIII4959"] + df["OIII5007"]) / (df["OII3726"] + df["OII3729"])
        logN2Hb = (df["NII6548"] + df["NII6583"]) / df["HBETA"]
        logO2Hb = (df["OII3726"] + df["OII3729"]) / df["HBETA"]

        # Decide which branch we're on
        logOH12 = np.full(df.shape[0], np.nan)
        pts_lower = logN2Hb < -0.6
        pts_upper = logN2Hb >= 0.6

        logOH12[pts_lower] = 7.932 + 0.944 * logO32 + 0.695 * logN2Hb + ( 0.970 - 0.291 * logO32 - 0.019 * logN2Hb) * logO2Hb
        logOH12[pts_upper] = 8.589 + 0.022 * logO32 + 0.399 * logN2Hb + (-0.137 + 0.164 * logO32 + 0.589 * logN2Hb) * logO2Hb

        # Does this calibration have limits?

    elif met_diagnostic == "Scal_PG16":
        # Scal - Pilyugin & Grebel (2016)
        logO3S2 = (df["OIII4959"] + df["OIII5007"]) / (df["SII6716"] + df["SII6731"])
        logN2Hb = (df["NII6548"] + df["NII6583"]) / df["HBETA"]
        logO2Hb = (df["OII3726"] + df["OII3729"]) / df["HBETA"]

        # Decide which branch we're on
        logOH12 = np.full_like(logO3S2, np.nan)
        pts_lower = logN2Hb < -0.6
        pts_upper = logN2Hb >= 0.6

        logOH12[pts_lower] = 8.072 + 0.789 * logO3S2 + 0.726 * logN2Hb + ( 1.069 - 0.170 * logO3S2 + 0.022 * logN2Hb) * logS2Hb
        logOH12[pts_upper] = 8.424 + 0.030 * logO3S2 + 0.751 * logN2Hb + (-0.349 + 0.182 * logO3S2 + 0.508 * logN2Hb) * logS2Hb

        # Does this calibration have limits?

    #//////////////////////////////////////////////////////////////////////////
    return logOH12


#//////////////////////////////////////////////////////////////////////////
# Metallicity diagnostics (to copy & paste into metalliciy.py)
# def get_metallicity(met_diagnostic, logR_met, 
#                     logU=None):

#     if np.isscalar(logR_met):
#         logR_met = np.array([logR_met])
#         wasscalar = True
#     else:
#         wasscalar = False


#     if met_diagnostic == "R23_KK04":
#         # R23 - Kobulnicky & Kewley (2004)
#         logOH12 = -999


#     elif met_diagnostic == "R23_C17":
#         # R23 - Curti et al. (2017)
#         logOH12 = -999


#     elif met_diagnostic == "N2O2_KD02":
#         # N2O2 - Kewley & Dopita (2002)
#         # Only reliable above Z > 0.5Zsun (log(O/H) + 12 > 8.6)
#         logOH12 = np.log10(1.54020 + 1.26602 * logR_met + 0.167977 * logR_met**2 ) + 8.93
#         good_pts = (logOH12 > 8.6) & (logOH12 < 9.4)  # upper limit eyeballed from their fig. 3
#         logOH12[~good_pts] = np.nan

#     elif met_diagnostic == "N2Ha_PP04":
#         # N2Ha - Pettini & Pagel (2004)
#         logOH12 = 9.37 + 2.03 * logR_met + 1.26 * logR_met**2 + 0.32 * logR_met**3
#         good_pts = (-2.5 < logR_met) & (logR_met < -0.3)
#         logOH12[~good_pts] = np.nan

#     elif met_diagnostic == "N2Ha_M13":
#         # N2Ha - Marino (2013)
#         logOH12 = 8.743 + 0.462 * logR_met
#         good_pts = (-1.6 < logR_met) & (logR_met < -0.2)
#         logOH12[~good_pts] = np.nan
    
#     if met_diagnostic == "O3N2_PP04":
#         # O3N2 - Pettini & Pagel (2004)
#         logOH12 = 8.73 - 0.32 * logR_met
#         good_pts = (-1 < logR_met) & (logR_met < 1.9)
#         logOH12[~good_pts] = np.nan

#     elif met_diagnostic == "O3N2_M13":
#         # O3N2 - Marino (2013)
#         logOH12 = 8.533 - 0.214 * logR_met
#         good_pts = (-1.1 < logR_met) & (logR_met < 1.7)
#         logOH12[~good_pts] = np.nan
    
#     if met_diagnostic == "N2S2Ha_D16":
#         # N2S2Ha - Dopita et al. (2016)
#         logOH12 = 8.77 + logR_met + 0.45 * (logR_met + 0.3)**5
#         good_pts = (-1.1 < logR_met) & (logR_met < 0.5)
#         logOH12[~good_pts] = np.nan

#     elif met_diagnostic == "ON_P10":
#         # ON - Pilyugin et al. (2016)
#         logOH12 = -999

#     elif met_diagnostic == "ONS_P10":
#         # ONS - Pilyugin et al. (2016)
#         logOH12 = -999

#         # Re-cast to scalar
#     if wasscalar:
#         logOH12 = np.float(logOH12)

#     return logOH12

###############################################################################
# Rcal calibration of Pilyugin & Grebel (2016)
def get_metallicity_Rcal(logO32, logN2Hb, logO2Hb):
    """
    IMPORTANT: In the Rcal and Scal calibrations,

    R3 = [OIII]4959,5007 / Hb
    S2 = [SII]6716,31 / Hb (= S2Hb)
    N2 = [NII]6548,83 / Hb (= N2Hb)
    R2 = [OII]3726,9 / Hb (= O2Hb)

    and 

    O3S2 = R3 / S2 = [OIII]4959,5007 / [SII]6716,31
    O3O2 = R3 / R2 = [OIII]4959,5007 / [OII]3726,9  * differs from the standard definition used in ionisation parameter computations

    """
    if np.isscalar(logO32):
        logO32 = np.array([logO32])
        logN2Hb = np.array([logN2Hb])
        logO2Hb = np.array([logO2Hb])
        wasscalar = True
    else:
        wasscalar = False

    assert len(logO32) == len(logN2Hb)
    assert len(logO32) == len(logO2Hb)
 
    # Decide which branch we're on
    logOH12_vals = np.full_like(logO32, np.nan)
    pts_lower = logN2Hb < -0.6
    pts_upper = logN2Hb >= 0.6

    logOH12_vals[pts_lower] = 7.932 + 0.944 * logO32 + 0.695 * logN2Hb + ( 0.970 - 0.291 * logO32 - 0.019 * logN2Hb) * logO2Hb
    logOH12_vals[pts_upper] = 8.589 + 0.022 * logO32 + 0.399 * logN2Hb + (-0.137 + 0.164 * logO32 + 0.589 * logN2Hb) * logO2Hb

    # Re-cast to scalar
    if wasscalar:
        logOH12 = np.float(logOH12)

    return logOH12

###############################################################################
# Scal calibration of Pilyugin & Grebel (2016)
def get_metallicity_Scal(logO3S2, logN2Hb, logS2Hb):
    """
    IMPORTANT: In the Rcal and Scal calibrations,

    R3 = [OIII]4959,5007 / Hb
    S2 = [SII]6716,31 / Hb (= S2Hb)
    N2 = [NII]6548,83 / Hb (= N2Hb)
    R2 = [OII]3726,9 / Hb (= O2Hb)

    and 

    O3S2 = R3 / S2 = [OIII]4959,5007 / [SII]6716,31
    O3O2 = R3 / R2 = [OIII]4959,5007 / [OII]3726,9  * differs from the standard definition used in ionisation parameter computations

    """

    if np.isscalar(logO3S2):
        logO3S2 = np.array([logO3S2])
        logN2Hb = np.array([logN2Hb])
        logS2Hb = np.array([logS2Hb])
        wasscalar = True
    else:
        wasscalar = False

    assert len(logO3S2) == len(logN2Hb)
    assert len(logO3S2) == len(logS2Hb)
 
    # Decide which branch we're on
    logOH12_vals = np.full_like(logO3S2, np.nan)
    pts_lower = logN2Hb < -0.6
    pts_upper = logN2Hb >= 0.6

    logOH12_vals[pts_lower] = 8.072 + 0.789 * logO3S2 + 0.726 * logN2Hb + ( 1.069 - 0.170 * logO3S2 + 0.022 * logN2Hb) * logS2Hb
    logOH12_vals[pts_upper] = 8.424 + 0.030 * logO3S2 + 0.751 * logN2Hb + (-0.349 + 0.182 * logO3S2 + 0.508 * logN2Hb) * logS2Hb

    # Re-cast to scalar
    if wasscalar:
        logOH12 = np.float(logOH12)

    return logOH12


###############################################################################
# N2S2Ha_D16
logR_met_vals = np.linspace(-1.5, 0.7, 100)
logOH12_vals = get_metallicity("N2S2Ha_D16", logR_met_vals)

# Check bounds
assert all(np.isnan(logOH12_vals[logR_met_vals < -1.1]))
assert all(np.isnan(logOH12_vals[logR_met_vals > 0.5]))

# Plot (compare to fig. 3)
fig, ax = plt.subplots(nrows=1, ncols=1)
ax.plot(logOH12_vals, logR_met_vals, "k-")
ax.set_xlabel("log(O/H) + 12")
ax.set_ylabel(r"$\log(R)$ (N2S2Ha_D16)")
ax.set_xlim([7.4, 9.4])
ax.set_ylim([-1.1, 0.6])
ax.grid()

###############################################################################
# N2O2_KD02
logR_met_vals = np.linspace(-1.5, 1.5, 100)
logOH12_vals = get_metallicity("N2O2_KD02", logR_met_vals)

# Check bounds
assert all(np.isnan(logOH12_vals[logOH12_vals < 8.6]))
assert all(np.isnan(logOH12_vals[logOH12_vals > 9.4]))

# Plot (compare to fig. 3)
fig, ax = plt.subplots(nrows=1, ncols=1)
ax.plot(logOH12_vals, logR_met_vals, "k-")
ax.set_xlabel("log(O/H) + 12")
ax.set_ylabel(r"$\log(R)$ (N2O2_KD02)")
ax.grid()

###############################################################################
# N2Ha
#//////////////////////////////////////////////////////////////////////////////
# N2Ha_PP04
logR_met_vals = np.linspace(-2.7, 0.0, 100)
logOH12_vals_PP04 = get_metallicity("N2Ha_PP04", logR_met_vals)

# Check bounds
assert all(np.isnan(logOH12_vals_PP04[logR_met_vals < -2.5]))
assert all(np.isnan(logOH12_vals_PP04[logR_met_vals > -0.3]))

#//////////////////////////////////////////////////////////////////////////////
# N2Ha_M13
logR_met_vals = np.linspace(-2.7, 0.0, 100)
logOH12_vals_M13 = get_metallicity("N2Ha_M13", logR_met_vals)

# Check bounds
assert all(np.isnan(logOH12_vals_M13[logR_met_vals < -1.6]))
assert all(np.isnan(logOH12_vals_M13[logR_met_vals > -0.2]))

#//////////////////////////////////////////////////////////////////////////////
# Plot (compare to fig. 1)
fig, ax = plt.subplots(nrows=1, ncols=1)
ax.plot(logR_met_vals, logOH12_vals_PP04, "k-", label="N2Ha_PP04")
ax.plot(logR_met_vals, logOH12_vals_M13, "r-", label="N2Ha_M13")
ax.set_ylabel("log(O/H) + 12")
ax.set_xlabel(r"$\log(R)$ (N2Ha)")
ax.legend()
ax.grid()

###############################################################################
# O3N2
#//////////////////////////////////////////////////////////////////////////////
# O3N2_PP04
logR_met_vals = np.linspace(-1.4, 3.5, 100)
logOH12_vals_PP04 = get_metallicity("O3N2_PP04", logR_met_vals)

# Check bounds
assert all(np.isnan(logOH12_vals_PP04[logR_met_vals > 1.9]))
assert all(np.isnan(logOH12_vals_PP04[logR_met_vals < -1.0]))

#//////////////////////////////////////////////////////////////////////////////
# O3N2_PP04
logR_met_vals = np.linspace(-1.4, 3.5, 100)
logOH12_vals_M13 = get_metallicity("O3N2_M13", logR_met_vals)

# Check bounds
assert all(np.isnan(logOH12_vals_M13[logR_met_vals > 1.7]))
assert all(np.isnan(logOH12_vals_M13[logR_met_vals < -1.1]))

#//////////////////////////////////////////////////////////////////////////////
# Plot (compare to fig. 1)
fig, ax = plt.subplots(nrows=1, ncols=1)
ax.plot(logR_met_vals, logOH12_vals_PP04, "k-", label="O3N2_PP04")
ax.plot(logR_met_vals, logOH12_vals_M13, "r-", label="O3N2_M13")
ax.set_ylabel("log(O/H) + 12")
ax.set_xlabel(r"$\log(R)$ (O3N2)")
ax.legend()
ax.grid()

###############################################################################
# Rcal and Scal





