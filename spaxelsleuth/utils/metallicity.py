import numpy as np
import pandas as pd
from scipy import constants
from time import time
import warnings

from spaxelsleuth.utils.misc import remove_col_suffix, add_col_suffix

from IPython.core.debugger import set_trace

import logging
logger = logging.getLogger(__name__)

# Dict containing the line lists for each metallicity/ionisation parameter diagnostic
line_list_dict = {
    # Kewley (2019) - log(O/H) + 12
    "N2Ha_K19": ["NII6583", "HALPHA"],
    "S2Ha_K19": ["SII6716+SII6731", "HALPHA"],
    "N2S2_K19": ["NII6583", "SII6716+SII6731"],
    "S23_K19": ["SII6716+SII6731", "SIII9069", "SIII9531", "HALPHA"],
    "O3N2_K19": ["OIII5007", "HBETA", "NII6583", "HALPHA"],
    "O2S2_K19": ["OII3726+OII3729", "SII6716+SII6731"],
    "O2Hb_K19": ["OII3726+OII3729", "HBETA"],
    "N2O2_K19": ["NII6583", "OII3726+OII3729"],
    "R23_K19": ["OIII4959+OIII5007", "OII3726+OII3729", "HBETA"],

    # Kewley (2019) - log(U)
    "O3O2_K19": ["OIII5007", "OII3726+OII3729"],
    "S32_K19": ["SIII9069", "SIII9531", "SII6716+SII6731"],

    # Others 
    "N2Ha_PP04": ["NII6583", "HALPHA"],
    "N2Ha_M13": ["NII6583", "HALPHA"],
    "O3N2_PP04": ["OIII5007", "HBETA", "HALPHA", "NII6583"],
    "O3N2_M13": ["OIII5007", "HBETA", "HALPHA", "NII6583"],
    "R23_KK04": ["NII6583", "OII3726+OII3729", "HBETA", "OIII4959+OIII5007", "OII3726+OII3729"],
    "O3O2_KK04": ["OIII4959+OIII5007", "OII3726+OII3729"], # KK04 - log(U) diagnostic. NOTE different O3O2 defn. to K19
    "N2S2Ha_D16": ["NII6583", "SII6716+SII6731", "HALPHA"],
    "N2O2_KD02": ["NII6583", "OII3726+OII3729"],
    "Rcal_PG16": ["OII3726+OII3729", "HBETA", "NII6548+NII6583", "OIII4959+OIII5007"],
    "Scal_PG16": ["HBETA", "NII6548+NII6583", "OIII4959+OIII5007", "SII6716+SII6731"],
    "ONS_P10": ["OII3726+OII3729", "OIII4959+OIII5007", "NII6548+NII6583", "SII6716+SII6731", "HBETA"],
    "ON_P10": ["OII3726+OII3729", "OIII4959+OIII5007", "NII6548+NII6583", "SII6716+SII6731", "HBETA"],
}

# Coefficients from Kewley (2019)
# Valid for log(P/k) = 5.0 and -3.98 < log(U) < -1.98
met_coeffs_K19 = {
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
        "A" : 12.4894,
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
        "RMS ERR" : 0.42,   # PERCENT
        "Zmin" : 8.53,
        "Zmax" : 9.23,
    },
}

# Valid for log(P/k) = 5.0
ion_coeffs_K19 = {
    "O3O2_K19" : {
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
    "S32_K19" : {
        "A" : 90.017,
        "B" : 21.934,
        "C" : -34.095,
        "D" : -5.0818,
        "E" : -1.4762,
        "F" : 4.1343,
        "G" : 0.3096,
        "H" : 0.1786,
        "I" : 0.1959,
        "J" : -0.1668,
        "RMS ERR" : 1.96,   # PERCENT
        "Zmin" : 7.63,
        "Zmax" : 9.23,
        "Umin" : -3.98,
        "Umax" : -2.48,
    },
}

###############################################################################
def _compute_logOH12(met_diagnostic, df, 
                     logU=None, 
                     compute_logU=False, ion_diagnostic=None, 
                     max_niters=10,
                     verbose=False):
    """
    Use strong-line metallicity diagnostics to compute the metallicity given a 
    DataFrame containing emission line fluxes. 

    Note that this function ONLY calculates the metallicity (and ionisation
    parameter if required) - NO errors are calculated in this function.

    INPUTS
    ---------------------------------------------------------------------------
        met_diagnostic:      str
            Metallicity diagnostic.

        df:                  pandas DataFrame
            DataFrame containing emission line fluxes. NOTE: it's assumed that 
            the input will have had column suffixes removed prior to being 
            passed to this function.

        logU:                float
            Constant log ionisation parameter to assume for diagnostics 
            requiring one - this includes R23_KK04 and all Kewley (2019)
            diagnostics.

        compute_logU:        bool
            If True, compute self-consistent metallicity and ionisation 
            parameters using an iterative approach. Only valid for metallicity
            diagnostics that use the ionisation parameter - this includes 
            R23_KK04 and all Kewley (2019) diagnostics.

        ion_diagnostic:      str 
            The ionisation parameter diagnostic to use if compute_logU is True.
            For the Kewley (2019) diagnostics, the only valid options are 
            O3O2_K19 or S32_K19. For R23_KK04 the only valid option is 
            O3O2_KK04.

        max_niters:          int
            Maximum number of iterations used to compute self-consistent 
            metallicity and ionisation parameters. 

        verbose:            bool
            If True, and if compute_logU, print out the number of converged 
            metallicity/ionisation parameter measurements after each iteration
            using the logger (only if configure_logger(level="DEBUG") has been
            used)
            
    OUTPUTS
    ---------------------------------------------------------------------------
        If logU is not specified -AND- compute_logU == False, this function 
        returns

            logOH12:       float OR numpy.array 
                log(O/H) + 12. If the input DataFrame is a single row, then a 
                float is returned. Otherwise, an array of log(O/H) + 12 values 
                is returned (corresponding to each row).

        Otherwise, this function returns 
            
            logOH12, logU:  tuple of (float OR numpy.array) 
                log(O/H) + 12 and log(U). If the input DataFrame is a single 
                row, then a float is returned for each value. Otherwise, 
                arrays of log(O/H) + 12 and log(U) values are returned 
                (corresponding to each row).

            
    """
    # Assume that linefns.ratio_fn() has already been run on the DataFrame, so
    # that doublets etc. are already there.
    for line in line_list_dict[met_diagnostic]:
        assert line in df,\
            f"Metallicity diagnostic {met_diagnostic} requires {line} which was not found in the DataFrame!"
    if compute_logU:
        for line in line_list_dict[ion_diagnostic]:
            assert line in df,\
                f"ionisation parameter diagnostic {ion_diagnostic} requires {line} which was not found in the DataFrame!"

    # K19 diagnostics
    if met_diagnostic.endswith("K19"):
        assert compute_logU or (logU is not None),\
            f"Metallicity diagnostic {met_diagnostic} requires log(U) to be specified!"
        
        # Compute the line ratio
        if met_diagnostic == "N2Ha_K19":
            logR = np.log10(df["NII6583"].values / df["HALPHA"].values)
        if met_diagnostic == "S2Ha_K19":
            logR = np.log10((df["SII6716+SII6731"].values) / df["HALPHA"].values)
        if met_diagnostic == "N2S2_K19":
            logR = np.log10(df["NII6583"].values / (df["SII6716+SII6731"].values))
        if met_diagnostic == "S23_K19": 
            logR = np.log10((df["SII6716+SII6731"].values + df["SIII9069"].values + df["SIII9531"].values) / df["HALPHA"].values)
        if met_diagnostic == "O3N2_K19":
            logR = np.log10((df["OIII5007"].values / df["HBETA"].values) / (df["NII6583"].values / df["HALPHA"].values))
        if met_diagnostic == "O2S2_K19":
            logR = np.log10((df["OII3726+OII3729"].values) / (df["SII6716+SII6731"].values))
        if met_diagnostic == "O2Hb_K19":
            logR = np.log10((df["OII3726+OII3729"].values) / df["HBETA"].values)
        if met_diagnostic == "N2O2_K19":
            logR = np.log10(df["NII6583"].values / (df["OII3726+OII3729"].values))
        if met_diagnostic == "R23_K19": 
            logR = np.log10((df["OIII4959+OIII5007"].values + df["OII3726+OII3729"].values) / df["HBETA"].values)

        # Compute metallicity
        logOH12_func = lambda x, y : \
              met_coeffs_K19[met_diagnostic]["A"] \
            + met_coeffs_K19[met_diagnostic]["B"] * x \
            + met_coeffs_K19[met_diagnostic]["C"] * y \
            + met_coeffs_K19[met_diagnostic]["D"] * x * y \
            + met_coeffs_K19[met_diagnostic]["E"] * x**2 \
            + met_coeffs_K19[met_diagnostic]["F"] * y**2 \
            + met_coeffs_K19[met_diagnostic]["G"] * x * y**2 \
            + met_coeffs_K19[met_diagnostic]["H"] * y * x**2 \
            + met_coeffs_K19[met_diagnostic]["I"] * x**3 \
            + met_coeffs_K19[met_diagnostic]["J"] * y**3
        
        if compute_logU:
            # Compute a self-consistent ionisation parameter using the O3O2
            # diagnostic
            logR_met = np.array(logR)
            if ion_diagnostic == "O3O2_K19":
                logR_ion = np.array(np.log10(df["OIII5007"].values / df["OII3726+OII3729"].values))
            if ion_diagnostic == "S23_K19":
                logR_ion = np.array(np.log10((df["SIII9069"].values + df["SIII9531"].values) / df["SII6716+SII6731"].values))

            # Ionisation parameter
            # x = log(R)
            # y = log(O/H) + 12
            logU_func = lambda x, y : \
                  ion_coeffs_K19[ion_diagnostic]["A"]   \
                + ion_coeffs_K19[ion_diagnostic]["B"] * x     \
                + ion_coeffs_K19[ion_diagnostic]["C"] * y     \
                + ion_coeffs_K19[ion_diagnostic]["D"] * x * y   \
                + ion_coeffs_K19[ion_diagnostic]["E"] * x**2  \
                + ion_coeffs_K19[ion_diagnostic]["F"] * y**2  \
                + ion_coeffs_K19[ion_diagnostic]["G"] * x * y**2    \
                + ion_coeffs_K19[ion_diagnostic]["H"] * y * x**2    \
                + ion_coeffs_K19[ion_diagnostic]["I"] * x**3  \
                + ion_coeffs_K19[ion_diagnostic]["J"] * y**3

            # Starting guesses
            logOH12_old = 8.0
            logU_old = -3.0
            for n in range(max_niters):
                # Recompute values
                logU_new = logU_func(x=logR_ion, y=logOH12_old)
                logOH12_new = logOH12_func(x=logR_met, y=logU_new)
                
                # Compute the consistency between this and the previous iteration
                diff_logOH12 = np.abs(logOH12_new - logOH12_old)
                diff_logU = np.abs(logU_new - logU_old)
                if verbose:
                    logger.debug(f"After {n} iterations: {len(diff_logOH12[diff_logOH12 >= 0.001])}/{len(diff_logOH12)} unconverged log(O/H) + 12 measurements, {len(diff_logU[diff_logU >= 0.001])}/{len(diff_logU)} unconverged log(U) measurements")
                if all(diff_logU < 0.001) and all(diff_logOH12 < 0.001):
                    break

                # Update variables 
                logOH12_old = logOH12_new
                logU_old = logU_new

            # Final values
            logOH12 = logOH12_new
            logU = logU_new

            # Mask out values where the convergence test fails
            good_pts = diff_logOH12 < 0.001
            good_pts &= diff_logU < 0.001

            # Apply limits 
            good_pts = logOH12 > met_coeffs_K19[met_diagnostic]["Zmin"]     # log(O/H) + 12 - metallicity limits
            good_pts &= logOH12 < met_coeffs_K19[met_diagnostic]["Zmax"]    # log(O/H) + 12 - metallicity limits
            good_pts &= logOH12 < ion_coeffs_K19[ion_diagnostic]["Zmax"]    # log(U) - metallicity limits
            good_pts &= logOH12 > ion_coeffs_K19[ion_diagnostic]["Zmin"]    # log(U) - metallicity limits
            good_pts &= logU < ion_coeffs_K19[ion_diagnostic]["Umax"]       # log(U) - ionisation parameter limits
            good_pts &= logU > ion_coeffs_K19[ion_diagnostic]["Umin"]       # log(U) - ionisation parameter limits
            logOH12[~good_pts] = np.nan
            logU[~good_pts] = np.nan

            return np.squeeze(logOH12), np.squeeze(logU)

        else:
            # Assume a fixed ionisation parameter
            logOH12 = logOH12_func(x=logR, y=logU)

            # Apply limits 
            good_pts = logOH12 > met_coeffs_K19[met_diagnostic]["Zmin"]     # log(O/H) + 12 - metallicity limits
            good_pts &= logOH12 < met_coeffs_K19[met_diagnostic]["Zmax"]    # log(O/H) + 12 - metallicity limits
            logOH12[~good_pts] = np.nan
            logU = np.full(df.shape[0], logU)
            logU[~good_pts] = np.nan

            return np.squeeze(logOH12), np.squeeze(logU)

    elif met_diagnostic == "R23_KK04":
        logger.warning(f"There may be an error in the implementation of the R23_KK04 diagnostic - proceed with caution!!")
        # R23 - Kobulnicky & Kewley (2004)
        # NOTE: from the paper, it's pretty clear that they only use the OII3726 in the denominator. 
        # HOWEVER, in the appendix of Poetrodjojo+2021, they seem to include the other line as well. So I'm not sure who to believe!! 
        logN2O2 = np.log10(df["NII6583"].values / df["OII3726+OII3729"].values)
        logO3O2 = np.log10(df["OIII4959+OIII5007"].values / df["OII3726+OII3729"].values)  # NOTE: their eqn. 10 suggests that the denominator should only include the 3727 line.
        logR23 = np.log10((df["OII3726+OII3729"].values + df["OIII4959+OIII5007"].values) / df["HBETA"].values)
        logOH12 = np.full(df.shape[0], np.nan)
 
        if compute_logU:
            # Compute a self-consistent ionisation parameter using the O3O2 diagnostic
            logOH12_old = np.full(df.shape[0], np.nan)
            logOH12_new = np.full(df.shape[0], np.nan)

            # Starting guesses 
            pts_lower = logN2O2 < -1.2  # NOTE: these initial guesses which tell us which branch to use are taken by eyeballing fig. 3 of KD02.
            pts_upper = logN2O2 >= -1.2  # NOTE: these initial guesses which tell us which branch to use are taken by eyeballing fig. 3 of KD02.
            logOH12_old[pts_lower] = 8.2
            logOH12_old[pts_upper] = 8.7
            logq_old = -99999  # Dummy value

            for n in range(max_niters):

                # Recompute values (eqn. 13)
                # NOTE: there is a serious transcription error in the version of this eqn. that appears in the appendix of Poeotrodjojo+2021: refer to the eqn. in the original paper
                logq_new = (32.81 - 1.153 * logO3O2**2\
                        + logOH12_old * (-3.396 - 0.025 * logO3O2 + 0.1444 * logO3O2**2))\
                       * (4.603 - 0.3119 * logO3O2 - 0.163 * logO3O2**2\
                          + logOH12_old * (-0.48 + 0.0271 * logO3O2 + 0.02037 * logO3O2**2))**(-1)
                
                # Compute metallicity (eqns. 16 and 17)
                # NOTE: this is a new and improved parameterisation of the diagnostic presented in KD02.
                pts_lower = logN2O2 < -1.2
                pts_upper = logN2O2 >= -1.2
                logOH12_new[pts_lower] = 9.40 + 4.65 * logR23[pts_lower] - 3.17 * logR23[pts_lower]**2 - logq_new[pts_lower] * (0.272 + 0.547 * logR23[pts_lower] - 0.513 * logR23[pts_lower]**2)
                logOH12_new[pts_upper] = 9.72 - 0.777 * logR23[pts_upper] - 0.951 * logR23[pts_upper]**2 - 0.072 * logR23[pts_upper]**3 - 0.811 * logR23[pts_upper]**4 - logq_new[pts_upper] * (0.0737 - 0.0713 * logR23[pts_upper] - 0.141 * logR23[pts_upper]**2 + 0.0373 * logR23[pts_upper]**3 - 0.058 * logR23[pts_upper]**4)

                # Compute the consistency between this and the previous iteration
                diff_logOH12 = np.abs(logOH12_new - logOH12_old)
                diff_logq = np.abs(logq_new - logq_old)
                # logger.debug(f"After {n} iterations: {len(diff_logOH12[diff_logOH12 >= 0.001])}/{len(diff_logOH12)} unconverged log(O/H) + 12 measurements, {len(diff_logq[diff_logq >= 0.001])}/{len(diff_logq)} unconverged log(q) measurements")
                if all(diff_logq < 0.001) and all(diff_logOH12 < 0.001):
                    break

                # Update variables 
                logOH12_old = logOH12_new
                logq_old = logq_new

            # Final values
            logOH12 = logOH12_new
            logU = logq_new - np.log10(constants.c * 1e2)

            # Mask out values where the convergence test fails
            good_pts = diff_logOH12 < 0.001
            good_pts &= diff_logq < 0.001            

            # Limits 
            good_pts &= logR23 < 1.0  # The diagnostic isn't defined for logR23 > 1.0, so remove these.
            logOH12[~good_pts] = np.nan
            logU[~good_pts] = np.nan

            return np.squeeze(logOH12), np.squeeze(logU)

        else:
            # Assume a fixed ionisation parameter
            logq = logU + np.log10(constants.c * 1e2)
            pts_lower = logN2O2 < -1.2
            pts_upper = logN2O2 >= -1.2
            logOH12[pts_lower] = 9.40 + 4.65 * logR23[pts_lower] - 3.17 * logR23[pts_lower]**2 - logq * (0.272 + 0.547 * logR23[pts_lower] - 0.513 * logR23[pts_lower]**2)
            logOH12[pts_upper] = 9.72 - 0.777 * logR23[pts_upper] - 0.951 * logR23[pts_upper]**2 - 0.072 * logR23[pts_upper]**3 - 0.811 * logR23[pts_upper]**4 - logq * (0.0737 - 0.0713 * logR23[pts_upper] - 0.141 * logR23[pts_upper]**2 + 0.0373 * logR23[pts_upper]**3 - 0.058 * logR23[pts_upper]**4)

            # The diagnostic isn't defined for logR23 > 1.0, so remove these.
            good_pts = logR23 < 1.0
            logOH12[~good_pts] = np.nan
            logU = np.full(df.shape[0], logU)
            logU[~good_pts] = np.nan

            return np.squeeze(logOH12), np.squeeze(logU)

    elif met_diagnostic == "N2O2_KD02":
        # N2O2 - Kewley & Dopita (2002)
        # Only reliable above Z > 0.5Zsun (log(O/H) + 12 > 8.6)
        # NOTE: the text in section 4.1 seems to imply that the denominator only includes OII3726, but the caption of fig. 3 says that it includes both.
        # Without any additional information, we assume that the denominator includes both lines (since in most surveys the doublet can't be resolved anyway)
        logger.warning(f"There may be an error in the implementation of the N2O2_KD02 diagnostic - proceed with caution!!")
        logR = np.log10(df["NII6583"].values / (df["OII3726+OII3729"].values))
        logOH12 = np.log10(1.54020 + 1.26602 * logR + 0.167977 * logR**2 ) + 8.93
        good_pts = (logOH12 > 8.6) & (logOH12 < 9.4)  # upper limit eyeballed from their fig. 3
        logOH12[~good_pts] = np.nan
        return np.squeeze(logOH12)

    elif met_diagnostic == "N2Ha_PP04":
        # N2Ha - Pettini & Pagel (2004)
        logR = np.log10(df["NII6583"].values / df["HALPHA"].values)
        logOH12 = 9.37 + 2.03 * logR + 1.26 * logR**2 + 0.32 * logR**3
        good_pts = (-2.5 < logR) & (logR < -0.3)
        logOH12[~good_pts] = np.nan
        return np.squeeze(logOH12)

    elif met_diagnostic == "N2Ha_M13":
        # N2Ha - Marino (2013)
        # NOTE: here we employ the Te-based calibration (eqn. 4), rather than the one based on CALIFA ONS data. 
        logR = np.log10(df["NII6583"].values / df["HALPHA"].values)
        logOH12 = 8.743 + 0.462 * logR
        good_pts = (-1.6 < logR) & (logR < -0.2)
        logOH12[~good_pts] = np.nan
        return np.squeeze(logOH12)
    
    if met_diagnostic == "O3N2_PP04":
        # O3N2 - Pettini & Pagel (2004)
        logR = np.log10((df["OIII5007"].values / df["HBETA"].values) / (df["NII6583"].values / df["HALPHA"].values))
        logOH12 = 8.73 - 0.32 * logR
        good_pts = (-1 < logR) & (logR < 1.9)
        logOH12[~good_pts] = np.nan
        return np.squeeze(logOH12)

    elif met_diagnostic == "O3N2_M13":
        # O3N2 - Marino (2013)
        logR = np.log10((df["OIII5007"].values / df["HBETA"].values) / (df["NII6583"].values / df["HALPHA"].values))
        logOH12 = 8.533 - 0.214 * logR
        good_pts = (-1.1 < logR) & (logR < 1.7)
        logOH12[~good_pts] = np.nan
        return np.squeeze(logOH12)
    
    if met_diagnostic == "N2S2Ha_D16":
        # N2S2Ha - Dopita et al. (2016)
        logR = np.log10(df["NII6583"].values / df["SII6716+SII6731"].values) + 0.264 * np.log10(df["NII6583"].values / df["HALPHA"].values)
        logOH12 = 8.77 + logR + 0.45 * (logR + 0.3)**5
        good_pts = (-1.1 < logR) & (logR < 0.5)  # Limits eyeballed from their fig. 3
        logOH12[~good_pts] = np.nan
        return np.squeeze(logOH12)

    elif met_diagnostic == "ONS_P10":
        # ONS - Pilyugin et al. (2016)
        logN2 = np.log10(df["NII6548+NII6583"].values / df["HBETA"].values)
        logS2 = np.log10(df["SII6716+SII6731"].values / df["HBETA"].values)
        logR3 = np.log10(df["OIII4959+OIII5007"].values / df["HBETA"].values)
        logR2 = np.log10(df["OII3726+OII3729"].values / df["HBETA"].values)
        R3 = df["OIII4959+OIII5007"].values / df["HBETA"].values
        R2 = df["OII3726+OII3729"].values / df["HBETA"].values
        P = R3 / (R3 + R2)  # NOTE: ratios of the non-log ratios

        logOH12 = np.full(df.shape[0], np.nan)
        pts_cool = logN2 >= -0.1
        pts_warm = (logN2 < -0.1) & ((logN2 - logS2) >= -0.25)
        pts_hot = (logN2 < -0.1) & ((logN2 - logS2) < -0.25)

        # Eqn. (17) from P10
        # Note that the below "warm" equation has a transcription error in Poetrodjojo+2021
        logOH12[pts_cool] = 8.277 + 0.657 * P[pts_cool] - 0.399 * logR3[pts_cool] - 0.061 * (logN2[pts_cool] - logR2[pts_cool]) + 0.005 * (logS2[pts_cool] - logR2[pts_cool])
        logOH12[pts_warm] = 8.816 - 0.733 * P[pts_warm] + 0.454 * logR3[pts_warm] + 0.710 * (logN2[pts_warm] - logR2[pts_warm]) - 0.337 * (logS2[pts_warm] - logR2[pts_warm])
        logOH12[pts_hot]  = 8.774 - 1.855 * P[pts_hot] + 1.517 * logR3[pts_hot] + 0.304 * (logN2[pts_hot] - logR2[pts_hot]) + 0.328 * (logS2[pts_hot] - logR2[pts_hot])

        return np.squeeze(logOH12)

    elif met_diagnostic == "ON_P10":
        # ON - Pilyugin et al. (2016)
        logN2 = np.log10(df["NII6548+NII6583"].values / df["HBETA"].values)
        logS2 = np.log10(df["SII6716+SII6731"].values / df["HBETA"].values)
        logR3 = np.log10(df["OIII4959+OIII5007"].values / df["HBETA"].values)
        logR2 = np.log10(df["OII3726+OII3729"].values / df["HBETA"].values)

        logOH12 = np.full(df.shape[0], np.nan)
        pts_cool = logN2 >= -0.1
        pts_warm = (logN2 < -0.1) & ((logN2 - logS2) >= -0.25)
        pts_hot = (logN2 < -0.1) & ((logN2 - logS2) < -0.25)

        # Eqn. (19) from P10
        logOH12[pts_cool] = 8.606 - 0.105 * logR3[pts_cool] - 0.410 * logR2[pts_cool] - 0.150 * (logN2[pts_cool] - logR2[pts_cool])
        logOH12[pts_warm] = 8.642 + 0.077 * logR3[pts_warm] + 0.411 * logR2[pts_warm] + 0.601 * (logN2[pts_warm] - logR2[pts_warm])
        logOH12[pts_hot]  = 8.013 + 0.905 * logR3[pts_hot] + 0.602 * logR2[pts_hot] + 0.751 * (logN2[pts_hot] - logR2[pts_hot])

        return np.squeeze(logOH12)

    elif met_diagnostic == "Rcal_PG16":
        # Rcal - Pilyugin & Grebel (2016)
        logO32 = np.log10((df["OIII4959+OIII5007"].values) / (df["OII3726+OII3729"].values))  # Their R3/R2
        logN2Hb = np.log10((df["NII6548+NII6583"].values) / df["HBETA"].values)  # Their N2
        logO2Hb = np.log10((df["OII3726+OII3729"].values) / df["HBETA"].values)  # Their R2

        # Decide which branch we're on
        logOH12 = np.full(df.shape[0], np.nan)
        pts_lower = logN2Hb < -0.6
        pts_upper = logN2Hb >= -0.6

        logOH12[pts_lower] = 7.932 + 0.944 * logO32[pts_lower] + 0.695 * logN2Hb[pts_lower] + ( 0.970 - 0.291 * logO32[pts_lower] - 0.019 * logN2Hb[pts_lower]) * logO2Hb[pts_lower]
        logOH12[pts_upper] = 8.589 + 0.022 * logO32[pts_upper] + 0.399 * logN2Hb[pts_upper] + (-0.137 + 0.164 * logO32[pts_upper] + 0.589 * logN2Hb[pts_upper]) * logO2Hb[pts_upper]

        # Does this calibration have limits?
        return np.squeeze(logOH12)

    elif met_diagnostic == "Scal_PG16":
        # Scal - Pilyugin & Grebel (2016)
        logO3S2 = np.log10((df["OIII4959+OIII5007"].values) / (df["SII6716+SII6731"].values))  # Their R3/S2
        logN2Hb = np.log10((df["NII6548+NII6583"].values) / df["HBETA"].values)  # Their N2 
        logS2Hb = np.log10((df["SII6716+SII6731"].values) / df["HBETA"].values)  # Their S2

        # Decide which branch we're on
        logOH12 = np.full_like(logO3S2, np.nan)
        pts_lower = logN2Hb < -0.6
        pts_upper = logN2Hb >= -0.6

        logOH12[pts_lower] = 8.072 + 0.789 * logO3S2[pts_lower] + 0.726 * logN2Hb[pts_lower] + ( 1.069 - 0.170 * logO3S2[pts_lower] + 0.022 * logN2Hb[pts_lower]) * logS2Hb[pts_lower]
        logOH12[pts_upper] = 8.424 + 0.030 * logO3S2[pts_upper] + 0.751 * logN2Hb[pts_upper] + (-0.349 + 0.182 * logO3S2[pts_upper] + 0.508 * logN2Hb[pts_upper]) * logS2Hb[pts_upper]

        # Does this calibration have limits?
        return np.squeeze(logOH12)

###############################################################################
def _met_helper_fn(args):
    """Helper function used in _get_metallicity() to compute metallicities across multiple threads."""
    met_diagnostic, df, logU, compute_logU, ion_diagnostic, niters = args
    df = df.copy()  # Make a copy to avoid the pandas SettingWithCopyWarning
    compute_errors = True if niters > 1 else False
    
    # Compute metallicities in ALL rows plus errors 
    logOH12_vals = np.full((niters, df.shape[0]), np.nan)
    if (logU is not None) or compute_logU:
        logU_vals = np.full((niters, df.shape[0]), np.nan)

    # Evaluate log(O/H) + 12 (and log(U) if compute_logU is True) niters times 
    # with random noise added to the emission line fluxes each time
    if compute_errors and compute_logU:
        logger.debug(f"computing log(O/H) + 12 and log(U) (+ errors) using diagnostics {met_diagnostic} and {ion_diagnostic} with {niters} iterations...")
    elif compute_errors:
        logger.debug(f"computing log(O/H) + 12 (+ errors) using diagnostic {met_diagnostic} with {niters} iterations...")
    else:
        logger.debug(f"computing log(O/H) + 12 using diagnostic {met_diagnostic}...")
    
    for nn in range(niters):
        # Make a copy of the DataFrame
        df_tmp = df.copy()
        
        # Add random error 
        if compute_errors:
            for eline in line_list_dict[met_diagnostic]:
                df_tmp[eline] += np.random.normal(scale=df_tmp[f"{eline} error"])
                cond_neg = df_tmp[eline] < 0
                df_tmp.loc[cond_neg, eline] = np.nan
            if compute_logU:
                for eline in line_list_dict[ion_diagnostic]:
                    df_tmp[eline] += np.random.normal(scale=df_tmp[f"{eline} error"])
                    cond_neg = df_tmp[eline] < 0
                    df_tmp.loc[cond_neg, eline] = np.nan

        # Compute corresponding metallicity
        if compute_logU:
            res = _compute_logOH12(met_diagnostic=met_diagnostic, df=df_tmp, 
                                compute_logU=compute_logU, ion_diagnostic=ion_diagnostic)
            logOH12_vals[nn] = res[0]
            logU_vals[nn] = res[1]

        elif logU is not None:
            res = _compute_logOH12(met_diagnostic=met_diagnostic, df=df_tmp, 
                                logU=logU)
            logOH12_vals[nn] = res[0]
            logU_vals[nn] = res[1]

        else:
            res = _compute_logOH12(met_diagnostic=met_diagnostic, df=df_tmp, 
                                                logU=None, compute_logU=False)
            logOH12_vals[nn] = res

    # Count number of NaN entries - these will be due to MC iterations bumping values into invalid ranges for the diagnostic(s) used
    frac_finite_measurements_logOH12 = np.nansum(~np.isnan(logOH12_vals), axis=0) / niters
    if compute_logU:
        frac_finite_measurements_logU = np.nansum(~np.isnan(logU_vals), axis=0) / niters
        valid_measurements = (frac_finite_measurements_logU > 0.5) & (frac_finite_measurements_logOH12 > 0.5)
        logger.info(f"For diagnostic {met_diagnostic}/{ion_diagnostic}, I am masking out {len(valid_measurements[~valid_measurements])} measurements in which >50% of MC iterations returned a NaN metallicity or ionisation parameter value")
    else:
        valid_measurements = frac_finite_measurements_logOH12 > 0.5
        logger.info(f"For diagnostic {met_diagnostic}, I am masking out {len(valid_measurements[~valid_measurements])} measurements in which >50% of MC iterations returned a NaN metallicity value")

    # Compute mean measurements from the MC runs, plus errors
    # NaN out metallicity/ionisation parameter measurements where >50% of measurements in either have failed 
    logOH12_mean = np.nanmean(logOH12_vals, axis=0)
    logOH12_mean[~valid_measurements] = np.nan
    if compute_logU:
        logU_mean = np.nanmean(logU_vals, axis=0)
        logU_mean[~valid_measurements] = np.nan
    if compute_errors:
        logOH12_q16 = np.quantile(logOH12_vals, q=0.16, axis=0)
        logOH12_q16[~valid_measurements] = np.nan
        logOH12_q84 = np.quantile(logOH12_vals, q=0.84, axis=0)
        logOH12_q84[~valid_measurements] = np.nan
        if compute_logU:
            logU_q16 = np.quantile(logU_vals, q=0.16, axis=0)
            logU_q16[~valid_measurements] = np.nan
            logU_q84 = np.quantile(logU_vals, q=0.84, axis=0)
            logU_q84[~valid_measurements] = np.nan

    # Add to DataFrame
    with warnings.catch_warnings():
        warnings.filterwarnings(action="ignore", category=RuntimeWarning, message="Mean of empty slice")
        if compute_logU:
            df.loc[:, f"log(O/H) + 12 ({met_diagnostic}/{ion_diagnostic})"] = logOH12_mean
            df.loc[:, f"log(U) ({met_diagnostic}/{ion_diagnostic})"] = logU_mean
            if compute_errors:
                df.loc[:, f"log(O/H) + 12 ({met_diagnostic}/{ion_diagnostic}) error (lower)"] = logOH12_mean - logOH12_q16
                df.loc[:, f"log(O/H) + 12 ({met_diagnostic}/{ion_diagnostic}) error (upper)"] = logOH12_q84 - logOH12_mean
                df.loc[:, f"log(U) ({met_diagnostic}/{ion_diagnostic}) error (lower)"] = logU_mean - logU_q16
                df.loc[:, f"log(U) ({met_diagnostic}/{ion_diagnostic}) error (upper)"] = logU_q84 - logU_mean
        
        else:
            df.loc[:, f"log(O/H) + 12 ({met_diagnostic})"] = logOH12_mean
            if compute_errors:
                df.loc[:, f"log(O/H) + 12 ({met_diagnostic}) error (lower)"] = logOH12_mean - logOH12_q16
                df.loc[:, f"log(O/H) + 12 ({met_diagnostic}) error (upper)"] = logOH12_q84 - logOH12_mean
            if logU is not None:
                df.loc[:, f"log(U) ({met_diagnostic})"] = logU_mean

    return df 

###############################################################################
def _get_metallicity(met_diagnostic, df,
                     logU=None, 
                     compute_logU=False, ion_diagnostic=None,
                     niters=1000):
    """
    A helper function that is used in calculate_metallicity().
    
    INPUTS 
    ---------------------------------------------------------------------------

    met_diagnostic:     str
        Strong-line metallicity diagnostic to use. Options:
            N2Ha_K19    N2Ha diagnostic from Kewley (2019).
            S2Ha_K19    S2Ha diagnostic from Kewley (2019).
            N2S2_K19    N2S2 diagnostic from Kewley (2019).
            S23_K19     S23 diagnostic from Kewley (2019).
            O3N2_K19    O3N2 diagnostic from Kewley (2019).
            O2S2_K19    O2S2 diagnostic from Kewley (2019).
            O2Hb_K19    O2Hb diagnostic from Kewley (2019).
            N2O2_K19    N2O2 diagnostic from Kewley (2019).
            R23_K19     R23 diagnostic from Kewley (2019).
            N2Ha_PP04   N2Ha diagnostic from Pilyugin & Peimbert (2004).
            N2Ha_M13    N2Ha diagnostic from Marino et al. (2013).
            O3N2_PP04   O3N2 diagnostic from Pilyugin & Peimbert (2004).
            O3N2_M13    O3N2 diagnostic from Marino et al. (2013).
            R23_KK04    R23 diagnostic from Kobulnicky & Kewley (2004).
            N2S2Ha_D16  N2S2Ha diagnostic from Dopita et al. (2016).
            N2O2_KD02   N2O2 diagnostic from Kewley & Dopita (2002).
            Rcal_PG16   Rcal diagnostic from Pilyugin & Grebel (2016).
            Scal_PG16   Scal diagnostic from Pilyugin & Grebel (2016).
            ONS_P10     ONS diagnostic from Pilyugin et al. (2010).
            ON_P10      ON diagnostic from Pilyugin et al. (2010).
        
    df:                  pandas DataFrame
        Pandas DataFrame containing emission line fluxes.

    logU:                float
        Constant dimensionless ionisation parameter (where U = q / c) to 
        assume in diagnostics that require log(U) to be specified. These 
        include all Kewley (2019) diagnostics and the R23 calibration of 
        Kobulnicky & Kewley (2004).

    compute_logU:        bool
        If True, iteratively compute self-consistent metallicities and 
        ionisation parameters using a strong-line ionisation parameter 
        diagnostic. This parameter ignores the logU input parameter.

    ion_diagnostic:      str 
        If compute_logU is set, ion_diagnostic specifies the strong-line 
        ionisation parameter diagnostic to use. If the metallicity diagnostic
        is from Kewley (2019) then ion_diagnostic may be "O3O2_K19" or 
        "S32_K19". If the metallicity diagnostic is "R23_KD04" then 
        ion_diagnostic must be "O3O2_KD04".

    compute_errors:      bool
        If True, estimate 1-sigma errors on log(O/H) + 12 and log(U) using a 
        Monte Carlo approach, in which the 1-sigma uncertainties on the 
        emission line fluxes are used generate a distribution in log(O/H) + 12 
        values, the mean and standard deviation of which are used to 
        evaluate the metallicity and corresponding uncertainty.

    niters:              int
        Number of MC iterations. 1000 is recommended.

    OUTPUTS 
    ---------------------------------------------------------------------------

    The input DataFrame with additional columns added containing the computed 
    metallicity (plus ionisation parameter) and associated errors if 
    compute_errors is set to True.

    """
    # Speed up execution by only passing rows where the metallicity can be calculated
    if "BPT" in df:
        cond_nomet = df["BPT"] != "SF"
    elif "BPT (numeric)" in df:
        cond_nomet = df["BPT (numeric)"] != 0
    for eline in line_list_dict[met_diagnostic]:
        cond_nomet |= df[f"{eline}"].isna()
        cond_nomet |= df[f"{eline} error"].isna()
    if compute_logU:
        for eline in line_list_dict[ion_diagnostic]:
            cond_nomet |= df[f"{eline}"].isna()
            cond_nomet |= df[f"{eline} error"].isna()

    df_nomet = df[cond_nomet].copy()
    df_met = df[~cond_nomet].copy()
    if ion_diagnostic is None:
        logger.debug(f"able to calculate {met_diagnostic} log(O/H) + 12  in {df_met.shape[0]:d}/{df.shape[0]:d} ({df_met.shape[0] / df.shape[0] * 100:.2f}%) of rows")
    else:
        logger.debug(f"able to calculate {met_diagnostic}/{ion_diagnostic} log(O/H) + 12  in {df_met.shape[0]:d}/{df.shape[0]:d} ({df_met.shape[0] / df.shape[0] * 100:.2f}%) of rows")

    # Compute metallicities in the subset of rows with valid line fluxes & BPT classifications
    df_met = _met_helper_fn([met_diagnostic, df_met, logU, compute_logU, ion_diagnostic, niters])

    # Add empty columns to stop Pandas from throwing an error at pd.concat
    new_cols = [c for c in df_met.columns if c not in df_nomet.columns]
    for c in new_cols:
        df_nomet[c] = np.nan

    # Merge back with original DataFrame
    logger.debug(f"concatenating DataFrames...")
    df = pd.concat([df_nomet, df_met])

    logger.debug(f"done!")
    return df

###############################################################################
def calculate_metallicity(df, met_diagnostic, 
                          logU=None, 
                          compute_logU=False, ion_diagnostic=None,
                          compute_errors=True, niters=1000,
                          s=None):
    """
    Compute metallicities using strong-line diagnostics.

    INPUTS 
    ---------------------------------------------------------------------------

    df:                  pandas DataFrame
        Pandas DataFrame containing emission line fluxes.

    met_diagnostic:      str
        Strong-line metallicity diagnostic to use. Options:
            N2Ha_K19    N2Ha diagnostic from Kewley (2019).
            S2Ha_K19    S2Ha diagnostic from Kewley (2019).
            N2S2_K19    N2S2 diagnostic from Kewley (2019).
            S23_K19     S23 diagnostic from Kewley (2019).
            O3N2_K19    O3N2 diagnostic from Kewley (2019).
            O2S2_K19    O2S2 diagnostic from Kewley (2019).
            O2Hb_K19    O2Hb diagnostic from Kewley (2019).
            N2O2_K19    N2O2 diagnostic from Kewley (2019).
            R23_K19     R23 diagnostic from Kewley (2019).
            N2Ha_PP04   N2Ha diagnostic from Pilyugin & Peimbert (2004).
            N2Ha_M13    N2Ha diagnostic from Marino et al. (2013).
            O3N2_PP04   O3N2 diagnostic from Pilyugin & Peimbert (2004).
            O3N2_M13    O3N2 diagnostic from Marino et al. (2013).
            R23_KK04    R23 diagnostic from Kobulnicky & Kewley (2004).
            N2S2Ha_D16  N2S2Ha diagnostic from Dopita et al. (2016).
            N2O2_KD02   N2O2 diagnostic from Kewley & Dopita (2002).
            Rcal_PG16   Rcal diagnostic from Pilyugin & Grebel (2016).
            Scal_PG16   Scal diagnostic from Pilyugin & Grebel (2016).
            ONS_P10     ONS diagnostic from Pilyugin et al. (2010).
            ON_P10      ON diagnostic from Pilyugin et al. (2010).

    logU:                float
        Constant dimensionless ionisation parameter (where U = q / c) to 
        assume in diagnostics that require log(U) to be specified. These 
        include all Kewley (2019) diagnostics and the R23 calibration of 
        Kobulnicky & Kewley (2004).
    
    compute_logU:        bool
        If True, iteratively compute self-consistent metallicities and 
        ionisation parameters using a strong-line ionisation parameter 
        diagnostic. This parameter ignores the logU input parameter.

    ion_diagnostic:      str 
        If compute_logU is set, ion_diagnostic specifies the strong-line 
        ionisation parameter diagnostic to use. If the metallicity diagnostic
        is from Kewley (2019) then ion_diagnostic may be "O3O2_K19" or 
        "S32_K19". If the metallicity diagnostic is "R23_KD04" then 
        ion_diagnostic must be "O3O2_KD04".

    compute_errors:      bool
        If True, estimate 1-sigma errors on log(O/H) + 12 and log(U) using a 
        Monte Carlo approach, in which the 1-sigma uncertainties on the 
        emission line fluxes are used generate a distribution in log(O/H) + 12 
        values, the mean and standard deviation of which are used to 
        evaluate the metallicity and corresponding uncertainty.

    niters:              int
        Number of MC iterations. 1000 is recommended.

    s:                   str 
        Column suffix to trim before carrying out computation - e.g. if 
        you want to compute metallicities using the "total" fluxes, and the 
        columns of the DataFrame look like 

            "HALPHA (total)", "HALPHA error (total)", etc.,

        then setting s=" (total)" will mean that this function "sees"

            "HALPHA", "HALPHA error".

        Useful for running this function on different emission line 
        components. The suffix is added back to the columns (and appended
        to any new columns that are added) before being returned. For 
        example, using the above example, the new added columns will be 

            "log(O/H) + 12 (N2O2_K19) (total)", 
            "log(O/H) + 12 (N2O2_K19) error (total)"


    OUTPUTS 
    ---------------------------------------------------------------------------

    The input DataFrame with additional columns added containing the computed 
    metallicity (plus ionisation parameter) and associated errors if 
    compute_errors is set to True.

    """
    logger.debug(f"computing metallicities for suffix {s} ({met_diagnostic}, logU={logU}, compute_logU={compute_logU}, ion_diagnostic={ion_diagnostic}, compute_errors={compute_errors}, niters={niters})...")
    t = time()

    # Input checks
    assert met_diagnostic in line_list_dict,\
        f"Metallicity diagnostic {met_diagnostic} is not valid!"
    for line in line_list_dict[met_diagnostic]:
        assert f"{line}{s}" in df,\
            f"Metallicity diagnostic {met_diagnostic} requires {line} which was not found in the DataFrame!"

    # Check that logU is specified if the diagnostic requires it 
    if met_diagnostic.endswith("K19") or met_diagnostic == "R23_KK04":
        assert (logU is not None) or compute_logU,\
            f"Metallicity diagnostic {met_diagnostic} requires either logU to specified OR compute_logU = True!"

    # Check that the ionisation parameter diagnostic is specified & that all 
    # required lines are present 
    if compute_logU:
        if met_diagnostic == "R23_KK04":
            assert ion_diagnostic == "O3O2_KK04", "If R23_KK04 is the chosen metallicity diagnostic, then the ionisation parameter diagnostic must be O3O2_KK04!"
        elif met_diagnostic.endswith("K19"):
            assert ion_diagnostic.endswith("K19"), "If the the chosen metallicity diagnostic is from Kewley (2019), then the ionisation parameter diagnostic must also be from Kewley (2019)!"
        for line in line_list_dict[ion_diagnostic]:
            assert f"{line}{s}" in df,\
                f"ionisation parameter diagnostic {ion_diagnostic} requires {line} which was not found in the DataFrame!"

    # Remove suffixes on columns
    df, suffix_cols, suffix_removed_cols, old_cols = remove_col_suffix(df, s)
    
    # Calculate the metallicity
    if compute_errors:
        logger.debug(f"computing metallicities with errors...")
        df = _get_metallicity(met_diagnostic=met_diagnostic, df=df, logU=logU, compute_logU=compute_logU, ion_diagnostic=ion_diagnostic, niters=niters)
    else:
        logger.debug(f"computing metallicities without errors...")
        df = _get_metallicity(met_diagnostic=met_diagnostic, df=df, logU=logU, compute_logU=compute_logU, ion_diagnostic=ion_diagnostic, niters=1)

    # Rename columns
    df = add_col_suffix(df, s, suffix_cols, suffix_removed_cols, old_cols)

    logger.debug(f"done! Total time = {int(np.floor((time() - t) / 3600)):d}:{int(np.floor((time() - t) / 60)):d}:{np.mod(time() - t, 60):02.5f}")
    return df 
