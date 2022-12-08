"""
File:       dqcut.py
Author:     Henry Zovaro
Email:      henry.zovaro@anu.edu.au

DESCRIPTION
------------------------------------------------------------------------------
This script contains utility functions useful for manipulating the DataFrames
created using make_df_sami.py.

The following functions are included:

get_wavelength_from_velocity()
    Convenience function for computing a Doppler-shifted wavelength given a 
    velocity and a rest-frame wavelength.

set_flags()
    A function for flagging cells affected by data quality & S/N cuts in a 
    given DataFrame.

apply_flags()
    Applies cuts to the flagged cells determined in set_flags().

compute_log_columns()
    Compute log quantities + errors for Halpha EW, sigma_gas, SFRs and [SII]
    ratios (for computing electron densities).

compute_gas_stellar_offsets()
    Compute the kinematic offsets between gas and stellar velocities and 
    velocity dispersions.

compute_component_offsets()
    Compute offsets in various quantities between successive kinematic 
    components - e.g. the difference in EW between components 1 and 2.

compute_extra_columns()
    Calls compute_log_columns(), compute_gas_stellar_offsets() and 
    compute_component_offsets(), plus computes emission line FWHMs and HALPHA
    emission line and continuum luminosities.

------------------------------------------------------------------------------
Copyright (C) 2022 Henry Zovaro
"""
###############################################################################
import numpy as np
from scipy import constants
from IPython.core.debugger import Tracer

from spaxelsleuth.utils.extcorr import eline_lambdas_A

###############################################################################
def get_wavelength_from_velocity(lambda_rest, v, units):
    """
    Convenience function for computing a Doppler-shifted wavelength given a 
    velocity and a rest-frame wavelength - used for converting SAMI fluxes 
    into amplitudes
    """
    assert units == 'm/s' or units == 'km/s', "units must be m/s or km/s!"
    if units == 'm/s':
        v_m_s = v
    elif units == 'km/s':
        v_m_s = v * 1e3
    lambda_obs = lambda_rest * np.sqrt((1 + v_m_s / constants.c) /
                                       (1 - v_m_s / constants.c))
    return lambda_obs

###############################################################################
def set_flags(df, eline_SNR_min, eline_list,
              sigma_gas_SNR_min=3, sigma_inst_kms=29.6):
    """
    A function for making data quality & S/N cuts on rows of a given DataFrame.

    This function can be used to determine whether certain cells pass or fail 
    a number of data quality and S/N criteria. 

    """
    def isdup(df):
        if any(df.columns.duplicated()):
            print("The following columns are duplicated:")
            for col in [df.columns[df.columns.duplicated()]]:
                print(col)
        else:
            print("No columns are duplicated")
        return

    ######################################################################
    # INITIALISE FLAGS: these will get set below.
    ###################################################################### 
    for nn in range(3):
        if f"v_gas (component {nn + 1})" in df:
            df[f"Beam smearing flag (component {nn + 1})"] = False
        if f"sigma_gas (component {nn + 1})" in df:
            df[f"Low sigma_gas S/N flag (component {nn + 1})"] = False
    df[f"Bad stellar kinematics"] = False

    for eline in eline_list:
        for nn in range(3):
            if f"{eline} (component {nn + 1})" in df:
                df[f"Low flux S/N flag - {eline} (component {nn + 1})"] = False
                df[f"Low flux fraction flag - {eline} (component {nn + 1})"] = False
                df[f"Low amplitude flag - {eline} (component {nn + 1})"] = False
                df[f"Missing flux flag - {eline} (component {nn + 1})"] = False
        if f"{eline} (total)" in df:
            df[f"Low flux S/N flag - {eline} (total)"] = False
            df[f"Low flux fraction flag - {eline} (total)"] = False
            df[f"Low amplitude flag - {eline} (total)"] = False
            df[f"Missing flux flag - {eline} (total)"] = False
    df["Missing components flag"] = False

    ######################################################################
    # Flag line fluxes that don't meet the emission line S/N requirement
    """
    If a single component doesn't meet the S/N requirement, Flag:
        - emission line fluxes (+ errors) for that component 
        - EWs (+ errors) for that component, if they exist 
        - if the emission line is HALPHA, also NaN out 
            - sigma_gas, v_gas, v_grad, SFR
    """
    ######################################################################
    print("////////////////////////////////////////////////////////////////////")
    print("In dqcut.set_flags(): Flagging low S/N components and spaxels...")
    for eline in eline_list:
        # Fluxes in individual components
        for nn in range(3):
            if f"{eline} (component {nn + 1})" in df.columns:
                cond_low_SN = df[f"{eline} S/N (component {nn + 1})"] < eline_SNR_min
                df.loc[cond_low_SN, f"Low flux S/N flag - {eline} (component {nn + 1})"] = True

        # Total fluxes
        if f"{eline} (total)" in df.columns:
            cond_low_SN = df[f"{eline} S/N (total)"] < eline_SNR_min
            df.loc[cond_low_SN, f"Low flux S/N flag - {eline} (total)"] = True

    ######################################################################
    # Flag emission lines with "missing" (i.e. NaN) fluxes in which the 
    # ERROR on the flux is not NaN
    ######################################################################
    print("In dqcut.set_flags(): Flagging components and spaxels with NaN fluxes and finite errors...")
    for eline in eline_list:
        # Fluxes in individual components
        for nn in range(3):
            if f"{eline} (component {nn + 1})" in df.columns:
                cond_missing_flux = df[f"{eline} (component {nn + 1})"].isna() & ~df[f"{eline} error (component {nn + 1})"].isna()
                df.loc[cond_missing_flux, f"Missing flux flag - {eline} (component {nn + 1})"] = True
                print(f"{eline} (component {nn + 1}): {df[cond_missing_flux].shape[0]:d} spaxels have missing fluxes in this component")

        # Total fluxes
        if f"{eline} (total)" in df.columns:
            cond_missing_flux = df[f"{eline} (total)"].isna() & ~df[f"{eline} error (total)"].isna()
            df.loc[cond_missing_flux, f"Missing flux flag - {eline} (total)"] = True
            print(f"{eline} (total): {df[cond_missing_flux].shape[0]:d} spaxels have missing total fluxes")

    ######################################################################
    # Flag rows where any component doesn't meet the amplitude 
    # requirement imposed by Avery+2021
    """
    For SAMI, we can only really calculate this for HALPHA, as we don't 
    have individual fluxes for the other lines, meaning we can't compute 
    the amplitudes of individual components. 

    Therefore, in the case where HALPHA is marked as a low S/N component, 
    then we assume that ALL measurements associated with this 
    component are garbage, since HALPHA is generally the strongest line. 
    Otherwise, we just NaN out the emission line fluxes.
    """
    ######################################################################
    # Compute the amplitude corresponding to each component
    print("In dqcut.set_flags(): Flagging components with amplitude < 3 * rms continuum noise...")
    for eline in eline_list:
        lambda_rest_A = eline_lambdas_A[eline]
        for nn in range(3):
            if f"{eline} (component {nn + 1})" in df.columns:
                # Compute the amplitude of the line
                lambda_obs_A = get_wavelength_from_velocity(lambda_rest=lambda_rest_A, 
                                                            v=df[f"v_gas (component {nn + 1})"], 
                                                            units='km/s')
                df[f"{eline} lambda_obs (component {nn + 1}) (Å)"] = lambda_obs_A
                df[f"{eline} sigma_gas (component {nn + 1}) (Å)"] = lambda_obs_A * df[f"sigma_gas (component {nn + 1})"] * 1e3 / constants.c
                df[f"{eline} A (component {nn + 1})"] = df[f"HALPHA (component {nn + 1})"] / df[f"{eline} sigma_gas (component {nn + 1}) (Å)"] / np.sqrt(2 * np.pi)
            
                # Flag bad components
                cond_bad_gasamp = df[f"{eline} A (component {nn + 1})"] < 3 * df["HALPHA continuum std. dev."]
                df.loc[cond_bad_gasamp, f"Low amplitude flag - {eline} (component {nn + 1})"] = True

        # If all components in a given spaxel have low s/n, then discard total values as well.
        if f"{eline} (component 1)" in df:
            cond_all_bad_components = (df["Number of components (original)"] == 1) &\
                                       df[f"Low amplitude flag - {eline} (component 1)"]
            if f"{eline} (component 2)" in df:
                cond_all_bad_components |= (df["Number of components (original)"] == 2) &\
                                        df[f"Low amplitude flag - {eline} (component 1)"] &\
                                        df[f"Low amplitude flag - {eline} (component 2)"]
            if f"{eline} (component 3)" in df:
                cond_all_bad_components |= (df["Number of components (original)"] == 3) &\
                                        df[f"Low amplitude flag - {eline} (component 1)"] &\
                                        df[f"Low amplitude flag - {eline} (component 2)"] &\
                                        df[f"Low amplitude flag - {eline} (component 3)"]
            df.loc[cond_all_bad_components, f"Low amplitude flag - {eline} (total)"] = True

        # if all([col in df for col in [f"{eline} (component 1)", f"{eline} (component 2)", f"{eline} (component3)"]]):
        #     cond_all_bad_components  = (df["Number of components (original)"] == 1) &\
        #                                 df[f"Low amplitude flag - {eline} (component 1)"]
        #     cond_all_bad_components |= (df["Number of components (original)"] == 2) &\
        #                                 df[f"Low amplitude flag - {eline} (component 1)"] &\
        #                                 df[f"Low amplitude flag - {eline} (component 2)"]
        #     cond_all_bad_components |= (df["Number of components (original)"] == 3) &\
        #                                 df[f"Low amplitude flag - {eline} (component 1)"] &\
        #                                 df[f"Low amplitude flag - {eline} (component 2)"] &\
        #                                 df[f"Low amplitude flag - {eline} (component 3)"]
        # elif all([col in df for col in [f"{eline} (component 1)", f"{eline} (component 2)"]]):
        #     cond_all_bad_components  = (df["Number of components (original)"] == 1) &\
        #                                 df[f"Low amplitude flag - {eline} (component 1)"]
        #     cond_all_bad_components |= (df["Number of components (original)"] == 2) &\
        #                                 df[f"Low amplitude flag - {eline} (component 1)"] &\
        #                                 df[f"Low amplitude flag - {eline} (component 2)"]
        # elif f"{eline} (component 1)" in df:
        #     cond_all_bad_components = (df["Number of components (original)"] == 1) &\
        #                                df[f"Low amplitude flag - {eline} (component 1)"]
            
        # df.loc[cond_all_bad_components, f"Low amplitude flag - {eline} (total)"] = True

    ######################################################################
    # Flag rows where the flux ratio of the broad:narrow component < 0.05 (using the method of Avery+2021)
    ######################################################################
    print("In dqcut.set_flags(): Flagging spaxels where the flux ratio of the broad:narrow component < 0.05...")
    for nn in range(1, 3):
        for eline in eline_list:
            if f"{eline} (component {nn + 1})" in df:
                cond_low_flux = df[f"{eline} A (component {nn + 1})"] < 0.05 * df[f"{eline} A (component 1)"]
                df.loc[cond_low_flux, f"Low flux fraction flag - {eline} (component {nn + 1})"] = True

    ######################################################################
    # Flag rows that don't meet the beam smearing requirement
    ######################################################################
    print("In dqcut.set_flags(): Flagging components likely to be affected by beam smearing...")
    # Gas kinematics: beam semaring criteria of Federrath+2017 and Zhou+2017.
    for nn in range(3):
        if f"v_grad (component {nn + 1})" in df and f"sigma_gas (component {nn + 1})" in df:
            cond_beam_smearing = df[f"sigma_gas (component {nn + 1})"] < 2 * df[f"v_grad (component {nn + 1})"]
            df.loc[cond_beam_smearing, f"Beam smearing flag (component {nn + 1})"] = True

    ######################################################################
    # Flag rows with insufficient S/N in sigma_gas
    ######################################################################
    print("In dqcut.set_flags(): Flagging components with low sigma_gas S/N...")
    # Gas kinematics: NaN out cells w/ sigma_gas S/N ratio < sigma_gas_SNR_min 
    # (For SAMI, the red arm resolution is 29.6 km/s - see p6 of Croom+2021)
    for nn in range(3):
        if f"sigma_gas (component {nn + 1})" in df:
            # 1. Define sigma_obs = sqrt(sigma_gas**2 + sigma_inst_kms**2).
            df[f"sigma_obs (component {nn + 1})"] = np.sqrt(df[f"sigma_gas (component {nn + 1})"]**2 + sigma_inst_kms**2)

            # 2. Define the S/N ratio of sigma_obs.
            # NOTE: here we assume that sigma_gas error (as output by LZIFU) 
            # really refers to the error on sigma_obs.
            if f"sigma_gas error (component {nn + 1})" in df:
                df[f"sigma_obs S/N (component {nn + 1})"] = df[f"sigma_obs (component {nn + 1})"] / df[f"sigma_gas error (component {nn + 1})"]

                # 3. Given our target SNR_gas, compute the target SNR_obs,
                # using the method in section 2.2.2 of Zhou+2017.
                df[f"sigma_obs target S/N (component {nn + 1})"] = sigma_gas_SNR_min * (1 + sigma_inst_kms**2 / df[f"sigma_gas (component {nn + 1})"]**2)
                cond_bad_sigma = df[f"sigma_obs S/N (component {nn + 1})"] < df[f"sigma_obs target S/N (component {nn + 1})"]
                df.loc[cond_bad_sigma, f"Low sigma_gas S/N flag (component {nn + 1})"] = True

    ######################################################################
    # Stellar kinematics DQ cut
    ######################################################################
    print("In dqcut.set_flags(): Flagging spaxels with unreliable stellar kinematics...")
    # Stellar kinematics: NaN out cells that don't meet the criteria given  
    # on page 18 of Croom+2021
    if all([c in df.columns for c in ["sigma_*", "v_*"]]):
        cond_bad_stekin = df["sigma_*"] <= 35
        cond_bad_stekin |= df["v_* error"] >= 30
        cond_bad_stekin |= df["sigma_* error"] >= df["sigma_*"] * 0.1 + 25
        cond_bad_stekin |= df["v_*"].isna()
        cond_bad_stekin |= df["v_* error"].isna()
        cond_bad_stekin |= df["sigma_*"].isna()
        cond_bad_stekin |= df["sigma_* error"].isna()
        df.loc[cond_bad_stekin, "Bad stellar kinematics"] = True

    return df
     
######################################################################
# Function for NaNing out cells based on the flags applied in dqcut()
def apply_flags(df,
                eline_list,
                line_flux_SNR_cut, 
                missing_fluxes_cut,
                line_amplitude_SNR_cut,
                flux_fraction_cut,
                sigma_gas_SNR_cut,
                vgrad_cut,
                stekin_cut):
    """
    Apply the data quality & S/N cuts on cells of a given DataFrame based on
    the flags that were defined in set_flags().

    The following flags control whether affected cells are masked or not (i.e., 
    set to NaN). 
    --------------------------------------------------------------------------

    line_flux_SNR_cut:      bool
        Whether to NaN emission line components AND total fluxes 
        (corresponding to emission lines in eline_list) below a specified S/N 
        threshold, given by eline_SNR_min. The S/N is simply the flux dividied 
        by the formal 1sigma uncertainty on the flux. 

    missing_fluxes_cut:     bool
        Whether to NaN out "missing" fluxes - i.e., cells in which the flux
        of an emission line (total or per component) is NaN, but the error 
        is not for some reason.    

    line_amplitude_SNR_cut: bool     
        Whether to NaN emission line components based on the amplitude of the
        Gaussian fit. If the amplitude A < 3 * the rms continuum noise 
        (measured in the vicinity of Halpha) then the flag is set to True.

    flux_fraction_cut:      bool
        Whether to NaN emission line components in which the amplitude is less
        than 0.05 * the amplitude of the narrowest component. Only applies to
        components > 1.         

    sigma_gas_SNR_cut:      bool 
        Whether to NaN gas velocity dispersions with a S/N < sigma_gas_SNR_min. 
        This follows the method of Zhou+2017.

    vgrad_cut:              bool
        Whether to NaN gas velocities and velocity dispersions in which 
            sigma_gas < 2 * v_grad 
        as per Federrath+2017 and Zhou+2017.

    stekin_cut:             bool
        Whether to NaN stellar kinematic quantities that do not meet the DQ 
        requirements listed in Croom+2021.    

    """
    print("////////////////////////////////////////////////////////////////////")
    if line_flux_SNR_cut:
        print("In dqcut.apply_flags(): Masking components that don't meet the S/N requirements...")
        for eline in eline_list:
            # Individual fluxes
            for nn in range(3):
                if f"{eline} (component {nn + 1})" in df:
                    cond_low_SN = df[f"Low flux S/N flag - {eline} (component {nn + 1})"]

                    # Cells to NaN
                    if eline == "HALPHA":
                        # Then NaN out EVERYTHING associated with this component - if we can't trust HALPHA then we probably can't trust anything else either!
                        cols_low_SN = [c for c in df.columns if f"(component {nn + 1})" in c and "flag" not in c]
                    else:
                        cols_low_SN = [c for c in df.columns if eline in c and f"(component {nn + 1})" in c and "flag" not in c]
                    df.loc[cond_low_SN, cols_low_SN] = np.nan

            if f"{eline} (total)" in df:
                # TOTAL fluxes
                cond_low_SN = df[f"Low flux S/N flag - {eline} (total)"]

                # Cells to NaN
                if eline == "HALPHA":
                    # Then NaN out EVERYTHING associated with this component - if we can't trust HALPHA then we probably can't trust anything else either!
                    cols_low_SN = [c for c in df.columns if f"(total)" in c and "flag" not in c]
                else:
                    cols_low_SN = [c for c in df.columns if eline in c and f"(total)" in c and "flag" not in c]
                df.loc[cond_low_SN, cols_low_SN] = np.nan

    if missing_fluxes_cut:
        print("In dqcut.apply_flags(): Masking components with missing fluxes...")
        for eline in eline_list:
            # Individual fluxes
            for nn in range(3):
                if f"{eline} (component {nn + 1})" in df:
                    cond_missing_flux = df[f"Missing flux flag - {eline} (component {nn + 1})"]

                    # Cells to NaN
                    if eline == "HALPHA":
                        # Then NaN out EVERYTHING associated with this component - if we can't trust HALPHA then we probably can't trust anything else either!
                        cols_missing_fluxes = [c for c in df.columns if f"(component {nn + 1})" in c and "flag" not in c]
                    else:
                        cols_missing_fluxes = [c for c in df.columns if eline in c and f"(component {nn + 1})" in c and "flag" not in c]
                    df.loc[cond_missing_flux, cols_missing_fluxes] = np.nan

            if f"{eline} (total)" in df:
                # TOTAL fluxes
                cond_missing_flux = df[f"Missing flux flag - {eline} (total)"]

                # Cells to NaN
                if eline == "HALPHA":
                    # Then NaN out EVERYTHING associated with this component - if we can't trust HALPHA then we probably can't trust anything else either!
                    cols_missing_fluxes = [c for c in df.columns if f"(total)" in c and "flag" not in c]
                else:
                    cols_missing_fluxes = [c for c in df.columns if eline in c and f"(total)" in c and "flag" not in c]
                df.loc[cond_missing_flux, cols_missing_fluxes] = np.nan

    if line_amplitude_SNR_cut:
        print("In dqcut.apply_flags(): Masking components that don't meet the amplitude requirements...")
        for eline in eline_list:
            for nn in range(3):
                if f"{eline} (component {nn + 1})" in df:
                    cond_low_amp = df[f"Low amplitude flag - {eline} (component {nn + 1})"]

                    # Cells to NaN
                    if eline == "HALPHA":
                        # Then NaN out EVERYTHING associated with this component - if we can't trust HALPHA then we probably can't trust anything else either!
                        cols_low_amp = [c for c in df.columns if f"(component {nn + 1})" in c and "flag" not in c]
                    else:
                        cols_low_amp = [c for c in df.columns if eline in c and f"(component {nn + 1})" in c and "flag" not in c]
                    df.loc[cond_low_amp, cols_low_amp] = np.nan

                # If ALL components in the spaxel fail the amplitude requirement then NaN them as well
                # Note that we only do this for HALPHA since we can't measure amplitudes for other lines in SAMI
                cond_low_amp = df[f"Low amplitude flag - {eline} (total)"]

                # Cells to NaN
                if eline == "HALPHA":
                    # Then NaN out EVERYTHING associated with this component - if we can't trust HALPHA then we probably can't trust anything else either!
                    cols_low_amp = [c for c in df.columns if f"(total)" in c and "flag" not in c]
                else:
                    cols_low_amp = [c for c in df.columns if eline in c and f"(total)" in c and "flag" not in c]
                df.loc[cond_low_amp, cols_low_amp] = np.nan

    if flux_fraction_cut:
        print("In dqcut.apply_flags(): Masking components 1, 2 where the flux ratio of this component:component 1 < 0.05...")
        for nn in range(1, 3):
            for eline in eline_list:
                if f"{eline} (component {nn + 1})" in df:
                    cond_low_flux_fraction = df[f"Low flux fraction flag - {eline} (component {nn + 1})"]

                    # Cells to NaN
                    if eline == "HALPHA":
                        # Then NaN out EVERYTHING associated with this component - if we can't trust HALPHA then we probably can't trust anything else either!
                        cols_low_flux_fraction += [c for c in df.columns if f"(component {nn + 1})" in c and "flag" not in c]
                    else:
                        cols_low_flux_fraction = [c for c in df.columns if eline in c and f"(component {nn + 1})" in c and "flag" not in c]
                    df.loc[cond_low_flux_fraction, cols_low_flux_fraction] = np.nan

    if vgrad_cut:
        print("In dqcut.apply_flags(): Masking components that don't meet the beam smearing requirement...")
        for nn in range(3):
            if f"v_grad (component {nn + 1})" in df:
                cond_beam_smearing = df[f"Beam smearing flag (component {nn + 1})"]

                # Cells to NaN
                cols_beam_smearing = [c for c in df.columns if f"component {nn + 1}" in c and ("v_gas" in c or "sigma_gas" in c) and "flag" not in c]
                cols_beam_smearing += [c for c in df.columns if "delta" in c and str(nn + 1) in c]
                df.loc[cond_beam_smearing, cols_beam_smearing] = np.nan

    if sigma_gas_SNR_cut:
        print("In dqcut.apply_flags(): Masking components with insufficient S/N in sigma_gas...")
        for nn in range(3):
            if f"sigma_obs S/N (component {nn + 1})" in df:
                cond_bad_sigma = df[f"Low sigma_gas S/N flag (component {nn + 1})"]
                
                # Cells to NaN
                cols_sigma_gas_SNR_cut = [c for c in df.columns if f"component {nn + 1}" in c and "sigma_gas" in c and "flag" not in c]
                cols_sigma_gas_SNR_cut += [c for c in df.columns if "delta" in c and str(nn + 1) in c]
                df.loc[cond_bad_sigma, cols_sigma_gas_SNR_cut] = np.nan

    if stekin_cut:
        print("In dqcut.apply_flags(): Masking spaxels with unreliable stellar kinematics...")
        if "v_*" in df and "sigma_*" in df:
            cond_bad_stekin = df["Bad stellar kinematics"]

            # Cells to NaN
            cols_stekin_cut = [c for c in df.columns if "v_*" in c or "sigma_*" in c]
            df.loc[cond_bad_stekin, cols_stekin_cut] = np.nan

    ######################################################################
    # Identify which spaxels have "missing components"
    ######################################################################
    print("////////////////////////////////////////////////////////////////////")
    print("In dqcut.apply_flags(): Flagging spaxels with 'missing components'...")
    """
    We define a "missing component" as one in which both the HALPHA flux and 
    velocity dispersion have been NaN'd for any reason, but we do NOT NaN 
    anything out based on this criterion, so that the user can NaN them out 
    if they need to based on their specific use case. 
    
    For example, a 2-component spaxel in which the 2nd component is a low-S/N
    component may still have high S/N *total* fluxes, in which case things 
    like e.g. line ratios for the *total* flux are most likely still reliable. 
    
    NOTE: for datasets other than SAMI, this criterion may pose a problem
    if e.g. there is a spaxel in which, say, [NII] is not NaN in component 1,
    but HALPHA is. 

    NOTE 2: The "Number of components" column really records the number of 
    *reliable* components ONLY IF they are in the right order, for lack of a 
    better phrase - take, for example, a spaxel that originally has 2 components.
    Say that component 1 (the narrowest component) has a low S/N in the gas 
    velocity dispersion, so sigma_gas (component 2) is masked out, but 
    HALPHA (component 2) is not, and that component 2 (the broad component)
    passes all DQ and S/N cuts. In this case, the spaxel will NOT be recorded
    as having "Number of components" = 1 or 2 because it fails cond_has_1 and
    cond_has_2 below. It will therefore have an undefined "Number of 
    components" and will be set to NaN. 

    Spaxels that still retain their original number of components after making 
    all DQ and S/N cuts can be selected as follows:

        df_good_quality_components = df[~df["Missing components flag"]]

    """
    if all([col in df for col in [f"HALPHA (component 1)", f"HALPHA (component 2)", f"HALPHA (component 3)",
                                  f"sigma_gas (component 1)", f"sigma_gas (component 2)", f"sigma_gas (component 3)"]]):
        cond_has_3 = ~np.isnan(df["HALPHA (component 1)"]) & ~np.isnan(df["sigma_gas (component 1)"]) &\
                     ~np.isnan(df["HALPHA (component 2)"]) & ~np.isnan(df["sigma_gas (component 2)"]) &\
                     ~np.isnan(df["HALPHA (component 3)"]) & ~np.isnan(df["sigma_gas (component 3)"]) 
        cond_has_2 = ~np.isnan(df["HALPHA (component 1)"]) & ~np.isnan(df["sigma_gas (component 1)"]) &\
                     ~np.isnan(df["HALPHA (component 2)"]) & ~np.isnan(df["sigma_gas (component 2)"]) &\
                      (np.isnan(df["HALPHA (component 3)"]) | np.isnan(df["sigma_gas (component 3)"]))  
        cond_has_1 = ~np.isnan(df["HALPHA (component 1)"]) & ~np.isnan(df["sigma_gas (component 1)"]) &\
                      (np.isnan(df["HALPHA (component 2)"]) | np.isnan(df["sigma_gas (component 2)"])) &\
                      (np.isnan(df["HALPHA (component 3)"]) | np.isnan(df["sigma_gas (component 3)"]))
        cond_has_0 =  (np.isnan(df["HALPHA (component 1)"]) | np.isnan(df["sigma_gas (component 1)"])) &\
                      (np.isnan(df["HALPHA (component 2)"]) | np.isnan(df["sigma_gas (component 2)"])) &\
                      (np.isnan(df["HALPHA (component 3)"]) | np.isnan(df["sigma_gas (component 3)"]))

        cond_still_has_1 = cond_has_1 & (df["Number of components (original)"] == 1)
        cond_still_has_2 = cond_has_2 & (df["Number of components (original)"] == 2)
        cond_still_has_3 = cond_has_3 & (df["Number of components (original)"] == 3)
        cond_still_has_0 = cond_has_0 & (df["Number of components (original)"] == 0)

        # TRUE if the number of components after making DQ cuts matches the "original" number of components
        cond_has_original = cond_still_has_1 | cond_still_has_2 | cond_still_has_3 | cond_still_has_0
        df.loc[~cond_has_original, "Missing components flag"] = True

        # Reset the number of components
        df.loc[cond_has_1, "Number of components"] = 1
        df.loc[cond_has_2, "Number of components"] = 2
        df.loc[cond_has_3, "Number of components"] = 3
        df.loc[cond_has_0, "Number of components"] = 0

    elif all([col in df for col in [f"HALPHA (component 1)", f"HALPHA (component 2)",
                                  f"sigma_gas (component 1)", f"sigma_gas (component 2)"]]):
        cond_has_2 = ~np.isnan(df["HALPHA (component 1)"]) & ~np.isnan(df["sigma_gas (component 1)"]) &\
                     ~np.isnan(df["HALPHA (component 2)"]) & ~np.isnan(df["sigma_gas (component 2)"])
        cond_has_1 = ~np.isnan(df["HALPHA (component 1)"]) & ~np.isnan(df["sigma_gas (component 1)"]) &\
                      (np.isnan(df["HALPHA (component 2)"]) | np.isnan(df["sigma_gas (component 2)"]))
        cond_has_0 =  (np.isnan(df["HALPHA (component 1)"]) | np.isnan(df["sigma_gas (component 1)"])) &\
                      (np.isnan(df["HALPHA (component 2)"]) | np.isnan(df["sigma_gas (component 2)"]))

        cond_still_has_1 = cond_has_1 & (df["Number of components (original)"] == 1)
        cond_still_has_2 = cond_has_2 & (df["Number of components (original)"] == 2)
        cond_still_has_0 = cond_has_0 & (df["Number of components (original)"] == 0)

        # TRUE if the number of components after making DQ cuts matches the "original" number of components
        cond_has_original = cond_still_has_1 | cond_still_has_2 | cond_still_has_0
        df.loc[~cond_has_original, "Missing components flag"] = True

        # Reset the number of components
        df.loc[cond_has_1, "Number of components"] = 1
        df.loc[cond_has_2, "Number of components"] = 2
        df.loc[cond_has_0, "Number of components"] = 0

    elif all([col in df for col in [f"HALPHA (component 1)", f"sigma_gas (component 1)"]]):
        cond_has_1 = ~np.isnan(df["HALPHA (component 1)"]) & ~np.isnan(df["sigma_gas (component 1)"])
        cond_has_0 =  np.isnan(df["HALPHA (component 1)"]) | np.isnan(df["sigma_gas (component 1)"])
        
        cond_still_has_1 = cond_has_1 & (df["Number of components (original)"] == 1)
        cond_still_has_0 = cond_has_0 & (df["Number of components (original)"] == 0)

        # TRUE if the number of components after making DQ cuts matches the "original" number of components
        cond_has_original = cond_still_has_1 | cond_has_0
        df.loc[~cond_has_original, "Missing components flag"] = True

        # Reset the number of components
        df.loc[cond_has_1, "Number of components"] = 1
        df.loc[cond_has_0, "Number of components"] = 0

    # Count how many spaxels have 'Number of components' != 'Number of components (original)'
    cond_mismatch = df["Number of components"] != df["Number of components (original)"]
    print(f"STATISTICS: {df[cond_mismatch].shape[0] / df.shape[0] * 100.:.2f}% of spaxels have individual components that have been masked out")

    ######################################################################
    # End
    ######################################################################
    # Drop rows that have been NaNed out
    if "ID" in df.columns: 
        df = df.dropna(subset=["ID"])

        # Cast ID column to int
        if all([type(c) == float for c in df["ID"]]):
            df = df.copy()  # Required to suppress SettingValueWithCopy warning
            df["ID"] = df["ID"].astype(int)

    return df

######################################################################
# Compute log quantities + errors for Halpha EW, sigma_gas and SFRs
def compute_log_columns(df):

    # Halpha flux and EW for individual components
    for col in ["HALPHA luminosity", "HALPHA continuum", "HALPHA EW", "sigma_gas", "S2 ratio"]:
        for s in ["(total)"] + [f"(component {nn})" for nn in [1, 2, 3]]:
            # Compute log quantities for total 
            if f"{col} {s}" in df:
                df[f"log {col} {s}"] = np.log10(df[f"{col} {s}"])
            if f"{col} error {s}" in df:
                df[f"log {col} error (lower) {s}"] = df[f"log {col} {s}"] - np.log10(df[f"{col} {s}"] - df[f"{col} error {s}"])
                df[f"log {col} error (upper) {s}"] = np.log10(df[f"{col} {s}"] + df[f"{col} error {s}"]) -  df[f"log {col} {s}"]

    # Compute log quantities for total SFR
    for col in ["SFR", "SFR surface density", "sSFR"]:
        for s in ["(total)"] + [f"(component {nn})" for nn in [1, 2, 3]]:
            if f"{col} {s}" in df:
                cond = ~np.isnan(df[f"{col} {s}"])
                cond &= df[f"{col} {s}"] > 0
                df.loc[cond, f"log {col} {s}"] = np.log10(df.loc[cond, f"{col} {s}"])
                if f"{col} error {s}" in df:
                    df.loc[cond, f"log {col} error (lower) {s}"] = df.loc[cond, f"log {col} {s}"] - np.log10(df.loc[cond, f"{col} {s}"] - df.loc[cond, f"{col} error {s}"])
                    df.loc[cond, f"log {col} error (upper) {s}"] = np.log10(df.loc[cond, f"{col} {s}"] + df.loc[cond, f"{col} error {s}"]) -  df.loc[cond, f"log {col} {s}"]
                
        # if f"SFR surface density {s}" in df:
        #     cond = ~np.isnan(df[f"SFR surface density {s}"])
        #     cond &= df[f"SFR surface density {s}"] > 0
        #     # Compute log quantities for total SFR surface density
        #     df.loc[cond, f"log SFR surface density {s}"] = np.log10(df.loc[cond, f"SFR surface density {s}"])
        #     if f"SFR surface density error {s}" in df:
        #         df.loc[cond, f"log SFR surface density error (lower) {s}"] = df.loc[cond, f"log SFR surface density {s}"] - np.log10(df.loc[cond, f"SFR surface density {s}"] - df.loc[cond, f"SFR surface density error {s}"])
        #         df.loc[cond, f"log SFR surface density error (upper) {s}"] = np.log10(df.loc[cond, f"SFR surface density {s}"] + df.loc[cond, f"SFR surface density error {s}"]) -  df.loc[cond, f"log SFR surface density {s}"]

        # if f"sSFR {s}" in df:
        #     cond = ~np.isnan(df[f"sSFR {s}"])
        #     cond &= df[f"sSFR {s}"] > 0
        #     # Compute log quantities for total sSFR
        #     df.loc[cond, f"log sSFR {s}"] = np.log10(df.loc[cond, f"sSFR {s}"])
        #     if f"sSFR error {s}" in df:
        #         df.loc[cond, f"log sSFR error (lower) {s}"] = df.loc[cond, f"log sSFR {s}"] - np.log10(df.loc[cond, f"sSFR {s}"] - df.loc[cond, f"sSFR error {s}"])
        #         df.loc[cond, f"log sSFR error (upper) {s}"] = np.log10(df.loc[cond, f"sSFR {s}"] + df.loc[cond, f"sSFR error {s}"]) -  df.loc[cond, f"log sSFR {s}"]

    return df

######################################################################
# Compute offsets between gas & stellar kinematics
def compute_gas_stellar_offsets(df):    
    if "v_*" in df and "sigma_*" in df:
        for nn in range(3):
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
def compute_component_offsets(df):
    
    for nn_1, nn_2 in ([2, 1], [3, 2], [3, 1]):

        #//////////////////////////////////////////////////////////////////////
        # Difference between gas velocity dispersion between components
        if all([col in df for col in [f"sigma_gas (component {nn_1})", f"sigma_gas (component {nn_2})"]]):
            df[f"delta sigma_gas ({nn_2}/{nn_1})"] = df[f"sigma_gas (component {nn_2})"] - df[f"sigma_gas (component {nn_1})"]
        
        # Error in the difference between gas velocity dispersion between components   
        if all([col in df for col in ["sigma_gas error (component 1)", "sigma_gas error (component 2)"]]):
            df[f"delta sigma_gas error ({nn_2}/{nn_1})"] = np.sqrt(df[f"sigma_gas error (component {nn_2})"]**2 +\
                                                                   df[f"sigma_gas error (component {nn_1})"]**2)

        #//////////////////////////////////////////////////////////////////////
        # DIfference between gas velocity between components (2/1)
        if all([col in df for col in [f"v_gas (component {nn_1})", f"v_gas (component {nn_2})"]]):     
            df[f"delta v_gas ({nn_2}/{nn_1})"] = df[f"v_gas (component {nn_2})"] - df[f"v_gas (component {nn_1})"]
        if all([col in df for col in [f"v_gas error (component {nn_2})", f"v_gas error (component {nn_1})"]]):  
            df[f"delta v_gas error ({nn_2}/{nn_1})"] = np.sqrt(df[f"v_gas error (component {nn_2})"]**2 +\
                                                               df[f"v_gas error (component {nn_1})"]**2)
        
        #//////////////////////////////////////////////////////////////////////
        # Ratio of HALPHA EWs between components   
        if all([col in df for col in [f"HALPHA EW (component {nn_1})", f"HALPHA EW (component {nn_2})"]]):     
            df[f"HALPHA EW ratio ({nn_2}/{nn_1})"] = df[f"HALPHA EW (component {nn_2})"] / df[f"HALPHA EW (component {nn_1})"]
        if all([col in df for col in [f"HALPHA EW error (component {nn_1})", f"HALPHA EW error (component {nn_2})"]]):     
            df[f"HALPHA EW ratio error ({nn_2}/{nn_1})"] = df[f"HALPHA EW ratio ({nn_2}/{nn_1})"] *\
                np.sqrt((df[f"HALPHA EW error (component {nn_2})"] / df[f"HALPHA EW (component {nn_2})"])**2 +\
                        (df[f"HALPHA EW error (component {nn_1})"] / df[f"HALPHA EW (component {nn_1})"])**2)

        #//////////////////////////////////////////////////////////////////////
        # Ratio of HALPHA EWs between components (log)
        if all([col in df for col in [f"log HALPHA EW (component {nn_2})", f"log HALPHA EW (component {nn_1})"]]):     
            df[f"Delta HALPHA EW ({nn_2}/{nn_1})"] = df[f"log HALPHA EW (component {nn_2})"] - df[f"log HALPHA EW (component {nn_1})"]

        #//////////////////////////////////////////////////////////////////////
        # Forbidden line ratios:
        for col in ["log O3", "log N2", "log S2", "log O1"]:
            if f"{col} (component {nn_1})" in df and f"{col} (component {nn_2})" in df:
                df[f"delta {col} ({nn_2}/{nn_1})"] = df[f"{col} (component {nn_2})"] - df[f"{col} (component {nn_1})"]
                df[f"delta {col} ({nn_2}/{nn_1}) error"] = np.sqrt(df[f"{col} error (component {nn_2})"]**2 + df[f"{col} error (component {nn_1})"]**2)

    #//////////////////////////////////////////////////////////////////////
    # Fractional of total Halpha EW in each component
    for nn in range(3):
        if all([col in df.columns for col in [f"HALPHA EW (component {nn + 1})", f"HALPHA EW (total)"]]):
            df[f"HALPHA EW/HALPHA EW (total) (component {nn + 1})"] = df[f"HALPHA EW (component {nn + 1})"] / df[f"HALPHA EW (total)"]

    return df

#########################################################################
def compute_extra_columns(df):
    """
    Add the following extra columns to the DataFrame:
     - log quantities + errors:
        - Halpha EW
        - sigma_gas
        - SFR
     - offsets between gas & stellar kinematics
     - offsets in Halpha EW, sigma_gas, v_gas between adjacent components
     - fraction of Halpha EW in each component
     - FWHM of emission lines

    This function should be called AFTER calling df_dqcut because it relies 
    on quantities that may be masked out due to S/N and DQ cuts.
    Technically we could make sure these columns get masked out in that 
    function but it would be cumbersome to do so...

    """
    
    # Halpha & continuum luminosity
    # HALPHA luminosity: units of erg s^-1 kpc^-2
    # HALPHA cont. luminosity: units of erg s^-1 Å-1 kpc^-2
    if all([col in df for col in ["D_L (Mpc)", "Bin size (square kpc)"]]):
        if all([col in df for col in ["HALPHA continuum", "HALPHA continuum error"]]):
            df[f"HALPHA continuum luminosity"] = df[f"HALPHA continuum"] * 1e-16 * 4 * np.pi * (df["D_L (Mpc)"] * 1e6 * 3.086e18)**2 * 1 / df["Bin size (square kpc)"]
            df[f"HALPHA continuum luminosity error"] = df[f"HALPHA continuum error"] * 1e-16 * 4 * np.pi * (df["D_L (Mpc)"] * 1e6 * 3.086e18)**2 * 1 / df["Bin size (square kpc)"]

        if all([col in df for col in ["HALPHA (total)", "HALPHA error (total)"]]):
            df[f"HALPHA luminosity (total)"] = df[f"HALPHA (total)"] * 1e-16 * 4 * np.pi * (df["D_L (Mpc)"] * 1e6 * 3.086e18)**2 * 1 / df["Bin size (square kpc)"]
            df[f"HALPHA luminosity error (total)"] = df[f"HALPHA error (total)"] * 1e-16 * 4 * np.pi * (df["D_L (Mpc)"] * 1e6 * 3.086e18)**2 * 1 / df["Bin size (square kpc)"]
        for nn in range(3):
            if all([col in df for col in [f"HALPHA (component {nn + 1})", f"HALPHA error (component {nn + 1})"]]):
                df[f"HALPHA luminosity (component {nn + 1})"] = df[f"HALPHA (component {nn + 1})"] * 1e-16 * 4 * np.pi * (df["D_L (Mpc)"] * 1e6 * 3.086e18)**2 * 1 / df["Bin size (square kpc)"]
                df[f"HALPHA luminosity error (component {nn + 1})"] = df[f"HALPHA error (component {nn + 1})"] * 1e-16 * 4 * np.pi * (df["D_L (Mpc)"] * 1e6 * 3.086e18)**2 * 1 / df["Bin size (square kpc)"]

    # Compute FWHM
    for nn in range(3):
        if f"sigma_gas (component {nn + 1})" in df:
            df[f"FWHM_gas (component {nn + 1})"] = df[f"sigma_gas (component {nn + 1})"] * 2 * np.sqrt(2 * np.log(2))
        if f"sigma_gas error (component {nn + 1})" in df:
            df[f"FWHM_gas error (component {nn + 1})"] = df[f"sigma_gas error (component {nn + 1})"] * 2 * np.sqrt(2 * np.log(2))

    # Stellar & gas kinematic offsets
    df = compute_gas_stellar_offsets(df)

    # Compute logs
    df = compute_log_columns(df)
    
    # Comptue component offsets
    df = compute_component_offsets(df)

    return df


