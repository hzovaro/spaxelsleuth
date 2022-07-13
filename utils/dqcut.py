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

dqcut()
    A function for making data quality & S/N cuts on rows of a given DataFrame.

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
def dqcut(df, ncomponents,
              eline_SNR_min, eline_list,
              missing_fluxes_cut=True,
              line_flux_SNR_cut=True, 
              line_amplitude_SNR_cut=True,
              flux_fraction_cut=False,
              sigma_gas_SNR_cut=True, sigma_gas_SNR_min=3, sigma_inst_kms=29.6,
              vgrad_cut=False,
              stekin_cut=True):
    """
    A function for making data quality & S/N cuts on rows of a given DataFrame.

    This function can be used to determine whether certain cells pass or fail 
    a number of data quality and S/N criteria. 

    The following flags control whether affected cells are masked or not (i.e., 
    set to NaN). 
    --------------------------------------------------------------------------

    missing_fluxes_cut:     bool
        Whether to NaN out "missing" fluxes - i.e., cells in which the flux
        of an emission line (total or per component) is NaN, but the error 
        is not for some reason.
    
    line_flux_SNR_cut:      bool
        Whether to NaN emission line components AND total fluxes 
        (corresponding to emission lines in eline_list) below a specified S/N 
        threshold, given by eline_SNR_min. The S/N is simply the flux dividied 
        by the formal 1sigma uncertainty on the flux. 

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
    for nn in range(ncomponents):
        df[f"Beam smearing flag (component {nn + 1})"] = False
        df[f"Low sigma_gas S/N flag (component {nn + 1})"] = False
        df[f"Bad stellar kinematics"] = False

    for eline in eline_list:
        for nn in range(ncomponents):
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
    print("In dqcut.dqcut(): Flagging low S/N components and spaxels...")
    for eline in eline_list:
        # Fluxes in individual components
        for nn in range(ncomponents):
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
    print("In dqcut.dqcut(): Flagging components and spaxels with NaN fluxes and finite errors...")
    for eline in eline_list:
        # Fluxes in individual components
        for nn in range(ncomponents):
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
    print("In dqcut.dqcut(): Flagging components with amplitude < 3 * rms continuum noise...")
    for eline in eline_list:
        if f"{eline} (component 1)" in df:
            lambda_rest_A = eline_lambdas_A[eline]
            for nn in range(ncomponents):
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
            if ncomponents == 3:
                cond_all_bad_components  = (df["Number of components (original)"] == 1) &\
                                            df[f"Low amplitude flag - {eline} (component 1)"]
                cond_all_bad_components |= (df["Number of components (original)"] == 2) &\
                                            df[f"Low amplitude flag - {eline} (component 1)"] &\
                                            df[f"Low amplitude flag - {eline} (component 2)"]
                cond_all_bad_components |= (df["Number of components (original)"] == 3) &\
                                            df[f"Low amplitude flag - {eline} (component 1)"] &\
                                            df[f"Low amplitude flag - {eline} (component 2)"] &\
                                            df[f"Low amplitude flag - {eline} (component 3)"]
            else:
                cond_all_bad_components = (df["Number of components (original)"] == 1) &\
                                           df[f"Low amplitude flag - {eline} (component 1)"]
                
            df.loc[cond_all_bad_components, f"Low amplitude flag - {eline} (total)"] = True

    ######################################################################
    # Flag rows where the flux ratio of the broad:narrow component < 0.05 (using the method of Avery+2021)
    ######################################################################
    print("In dqcut.dqcut(): Flagging spaxels where the flux ratio of the broad:narrow component < 0.05...")
    if ncomponents > 1:
        for nn in range(1, ncomponents):
            for eline in eline_list:
                if f"{eline} (component {nn + 1})" in df:
                    cond_low_flux = df[f"{eline} A (component {nn + 1})"] < 0.05 * df[f"{eline} A (component 1)"]
                    df.loc[cond_low_flux, f"Low flux fraction flag - {eline} (component {nn + 1})"] = True

    ######################################################################
    # Flag rows that don't meet the beam smearing requirement
    ######################################################################
    print("In dqcut.dqcut(): Flagging components likely to be affected by beam smearing...")
    # Gas kinematics: beam semaring criteria of Federrath+2017 and Zhou+2017.
    for nn in range(ncomponents):
        cond_beam_smearing = df[f"sigma_gas (component {nn + 1})"] < 2 * df[f"v_grad (component {nn + 1})"]
        df.loc[cond_beam_smearing, f"Beam smearing flag (component {nn + 1})"] = True

    ######################################################################
    # Flag rows with insufficient S/N in sigma_gas
    ######################################################################
    print("In dqcut.dqcut(): Flagging components with low sigma_gas S/N...")
    # Gas kinematics: NaN out cells w/ sigma_gas S/N ratio < sigma_gas_SNR_min 
    # (For SAMI, the red arm resolution is 29.6 km/s - see p6 of Croom+2021)
    for nn in range(ncomponents):
        # 1. Define sigma_obs = sqrt(sigma_gas**2 + sigma_inst_kms**2).
        df[f"sigma_obs (component {nn + 1})"] = np.sqrt(df[f"sigma_gas (component {nn + 1})"]**2 + sigma_inst_kms**2)

        # 2. Define the S/N ratio of sigma_obs.
        # NOTE: here we assume that sigma_gas error (as output by LZIFU) 
        # really refers to the error on sigma_obs.
        df[f"sigma_obs S/N (component {nn + 1})"] = df[f"sigma_obs (component {nn + 1})"] / df[f"sigma_gas error (component {nn + 1})"]

        # 3. Given our target SNR_gas, compute the target SNR_obs,
        # using the method in section 2.2.2 of Zhou+2017.
        df[f"sigma_obs target S/N (component {nn + 1})"] = sigma_gas_SNR_min * (1 + sigma_inst_kms**2 / df[f"sigma_gas (component {nn + 1})"]**2)
        cond_bad_sigma = df[f"sigma_obs S/N (component {nn + 1})"] < df[f"sigma_obs target S/N (component {nn + 1})"]
        df.loc[cond_bad_sigma, f"Low sigma_gas S/N flag (component {nn + 1})"] = True

    ######################################################################
    # Stellar kinematics DQ cut
    ######################################################################
    print("In dqcut.dqcut(): Flagging spaxels with unreliable stellar kinematics...")
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
        
    ######################################################################
    # NaN out columns!
    ######################################################################
    print("////////////////////////////////////////////////////////////////////")
    if line_flux_SNR_cut:
        print("In dqcut.dqcut(): Masking components that don't meet the S/N requirements...")
        for eline in eline_list:
            if f"{eline} (component 1)" in df:
                # Individual fluxes
                for nn in range(ncomponents):
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
        print("In dqcut.dqcut(): Masking components with missing fluxes...")
        for eline in eline_list:
            if f"{eline} (component 1)" in df:
                # Individual fluxes
                for nn in range(ncomponents):
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
        print("In dqcut.dqcut(): Masking components that don't meet the amplitude requirements...")
        for eline in eline_list:
            if f"{eline} (component 1)" in df:
                for nn in range(ncomponents):
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
        print("In dqcut.dqcut(): Masking components 1, 2 where the flux ratio of this component:component 1 < 0.05...")
        if ncomponents > 1:
            for nn in range(1, ncomponents):
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
        print("In dqcut.dqcut(): Masking components that don't meet the beam smearing requirement...")
        for nn in range(ncomponents):
            cond_beam_smearing = df[f"Beam smearing flag (component {nn + 1})"]

            # Cells to NaN
            cols_beam_smearing = [c for c in df.columns if f"component {nn + 1}" in c and ("v_gas" in c or "sigma_gas" in c) and "flag" not in c]
            cols_beam_smearing += [c for c in df.columns if "delta" in c and str(nn + 1) in c]
            df.loc[cond_beam_smearing, cols_beam_smearing] = np.nan

    if sigma_gas_SNR_cut:
        print("In dqcut.dqcut(): Masking components with insufficient S/N in sigma_gas...")
        for nn in range(ncomponents):
            cond_bad_sigma = df[f"Low sigma_gas S/N flag (component {nn + 1})"]
            
            # Cells to NaN
            cols_sigma_gas_SNR_cut = [c for c in df.columns if f"component {nn + 1}" in c and "sigma_gas" in c and "flag" not in c]
            cols_sigma_gas_SNR_cut += [c for c in df.columns if "delta" in c and str(nn + 1) in c]
            df.loc[cond_bad_sigma, cols_sigma_gas_SNR_cut] = np.nan

    if stekin_cut:
        print("In dqcut.dqcut(): Masking spaxels with unreliable stellar kinematics...")
        cond_bad_stekin = df["Bad stellar kinematics"]

        # Cells to NaN
        cols_stekin_cut = [c for c in df.columns if "v_*" in c or "sigma_*" in c]
        df.loc[cond_bad_stekin, cols_stekin_cut] = np.nan

    ######################################################################
    # Identify which spaxels have "missing components"
    ######################################################################
    print("////////////////////////////////////////////////////////////////////")
    print("In dqcut.dqcut(): Flagging spaxels with 'missing components'...")
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
    if ncomponents == 3:
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

    elif ncomponents == 1:
        cond_has_1 = ~np.isnan(df["HALPHA (component 1)"]) & ~np.isnan(df["sigma_gas (component 1)"])
        cond_has_0 =  np.isnan(df["HALPHA (component 1)"]) | np.isnan(df["sigma_gas (component 1)"])
        
        cond_still_has_1 = cond_has_1 & (df["Number of components (original)"] == 1)
        cond_still_has_1 = cond_has_0 & (df["Number of components (original)"] == 0)

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
    if "catid" in df.columns: 
        df = df.dropna(subset=["catid"])

        # Cast catid column to int
        if all([type(c) == float for c in df["catid"]]):
            df = df.copy()  # Required to suppress SettingValueWithCopy warning
            df["catid"] = df["catid"].astype(int)

    return df

######################################################################
# Compute log quantities + errors for Halpha EW, sigma_gas and SFRs
def compute_log_columns(df, ncomponents):
    # Halpha flux and EW for individual components
    for nn in range(ncomponents):          
        # log quantities
        df[f"log HALPHA luminosity (component {nn + 1})"] = np.log10(df[f"HALPHA luminosity (component {nn + 1})"])
        df[f"log HALPHA EW (component {nn + 1})"] = np.log10(df[f"HALPHA EW (component {nn + 1})"])
        df[f"log sigma_gas (component {nn + 1})"] = np.log10(df[f"sigma_gas (component {nn + 1})"])

        # Compute errors for log quantities
        df[f"log HALPHA luminosity error (lower) (component {nn + 1})"] = df[f"log HALPHA luminosity (component {nn + 1})"] - np.log10(df[f"HALPHA luminosity (component {nn + 1})"] - df[f"HALPHA luminosity error (component {nn + 1})"])
        df[f"log HALPHA luminosity error (upper) (component {nn + 1})"] = np.log10(df[f"HALPHA luminosity (component {nn + 1})"] + df[f"HALPHA luminosity error (component {nn + 1})"]) - df[f"log HALPHA luminosity (component {nn + 1})"]

        df[f"log HALPHA EW error (lower) (component {nn + 1})"] = df[f"log HALPHA EW (component {nn + 1})"] - np.log10(df[f"HALPHA EW (component {nn + 1})"] - df[f"HALPHA EW error (component {nn + 1})"])
        df[f"log HALPHA EW error (upper) (component {nn + 1})"] = np.log10(df[f"HALPHA EW (component {nn + 1})"] + df[f"HALPHA EW error (component {nn + 1})"]) - df[f"log HALPHA EW (component {nn + 1})"]
        
        df[f"log sigma_gas error (lower) (component {nn + 1})"] = df[f"log sigma_gas (component {nn + 1})"] - np.log10(df[f"sigma_gas (component {nn + 1})"] - df[f"sigma_gas error (component {nn + 1})"])
        df[f"log sigma_gas error (upper) (component {nn + 1})"] = np.log10(df[f"sigma_gas (component {nn + 1})"] + df[f"sigma_gas error (component {nn + 1})"]) - df[f"log sigma_gas (component {nn + 1})"]
        
    # Compute log quantities for total HALPHA EW
    df[f"log HALPHA luminosity (total)"] = np.log10(df[f"HALPHA luminosity (total)"])
    df["log HALPHA luminosity error (lower) (total)"] = df["log HALPHA luminosity (total)"] - np.log10(df["HALPHA luminosity (total)"] - df["HALPHA luminosity error (total)"])
    df["log HALPHA luminosity error (upper) (total)"] = np.log10(df["HALPHA luminosity (total)"] + df["HALPHA luminosity error (total)"]) -  df["log HALPHA luminosity (total)"]
    
    df[f"log HALPHA continuum luminosity"] = np.log10(df[f"HALPHA continuum luminosity"])
    df["log HALPHA continuum luminosity error (lower)"] = df["log HALPHA continuum luminosity"] - np.log10(df["HALPHA continuum luminosity"] - df["HALPHA continuum luminosity error"])
    df["log HALPHA continuum luminosity error (upper)"] = np.log10(df["HALPHA continuum luminosity"] + df["HALPHA continuum luminosity error"]) -  df["log HALPHA continuum luminosity"]
    
    df["log HALPHA EW (total)"] = np.log10(df["HALPHA EW (total)"])
    df["log HALPHA EW error (lower) (total)"] = df["log HALPHA EW (total)"] - np.log10(df["HALPHA EW (total)"] - df["HALPHA EW error (total)"])
    df["log HALPHA EW error (upper) (total)"] = np.log10(df["HALPHA EW (total)"] + df["HALPHA EW error (total)"]) -  df["log HALPHA EW (total)"]

    # Compute log quantities for total HALPHA EW
    for nn in range(ncomponents):
        if f"S2 ratio (component {nn + 1})" in df.columns:
            df[f"log S2 ratio (component {nn + 1})"] = np.log10(df[f"S2 ratio (component {nn + 1})"])
            df[f"log S2 ratio error (lower) (component {nn + 1})"] = df[f"log S2 ratio (component {nn + 1})"] - np.log10(df[f"S2 ratio (component {nn + 1})"] - df[f"S2 ratio error (component {nn + 1})"])
            df[f"log S2 ratio error (upper) (component {nn + 1})"] = np.log10(df[f"S2 ratio (component {nn + 1})"] + df[f"S2 ratio error (component {nn + 1})"]) -  df[f"log S2 ratio (component {nn + 1})"]
    if f"S2 ratio (total)" in df.columns:    
        df[f"log S2 ratio (total)"] = np.log10(df["S2 ratio (total)"])
        df[f"log S2 ratio error (lower) (total)"] = df[f"log S2 ratio (total)"] - np.log10(df["S2 ratio (total)"] - df["S2 ratio error (total)"])
        df[f"log S2 ratio error (upper) (total)"] = np.log10(df["S2 ratio (total)"] + df["S2 ratio error (total)"]) -  df[f"log S2 ratio (total)"]

    # Compute log quantities for total SFR
    for s in ["(total)", "(component 1)"]:
        if f"SFR {s}" in df.columns:
            cond = ~np.isnan(df[f"SFR {s}"])
            cond &= df[f"SFR {s}"] > 0
            df.loc[cond, f"log SFR {s}"] = np.log10(df.loc[cond, f"SFR {s}"])
            df.loc[cond, f"log SFR error (lower) {s}"] = df.loc[cond, f"log SFR {s}"] - np.log10(df.loc[cond, f"SFR {s}"] - df.loc[cond, f"SFR error {s}"])
            df.loc[cond, f"log SFR error (upper) {s}"] = np.log10(df.loc[cond, f"SFR {s}"] + df.loc[cond, f"SFR error {s}"]) -  df.loc[cond, f"log SFR {s}"]
            
        if f"SFR surface density {s}" in df.columns:
            cond = ~np.isnan(df[f"SFR surface density {s}"])
            cond &= df[f"SFR surface density {s}"] > 0
            # Compute log quantities for total SFR surface density
            df.loc[cond, f"log SFR surface density {s}"] = np.log10(df.loc[cond, f"SFR surface density {s}"])
            df.loc[cond, f"log SFR surface density error (lower) {s}"] = df.loc[cond, f"log SFR surface density {s}"] - np.log10(df.loc[cond, f"SFR surface density {s}"] - df.loc[cond, f"SFR surface density error {s}"])
            df.loc[cond, f"log SFR surface density error (upper) {s}"] = np.log10(df.loc[cond, f"SFR surface density {s}"] + df.loc[cond, f"SFR surface density error {s}"]) -  df.loc[cond, f"log SFR surface density {s}"]

    return df

######################################################################
# Compute offsets between gas & stellar kinematics
def compute_gas_stellar_offsets(df, ncomponents):
    assert ("v_*" in df.columns) and ("sigma_*" in df.columns), "v_* and sigma_* must be in the DataFrame to compute gas & stellar offsets!"
    for nn in range(ncomponents):
        df[f"sigma_gas - sigma_* (component {nn + 1})"] = df[f"sigma_gas (component {nn + 1})"] - df["sigma_*"]
        df[f"sigma_gas - sigma_* error (component {nn + 1})"] = np.sqrt(df[f"sigma_gas error (component {nn + 1})"]**2 + df["sigma_* error"]**2)

        df[f"sigma_gas^2 - sigma_*^2 (component {nn + 1})"] = df[f"sigma_gas (component {nn + 1})"]**2 - df["sigma_*"]**2
        df[f"sigma_gas^2 - sigma_*^2 error (component {nn + 1})"] = 2 * np.sqrt(df[f"sigma_gas (component {nn + 1})"]**2 * df[f"sigma_gas error (component {nn + 1})"]**2 +\
                                                                            df["sigma_*"]**2 * df["sigma_* error"]**2)
        
        df[f"v_gas - v_* (component {nn + 1})"] = df[f"v_gas (component {nn + 1})"] - df["v_*"]
        df[f"v_gas - v_* error (component {nn + 1})"] = np.sqrt(df[f"v_gas error (component {nn + 1})"]**2 + df["v_* error"]**2)
        
        df[f"sigma_gas/sigma_* (component {nn + 1})"] = df[f"sigma_gas (component {nn + 1})"] / df["sigma_*"]
        df[f"sigma_gas/sigma_* error (component {nn + 1})"] =\
                df[f"sigma_gas/sigma_* (component {nn + 1})"] *\
                np.sqrt((df[f"sigma_gas error (component {nn + 1})"] / df[f"sigma_gas (component {nn + 1})"])**2 +\
                        (df["sigma_* error"] / df["sigma_*"])**2)
    return df

######################################################################
# Compute differences in Halpha EW, sigma_gas between different components
def compute_component_offsets(df, ncomponents):
    assert ncomponents == 3, "ncomponents must be 3 to compute offsets between different components!"
    if ncomponents == 3:
        # Difference between gas velocity dispersion between components
        df["delta sigma_gas (2/1)"] = df["sigma_gas (component 2)"] - df["sigma_gas (component 1)"]
        df["delta sigma_gas (3/2)"] = df["sigma_gas (component 3)"] - df["sigma_gas (component 2)"]

        df["delta sigma_gas error (2/1)"] = np.sqrt(df["sigma_gas error (component 2)"]**2 +\
                                                         df["sigma_gas error (component 1)"]**2)
        df["delta sigma_gas error (3/2)"] = np.sqrt(df["sigma_gas error (component 2)"]**2 +\
                                                         df["sigma_gas error (component 3)"]**2)
        
        # DIfference between gas velocity between components
        df["delta v_gas (2/1)"] = df["v_gas (component 2)"] - df["v_gas (component 1)"]
        df["delta v_gas (3/2)"] = df["v_gas (component 3)"] - df["v_gas (component 2)"]
        df["delta v_gas error (2/1)"] = np.sqrt(df["v_gas error (component 2)"]**2 +\
                                                     df["v_gas error (component 1)"]**2)
        df["delta v_gas error (3/2)"] = np.sqrt(df["v_gas error (component 2)"]**2 +\
                                                     df["v_gas error (component 3)"]**2)
        
        # Ratio of HALPHA EWs between components
        df["HALPHA EW ratio (2/1)"] = df["HALPHA EW (component 2)"] / df["HALPHA EW (component 1)"]
        df["HALPHA EW ratio (3/2)"] = df["HALPHA EW (component 3)"] / df["HALPHA EW (component 2)"]
        df["HALPHA EW ratio error (2/1)"] = df["HALPHA EW ratio (2/1)"] *\
            np.sqrt((df["HALPHA EW error (component 2)"] / df["HALPHA EW (component 2)"])**2 +\
                    (df["HALPHA EW error (component 1)"] / df["HALPHA EW (component 1)"])**2)
        df["HALPHA EW ratio error (3/2)"] = df["HALPHA EW ratio (3/2)"] *\
            np.sqrt((df["HALPHA EW error (component 3)"] / df["HALPHA EW (component 3)"])**2 +\
                    (df["HALPHA EW error (component 2)"] / df["HALPHA EW (component 2)"])**2)
        
        # Ratio of HALPHA EWs between components (log)
        df["Delta HALPHA EW (1/2)"] = df["log HALPHA EW (component 1)"] - df["log HALPHA EW (component 2)"]
        df["Delta HALPHA EW (2/3)"] = df["log HALPHA EW (component 2)"] - df["log HALPHA EW (component 3)"]

        # Fractional of total Halpha EW in each component
        for nn in range(3):
            df[f"HALPHA EW/HALPHA EW (total) (component {nn + 1})"] = df[f"HALPHA EW (component {nn + 1})"] / df[f"HALPHA EW (total)"]

        # Forbidden line ratios:
        for col in ["log O3", "log N2", "log S2", "log O1"]:
            if f"{col} (component 1)" in df.columns and f"{col} (component 2)" in df.columns:
                df[f"delta {col} (2/1)"] = df[f"{col} (component 2)"] - df[f"{col} (component 1)"]
                df[f"delta {col} (2/1) error"] = np.sqrt(df[f"{col} (component 2)"]**2 + df[f"{col} (component 1)"]**2)
            if f"{col} (component 2)" in df.columns and f"{col} (component 3)" in df.columns:
                df[f"delta {col} (3/2)"] = df[f"{col} (component 3)"] - df[f"{col} (component 2)"]
                df[f"delta {col} (3/2) error"] = np.sqrt(df[f"{col} (component 3)"]**2 + df[f"{col} (component 2)"]**2)

    return df

#########################################################################
def compute_extra_columns(df, ncomponents):
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
    df[f"HALPHA continuum luminosity"] = df[f"HALPHA continuum"] * 1e-16 * 4 * np.pi * (df["D_L (Mpc)"] * 1e6 * 3.086e18)**2 * 1 / df["Bin size (square kpc)"]
    df[f"HALPHA continuum luminosity error"] = df[f"HALPHA continuum error"] * 1e-16 * 4 * np.pi * (df["D_L (Mpc)"] * 1e6 * 3.086e18)**2 * 1 / df["Bin size (square kpc)"]
    df[f"HALPHA luminosity (total)"] = df[f"HALPHA (total)"] * 1e-16 * 4 * np.pi * (df["D_L (Mpc)"] * 1e6 * 3.086e18)**2 * 1 / df["Bin size (square kpc)"]
    df[f"HALPHA luminosity error (total)"] = df[f"HALPHA error (total)"] * 1e-16 * 4 * np.pi * (df["D_L (Mpc)"] * 1e6 * 3.086e18)**2 * 1 / df["Bin size (square kpc)"]
    for nn in range(ncomponents):
        df[f"HALPHA luminosity (component {nn + 1})"] = df[f"HALPHA (component {nn + 1})"] * 1e-16 * 4 * np.pi * (df["D_L (Mpc)"] * 1e6 * 3.086e18)**2 * 1 / df["Bin size (square kpc)"]
        df[f"HALPHA luminosity error (component {nn + 1})"] = df[f"HALPHA error (component {nn + 1})"] * 1e-16 * 4 * np.pi * (df["D_L (Mpc)"] * 1e6 * 3.086e18)**2 * 1 / df["Bin size (square kpc)"]

    # Compute FWHM
    for nn in range(ncomponents):
        df[f"FWHM_gas (component {nn + 1})"] = df[f"sigma_gas (component {nn + 1})"] * 2 * np.sqrt(2 * np.log(2))
        df[f"FWHM_gas error (component {nn + 1})"] = df[f"sigma_gas error (component {nn + 1})"] * 2 * np.sqrt(2 * np.log(2))

    # Stellar & gas kinematic offsets
    if "v_*" in df.columns and "sigma_*" in df.columns:
        df = compute_gas_stellar_offsets(df, ncomponents=ncomponents)

    # Compute logs
    df = compute_log_columns(df, ncomponents=ncomponents)
    
    if ncomponents > 1:
        df = compute_component_offsets(df, ncomponents=ncomponents)

    return df

######################################################################
# TESTING
######################################################################
if __name__ == "__main__":

    import pandas as pd
    import os
    from copy import deepcopy

    import matplotlib.pyplot as plt
    plt.ion()

    ######################################################################
    # Load the SAMI DF
    ######################################################################
    data_path = "/priv/meggs3/u5708159/SAMI/sami_dr3/"

    bin_type = "default"
    ncomponents = "recom"
    eline_list = ["HALPHA", "HBETA", "NII6583", "OI6300", "OII3726+OII3729", "OIII5007", "SII6716", "SII6731"]
    
    # DQ options
    stekin_cut = True
    vgrad_cut = False
    eline_SNR_cut = True 

    eline_SNR_min = 5
    sigma_gas_SNR_cut = True
    sigma_gas_SNR_min = 3
    sigma_inst_kms = 29.6

    df_fname = f"sami_{bin_type}_{ncomponents}-comp.hd5"
    print("Loading DataFrame...")
    df = pd.read_hdf(os.path.join(data_path, df_fname),
                    key=f"{aperture_type} {bin_type}, {ncomponents}-comp" if bin_type == "aperture" else f"{bin_type}, {ncomponents}-comp")

    ######################################################################
    # Now, we can test...
    ######################################################################

    fig, axs = plt.subplots(nrows=1, ncols=3)

    df = compute_extra_columns(df, ncomponents=3 if ncomponents == "recom" else 1)
    axs[0].hist(df["log HALPHA EW (component 1)"], color="k", histtype="step", range=(-1, 3), bins=20, alpha=0.5)
    axs[1].hist(df["sigma_gas (component 1)"], color="k", histtype="step", range=(0, 50), bins=20, alpha=0.5)
    axs[2].hist(df["sigma_gas - sigma_* (component 1)"], color="k", histtype="step", range=(0, 50), bins=20, alpha=0.5)


    df_cut = deepcopy(df)

    eline_SNR_min = 5
    df_cut = df_dqcut(df=df_cut, ncomponents=3 if ncomponents == "recom" else 1,
                   eline_SNR_min=eline_SNR_min, eline_list=eline_list,
                   sigma_gas_SNR_cut=sigma_gas_SNR_cut, sigma_gas_SNR_min=sigma_gas_SNR_min, sigma_inst_kms=sigma_inst_kms,
                   vgrad_cut=False,
                   stekin_cut=True)

    df_cut = compute_extra_columns(df_cut, ncomponents=3 if ncomponents == "recom" else 1)
    axs[0].hist(df_cut["log HALPHA EW (component 1)"], color="k", histtype="step", range=(-1, 3), bins=20)
    axs[1].hist(df_cut["sigma_gas (component 1)"], color="k", histtype="step", range=(0, 50), bins=20)
    axs[2].hist(df_cut["sigma_gas - sigma_* (component 1)"], color="k", histtype="step", range=(0, 50), bins=20)

    # CHECK: no rows with S/N < 5
    fig, axs = plt.subplots(nrows=1, ncols=len(eline_list), figsize=(20, 5))
    fig.subplots_adjust(wspace=0)
    for rr, eline in enumerate(eline_list):
        axs[rr].hist(df[f"{eline} S/N (total)"], range=(0, 10), bins=20, color="k", histtype="step", alpha=0.5)
        axs[rr].hist(df_cut[f"{eline} S/N (total)"], range=(0, 10), bins=20, color="k", histtype="step", alpha=1.0)
        axs[rr].set_xlabel(f"{eline} S/N (total)")

    # CHECK: no rows with SFR = 0
    assert np.all(df_cut["SFR"] != 0)

    # CHECK: no rows with inclination = 0 and SFR != nan
    assert df_cut[(df_cut["Inclination i (degrees)"] == 0) & (df_cut["SFR"] > 0)].shape[0] == 0

    # CHECK: "Number of components" is 0, 1, 2 or 3
    assert np.all(df_cut["Number of components"].unique() == [0, 1, 2, 3])

    # CHECK: all emission line fluxes with S/N < SNR_min are NaN
    for eline in eline_list:
        assert np.all(np.isnan(df_cut.loc[df_cut[f"{eline} S/N (total)"] < eline_SNR_min, f"{eline} (total)"]))

    # CHECK: all Halpha components below S/N limit are NaN
    for nn in range(3 if ncomponents == "recom" else 1):
        assert np.all(np.isnan(df_cut.loc[df_cut[f"HALPHA S/N (component {nn + 1})"] < eline_SNR_min, f"HALPHA (component {nn + 1})"]))
        assert np.all(np.isnan(df_cut.loc[df_cut[f"HALPHA S/N (component {nn + 1})"] < eline_SNR_min, f"HALPHA error (component {nn + 1})"]))
        assert np.all(np.isnan(df_cut.loc[df_cut[f"HALPHA S/N (component {nn + 1})"] < eline_SNR_min, f"HALPHA EW (component {nn + 1})"]))
        assert np.all(np.isnan(df_cut.loc[df_cut[f"HALPHA S/N (component {nn + 1})"] < eline_SNR_min, f"HALPHA EW error (component {nn + 1})"]))
        assert np.all(np.isnan(df_cut.loc[df_cut[f"HALPHA S/N (component {nn + 1})"] < eline_SNR_min, f"log HALPHA EW (component {nn + 1})"]))
        assert np.all(np.isnan(df_cut.loc[df_cut[f"HALPHA S/N (component {nn + 1})"] < eline_SNR_min, f"log HALPHA EW error (upper) (component {nn + 1})"]))
        assert np.all(np.isnan(df_cut.loc[df_cut[f"HALPHA S/N (component {nn + 1})"] < eline_SNR_min, f"log HALPHA EW error (lower) (component {nn + 1})"]))

    # CHECK: all sigma_gas components with S/N < S/N target are NaN
    for nn in range(3 if ncomponents == "recom" else 1):
        assert np.all(np.isnan(df_cut.loc[df_cut[f"sigma_obs S/N (component {nn + 1})"] < df_cut[f"sigma_obs target S/N (component {nn + 1})"], f"sigma_gas (component {nn + 1})"]))
        assert np.all(np.isnan(df_cut.loc[df_cut[f"sigma_obs S/N (component {nn + 1})"] < df_cut[f"sigma_obs target S/N (component {nn + 1})"], f"sigma_gas error (component {nn + 1})"]))

    # CHECK: all sigma_gas components that don't meet the v_grad requirement are NaN
    if vgrad_cut:
        for nn in range(3 if ncomponents == "recom" else 1):
            assert np.all(np.isnan(df_cut.loc[df_cut[f"v_grad (component {nn + 1})"] > 2 * df_cut[f"sigma_gas (component {nn + 1})"], f"sigma_gas (component {nn + 1})"]))
            assert np.all(np.isnan(df_cut.loc[df_cut[f"v_grad (component {nn + 1})"] > 2 * df_cut[f"sigma_gas (component {nn + 1})"], f"sigma_gas error (component {nn + 1})"]))

    # 
    fig, axs = plt.subplots(nrows=1, ncols=len(eline_list), figsize=(20, 5))
    fig.subplots_adjust(wspace=0)
    for rr, eline in enumerate(eline_list):
        axs[rr].hist(df[f"{eline} S/N (total)"], range=(0, 10), bins=20, color="k", histtype="step", alpha=0.5)
        axs[rr].hist(df_cut[f"{eline} S/N (total)"], range=(0, 10), bins=20, color="k", histtype="step", alpha=1.0)
        axs[rr].set_xlabel(f"{eline} S/N (total)")



