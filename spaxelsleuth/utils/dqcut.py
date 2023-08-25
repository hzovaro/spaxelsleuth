import numpy as np
import pandas as pd 
from scipy import constants
import warnings

from spaxelsleuth.utils.elines import eline_lambdas_A
from spaxelsleuth.utils.continuum import compute_continuum_intensity
from spaxelsleuth.utils.misc import in_dataframe
from spaxelsleuth.utils.velocity import get_wavelength_from_velocity

import logging
logger = logging.getLogger(__name__)

###############################################################################
def compute_SN(df, ncomponents_max, eline_list):
    """Compute the flux S/N in the provided emission lines."""
    logger.debug("computing emission line S/N ratios...")
    for eline in eline_list:
        # Compute S/N
        for nn in range(ncomponents_max):
            if in_dataframe(df, [f"{eline} (component {nn + 1})", f"{eline} error (component {nn + 1})"]):
                df[f"{eline} S/N (component {nn + 1})"] = df[f"{eline} (component {nn + 1})"] / df[f"{eline} error (component {nn + 1})"]

        # Compute the S/N in the TOTAL line flux
        if in_dataframe(df, [f"{eline} (total)", f"{eline} error (total)"]):
            df[f"{eline} S/N (total)"] = df[f"{eline} (total)"] / df[f"{eline} error (total)"]

    return df

###############################################################################
def compute_HALPHA_amplitude_to_noise(data_cube, var_cube, lambda_vals_rest_A, v_star_map, v_map, dv):
    """Measure the HALPHA amplitude-to-noise.
        We measure this as
              (peak spectral value in window around Ha - mean R continuum flux density) / standard deviation in R continuum flux density
        As such, this value can be negative."""
    logger.debug("computing HALPHA amplitude-to-noise...")
    lambda_vals_rest_A_cube = np.zeros(data_cube.shape)
    lambda_vals_rest_A_cube[:] = lambda_vals_rest_A[:, None, None]

    # Get the HALPHA continuum & std. dev.
    cont_HALPHA_map, cont_HALPHA_map_std, cont_HALPHA_map_err = compute_continuum_intensity(data_cube=data_cube, var_cube=var_cube, lambda_vals_rest_A=lambda_vals_rest_A, start_A=6500, stop_A=6540, v_map=v_star_map)

    # Wavelength window in which to compute A/N
    lambda_max_A = get_wavelength_from_velocity(6562.8, v_map + dv, units="km/s")
    lambda_min_A = get_wavelength_from_velocity(6562.8, v_map - dv, units="km/s")

    # Measure HALPHA amplitude-to-noise
    A_HALPHA_mask = (lambda_vals_rest_A_cube > lambda_min_A) & (lambda_vals_rest_A_cube < lambda_max_A)
    data_cube_masked_R = np.copy(data_cube)
    data_cube_masked_R[~A_HALPHA_mask] = np.nan
    with warnings.catch_warnings():
        warnings.filterwarnings(action="ignore", category=RuntimeWarning, message="All-NaN slice encountered")
        A_HALPHA_map = np.nanmax(data_cube_masked_R, axis=0)
    AN_HALPHA_map = (A_HALPHA_map - cont_HALPHA_map) / cont_HALPHA_map_std

    return AN_HALPHA_map

###############################################################################
def set_flags(df, eline_SNR_min, eline_list, ncomponents_max,
              sigma_gas_SNR_min=3, **kwargs):
    """Set data quality & S/N flags.
    This function can be used to determine whether certain cells pass or fail 
    a number of data quality and S/N criteria. 
    """
    logger.debug("setting data quality and S/N flags...")
    with warnings.catch_warnings():
        warnings.filterwarnings(action="ignore", category=pd.errors.PerformanceWarning, message="DataFrame is highly fragmented.")
        ######################################################################
        # INITIALISE FLAGS: these will get set below.
        ###################################################################### 
        for nn in range(ncomponents_max):
            if f"v_gas (component {nn + 1})" in df:
                df[f"Beam smearing flag (component {nn + 1})"] = False
            if f"sigma_gas (component {nn + 1})" in df:
                df[f"Low sigma_gas S/N flag (component {nn + 1})"] = False
        df[f"Bad stellar kinematics"] = False

        for eline in eline_list:
            for nn in range(ncomponents_max):
                if f"{eline} (component {nn + 1})" in df:
                    df[f"Low flux S/N flag - {eline} (component {nn + 1})"] = False
                    if nn >= 1:
                        df[f"Low flux fraction flag - {eline} (component {nn + 1})"] = False
                    df[f"Low amplitude flag - {eline} (component {nn + 1})"] = False
                    df[f"Missing flux flag - {eline} (component {nn + 1})"] = False
            if f"{eline} (total)" in df:
                df[f"Low flux S/N flag - {eline} (total)"] = False
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
        logger.debug("flagging low S/N components and spaxels...")
        for eline in eline_list:
            # Fluxes in individual components
            for nn in range(ncomponents_max):
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
        logger.debug("flagging components and spaxels with NaN fluxes and finite errors...")
        for eline in eline_list:
            # Fluxes in individual components
            for nn in range(ncomponents_max):
                # Should we change this to include fluxes that are NaN?
                if f"{eline} (component {nn + 1})" in df.columns:
                    cond_missing_flux = df[f"{eline} (component {nn + 1})"].isna() & ~df[f"{eline} error (component {nn + 1})"].isna()
                    df.loc[cond_missing_flux, f"Missing flux flag - {eline} (component {nn + 1})"] = True
                    logger.debug(f"{eline} (component {nn + 1}): {df[cond_missing_flux].shape[0]:d} spaxels have missing fluxes in this component")

            # Total fluxes
            if f"{eline} (total)" in df.columns:
                cond_missing_flux = df[f"{eline} (total)"].isna() & ~df[f"{eline} error (total)"].isna()
                df.loc[cond_missing_flux, f"Missing flux flag - {eline} (total)"] = True
                logger.debug(f"{eline} (total): {df[cond_missing_flux].shape[0]:d} spaxels have missing total fluxes")

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
        logger.debug("flagging components with amplitude < 3 * rms continuum noise...")
        for eline in eline_list:
            lambda_rest_A = eline_lambdas_A[eline]
            for nn in range(ncomponents_max):
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

        ######################################################################
        # Flag rows where the flux ratio of the broad:narrow component < 0.05 (using the method of Avery+2021)
        ######################################################################
        logger.debug("flagging spaxels where the flux ratio of the broad:narrow component < 0.05...")
        for nn in range(1, ncomponents_max):
            for eline in eline_list:
                if f"{eline} (component {nn + 1})" in df:
                    cond_low_flux = df[f"{eline} A (component {nn + 1})"] < 0.05 * df[f"{eline} A (component 1)"]
                    df.loc[cond_low_flux, f"Low flux fraction flag - {eline} (component {nn + 1})"] = True

        ######################################################################
        # Flag rows that don't meet the beam smearing requirement
        ######################################################################
        logger.debug("flagging components likely to be affected by beam smearing...")
        # Gas kinematics: beam semaring criteria of Federrath+2017 and Zhou+2017.
        for nn in range(ncomponents_max):
            if f"v_grad (component {nn + 1})" in df and f"sigma_gas (component {nn + 1})" in df:
                cond_beam_smearing = df[f"sigma_gas (component {nn + 1})"] < 2 * df[f"v_grad (component {nn + 1})"]
                df.loc[cond_beam_smearing, f"Beam smearing flag (component {nn + 1})"] = True

        ######################################################################
        # Flag rows with insufficient S/N in sigma_gas
        ######################################################################
        logger.debug("flagging components with low sigma_gas S/N...")
        # Gas kinematics: NaN out cells w/ sigma_gas S/N ratio < sigma_gas_SNR_min 
        # (For SAMI, the red arm resolution is 29.6 km/s - see p6 of Croom+2021)
        for nn in range(ncomponents_max):
            if f"sigma_gas (component {nn + 1})" in df:
                # 1. Define sigma_obs = sqrt(sigma_gas**2 + sigma_inst_kms**2).
                df[f"sigma_obs (component {nn + 1})"] = np.sqrt(df[f"sigma_gas (component {nn + 1})"]**2 + kwargs["sigma_inst_kms"]**2)

                # 2. Define the S/N ratio of sigma_obs.
                # NOTE: here we assume that sigma_gas error (as output by LZIFU) 
                # really refers to the error on sigma_obs.
                if f"sigma_gas error (component {nn + 1})" in df:
                    df[f"sigma_obs S/N (component {nn + 1})"] = df[f"sigma_obs (component {nn + 1})"] / df[f"sigma_gas error (component {nn + 1})"]

                    # 3. Given our target SNR_gas, compute the target SNR_obs,
                    # using the method in section 2.2.2 of Zhou+2017.
                    df[f"sigma_obs target S/N (component {nn + 1})"] = sigma_gas_SNR_min * (1 + kwargs["sigma_inst_kms"]**2 / df[f"sigma_gas (component {nn + 1})"]**2)
                    cond_bad_sigma = df[f"sigma_obs S/N (component {nn + 1})"] < df[f"sigma_obs target S/N (component {nn + 1})"]
                    df.loc[cond_bad_sigma, f"Low sigma_gas S/N flag (component {nn + 1})"] = True

        ######################################################################
        # Stellar kinematics DQ cut
        ######################################################################
        logger.debug("flagging spaxels with unreliable stellar kinematics...")
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
                ncomponents_max,
                eline_list,
                line_flux_SNR_cut, 
                missing_fluxes_cut,
                line_amplitude_SNR_cut,
                flux_fraction_cut,
                sigma_gas_SNR_cut,
                vgrad_cut,
                stekin_cut,
                base_missing_flux_components_on_HALPHA=True,
                **kwargs):
    """Apply the data quality & S/N cuts based on the flags that were defined in set_flags().

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

    base_missing_flux_components_on_HALPHA  bool
        If True, then use HALPHA fluxes to determine whether components are 
        "missing" or not. i.e. - if HALPHA is NaN in component 1 but sigma_gas 
        is not, then this component is counted as "missing". If false,
        only sigma_gas is used in this determination, so this component would 
        NOT be counted as missing. See below for more details.

    MISSING COMPONENTS 
    --------------------------------------------------------------------------
    In datasets where emission lines have been fitted with multiple Gaussian 
    components, it is important to consider how to handle spaxels where one 
    or more individual components fails to meet S/N or data quality criteria. 
    We refer to these as "**missing components**".

    For example, a 2-component spaxel in which the 2nd component is a low-S/N 
    component may still have high S/N *total* fluxes, in which case things 
    like e.g. line ratios for the *total* flux are most likely still reliable. 
    In this case, you would probably want to mask out the fluxes and kinematics 
    of the low-S/N component, but keep the total fluxes.

    By default, (i.e. when base_missing_flux_components_on_HALPHA = True), 
    we define a "missing component" as one in which both the HALPHA flux and 
    velocity dispersion have been masked out for any reason. 
    
    If base_missing_flux_components_on_HALPHA = False, it is based only on 
    the velocity dispersion. 

    Note that while spaxelsleuth will flag affected spaxels (denoted by the 
    Missing components flag column), it will not automatically mask out 
    anything out based on this criterion, allowing the user to control how 
    spaxels with missing components are handled based on their specific use 
    case. 

    `utils.dqcut.apply_flags()` adds an additional `Number of components` column 
    to the DataFrame (not to be confused with the `Number of components (original)`
    column, which records the number of kinematic components in each spaxel that 
    were originally fitted to the data). `Number of components` records the number 
    of *reliable* components ONLY IF they are in the right order. For example, 
    consider a spaxel that originally has 2 components. Say that component 1 has a
    low S/N in the gas velocity dispersion, so sigma_gas (component 1) is masked 
    out, but HALPHA (component 1) is not, and that component 2 passes all DQ and S/N 
    cuts. In this case, the spaxel will NOT be recorded as having `Number of 
    components = 1 or 2` because component 1 fails to meet the DQ and S/N criteria. 
    It will therefore have an undefined `Number of components` and will be set to NaN. 

    Spaxels that still retain their original number of components after making all 
    DQ and S/N cuts can be selected as follows:

        df_good_quality_components = df[~df["Missing components flag"]]
 

    """
    logger.debug("applying data quality and S/N flags...")
    with warnings.catch_warnings():
        warnings.filterwarnings(action="ignore", category=pd.errors.PerformanceWarning, message="DataFrame is highly fragmented.")
        if line_flux_SNR_cut:
            logger.debug("masking components that don't meet the S/N requirements...")
            for eline in eline_list:
                # Individual fluxes
                for nn in range(ncomponents_max):
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
            logger.debug("masking components with missing fluxes...")
            for eline in eline_list:
                # Individual fluxes
                for nn in range(ncomponents_max):
                    if f"{eline} (component {nn + 1})" in df:
                        cond_missing_flux = df[f"Missing flux flag - {eline} (component {nn + 1})"]

                        # Cells to NaN
                        if eline == "HALPHA":
                            # Then NaN out EVERYTHING associated with this component - if we can't trust HALPHA then we probably can't trust anything else either!
                            cols_missing_fluxes = [c for c in df.columns if f"(component {nn + 1})" in c and "flag" not in c]
                        else:
                            cols_missing_fluxes = [c for c in df.columns if eline in c and f"(component {nn + 1})" in c and "flag" not in c]
                        df.loc[cond_missing_flux, cols_missing_fluxes] = np.nan

                # NOTE: I discovered that there are quite a few spaxels with NaN fluxes in the total line 
                # maps (i.e. data[0]) but non-zero errors in the corresponding error map (i.e. data_err[0]) in the LZIFU fits files.
                # These spaxels get flagged by the below lines.
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
            logger.debug("masking components that don't meet the amplitude requirements...")
            for eline in eline_list:
                for nn in range(ncomponents_max):
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
            logger.debug("masking components 1, 2 where the flux ratio of this component:component 1 < 0.05...")
            for nn in range(1, ncomponents_max):
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
            logger.debug("masking components that don't meet the beam smearing requirement...")
            for nn in range(ncomponents_max):
                if f"v_grad (component {nn + 1})" in df:
                    cond_beam_smearing = df[f"Beam smearing flag (component {nn + 1})"]

                    # Cells to NaN
                    cols_beam_smearing = [c for c in df.columns if f"component {nn + 1}" in c and ("v_gas" in c or "sigma_gas" in c) and "flag" not in c]
                    cols_beam_smearing += [c for c in df.columns if "delta" in c and str(nn + 1) in c]
                    df.loc[cond_beam_smearing, cols_beam_smearing] = np.nan

        if sigma_gas_SNR_cut:
            logger.debug("masking components with insufficient S/N in sigma_gas...")
            for nn in range(ncomponents_max):
                if f"sigma_obs S/N (component {nn + 1})" in df:
                    cond_bad_sigma = df[f"Low sigma_gas S/N flag (component {nn + 1})"]
                    
                    # Cells to NaN
                    cols_sigma_gas_SNR_cut = [c for c in df.columns if f"component {nn + 1}" in c and "sigma_gas" in c and "flag" not in c]
                    cols_sigma_gas_SNR_cut += [c for c in df.columns if "delta" in c and str(nn + 1) in c]
                    df.loc[cond_bad_sigma, cols_sigma_gas_SNR_cut] = np.nan

        if stekin_cut:
            logger.debug("masking spaxels with unreliable stellar kinematics...")
            if "v_*" in df and "sigma_*" in df:
                cond_bad_stekin = df["Bad stellar kinematics"]

                # Cells to NaN
                cols_stekin_cut = [c for c in df.columns if "v_*" in c or "sigma_*" in c]
                df.loc[cond_bad_stekin, cols_stekin_cut] = np.nan

        ######################################################################
        # Identify which spaxels have "missing components"
        ######################################################################
        logger.debug("flagging spaxels with 'missing components'...")
        """
        In datasets where emission lines have been fitted with multiple Gaussian 
        components, it is important to consider how to handle spaxels where one 
        or more individual components fails to meet S/N or data quality criteria. 
        We refer to these as "missing components".

        For example, a 2-component spaxel in which the 2nd component is a low-S/N 
        component may still have high S/N *total* fluxes, in which case things 
        like e.g. line ratios for the *total* flux are most likely still reliable. 
        In this case, you would probably want to mask out the fluxes and kinematics 
        of the low-S/N component, but keep the total fluxes.

        By default, (i.e. when base_missing_flux_components_on_HALPHA = True), 
        we define a "missing component" as one in which both the HALPHA flux and 
        velocity dispersion have been masked out for any reason. 
        
        If base_missing_flux_components_on_HALPHA = False, it is based only on 
        the velocity dispersion. 

        Note that while `spaxelsleuth` will flag affected spaxels (denoted by the 
        `Missing components flag` column), it will not automatically mask out 
        anything out based on this criterion, allowing the user to control how spaxels 
        with missing components are handled based on their specific use case. 

        `utils.dqcut.apply_flags()` adds an additional `Number of components` column 
        to the DataFrame (not to be confused with the `Number of components (original)`
        column, which records the number of kinematic components in each spaxel that 
        were originally fitted to the data). `Number of components` records the number 
        of *reliable* components ONLY IF they are in the right order. For example, 
        consider a spaxel that originally has 2 components. Say that component 1 has a
        low S/N in the gas velocity dispersion, so sigma_gas (component 1) is masked 
        out, but HALPHA (component 1) is not, and that component 2 passes all DQ and S/N 
        cuts. In this case, the spaxel will NOT be recorded as having `Number of 
        components = 1 or 2` because component 1 fails to meet the DQ and S/N criteria. 
        It will therefore have an undefined `Number of components` and will be set to NaN. 

        Spaxels that still retain their original number of components after making all 
        DQ and S/N cuts can be selected as follows:

            df_good_quality_components = df[~df["Missing components flag"]]

        """
        if base_missing_flux_components_on_HALPHA:
            logger.debug("using HALPHA to determine 'missing components'...")
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
        
        else:
            # Base decision only on sigma_gas.
            logger.warn("using only sigma_gas to define 'missing components'...")
            if all([col in df for col in [f"sigma_gas (component 1)", f"sigma_gas (component 2)", f"sigma_gas (component 3)"]]):
                cond_has_3 = ~np.isnan(df["sigma_gas (component 1)"]) &\
                            ~np.isnan(df["sigma_gas (component 2)"]) &\
                            ~np.isnan(df["sigma_gas (component 3)"]) 
                cond_has_2 = ~np.isnan(df["sigma_gas (component 1)"]) &\
                            ~np.isnan(df["sigma_gas (component 2)"]) &\
                            np.isnan(df["sigma_gas (component 3)"])  
                cond_has_1 = ~np.isnan(df["sigma_gas (component 1)"]) &\
                            np.isnan(df["sigma_gas (component 2)"]) &\
                            np.isnan(df["sigma_gas (component 3)"])
                cond_has_0 =  np.isnan(df["sigma_gas (component 1)"]) &\
                            np.isnan(df["sigma_gas (component 2)"]) &\
                            np.isnan(df["sigma_gas (component 3)"])

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

            elif all([col in df for col in [f"sigma_gas (component 1)", f"sigma_gas (component 2)"]]):
                cond_has_2 = ~np.isnan(df["sigma_gas (component 1)"]) &\
                            ~np.isnan(df["sigma_gas (component 2)"])
                cond_has_1 = ~np.isnan(df["sigma_gas (component 1)"]) &\
                            np.isnan(df["sigma_gas (component 2)"])
                cond_has_0 =  np.isnan(df["sigma_gas (component 1)"]) &\
                            np.isnan(df["sigma_gas (component 2)"])

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

            elif all([col in df for col in [f"sigma_gas (component 1)"]]):
                cond_has_1 = ~np.isnan(df["sigma_gas (component 1)"])
                cond_has_0 =  np.isnan(df["sigma_gas (component 1)"])
                
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
        logger.debug(f"statistics: {df[cond_mismatch].shape[0] / df.shape[0] * 100.:.2f}% of spaxels have individual components that have been masked out")

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