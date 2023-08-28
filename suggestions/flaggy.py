def set_flags(spaxels, eline_SNR_min, eline_list, ncomponents_max,
              sigma_gas_SNR_min=3, **kwargs):
    """Set data quality & S/N flags.
    This function can be used to determine whether certain cells pass or fail 
    a number of data quality and S/N criteria. 

    Mutates data.
    """
    logger.debug("setting data quality and S/N flags...")
    with warnings.catch_warnings():
        warnings.filterwarnings(action="ignore", category=pd.errors.PerformanceWarning, message="DataFrame is highly fragmented.")
        ######################################################################
        # INITIALISE FLAGS: these will get set below.
        ########################################### d########################### 
        for component in spaxel.components:
            component.beam_smear = False
            component.low_sigma_gas_sn = False
        spaxel.ok_stellar_kinematics = True

        for eline in spaxel.elines:
            for component in eline.components:
                component.low_flux_sn = False
                if component.index >= 1:
                    component.low_flux_fraction = False
                component.low_amp = False
                component.missing_flux = False
            eline.low_flux_sn = False
            eline.low_amp = False
            eline.missing_flux = False
        spaxel.missing_components = False

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
            for component in eline.components:
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
  