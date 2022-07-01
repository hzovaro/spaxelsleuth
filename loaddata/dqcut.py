import numpy as np
from scipy import constants
from IPython.core.debugger import Tracer
"""
Convenience function for computing a Doppler-shifted wavelength given a 
velocity and a rest-frame wavelength - used for converting SAMI fluxes 
into amplitudes
"""
def get_wavelength_from_velocity(lambda_rest, v, units):
    assert units == 'm/s' or units == 'km/s', "units must be m/s or km/s!"
    if units == 'm/s':
        v_m_s = v
    elif units == 'km/s':
        v_m_s = v * 1e3
    lambda_obs = lambda_rest * np.sqrt((1 + v_m_s / constants.c) /
                                       (1 - v_m_s / constants.c))
    return lambda_obs


"""
A function for making data quality & S/N cuts on rows of a given DataFrame.
"""
def dqcut(df, ncomponents,
             eline_SNR_min, eline_list,
             sigma_gas_SNR_cut=True, sigma_gas_SNR_min=3, sigma_inst_kms=29.6,
             vgrad_cut=False,
             line_amplitude_SNR_cut=True,
             stekin_cut=True):

    ######################################################################
    # INITIALISE FLAGS: these will get set below.
    ###################################################################### 
    for ii in range(ncomponents):
        df[f"Beam smearing flag (component {ii})"] = False
        df[f"sigma_gas S/N flag (component {ii})"] = False
        df[f"Bad stellar kinematics"] = False

    ######################################################################
    # NaN out line fluxes that don't meet the emission line S/N requirement
    ######################################################################
    """
    Note that we don't mask out the kinematic information here -
    if we were to do that here, we could end up in a situation where
    a spaxel w/ 2 components in which the 2nd component is low-S/N
    gets masked out, and so ends up looking like a 1-component
    spaxel, in which case the lines below won't mask it out. 
    So instead, such a spaxel will be filtered out by the below lines
    and it won't be included in our final sample.
    """
    for eline in eline_list:
        for ii in range(ncomponents):
            if f"{eline} (component {ii})" in df.columns:
                df.loc[df[f"{eline} S/N (component {ii})"] < eline_SNR_min, f"{eline} (component {ii})"] = np.nan
                df.loc[df[f"{eline} S/N (component {ii})"] < eline_SNR_min, f"{eline} error (component {ii})"] = np.nan

        # NaN out the TOTAL flux
        df.loc[df[f"{eline} S/N (total)"] < eline_SNR_min, f"{eline} (total)"] = np.nan
        df.loc[df[f"{eline} S/N (total)"] < eline_SNR_min, f"{eline} error (total)"] = np.nan

    ######################################################################
    # NaN out rows where any component doesn't meet the amplitude 
    # requirement imposed by Avery+2021
    # Note that we do NOT NaN out the TOTAL fluxes in lines where the 
    # fluxes of INDIVIDUAL fluxes do not meet our S/N requirement.
    ######################################################################
    # Compute the amplitude corresponding to each component
    # WARNING: only implemented for HALPHA!!! 
    # TODO: implement other lines
    for eline in ["HALPHA"]:
        lambda_rest_A = 6562.8  
        for ii in range(ncomponents):
            if f"{eline} (component {ii})" in df.columns:
                # Compute the amplitude of the line
                lambda_obs_A = get_wavelength_from_velocity(lambda_rest=lambda_rest_A, 
                                                            v=df[f"v_gas (component {ii})"], 
                                                            units='km/s')
                df[f"{eline} lambda_obs (component {ii}) (Å)"] = lambda_obs_A
                df[f"{eline} sigma_gas (component {ii}) (Å)"] = lambda_obs_A * df[f"sigma_gas (component {ii})"] * 1e3 / constants.c
                df[f"{eline} A (component {ii})"] = df[f"HALPHA (component {ii})"] / df[f"{eline} sigma_gas (component {ii}) (Å)"] / np.sqrt(2 * np.pi)
            
                # Flag bad components
                df[f"Low S/N component - {eline} (component {ii})"] = True 
                cond_bad_gasamp = df[f"{eline} A (component {ii})"] >= 3 * df["HALPHA continuum std. dev."]
                df.loc[cond_bad_gasamp, f"Low S/N component - {eline} (component {ii})"] = False

                # NaN out fluxes and kinematics associated with this component and this line
                if line_amplitude_SNR_cut:
                    cols = [f"{eline} (component {ii})", f"{eline} error (component {ii})"]
                    # Only NaN out the corresponding velocities if the line is HALPHA, since this is usually the strongest line
                    # Also NaN out the EW in that case
                    if eline == "HALPHA":
                        cols += [f"v_gas (component {ii})",
                                 f"sigma_gas (component {ii})",
                                 f"v_gas error (component {ii})",
                                 f"sigma_gas error (component {ii})",]
                        cols += [f"HALPHA EW (component {ii})", f"HALPHA EW error (component {ii})"]
                    df.loc[df[f"Low S/N component - {eline} (component {ii})"], cols] = np.nan

        # IF ALL COMPONENTS HAVE LOW S/N, THEN DISCARD TOTAL VALUES AS WELL
        if line_amplitude_SNR_cut:
            if ncomponents == 3:
                cond_all_bad_components = df[f"Low S/N component - {eline} (component 0)"] &\
                                          df[f"Low S/N component - {eline} (component 1)"] &\
                                          df[f"Low S/N component - {eline} (component 2)"]
            else:
                cond_all_bad_components = df[f"Low S/N component - {eline} (component 0)"] 
            if any(cond_all_bad_components):
                cols = [f"{eline} (total)", f"{eline} error (total)"]
                # Also zero SFR measurements
                if eline == "HALPHA":
                    cols += [c for c in df.columns if "SFR" in c]
                    cols += ["HALPHA EW (total)", "HALPHA EW error (total)"]
                df.loc[cond_all_bad_components, cols] = np.nan

    # ######################################################################
    # # NaN out rows where the flux ratio of the broad:narrow component < 0.05 
    # # (using the method of Avery+2021)
    # ######################################################################
    # if ncomponents > 1:
    #     for ii in [1, 2]:
    #         cond_low_flux = df[f"HALPHA A (component {ii})"] < 0.05 * df["HALPHA A (component 0)"]
    #         df.loc[cond_low_flux, f"Low flux component (component {ii})"] = True
            
    #         # NaN out rows 
    #         cols = [f"HALPHA (component {ii})", f"HALPHA error (component {ii})"]
    #         cols += [f"HALPHA EW (component {ii})", f"HALPHA EW error (component {ii})"]
    #         cols += [f"v_gas (component {ii})",
    #                  f"sigma_gas (component {ii})",
    #                  f"v_gas error (component {ii})",
    #                  f"sigma_gas error (component {ii})",]
    #         df.loc[df[f"Low flux component (component {ii})"], cols] = np.nan

    ######################################################################
    # NaN out the Halpha EW in each component where the HALPHA fluxes have 
    # been NaN'd out
    ######################################################################
    # # Mask out the HALPHA EW in rows where the Halpha doesn't meet the S/N requirement.
    # if "HALPHA (component 0)" in df.columns:
    #     for ii in range(ncomponents):
    #         df.loc[df[f"HALPHA (component {ii})"].isna(), f"HALPHA EW (component {ii})"] = np.nan
    #         df.loc[df[f"HALPHA (component {ii})"].isna(), f"HALPHA EW error (component {ii})"] = np.nan

    # # NaN out the TOTAL Halpha S/N if it doesn't meet the requirement
    # df.loc[df[f"HALPHA (total)"].isna(), f"HALPHA EW (total)"] = np.nan
    # df.loc[df[f"HALPHA (total)"].isna(), f"HALPHA EW error (total)"] = np.nan

    ######################################################################
    # NaN out rows that don't meet the beam smearing requirement
    ######################################################################
    # Gas kinematics: beam semaring criteria of Federrath+2017 and Zhou+2017.
    for ii in range(ncomponents):
        cond_beam_smearing = df[f"sigma_gas (component {ii})"] < 2 * df[f"v_grad (component {ii})"]
        df.loc[cond_beam_smearing, f"Beam smearing flag (component {ii})"] = True

        # NaN out offending cells
        if vgrad_cut:
            df.loc[cond_beam_smearing, 
                   [f"v_gas (component {ii})",
                    f"sigma_gas (component {ii})", 
                    f"v_gas error (component {ii})",
                    f"sigma_gas error (component {ii})",]] = np.nan

    ######################################################################
    # NaN out rows with insufficient S/N in sigma_gas
    ######################################################################
    # Gas kinematics: NaN out cells w/ sigma_gas S/N ratio < sigma_gas_SNR_min 
    # (For SAMI, the red arm resolution is 29.6 km/s - see p6 of Croom+2021)
    for ii in range(ncomponents):
        # 1. Define sigma_obs = sqrt(sigma_gas**2 + sigma_inst_kms**2).
        df[f"sigma_obs (component {ii})"] = np.sqrt(df[f"sigma_gas (component {ii})"]**2 + sigma_inst_kms**2)

        # 2. Define the S/N ratio of sigma_obs.
        # NOTE: here we assume that sigma_gas error (as output by LZIFU) 
        # really refers to the error on sigma_obs.
        df[f"sigma_obs S/N (component {ii})"] = df[f"sigma_obs (component {ii})"] / df[f"sigma_gas error (component {ii})"]

        # 3. Given our target SNR_gas, compute the target SNR_obs,
        # using the method in section 2.2.2 of Zhou+2017.
        df[f"sigma_obs target S/N (component {ii})"] = sigma_gas_SNR_min * (1 + sigma_inst_kms**2 / df[f"sigma_gas (component {ii})"]**2)
        cond_bad_sigma = df[f"sigma_obs S/N (component {ii})"] < df[f"sigma_obs target S/N (component {ii})"]
        df.loc[cond_bad_sigma, f"sigma_gas S/N flag (component {ii})"] = True

        # NaN out offending cells
        if sigma_gas_SNR_cut:
            df.loc[cond_bad_sigma, 
                   [f"sigma_gas (component {ii})", 
                    f"sigma_gas error (component {ii})",]] = np.nan

    ######################################################################
    # Determine how to compute the number of components in each 
    # spaxel
    ######################################################################
    """
    2 ways to do this -
    1. Conservative: 
        cut spaxels in which ANY component has low S/N - i.e. NaN out ALL
        emission-line measurements including fluxes and kinematics for ALL
        components and for total values as well.
        Re-set the number of components in each spaxel 

    2. Relaxed:
        leave everything as-is, including the original number of components
        fitted by LZIFU. HOWEVER - the downside of this is that when making
        selections of spaxels based on the original number of components 
        fitted, there will be missing data in components with insufficient
        S/N.
    
    """
    # Identify rows with non-NaN Halpha fluxes AND sigma_gas values. i.e., get 
    # rid of the orphan components.
    if ncomponents == 3:
        cond_has_3 = (df["Number of components (original)"] == 3) &\
                     ~np.isnan(df["HALPHA (component 0)"]) & ~np.isnan(df["sigma_gas (component 0)"]) &\
                     ~np.isnan(df["HALPHA (component 1)"]) & ~np.isnan(df["sigma_gas (component 1)"]) &\
                     ~np.isnan(df["HALPHA (component 2)"]) & ~np.isnan(df["sigma_gas (component 2)"]) 
        cond_has_2 = (df["Number of components (original)"] == 2) &\
                     ~np.isnan(df["HALPHA (component 0)"]) & ~np.isnan(df["sigma_gas (component 0)"]) &\
                     ~np.isnan(df["HALPHA (component 1)"]) & ~np.isnan(df["sigma_gas (component 1)"]) &\
                      np.isnan(df["HALPHA (component 2)"]) &  np.isnan(df["sigma_gas (component 2)"])  
        cond_has_1 = (df["Number of components (original)"] == 1) &\
                     ~np.isnan(df["HALPHA (component 0)"]) & ~np.isnan(df["sigma_gas (component 0)"]) &\
                      np.isnan(df["HALPHA (component 1)"]) &  np.isnan(df["sigma_gas (component 1)"]) &\
                      np.isnan(df["HALPHA (component 2)"]) &  np.isnan(df["sigma_gas (component 2)"])
        cond_has_any = cond_has_1 | cond_has_2 | cond_has_3
        
        # Define columns to NaN out 
        cols_to_nan = [f"{e} (total)" for e in eline_list]
        cols_to_nan += [f"{e} error (total)" for e in eline_list]
        for ii in range(3):
            for e in eline_list:
                cols_to_nan += [f"{e} (component {ii})" for e in eline_list if f"{e} (component {ii})" in df.columns]
                cols_to_nan += [f"{e} error (component {ii})" for e in eline_list if f"{e} error (component {ii})" in df.columns]
        
        # Add EWs, kinematic quantities
        cols_to_nan += [
                "HALPHA EW (component 0)", "HALPHA EW (component 1)", "HALPHA EW (component 2)", "HALPHA EW (total)",
                "HALPHA EW error (component 0)", "HALPHA EW error (component 1)", "HALPHA EW error (component 2)", "HALPHA EW error (total)",
                "sigma_gas (component 0)", "sigma_gas (component 1)", "sigma_gas (component 2)",
                "sigma_obs (component 0)", "sigma_obs (component 1)", "sigma_obs (component 2)",
                "v_gas (component 0)", "v_gas (component 1)", "v_gas (component 2)",
                "sigma_gas error (component 0)", "sigma_gas error (component 1)", "sigma_gas error (component 2)",
                "v_gas error (component 0)", "v_gas error (component 1)", "v_gas error (component 2)"]

        # NaN them out.
        df.loc[~cond_has_any, cols_to_nan] = np.nan

        # Reset the number of components
        df.loc[cond_has_1, "Number of components"] = 1
        df.loc[cond_has_2, "Number of components"] = 2
        df.loc[cond_has_3, "Number of components"] = 3
        df.loc[~cond_has_any, "Number of components"] = 0

    elif ncomponents == 1:
        cond_has_1 = (df["Number of components (original)"] == 1) & ~np.isnan(df["HALPHA (component 0)"]) & ~np.isnan(df["sigma_gas (component 0)"])

        # Define columns to NaN out 
        cols_to_nan = [f"{e} (total)" for e in eline_list]
        cols_to_nan += [f"{e} error (total)" for e in eline_list]
        cols_to_nan += [f"{e} (component 0)" for e in eline_list if f"{e} (component 0)" in df.columns]
        cols_to_nan += [f"{e} error (component 0)" for e in eline_list if f"{e} error (component 0)" in df.columns]
    
        # Add EWs, kinematic quantities
        cols_to_nan += [
                "HALPHA EW (component 0)", "HALPHA EW (total)",
                "HALPHA EW error (component 0)", "HALPHA EW error (total)",
                "sigma_gas (component 0)", "sigma_obs (component 0)", "v_gas (component 0)",
                "sigma_gas error (component 0)", "v_gas error (component 0)"]

        # NaN them out
        df.loc[~cond_has_1, cols_to_nan] = np.nan

        # Reset the number of components
        df.loc[cond_has_1, "Number of components"] = 1
        df.loc[~cond_has_1, "Number of components"] = 0

    else:
        raise ValueError("Only 3- or 1-component fits have been implemented!")

    ######################################################################
    # Stellar kinematics DQ cut
    ######################################################################
    # Stellar kinematics: NaN out cells that don't meet the criteria given  
    # on page 18 of Croom+2021
    if all([c in df.columns for c in ["sigma_*", "v_*"]]):
        cond_bad_stekin = df["sigma_*"] <= 35
        cond_bad_stekin |= df["v_* error"] >= 30
        cond_bad_stekin |= df["sigma_* error"] >= df["sigma_*"] * 0.1 + 25
        df.loc[cond_bad_stekin, "Bad stellar kinematics"] = True

        if stekin_cut:
            # Only NaN out the stellar kinematic info, since it's unrelated to the gas kinematics & other quantities
            df.loc[cond_bad_stekin, ["v_*", "v_* error", 
                                     "sigma_*", "sigma_* error"]] = np.nan

    ######################################################################
    # SFR and SFR surface density DQ cuts
    ######################################################################
    # Set components w/ SFR = 0 to NaN
    # If the inclination is undefined, also set the SFR and SFR surface density to NaN.
    if "SFR" in df.columns:
        cond_zero_SFR = df["SFR"] == 0
        df.loc[cond_zero_SFR, "SFR"] = np.nan
        df.loc[cond_zero_SFR, "SFR error"] = np.nan
    if "SFR surface density" in df.columns:
        cond_zero_SFR = df["SFR"] == 0
        df.loc[cond_zero_SFR, "SFR surface density"] = np.nan
        df.loc[cond_zero_SFR, "SFR surface density error"] = np.nan
        df.loc[np.isnan(df["Inclination i (degrees)"]), "SFR surface density"] = np.nan
        df.loc[np.isnan(df["Inclination i (degrees)"]), "SFR surface density error"] = np.nan

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
    for ii in range(ncomponents):          
        # log quantities
        df[f"log HALPHA luminosity (component {ii})"] = np.log10(df[f"HALPHA luminosity (component {ii})"])
        df[f"log HALPHA EW (component {ii})"] = np.log10(df[f"HALPHA EW (component {ii})"])
        df[f"log sigma_gas (component {ii})"] = np.log10(df[f"sigma_gas (component {ii})"])

        # Compute errors for log quantities
        df[f"log HALPHA luminosity error (lower) (component {ii})"] = df[f"log HALPHA luminosity (component {ii})"] - np.log10(df[f"HALPHA luminosity (component {ii})"] - df[f"HALPHA luminosity error (component {ii})"])
        df[f"log HALPHA luminosity error (upper) (component {ii})"] = np.log10(df[f"HALPHA luminosity (component {ii})"] + df[f"HALPHA luminosity error (component {ii})"]) - df[f"log HALPHA luminosity (component {ii})"]

        df[f"log HALPHA EW error (lower) (component {ii})"] = df[f"log HALPHA EW (component {ii})"] - np.log10(df[f"HALPHA EW (component {ii})"] - df[f"HALPHA EW error (component {ii})"])
        df[f"log HALPHA EW error (upper) (component {ii})"] = np.log10(df[f"HALPHA EW (component {ii})"] + df[f"HALPHA EW error (component {ii})"]) - df[f"log HALPHA EW (component {ii})"]
        
        df[f"log sigma_gas error (lower) (component {ii})"] = df[f"log sigma_gas (component {ii})"] - np.log10(df[f"sigma_gas (component {ii})"] - df[f"sigma_gas error (component {ii})"])
        df[f"log sigma_gas error (upper) (component {ii})"] = np.log10(df[f"sigma_gas (component {ii})"] + df[f"sigma_gas error (component {ii})"]) - df[f"log sigma_gas (component {ii})"]
        
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
    for ii in range(ncomponents):
        if f"S2 ratio (component {ii})" in df.columns:
            df[f"log S2 ratio (component {ii})"] = np.log10(df[f"S2 ratio (component {ii})"])
            df[f"log S2 ratio error (lower) (component {ii})"] = df[f"log S2 ratio (component {ii})"] - np.log10(df[f"S2 ratio (component {ii})"] - df[f"S2 ratio error (component {ii})"])
            df[f"log S2 ratio error (upper) (component {ii})"] = np.log10(df[f"S2 ratio (component {ii})"] + df[f"S2 ratio error (component {ii})"]) -  df[f"log S2 ratio (component {ii})"]
    if f"S2 ratio (total)" in df.columns:    
        df[f"log S2 ratio (total)"] = np.log10(df["S2 ratio (total)"])
        df[f"log S2 ratio error (lower) (total)"] = df[f"log S2 ratio (total)"] - np.log10(df["S2 ratio (total)"] - df["S2 ratio error (total)"])
        df[f"log S2 ratio error (upper) (total)"] = np.log10(df["S2 ratio (total)"] + df["S2 ratio error (total)"]) -  df[f"log S2 ratio (total)"]

    # Compute log quantities for total SFR
    for s in ["(total)", "(component 0)"]:
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
    for ii in range(ncomponents):
        df[f"sigma_gas - sigma_* (component {ii})"] = df[f"sigma_gas (component {ii})"] - df["sigma_*"]
        df[f"sigma_gas - sigma_* error (component {ii})"] = np.sqrt(df[f"sigma_gas error (component {ii})"]**2 + df["sigma_* error"]**2)

        df[f"sigma_gas^2 - sigma_*^2 (component {ii})"] = df[f"sigma_gas (component {ii})"]**2 - df["sigma_*"]**2
        df[f"sigma_gas^2 - sigma_*^2 error (component {ii})"] = 2 * np.sqrt(df[f"sigma_gas (component {ii})"]**2 * df[f"sigma_gas error (component {ii})"]**2 +\
                                                                            df["sigma_*"]**2 * df["sigma_* error"]**2)
        
        df[f"v_gas - v_* (component {ii})"] = df[f"v_gas (component {ii})"] - df["v_*"]
        df[f"v_gas - v_* error (component {ii})"] = np.sqrt(df[f"v_gas error (component {ii})"]**2 + df["v_* error"]**2)
        
        df[f"sigma_gas/sigma_* (component {ii})"] = df[f"sigma_gas (component {ii})"] / df["sigma_*"]
        df[f"sigma_gas/sigma_* error (component {ii})"] =\
                df[f"sigma_gas/sigma_* (component {ii})"] *\
                np.sqrt((df[f"sigma_gas error (component {ii})"] / df[f"sigma_gas (component {ii})"])**2 +\
                        (df["sigma_* error"] / df["sigma_*"])**2)
    return df

######################################################################
# Compute differences in Halpha EW, sigma_gas between different components
def compute_component_offsets(df, ncomponents):
    assert ncomponents == 3, "ncomponents must be 3 to compute offsets between different components!"
    if ncomponents == 3:
        # Difference between gas velocity dispersion between components
        df["delta sigma_gas (1/0)"] = df["sigma_gas (component 1)"] - df["sigma_gas (component 0)"]
        df["delta sigma_gas (2/1)"] = df["sigma_gas (component 2)"] - df["sigma_gas (component 1)"]

        df["delta sigma_gas error (1/0)"] = np.sqrt(df["sigma_gas error (component 1)"]**2 +\
                                                         df["sigma_gas error (component 0)"]**2)
        df["delta sigma_gas error (2/1)"] = np.sqrt(df["sigma_gas error (component 1)"]**2 +\
                                                         df["sigma_gas error (component 2)"]**2)
        
        # DIfference between gas velocity between components
        df["delta v_gas (1/0)"] = df["v_gas (component 1)"] - df["v_gas (component 0)"]
        df["delta v_gas (2/1)"] = df["v_gas (component 2)"] - df["v_gas (component 1)"]
        df["delta v_gas error (1/0)"] = np.sqrt(df["v_gas error (component 1)"]**2 +\
                                                     df["v_gas error (component 0)"]**2)
        df["delta v_gas error (2/1)"] = np.sqrt(df["v_gas error (component 1)"]**2 +\
                                                     df["v_gas error (component 2)"]**2)
        
        # Ratio of HALPHA EWs between components
        df["HALPHA EW ratio (1/0)"] = df["HALPHA EW (component 1)"] / df["HALPHA EW (component 0)"]
        df["HALPHA EW ratio (2/1)"] = df["HALPHA EW (component 2)"] / df["HALPHA EW (component 1)"]
        df["HALPHA EW ratio error (1/0)"] = df["HALPHA EW ratio (1/0)"] *\
            np.sqrt((df["HALPHA EW error (component 1)"] / df["HALPHA EW (component 1)"])**2 +\
                    (df["HALPHA EW error (component 0)"] / df["HALPHA EW (component 0)"])**2)
        df["HALPHA EW ratio error (2/1)"] = df["HALPHA EW ratio (2/1)"] *\
            np.sqrt((df["HALPHA EW error (component 2)"] / df["HALPHA EW (component 2)"])**2 +\
                    (df["HALPHA EW error (component 1)"] / df["HALPHA EW (component 1)"])**2)
        
        # Ratio of HALPHA EWs between components (log)
        df["Delta HALPHA EW (0/1)"] = df["log HALPHA EW (component 0)"] - df["log HALPHA EW (component 1)"]
        df["Delta HALPHA EW (1/2)"] = df["log HALPHA EW (component 1)"] - df["log HALPHA EW (component 2)"]

        # Fractional of total Halpha EW in each component
        for ii in range(3):
            df[f"HALPHA EW/HALPHA EW (total) (component {ii})"] = df[f"HALPHA EW (component {ii})"] / df[f"HALPHA EW (total)"]

        # Forbidden line ratios:
        for col in ["log O3", "log N2", "log S2", "log O1"]:
            if f"{col} (component 0)" in df.columns and f"{col} (component 1)" in df.columns:
                df[f"delta {col} (1/0)"] = df[f"{col} (component 1)"] - df[f"{col} (component 0)"]
                df[f"delta {col} (1/0) error"] = np.sqrt(df[f"{col} (component 1)"]**2 + df[f"{col} (component 0)"]**2)
            if f"{col} (component 1)" in df.columns and f"{col} (component 2)" in df.columns:
                df[f"delta {col} (2/1)"] = df[f"{col} (component 2)"] - df[f"{col} (component 1)"]
                df[f"delta {col} (2/1) error"] = np.sqrt(df[f"{col} (component 2)"]**2 + df[f"{col} (component 1)"]**2)

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
    for ii in range(ncomponents):
        df[f"HALPHA luminosity (component {ii})"] = df[f"HALPHA (component {ii})"] * 1e-16 * 4 * np.pi * (df["D_L (Mpc)"] * 1e6 * 3.086e18)**2 * 1 / df["Bin size (square kpc)"]
        df[f"HALPHA luminosity error (component {ii})"] = df[f"HALPHA error (component {ii})"] * 1e-16 * 4 * np.pi * (df["D_L (Mpc)"] * 1e6 * 3.086e18)**2 * 1 / df["Bin size (square kpc)"]

    # Compute FWHM
    for ii in range(ncomponents):
        df[f"FWHM_gas (component {ii})"] = df[f"sigma_gas (component {ii})"] * 2 * np.sqrt(2 * np.log(2))
        df[f"FWHM_gas error (component {ii})"] = df[f"sigma_gas error (component {ii})"] * 2 * np.sqrt(2 * np.log(2))

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
    axs[0].hist(df["log HALPHA EW (component 0)"], color="k", histtype="step", range=(-1, 3), bins=20, alpha=0.5)
    axs[1].hist(df["sigma_gas (component 0)"], color="k", histtype="step", range=(0, 50), bins=20, alpha=0.5)
    axs[2].hist(df["sigma_gas - sigma_* (component 0)"], color="k", histtype="step", range=(0, 50), bins=20, alpha=0.5)


    df_cut = deepcopy(df)

    eline_SNR_min = 5
    df_cut = df_dqcut(df=df_cut, ncomponents=3 if ncomponents == "recom" else 1,
                   eline_SNR_min=eline_SNR_min, eline_list=eline_list,
                   sigma_gas_SNR_cut=sigma_gas_SNR_cut, sigma_gas_SNR_min=sigma_gas_SNR_min, sigma_inst_kms=sigma_inst_kms,
                   vgrad_cut=False,
                   stekin_cut=True)

    df_cut = compute_extra_columns(df_cut, ncomponents=3 if ncomponents == "recom" else 1)
    axs[0].hist(df_cut["log HALPHA EW (component 0)"], color="k", histtype="step", range=(-1, 3), bins=20)
    axs[1].hist(df_cut["sigma_gas (component 0)"], color="k", histtype="step", range=(0, 50), bins=20)
    axs[2].hist(df_cut["sigma_gas - sigma_* (component 0)"], color="k", histtype="step", range=(0, 50), bins=20)

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
    for ii in range(3 if ncomponents == "recom" else 1):
        assert np.all(np.isnan(df_cut.loc[df_cut[f"HALPHA S/N (component {ii})"] < eline_SNR_min, f"HALPHA (component {ii})"]))
        assert np.all(np.isnan(df_cut.loc[df_cut[f"HALPHA S/N (component {ii})"] < eline_SNR_min, f"HALPHA error (component {ii})"]))
        assert np.all(np.isnan(df_cut.loc[df_cut[f"HALPHA S/N (component {ii})"] < eline_SNR_min, f"HALPHA EW (component {ii})"]))
        assert np.all(np.isnan(df_cut.loc[df_cut[f"HALPHA S/N (component {ii})"] < eline_SNR_min, f"HALPHA EW error (component {ii})"]))
        assert np.all(np.isnan(df_cut.loc[df_cut[f"HALPHA S/N (component {ii})"] < eline_SNR_min, f"log HALPHA EW (component {ii})"]))
        assert np.all(np.isnan(df_cut.loc[df_cut[f"HALPHA S/N (component {ii})"] < eline_SNR_min, f"log HALPHA EW error (upper) (component {ii})"]))
        assert np.all(np.isnan(df_cut.loc[df_cut[f"HALPHA S/N (component {ii})"] < eline_SNR_min, f"log HALPHA EW error (lower) (component {ii})"]))

    # CHECK: all sigma_gas components with S/N < S/N target are NaN
    for ii in range(3 if ncomponents == "recom" else 1):
        assert np.all(np.isnan(df_cut.loc[df_cut[f"sigma_obs S/N (component {ii})"] < df_cut[f"sigma_obs target S/N (component {ii})"], f"sigma_gas (component {ii})"]))
        assert np.all(np.isnan(df_cut.loc[df_cut[f"sigma_obs S/N (component {ii})"] < df_cut[f"sigma_obs target S/N (component {ii})"], f"sigma_gas error (component {ii})"]))

    # CHECK: all sigma_gas components that don't meet the v_grad requirement are NaN
    if vgrad_cut:
        for ii in range(3 if ncomponents == "recom" else 1):
            assert np.all(np.isnan(df_cut.loc[df_cut[f"v_grad (component {ii})"] > 2 * df_cut[f"sigma_gas (component {ii})"], f"sigma_gas (component {ii})"]))
            assert np.all(np.isnan(df_cut.loc[df_cut[f"v_grad (component {ii})"] > 2 * df_cut[f"sigma_gas (component {ii})"], f"sigma_gas error (component {ii})"]))

    # 
    fig, axs = plt.subplots(nrows=1, ncols=len(eline_list), figsize=(20, 5))
    fig.subplots_adjust(wspace=0)
    for rr, eline in enumerate(eline_list):
        axs[rr].hist(df[f"{eline} S/N (total)"], range=(0, 10), bins=20, color="k", histtype="step", alpha=0.5)
        axs[rr].hist(df_cut[f"{eline} S/N (total)"], range=(0, 10), bins=20, color="k", histtype="step", alpha=1.0)
        axs[rr].set_xlabel(f"{eline} S/N (total)")



