import numpy as np

from spaxelsleuth.config import settings
from spaxelsleuth.utils import continuum, dqcut, linefns, metallicity, extcorr, misc
from spaxelsleuth.utils.misc import in_dataframe

import logging
logger = logging.getLogger(__name__)

###############################################################################
def add_columns(survey, df, **kwargs):
    """Computes quantities such as metallicities, extinctions, etc. for each row in df."""
    logger.info(f"adding columns to the DataFrame...")

    # Figure out the maximum number of components that has been fitted to each spaxel
    ncomponents_max = 0
    while True:
        if not in_dataframe(df, f"sigma_gas (component {ncomponents_max + 1})"):
            break
        ncomponents_max += 1

    # Compute the ORIGINAL number of components in each spaxel: define these as those in which sigma_gas is not NaN
    ncomponents_original = (~df[f"sigma_gas (component 1)"].isna()).astype(int)
    for nn in range(1, ncomponents_max):
        ncomponents_original += (~df[f"sigma_gas (component {nn + 1})"].isna()).astype(int)
    df["Number of components (original)"] = ncomponents_original

    # Options that are survey-specific 
    eline_list = settings[survey]["eline_list"]
    if survey == "sami":
        compute_sfr = False
        kwargs["base_missing_flux_components_on_HALPHA"] = True
    else:
        compute_sfr = True
        kwargs["base_missing_flux_components_on_HALPHA"] = False

    ######################################################################
    # Calculate equivalent widths
    df = continuum.compute_EW(df, ncomponents_max, eline_list=["HALPHA"])
    
    ######################################################################
    # Compute S/N and A/N in all lines
    df = dqcut.compute_SN(df, ncomponents_max, eline_list)
    df = dqcut.compute_AN(df, ncomponents_max, eline_list)

    ######################################################################
    # DQ and S/N CUTS
    if "sigma_inst_kms" not in kwargs.keys():
        try:
            sigma_inst_kms = settings[survey]["sigma_inst_kms"]
        except KeyError:
            logger.warning(f"I could not find find sigma_inst_kms in settings[{survey}] or in kwargs, so I am assuming sigma_inst_kms = 0!")
            sigma_inst_kms = 0
    else:
        sigma_inst_kms = kwargs["sigma_inst_kms"]

    logger.info(f"setting & aplying data quality and S/N cuts...")
    df = dqcut.set_flags(df=df, 
                         eline_SNR_min=kwargs["eline_SNR_min"],
                         eline_ANR_min=kwargs["eline_ANR_min"],
                         eline_list=eline_list,
                         ncomponents_max=ncomponents_max,
                         sigma_inst_kms=settings[survey]["sigma_inst_kms"],
                         sigma_gas_SNR_min=kwargs["sigma_gas_SNR_min"]
                         )
    df = dqcut.apply_flags(df=df, 
                           ncomponents_max=ncomponents_max, 
                           eline_list=settings[survey]["eline_list"],
                           **kwargs)

    ######################################################################
    # Fix SFR columns
    # NaN the SFR surface density if the inclination is undefined
    if in_dataframe(df, "i (degrees)"):
        cond_NaN_inclination = np.isnan(df["i (degrees)"])
        cols = [c for c in df.columns if "SFR surface density" in c]
        df.loc[cond_NaN_inclination, cols] = np.nan

    # NaN the SFR if the SFR == 0
    # Note: I'm not entirely sure why there are spaxels with SFR == 0
    # in the first place.
    if in_dataframe(df, "SFR (total)"):
        cond_zero_SFR = df["SFR (total)"]  == 0
        cols = [c for c in df.columns if "SFR" in c]
        df.loc[cond_zero_SFR, cols] = np.nan

    # NaN out SFR quantities if the HALPHA flux is NaN
    # need to do this AFTER applying S/N and DQ cuts above.
    if in_dataframe(df, "HALPHA (total)"):
        cond_Ha_isnan = df["HALPHA (total)"].isna()
        cols_sfr = [c for c in df.columns if "SFR" in c]
        for col in cols_sfr:
            df.loc[cond_Ha_isnan, col] = np.nan

    ######################################################################
    # EXTINCTION CORRECTION
    # Compute A_V & correct emission line fluxes (but not EWs!)
    if kwargs["correct_extinction"]:
        logger.info(f"correcting emission line fluxes (but not EWs) for extinction...")
        # Compute A_V using total Halpha and Hbeta emission line fluxes
        df = extcorr.compute_A_V(df,
                                         reddening_curve="fm07",
                                         balmer_SNR_min=5,
                                         s=f" (total)")

        # Apply the extinction correction to total emission line fluxes
        df = extcorr.apply_extinction_correction(df,
                                        reddening_curve="fm07",
                                        eline_list=[e for e in eline_list if f"{e} (total)" in df],
                                        a_v_col_name="A_V (total)",
                                        nthreads=kwargs["nthreads"],
                                        s=f" (total)")

        # Apply the extinction correction to fluxes of  individual components
        for nn in range(ncomponents_max):
            df = extcorr.apply_extinction_correction(df,
                                            reddening_curve="fm07",
                                            eline_list=[e for e in eline_list if f"{e} (component {nn + 1})" in df],
                                            a_v_col_name="A_V (total)",
                                            nthreads=kwargs["nthreads"],
                                            s=f" (component {nn + 1})")

        df["correct_extinction"] = True
    else:
        logger.info(f"skipping extinction correction...")
        df["correct_extinction"] = False
    df = df.sort_index()

    ######################################################################
    # EVALUATE LINE RATIOS & SPECTRAL CLASSIFICATIONS
    logger.info(f"computing emission line ratios and BPT categories...")
    df = linefns.ratio_fn(df, s=f" (total)")
    df = linefns.bpt_fn(df, s=f" (total)")

    ######################################################################
    # EVALUATE ADDITIONAL COLUMNS - log quantites, etc.
    logger.info(f"computing additional quantities...")
    df = continuum.compute_continuum_luminosity(df, flux_units=settings[survey]["flux_units"])
    df = linefns.compute_eline_luminosity(df, ncomponents_max, eline_list=["HALPHA"], flux_units=settings[survey]["flux_units"])
    if compute_sfr:
        df = linefns.compute_SFR(df, ncomponents_max)
    df = linefns.compute_FWHM(df, ncomponents_max)
    df = misc.compute_gas_stellar_offsets(df, ncomponents_max)
    df = misc.compute_log_columns(df, ncomponents_max)
    df = misc.compute_component_offsets(df, ncomponents_max)

    ######################################################################
    # GEOMETRY
    if "r (relative to galaxy centre, deprojected, arcsec)" in df and "R_e (arcsec)" in df:
        df["r/R_e"] = df["r (relative to galaxy centre, deprojected, arcsec)"] / df[
                "R_e (arcsec)"]

    ######################################################################
    # EVALUATE METALLICITY (only for spaxels with extinction correction)
    logger.info(f"computing metallicities...")
    for diagnostic in kwargs["metallicity_diagnostics"]:        
        if diagnostic.endswith("K19"):
            df = metallicity.calculate_metallicity(met_diagnostic=diagnostic, compute_logU=True, ion_diagnostic="O3O2_K19", compute_errors=True, niters=1000, df=df, s=" (total)")
        elif diagnostic.endswith("KK04"):
            df = metallicity.calculate_metallicity(met_diagnostic=diagnostic, compute_logU=True, ion_diagnostic="O3O2_KK04", compute_errors=True, niters=1000, df=df, s=" (total)")
        else:
            df = metallicity.calculate_metallicity(met_diagnostic=diagnostic, compute_errors=True, niters=1000, df=df, s=" (total)")        

    ###############################################################################
    # Save input flags to the DataFrame
    logger.info(f"adding flags to DataFrame...")
    # TODO add all kwargs here instead? 
    for flag in ["eline_SNR_min", "eline_ANR_min", "sigma_gas_SNR_min",
                 "line_flux_SNR_cut", "missing_fluxes_cut", "missing_kinematics_cut", 
                 "line_amplitude_SNR_cut", "flux_fraction_cut", "vgrad_cut", 
                 "sigma_gas_SNR_cut", "stekin_cut"]:
        df[flag] = kwargs[flag]

    return df