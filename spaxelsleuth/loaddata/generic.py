"""
In here: take a "generic" DataFrame & calculate ALL THINGS that are NOT specific to certain surveys 
e.g. metallicities, extinction, etc. 

Input: Pandas DataFrame 

Output: same Pandas DataFrame but with additional columns added. 

Steps to include:
- Calculate equivalent widths
- Compute S/N in all lines
- Fix SFR columns
- DQ and S/N CUTS
- NaN out SFR quantities if the HALPHA flux is NaN 
- EXTINCTION CORRECTION
- EVALUATE LINE RATIOS & SPECTRAL CLASSIFICATIONS
- EVALUATE ADDITIONAL COLUMNS - log quantites, etc.
- EVALUATE METALLICITY (only for spaxels with extinction correction)
- Save input flags to the DataFrame so that we can keep track
- Save to .hd5 & .csv.
"""
import numpy as np

# TODO: import this from a config file 
NCOMPONENTS_MAX = 3

# TODO: figure out how to tidy up this import
from spaxelsleuth.utils import dqcut, linefns, metallicity, extcorr

#//////////////////////////////////////////////////////////////////////////////
def add_columns(df, **kwargs):
    """Computes quantities such as metallicities, extinctions, etc. for each row in df."""
    # TODO: before each of these, check that columns exist. Define a handy function for doing this.

    # Utility function for checking necessary columns
    def in_dataframe(cols) -> bool:
        """Returns True if all colums in cols are present in DataFrame df."""
        if type(cols) == list:
            return all([c in df for c in cols])
        elif type(cols) == str:
            return cols in df
        else:
            raise ValueError("cols must be str or list of str!")

    status_str = "In generic.add_columns():"

    # Component indices for emission line-derived quantities
    range_ncomponents_elines =\
        range(3 if kwargs["ncomponents"] == "recom" else 1) if not kwargs["__use_lzifu_fits"] \
        else range(3 if kwargs["__lzifu_ncomponents"] == "recom" else int(kwargs["__lzifu_ncomponents"]))        

    ###############################################################################
    # Calculate equivalent widths
    ###############################################################################
    if in_dataframe(["HALPHA continuum"]):
        # Zero out -ve continuum values
        df.loc[df["HALPHA continuum"] < 0, "HALPHA continuum"] = 0
    
        # Compute EW in each component
        for nn in range(NCOMPONENTS_MAX):
            if in_dataframe(f"HALPHA (component {nn + 1})"):
                df[f"HALPHA EW (component {nn + 1})"] = df[f"HALPHA (component {nn + 1})"] / df["HALPHA continuum"]
                # Compute associated errors 
                if in_dataframe([f"HALPHA error (component {nn + 1})", "HALPHA continuum error"]):
                    df[f"HALPHA EW error (component {nn + 1})"] = df[f"HALPHA EW (component {nn + 1})"] *\
                        np.sqrt((df[f"HALPHA error (component {nn + 1})"] / df[f"HALPHA (component {nn + 1})"])**2 +\
                                (df[f"HALPHA continuum error"] / df[f"HALPHA continuum"])**2) 
            
                # If the continuum level <= 0, then the EW is undefined, so set to NaN.
                df.loc[df["HALPHA continuum"] <= 0, [f"HALPHA EW (component {nn + 1})"]] = np.nan  
                if in_dataframe([f"HALPHA EW error (component {nn + 1})"]):
                    df.loc[df["HALPHA continuum"] <= 0, [f"HALPHA EW error (component {nn + 1})"]] = np.nan  

        # Calculate total EW
        if in_dataframe("HALPHA (total)"):
            df[f"HALPHA EW (total)"] = df[f"HALPHA (total)"] / df["HALPHA continuum"]
            if in_dataframe(["HALPHA error (total)", "HALPHA continuum error"]):
                df[f"HALPHA EW error (total)"] = df[f"HALPHA EW (total)"] *\
                    np.sqrt((df[f"HALPHA error (total)"] / df[f"HALPHA (total)"])**2 +\
                            (df[f"HALPHA continuum error"] / df[f"HALPHA continuum"])**2) 
            
            # If the continuum level <= 0, then the EW is undefined, so set to NaN.
            df.loc[df["HALPHA continuum"] <= 0, [f"HALPHA EW (total)", f"HALPHA EW error (total)"]] = np.nan  

    ######################################################################
    # Compute S/N in all lines
    ######################################################################
    for eline in kwargs["eline_list"]:
        # Compute S/N 
        for nn in range(NCOMPONENTS_MAX):
            if in_dataframe([f"{eline} (component {nn + 1})", f"{eline} error (component {nn + 1})"]):
                df[f"{eline} S/N (component {nn + 1})"] = df[f"{eline} (component {nn + 1})"] / df[f"{eline} error (component {nn + 1})"]
        
        # Compute the S/N in the TOTAL line flux
        if in_dataframe([f"{eline} (total)", f"{eline} error (total)"]):
            df[f"{eline} S/N (total)"] = df[f"{eline} (total)"] / df[f"{eline} error (total)"]

    ######################################################################
    # Fix SFR columns
    ######################################################################
    # NaN the SFR surface density if the inclination is undefined
    if in_dataframe("i (degrees)"):
        cond_NaN_inclination = np.isnan(df["i (degrees)"])
        cols = [c for c in df.columns if "SFR surface density" in c]
        df.loc[cond_NaN_inclination, cols] = np.nan

    # NaN the SFR if the SFR == 0
    # Note: I'm not entirely sure why there are spaxels with SFR == 0
    # in the first place.
    if in_dataframe("SFR (total)"):
        cond_zero_SFR = df["SFR (total)"]  == 0
        cols = [c for c in df.columns if "SFR" in c]
        df.loc[cond_zero_SFR, cols] = np.nan

    ######################################################################
    # DQ and S/N CUTS
    ######################################################################
    df = dqcut.set_flags(df=df, 
                  eline_SNR_min=kwargs["eline_SNR_min"], eline_list=kwargs["eline_list"],
                  sigma_gas_SNR_min=kwargs["sigma_gas_SNR_min"],
                  sigma_inst_kms=29.6)

    # Apply S/N and DQ cuts
    df = dqcut.apply_flags(df=df, 
                                   line_flux_SNR_cut=kwargs["line_flux_SNR_cut"],
                                   missing_fluxes_cut=kwargs["missing_fluxes_cut"],
                                   line_amplitude_SNR_cut=kwargs["line_amplitude_SNR_cut"],
                                   flux_fraction_cut=kwargs["flux_fraction_cut"],
                                   vgrad_cut=kwargs["vgrad_cut"],
                                   sigma_gas_SNR_cut=kwargs["sigma_gas_SNR_cut"],
                                   stekin_cut=kwargs["stekin_cut"],
                                   eline_list=kwargs["eline_list"])    

    ######################################################################
    # NaN out SFR quantities if the HALPHA flux is NaN
    # need to do this AFTER applying S/N and DQ cuts above.
    ###################################################################### 
    if in_dataframe("HALPHA (total)"):
        cond_Ha_isnan = df["HALPHA (total)"].isna()
        cols_sfr = [c for c in df.columns if "SFR" in c]
        for col in cols_sfr:
            df.loc[cond_Ha_isnan, col] = np.nan
    
    ######################################################################
    # Make a copy of the DataFrame with EXTINCTION CORRECTION
    # Correct emission line fluxes (but not EWs!)
    # NOTE: extinction.fm07 assumes R_V = 3.1 so do not change R_V from 
    # this value!!!
    ######################################################################
    if kwargs["correct_extinction"]:
        print(f"{status_str}: Correcting emission line fluxes (but not EWs) for extinction...")
        # Compute A_V using total Halpha and Hbeta emission line fluxes
        df = extcorr.compute_A_V(df,
                                         reddening_curve="fm07", 
                                         balmer_SNR_min=5,
                                         s=f" (total)")

        # Apply the extinction correction to total emission line fluxes
        df = extcorr.apply_extinction_correction(df, 
                                        reddening_curve="fm07", 
                                        eline_list=[e for e in kwargs["eline_list"] if f"{e} (total)" in df],
                                        a_v_col_name="A_V (total)",
                                        nthreads=kwargs["nthreads_max"],
                                        s=f" (total)")
        
        # Apply the extinction correction to fluxes of  individual components
        for nn in range(NCOMPONENTS_MAX):
            df = extcorr.apply_extinction_correction(df, 
                                            reddening_curve="fm07", 
                                            eline_list=[e for e in kwargs["eline_list"] if f"{e} (component {nn + 1})" in df],
                                            a_v_col_name="A_V (total)",
                                            nthreads=kwargs["nthreads_max"],
                                            s=f" (component {nn + 1})")

        df["Corrected for extinction?"] = True
    else:
        df["Corrected for extinction?"] = False
    df = df.sort_index()

    ######################################################################
    # EVALUATE LINE RATIOS & SPECTRAL CLASSIFICATIONS
    ######################################################################
    df = linefns.ratio_fn(df, s=f" (total)")
    df = linefns.bpt_fn(df, s=f" (total)")

    ######################################################################
    # EVALUATE ADDITIONAL COLUMNS - log quantites, etc.
    ######################################################################
    df = dqcut.compute_extra_columns(df)

    ######################################################################
    # EVALUATE METALLICITY (only for spaxels with extinction correction)
    ######################################################################
    if not kwargs["debug"]:
        df = metallicity.calculate_metallicity(met_diagnostic="N2Ha_PP04", compute_errors=True, niters=1000, df=df, s=" (total)")
        df = metallicity.calculate_metallicity(met_diagnostic="N2Ha_M13", compute_errors=True, niters=1000, df=df, s=" (total)")
        df = metallicity.calculate_metallicity(met_diagnostic="O3N2_PP04", compute_errors=True, niters=1000, df=df, s=" (total)")
        df = metallicity.calculate_metallicity(met_diagnostic="O3N2_M13", compute_errors=True, niters=1000, df=df, s=" (total)")
        df = metallicity.calculate_metallicity(met_diagnostic="N2S2Ha_D16", compute_errors=True, niters=1000, df=df, s=" (total)")
        df = metallicity.calculate_metallicity(met_diagnostic="N2O2_KD02", compute_errors=True, niters=1000, df=df, s=" (total)")
        df = metallicity.calculate_metallicity(met_diagnostic="Rcal_PG16", compute_errors=True, niters=1000, df=df, s=" (total)")
        df = metallicity.calculate_metallicity(met_diagnostic="Scal_PG16", compute_errors=True, niters=1000, df=df, s=" (total)")
        df = metallicity.calculate_metallicity(met_diagnostic="ON_P10", compute_errors=True, niters=1000, df=df, s=" (total)")
        df = metallicity.calculate_metallicity(met_diagnostic="ONS_P10", compute_errors=True, niters=1000, df=df, s=" (total)")
        df = metallicity.calculate_metallicity(met_diagnostic="N2Ha_K19", compute_logU=True, ion_diagnostic="O3O2_K19", compute_errors=True, niters=1000, df=df, s=" (total)")
        df = metallicity.calculate_metallicity(met_diagnostic="O3N2_K19", compute_logU=True, ion_diagnostic="O3O2_K19", compute_errors=True, niters=1000, df=df, s=" (total)")
        df = metallicity.calculate_metallicity(met_diagnostic="N2O2_K19", compute_logU=True, ion_diagnostic="O3O2_K19", compute_errors=True, niters=1000, df=df, s=" (total)")
        df = metallicity.calculate_metallicity(met_diagnostic="R23_KK04", compute_logU=True, ion_diagnostic="O3O2_KK04", compute_errors=True, niters=1000, df=df, s=" (total)")
    else:
        df = metallicity.calculate_metallicity(met_diagnostic="N2Ha_PP04", compute_errors=True, niters=1000, df=df, s=" (total)")
        df = metallicity.calculate_metallicity(met_diagnostic="N2Ha_K19", compute_logU=True, ion_diagnostic="O3O2_K19", compute_errors=True, niters=1000, df=df, s=" (total)")

    ###############################################################################
    # Save input flags to the DataFrame so that we can keep track
    ###############################################################################
    df["eline_SNR_min"] = kwargs["eline_SNR_min"]
    df["sigma_gas_SNR_min"] = kwargs["sigma_gas_SNR_min"]
    df["Extinction correction applied"] = kwargs["correct_extinction"]
    df["line_flux_SNR_cut"] = kwargs["line_flux_SNR_cut"]
    df["missing_fluxes_cut"] = kwargs["missing_fluxes_cut"]
    df["line_amplitude_SNR_cut"] = kwargs["line_amplitude_SNR_cut"]
    df["flux_fraction_cut"] = kwargs["flux_fraction_cut"]
    df["vgrad_cut"] = kwargs["vgrad_cut"]
    df["sigma_gas_SNR_cut"] = kwargs["sigma_gas_SNR_cut"]
    df["stekin_cut"] = kwargs["stekin_cut"]
    
    # TODO: why is this here?
    for col in [c for c in df.columns if "SFR" in c and "error" not in c]:
        assert f"{col}" in df.columns
        assert all(df.loc[df["HALPHA (total)"].isna(), f"{col}"].isna())

    return df