# Imports
import pandas as pd
from pathlib import Path
import pkgutil
import numpy as np

from spaxelsleuth.config import settings
from spaxelsleuth.utils import linefns, metallicity, extcorr

import logging
logger = logging.getLogger(__name__)

###############################################################################
# Paths
input_path = Path(settings["sami"]["input_path"])
output_path = Path(settings["sami"]["output_path"])


###############################################################################
def make_sami_aperture_df(eline_SNR_min,
                          line_flux_SNR_cut=True,
                          missing_fluxes_cut=True,
                          sigma_gas_SNR_cut=True,
                          sigma_gas_SNR_min=3,
                          metallicity_diagnostics=[
                              "N2Ha_PP04",
                              "N2Ha_M13",
                              "O3N2_PP04",
                              "O3N2_M13",
                              "N2S2Ha_D16",
                              "N2O2_KD02",
                              "Rcal_PG16",
                              "Scal_PG16",
                              "ON_P10",
                              "ONS_P10",
                              "N2Ha_K19",
                              "O3N2_K19",
                              "N2O2_K19",
                              "R23_KK04",
                          ],
                          nthreads=20,
                          correct_extinction=True):
    """
    This is a convenience function for extracting the data products obtained 
    from the single-component emission line fits to the aperture spectra 
    that is available through DataCentral in table "EmissionLine1compDR3". 
    Various quantities are computed using the data products, such as A_V and 
    emission line ratios. The data products and computed quantities are saved 
    to a DataFrame. 

    USAGE
    ---------------------------------------------------------------------------
            
            >>> from spaxelsleuth.loaddata.sami import make_sami_aperture_df
            >>> make_sami_aperture_df(eline_SNR_min=5)

    INPUTS
    ---------------------------------------------------------------------------
    eline_SNR_min:      int 
        Minimum emission line flux S/N to assume.

    line_flux_SNR_cut:          bool
        If True, make a S/N cut on all emission line components and associated
        quantities with flux S/N below eline_SNR_min.

    missing_fluxes_cut:         bool
        If True, mask out emission line fluxes and associated quantities when 
        either the flux or the error on the flux are NaN.

    sigma_gas_SNR_cut:          bool
        If True, mask component velocity dispersions where the S/N on the 
        velocity dispersion measurement is below sigma_gas_SNR_min. 

    sigma_gas_SNR_min:          int
        Minimum velocity dipersion S/N to accept.

    correct_extinction:         bool 
        If True, correct emission line fluxes for extinction.

    nthreads:                   int            
        Maximum number of threads to use during extinction correction 
        calculation.    

    OUTPUTS
    ---------------------------------------------------------------------------
    The DataFrame is saved to 

        settings["sami"]["output_path"]/sami_apertures_<correct_extinction>_minSNR=<eline_SNR_min>.hd5

    """
    #######################################################################
    # FILENAMES
    df_metadata_fname = "sami_dr3_metadata.hd5"

    # Output file names
    if correct_extinction:
        df_fname = f"sami_apertures_extcorr_minSNR={eline_SNR_min}.hd5"
    else:
        df_fname = f"sami_apertures_minSNR={eline_SNR_min}.hd5"
    logger.info(f"saving to files {df_fname}...")

    ###########################################################################
    # Open the .csv file containing the table
    data_path = Path(pkgutil.get_loader(__name__).get_filename()).parent.parent / "data"

    # Emission line info
    df_ap_elines = pd.read_csv(data_path / "sami_EmissionLine1compDR3.csv")
    df_ap_elines = df_ap_elines.set_index("catid").drop("Unnamed: 0", axis=1)
    df_ap_elines = df_ap_elines.rename(columns={"catid": "ID"})
    logger.info(f"df_ap_elines has {len(df_ap_elines.index.unique())} galaxies")

    # SSP info
    df_ap_ssp = pd.read_csv(data_path / "sami_SSPAperturesDR3.csv")
    df_ap_ssp = df_ap_ssp.set_index("catid").drop("Unnamed: 0", axis=1)
    df_ap_ssp = df_ap_ssp.rename(columns={"catid": "ID"})
    logger.info(f"df_ap_ssp has {len(df_ap_ssp.index.unique())} galaxies")

    # Stellar indices
    df_ap_indices = pd.read_csv(data_path / "sami_IndexAperturesDR3.csv")
    df_ap_indices = df_ap_indices.set_index("catid").drop("Unnamed: 0", axis=1)
    df_ap_indices = df_ap_indices.rename(columns={"catid": "ID"})
    logger.info(f"df_ap_indices has {len(df_ap_indices.index.unique())} galaxies")

    # Merge
    df_ap = df_ap_elines.merge(df_ap_ssp,
                               how="outer",
                               left_index=True,
                               right_index=True).drop(["cubeid_x", "cubeid_y"],
                                                      axis=1)
    stellar_idx_cols = [c for c in df_ap_indices if c not in df_ap]
    df_ap = df_ap.merge(df_ap_indices[stellar_idx_cols],
                        how="outer",
                        left_index=True,
                        right_index=True)
    logger.info(f"after merging, df_ap has {len(df_ap.index.unique())} galaxies")

    # Drop duplicate rows
    df_ap = df_ap[~df_ap.index.duplicated(keep="first")]

    ###########################################################################
    # Merge with metadata DataFrame
    df_metadata = pd.read_hdf(output_path / df_metadata_fname, key="metadata")
    logger.info(
        f"before merging, there are {len([g for g in df_metadata.index if g not in df_ap.index])} galaxies in df_metadata that are missing from df_ap"
    )
    logger.info(
        f"before merging, there are {len([g for g in df_ap.index if g not in df_metadata.index])} galaxies in df_ap that are missing from df_metadata"
    )
    df_ap = df_ap.merge(df_metadata,
                        how="outer",
                        left_index=True,
                        right_index=True)

    ###########################################################################
    # Rename columns
    for ap in [
            "_1_4_arcsecond", "_2_arcsecond", "_3_arcsecond", "_4_arcsecond",
            "_re_mge", "_re", "_3kpc_round"
    ]:
        cols_ap = [col for col in df_ap.columns if ap in col]
        rename_dict = {}
        for old_col in cols_ap:
            if old_col.endswith("_err"):
                new_col = old_col.split(
                    ap)[0].upper() + f" error ({ap.replace('_', ' ')[1:]})"
            else:
                new_col = old_col.split(
                    ap)[0].upper() + f" ({ap.replace('_', ' ')[1:]})"
            rename_dict[old_col] = new_col
            logger.info(f"{old_col} --> {new_col}")
        df_ap = df_ap.rename(columns=rename_dict)

    old_cols_v = [col for col in df_ap.columns if "V_GAS" in col]
    new_cols_v = [col.replace("V_GAS", "v_gas") for col in old_cols_v]
    old_cols_sigma = [col for col in df_ap.columns if "VDISP_GAS" in col]
    new_cols_sigma = [
        col.replace("VDISP_GAS", "sigma_gas") for col in old_cols_sigma
    ]
    rename_dict = dict(
        zip(old_cols_v + old_cols_sigma, new_cols_v + new_cols_sigma))
    df_ap = df_ap.rename(columns=rename_dict)

    old_cols_re = [col for col in df_ap.columns if "(re)" in col]
    new_cols_re = [col.replace("(re)", "(R_e)") for col in old_cols_re]
    old_cols_mge = [col for col in df_ap.columns if "mge)" in col]
    new_cols_mge = [
        col.replace("(re mge)", "(R_e (MGE))") for col in old_cols_mge
    ]
    rename_dict = dict(
        zip(old_cols_re + old_cols_mge, new_cols_re + new_cols_mge))
    df_ap = df_ap.rename(columns=rename_dict)

    old_cols_14 = [col for col in df_ap.columns if "(1 4" in col]
    new_cols_14 = [col.replace("(1 4", "(1.4") for col in old_cols_14]
    rename_dict = dict(zip(old_cols_14, new_cols_14))
    df_ap = df_ap.rename(columns=rename_dict)

    # Rename stellar age/metallicity measurements
    old_cols_ssp = [col for col in df_ap.columns if "AGE" in col] +\
                   [col for col in df_ap.columns if "Z" in col] +\
                   [col for col in df_ap.columns if "ALPHA" in col and "HALPHA" not in col]
    new_cols_ssp = [col.replace("AGE", "Stellar age (Gyr)") for col in [col for col in df_ap.columns if "AGE" in col]] +\
                   [col.replace("Z", "Stellar [Z/H]") for col in [col for col in df_ap.columns if "Z" in col]] +\
                   [col.replace("ALPHA", "Stellar [alpha/Fe]") for col in [col for col in df_ap.columns if "ALPHA" in col and "HALPHA" not in col]]
    rename_dict = dict(zip(old_cols_ssp, new_cols_ssp))
    df_ap = df_ap.rename(columns=rename_dict)

    # Rename stellar indices
    old_cols_idxs = [col for col in df_ap.columns if "HDELTAA" in col] +\
                    [col for col in df_ap.columns if "HDELTAF" in col] +\
                    [col for col in df_ap.columns if "HGAMMAA" in col] +\
                    [col for col in df_ap.columns if "HGAMMAF" in col] +\
                    [col for col in df_ap.columns if "MGB" in col] +\
                    [col for col in df_ap.columns if "FE4383" in col] +\
                    [col for col in df_ap.columns if "FE4668" in col] +\
                    [col for col in df_ap.columns if "FE5015" in col] +\
                    [col for col in df_ap.columns if "FE5270" in col] +\
                    [col for col in df_ap.columns if "FE5335" in col] +\
                    [col for col in df_ap.columns if "CN1" in col] +\
                    [col for col in df_ap.columns if "CN2" in col] +\
                    [col for col in df_ap.columns if "CA4227" in col] +\
                    [col for col in df_ap.columns if "G4300" in col] +\
                    [col for col in df_ap.columns if "MG1" in col] +\
                    [col for col in df_ap.columns if "MG2" in col] +\
                    [col for col in df_ap.columns if "FE4531" in col] +\
                    [col for col in df_ap.columns if "FE5406" in col] +\
                    [col for col in df_ap.columns if "CA4455" in col]
    new_cols_idxs = [col.replace("HDELTAA", "HDELTA_A") for col in [col for col in df_ap.columns if "HDELTAA" in col]] +\
                    [col.replace("HDELTAF", "HDELTA_F") for col in [col for col in df_ap.columns if "HDELTAF" in col]] +\
                    [col.replace("HGAMMAA", "HGAMMA_A") for col in [col for col in df_ap.columns if "HGAMMAA" in col]] +\
                    [col.replace("HGAMMAF", "HGAMMA_F") for col in [col for col in df_ap.columns if "HGAMMAF" in col]] +\
                    [col.replace("MGB", "Mg_b") for col in [col for col in df_ap.columns if "MGB" in col]] +\
                    [col.replace("FE4383", "Fe_4383") for col in [col for col in df_ap.columns if "FE4383" in col]] +\
                    [col.replace("FE4668", "Fe_4668") for col in [col for col in df_ap.columns if "FE4668" in col]] +\
                    [col.replace("FE5015", "Fe_5015") for col in [col for col in df_ap.columns if "FE5015" in col]] +\
                    [col.replace("FE5270", "Fe_5270") for col in [col for col in df_ap.columns if "FE5270" in col]] +\
                    [col.replace("FE5335", "Fe_5335") for col in [col for col in df_ap.columns if "FE5335" in col]] +\
                    [col.replace("CN1", "CN_1") for col in [col for col in df_ap.columns if "CN1" in col]] +\
                    [col.replace("CN2", "CN_2") for col in [col for col in df_ap.columns if "CN2" in col]] +\
                    [col.replace("CA4227", "Ca_4227") for col in [col for col in df_ap.columns if "CA4227" in col]] +\
                    [col.replace("G4300", "G_4300") for col in [col for col in df_ap.columns if "G4300" in col]] +\
                    [col.replace("MG1", "Mg_1") for col in [col for col in df_ap.columns if "MG1" in col]] +\
                    [col.replace("MG2", "Mg_2") for col in [col for col in df_ap.columns if "MG2" in col]] +\
                    [col.replace("FE4531", "Fe_4531") for col in [col for col in df_ap.columns if "FE4531" in col]] +\
                    [col.replace("FE5406", "Fe_5406") for col in [col for col in df_ap.columns if "FE5406" in col]] +\
                    [col.replace("CA4455", "Ca_4455") for col in [col for col in df_ap.columns if "CA4455" in col]]
    rename_dict = dict(zip(old_cols_idxs, new_cols_idxs))
    df_ap = df_ap.rename(columns=rename_dict)

    ######################################################################
    # Compute SFR surface densities & log quantities
    for ap in [
            "1.4 arcsecond", "2 arcsecond", "3 arcsecond", "4 arcsecond",
            "R_e", "R_e (MGE)", "3kpc round"
    ]:
        if ap.endswith("arcsecond"):
            r_kpc = float(ap.split(" arcsecond")[0]) * df_ap["kpc per arcsec"]
        elif ap == "3kpc round":
            r_kpc = 3
        elif ap == "R_e (MGE)":
            r_kpc = df_ap["R_e (MGE) (kpc)"]
        elif ap == "R_e":
            r_kpc = df_ap["R_e (kpc)"]
        A_kpc2 = np.pi * r_kpc**2

        # SFR surface density
        df_ap[f"SFR surface density ({ap})"] = df_ap[f"SFR ({ap})"] / A_kpc2
        df_ap[f"SFR surface density error ({ap})"] = df_ap[
            f"SFR error ({ap})"] / A_kpc2

        # Log SFR surface density
        df_ap[f"log SFR surface density ({ap})"] = np.log10(
            df_ap[f"SFR surface density ({ap})"])
        df_ap[f"log SFR surface density error (upper) ({ap})"] = np.log10(
            df_ap[f"SFR surface density ({ap})"] +
            df_ap[f"SFR surface density error ({ap})"]
        ) - df_ap[f"log SFR surface density ({ap})"]
        df_ap[f"log SFR surface density error (lower) ({ap})"] = df_ap[
            f"log SFR surface density ({ap})"] - np.log10(
                df_ap[f"SFR surface density ({ap})"] -
                df_ap[f"SFR surface density error ({ap})"])

        # log SFR
        df_ap[f"log SFR ({ap})"] = np.log10(df_ap[f"SFR ({ap})"])
        df_ap[f"log SFR error (upper) ({ap})"] = np.log10(
            df_ap[f"SFR ({ap})"] +
            df_ap[f"SFR error ({ap})"]) - df_ap[f"log SFR ({ap})"]
        df_ap[f"log SFR error (lower) ({ap})"] = df_ap[
            f"log SFR ({ap})"] - np.log10(df_ap[f"SFR ({ap})"] -
                                          df_ap[f"SFR error ({ap})"])

    ######################################################################
    # Compute specific SFRs
    for ap in [
            "1.4 arcsecond", "2 arcsecond", "3 arcsecond", "4 arcsecond",
            "R_e", "R_e (MGE)", "3kpc round"
    ]:
        df_ap[f"log sSFR ({ap})"] = df_ap[f"log SFR ({ap})"] - df_ap["log M_*"]

    ######################################################################
    # Compute S/N for each emission line in each aperture
    aps = [
        "1.4 arcsecond", "2 arcsecond", "3 arcsecond", "4 arcsecond", "R_e",
        "R_e (MGE)", "3kpc round"
    ]
    eline_list = [
        "OII3726", "OII3729", "NEIII3869", "HEPSILON", "HDELTA", "HGAMMA",
        "OIII4363", "HBETA", "OIII5007", "OI6300", "HALPHA", "NII6583",
        "SII6716", "SII6731"
    ]

    for eline in eline_list:
        # Compute S/N
        for ap in aps:
            if f"{eline} ({ap})" in df_ap.columns:
                df_ap[f"{eline} S/N ({ap})"] = df_ap[
                    f"{eline} ({ap})"] / df_ap[f"{eline} error ({ap})"]

    ######################################################################
    # Initialise DQ and S/N flags
    for ap in aps:
        df_ap[f"Low sigma_gas S/N flag ({ap})"] = False
        for eline in eline_list:
            df_ap[f"Missing flux flag - {eline} ({ap})"] = False
            df_ap[f"Low flux S/N flag - {eline} ({ap})"] = False

    ######################################################################
    # Fix SFR columns
    # NaN the SFR if the SFR == 0
    for ap in aps:
        cond_zero_SFR = df_ap[f"SFR ({ap})"] == 0
        cols = [c for c in df_ap.columns if "SFR" in c and f"({ap})" in c]
        df_ap.loc[cond_zero_SFR, cols] = np.nan

    ######################################################################
    # DQ and S/N CUTS
    ######################################################################
    # Flag low S/N emission lines
    logger.info(f"flagging low S/N components and spaxels...")
    for eline in eline_list:
        # Fluxes in individual components
        for ap in aps:
            if f"{eline} ({ap})" in df_ap.columns:
                cond_low_SN = df_ap[f"{eline} S/N ({ap})"] < eline_SNR_min
                df_ap.loc[cond_low_SN,
                          f"Low flux S/N flag - {eline} ({ap})"] = True

    ######################################################################
    # Flag emission lines with "missing" (i.e. NaN) fluxes in which the
    # ERROR on the flux is not NaN
    logger.info(
        f"flagging components and galaxies with NaN fluxes and finite errors..."
    )
    for eline in eline_list:
        # Fluxes in individual components
        for ap in aps:
            if f"{eline} ({ap})" in df_ap.columns:
                cond_missing_flux = df_ap[f"{eline} ({ap})"].isna(
                ) & ~df_ap[f"{eline} error ({ap})"].isna()
                df_ap.loc[cond_missing_flux,
                          f"Missing flux flag - {eline} ({ap})"] = True
                logger.info(
                    f"{eline} ({ap}): {df_ap[cond_missing_flux].shape[0]:d} galaxies have missing fluxes in this component"
                )

    ######################################################################
    # Flag rows with insufficient S/N in sigma_gas
    logger.info(f"flagging components with low sigma_gas S/N...")
    sigma_gas_SNR_min = 3
    # Gas kinematics: NaN out cells w/ sigma_gas S/N ratio < sigma_gas_SNR_min
    # (For SAMI, the red arm resolution is 29.6 km/s - see p6 of Croom+2021)
    for ap in aps:
        # 1. Define sigma_obs = sqrt(sigma_gas**2 + sigma_inst_kms**2).
        df_ap[f"sigma_obs ({ap})"] = np.sqrt(
            df_ap[f"sigma_gas ({ap})"]**2 +
            settings["sami"]["sigma_inst_kms"]**2)

        # 2. Define the S/N ratio of sigma_obs.
        # NOTE: here we assume that sigma_gas error (as output by LZIFU)
        # really refers to the error on sigma_obs.
        df_ap[f"sigma_obs S/N ({ap})"] = df_ap[f"sigma_obs ({ap})"] / df_ap[
            f"sigma_gas error ({ap})"]

        # 3. Given our target SNR_gas, compute the target SNR_obs,
        # using the method in section 2.2.2 of Zhou+2017.
        df_ap[f"sigma_obs target S/N ({ap})"] = sigma_gas_SNR_min * (
            1 + settings["sami"]["sigma_inst_kms"]**2 /
            df_ap[f"sigma_gas ({ap})"]**2)
        cond_bad_sigma = df_ap[f"sigma_obs S/N ({ap})"] < df_ap[
            f"sigma_obs target S/N ({ap})"]
        df_ap.loc[cond_bad_sigma, f"Low sigma_gas S/N flag ({ap})"] = True

    ######################################################################
    # NaN out offending cells
    if line_flux_SNR_cut:
        logger.info(
            f"masking components that don't meet the S/N requirements..."
        )
        for eline in eline_list:
            # Individual fluxes
            for ap in aps:
                cond_low_SN = df_ap[f"Low flux S/N flag - {eline} ({ap})"]

                # Cells to NaN
                if eline == "HALPHA":
                    # Then NaN out EVERYTHING associated with this component - if we can't trust HALPHA then we probably can't trust anything else either!
                    cols_low_SN = [
                        c for c in df_ap.columns
                        if f"({ap})" in c and "flag" not in c
                    ]
                else:
                    cols_low_SN = [
                        c for c in df_ap.columns
                        if eline in c and f"({ap})" in c and "flag" not in c
                    ]
                df_ap.loc[cond_low_SN, cols_low_SN] = np.nan

    if missing_fluxes_cut:
        logger.info(f"masking components with missing fluxes...")
        for eline in eline_list:
            # Individual fluxes
            for ap in aps:
                cond_missing_flux = df_ap[
                    f"Missing flux flag - {eline} ({ap})"]

                # Cells to NaN
                if eline == "HALPHA":
                    # Then NaN out EVERYTHING associated with this component - if we can't trust HALPHA then we probably can't trust anything else either!
                    cols_missing_fluxes = [
                        c for c in df_ap.columns
                        if f"({ap})" in c and "flag" not in c
                    ]
                else:
                    cols_missing_fluxes = [
                        c for c in df_ap.columns
                        if eline in c and f"({ap})" in c and "flag" not in c
                    ]
                df_ap.loc[cond_missing_flux, cols_missing_fluxes] = np.nan

    if sigma_gas_SNR_cut:
        logger.info(
            f"masking components with insufficient S/N in sigma_gas..."
        )
        for ap in aps:
            cond_bad_sigma = df_ap[f"Low sigma_gas S/N flag ({ap})"]

            # Cells to NaN
            cols_sigma_gas_SNR_cut = [
                c for c in df_ap.columns
                if f"({ap})" in c and "sigma_gas" in c and "flag" not in c
            ]
            df_ap.loc[cond_bad_sigma, cols_sigma_gas_SNR_cut] = np.nan

    ######################################################################
    # Correct emission line fluxes for extinction (but not EWs!)
    if correct_extinction:
        logger.info(
            f"correcting emission line fluxes (but not EWs) for extinction..."
        )
        for ap in aps:
            # Compute A_V using total Halpha and Hbeta emission line fluxes
            df_ap = extcorr.compute_A_V(df_ap,
                                              reddening_curve="fm07",
                                              balmer_SNR_min=5,
                                              s=f" ({ap})")

            # Apply the extinction correction to emission line fluxes
            df_ap = extcorr.apply_extinction_correction(
                df_ap,
                reddening_curve="fm07",
                eline_list=[e for e in eline_list if f"{e} ({ap})" in df_ap],
                a_v_col_name=f"A_V ({ap})",
                nthreads=nthreads,
                s=f" ({ap})")
    df_ap["correct_extinction"] = correct_extinction
    df_ap = df_ap.sort_index()

    ######################################################################
    # EVALUATE LINE RATIOS & SPECTRAL CLASSIFICATIONS
    for ap in aps:
        df_ap = linefns.ratio_fn(df_ap, s=f" ({ap})")
        df_ap = linefns.bpt_fn(df_ap, s=f" ({ap})")

    ######################################################################
    # EVALUATE METALLICITY
    for ap in aps:
        for diagnostic in metallicity_diagnostics:
            if diagnostic.endswith("K19"):
                df_ap = metallicity.calculate_metallicity(
                    met_diagnostic=diagnostic,
                    compute_logU=True,
                    ion_diagnostic="O3O2_K19",
                    compute_errors=True,
                    niters=1000,
                    df=df_ap,
                    s=f" ({ap})")
            elif diagnostic.endswith("KK04"):
                df_ap = metallicity.calculate_metallicity(
                    met_diagnostic=diagnostic,
                    compute_logU=True,
                    ion_diagnostic="O3O2_KK04",
                    compute_errors=True,
                    niters=1000,
                    df=df_ap,
                    s=f" ({ap})")
            else:
                df_ap = metallicity.calculate_metallicity(
                    met_diagnostic=diagnostic,
                    compute_errors=True,
                    niters=1000,
                    df=df_ap,
                    s=f" ({ap})")

    ###############################################################################
    # Save input flags to the DataFrame so that we can keep track
    df_ap["line_flux_SNR_cut"] = line_flux_SNR_cut
    df_ap["eline_SNR_min"] = eline_SNR_min
    df_ap["missing_fluxes_cut"] = missing_fluxes_cut
    df_ap["sigma_gas_SNR_min"] = sigma_gas_SNR_min
    df_ap["sigma_gas_SNR_cut"] = sigma_gas_SNR_cut

    ###############################################################################
    # Save to .hd5 & .csv
    logger.info(f"saving to file {output_path / df_fname}...")
    df_ap.to_hdf(output_path / df_fname, key=f"1-comp aperture fit")
    logger.info(f"finished!")
    return