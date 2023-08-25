# Imports
import datetime
import multiprocessing
import numpy as np
import os
import pandas as pd
from pathlib import Path
import warnings

from astropy.cosmology import FlatLambdaCDM
from astropy.io import fits

from spaxelsleuth.config import settings
from spaxelsleuth.utils.continuum import compute_d4000, compute_continuum_intensity
from spaxelsleuth.utils.geometry import deproject_coordinates
from spaxelsleuth.utils.dqcut import compute_HALPHA_amplitude_to_noise
from spaxelsleuth.utils.velocity import compute_v_grad
from spaxelsleuth.utils.addcolumns import add_columns
from spaxelsleuth.utils.misc import morph_dict, morph_num_to_str
from spaxelsleuth.utils.linefns import bpt_num_to_str

import logging
logger = logging.getLogger(__name__)

###############################################################################
# Paths
input_path = Path(settings["sami"]["input_path"])
output_path = Path(settings["sami"]["output_path"])
data_cube_path = Path(settings["sami"]["data_cube_path"])
__lzifu_products_path = Path(settings["sami"]["lzifu_products_path"])


###############################################################################
# For computing median continuum S/N values in make_sami_metadata_df()
def _compute_snr(args, plotit=False):
    gal, df_metadata = args

    # Load the red & blue data cubes.
    try:
        hdulist_R_cube = fits.open(
            Path(data_cube_path) / f"ifs/{gal}/{gal}_A_cube_red.fits.gz")
        hdulist_B_cube = fits.open(
            Path(data_cube_path) / f"ifs/{gal}/{gal}_A_cube_blue.fits.gz")
    except FileNotFoundError:
        logger.warning(f"data cubes not found for galaxy {gal} - cannot compute S/N")
        return [
            gal, np.nan, np.nan, np.nan, np.nan, np.nan,
            np.nan, np.nan, np.nan
        ]
    data_cube_B = hdulist_B_cube[0].data
    var_cube_B = hdulist_B_cube[1].data
    data_cube_R = hdulist_R_cube[0].data
    var_cube_R = hdulist_R_cube[1].data
    hdulist_R_cube.close()
    hdulist_B_cube.close()

    # Compute an image showing the median S/N in each spaxel.
    im_SNR_B = np.nanmedian(data_cube_B / np.sqrt(var_cube_B), axis=0)
    im_SNR_R = np.nanmedian(data_cube_R / np.sqrt(var_cube_R), axis=0)

    #######################################################################
    # Use R_e to compute the median S/N within 1, 1.5, 2 R_e.
    # Transform coordinates into the galaxy plane
    e = df_metadata.loc[gal, "e"]
    PA = df_metadata.loc[gal, "PA (degrees)"]
    i_rad = df_metadata.loc[gal, "i (degrees)"]
    beta_rad = np.deg2rad(PA - 90)
    i_rad = 0 if np.isnan(i_rad) else i_rad

    # De-project the centroids to the coordinate system of the galaxy plane
    ny, nx = data_cube_B.shape[1:]
    ys, xs = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
    x_cc = xs - settings["sami"]["x0_px"]  # pixels
    y_cc = ys - settings["sami"]["x0_px"]  # pixels
    x_prime = x_cc * np.cos(beta_rad) + y_cc * np.sin(beta_rad)
    y_prime_projec = (-x_cc * np.sin(beta_rad) + y_cc * np.cos(beta_rad))
    y_prime = (-x_cc * np.sin(beta_rad) +
               y_cc * np.cos(beta_rad)) / np.cos(i_rad)
    r_prime = np.sqrt(x_prime**2 + y_prime**2)

    # Convert to arcsec
    r_prime_as = r_prime * settings["sami"]["as_per_px"]

    # Masks enclosing differen multiples of R_e
    mask_1Re = r_prime_as < df_metadata.loc[gal, "R_e (arcsec)"]
    mask_15Re = r_prime_as < 1.5 * df_metadata.loc[gal, "R_e (arcsec)"]
    mask_2Re = r_prime_as < 2 * df_metadata.loc[gal, "R_e (arcsec)"]

    # Compute median SNR within 1, 1.5, 2R_e
    SNR_full_B = np.nanmedian(im_SNR_B)
    SNR_full_R = np.nanmedian(im_SNR_R)
    SNR_1Re_B = np.nanmedian(im_SNR_B[mask_1Re])
    SNR_1Re_R = np.nanmedian(im_SNR_R[mask_1Re])
    SNR_15Re_B = np.nanmedian(im_SNR_B[mask_15Re])
    SNR_15Re_R = np.nanmedian(im_SNR_R[mask_15Re])
    SNR_2Re_B = np.nanmedian(im_SNR_B[mask_2Re])
    SNR_2Re_R = np.nanmedian(im_SNR_R[mask_2Re])

    #######################################################################
    # End
    logger.info(f"finished processing {gal}")
    return [
        gal, SNR_full_B, SNR_full_R, SNR_1Re_B, SNR_1Re_R, SNR_15Re_B,
        SNR_15Re_R, SNR_2Re_B, SNR_2Re_R
    ]


###############################################################################
def make_sami_metadata_df(recompute_continuum_SNRs=False, nthreads=None):
    """Create the SAMI "metadata" DataFrame.

    DESCRIPTION
    ---------------------------------------------------------------------------
    This function is used to create a DataFrame containing "metadata", including
    stellar masses, spectroscopic redshifts, morphologies and other information
    for each galaxy in SAMI. In addition to the provided values in the input
    catalogues, the angular scale (in kpc per arcsecond) and inclination are 
    computed for each galaxy.

    This script must be run before make_sami_df() as the resulting DataFrame
    is used there.

    The information used here is from the catalogues are available at 
    https://datacentral.org.au/. 

    Details:
        - Distances are computed from the redshifts assuming a flat ΛCDM cosmology 
    with H0 = 70 km/s/Mpc, ΩM = 0.3 and ΩΛ = 0.7. Flow-corrected redshifts are 
    used to compute distances when available. 
        - Morphologies are taken from the `VisualMorphologyDR3` catalogue. For 
    simplicity, the `?`, `No agreement` and `Unknown` categories are all merged 
    into a single category labelled `Unknown`.
        - MGE effective radius measurements are taken from the `MGEPhotomUnregDR3` 
    catalogue. For galaxies for which measurements from both VST and SDSS 
    photometry are available, only the VST measurements are kept.

    USAGE
    ---------------------------------------------------------------------------
            
            >>> from spaxelsleuth.loaddata.sami import make_sami_metadata_df
            >>> make_sami_metadata_df()

    INPUTS
    ---------------------------------------------------------------------------
    recompute_continuum_SNRs:   bool
        If True, compute median continuum S/N values 

    nthreads:                   int
        Number of threads used to parallelise the continuum S/N values.

    OUTPUTS
    ---------------------------------------------------------------------------
    The DataFrame is saved to 

        settings["sami"]["output_path"]/sami_dr3_metadata.hd5

    The DataFrame containing continuum S/N values is saved to 

        settings["sami"]["output_path"]/sami_dr3_aperture_snrs.hd5

    PREREQUISITES
    ---------------------------------------------------------------------------
    Tables containing metadata for SAMI galaxies are required for this script. 
    These have been included in the ../data/ directory. 

    These tables were downloaded in CSV format from 
        
        https://datacentral.org.au/services/schema/
        
    where they can be found under the following tabs:

        --> SAMI
            --> Data Release 3
                --> Catalogues 
                    --> SAMI 
                        --> CubeObs:
                            - sami_CubeObs
                        --> Other
                            - InputCatGAMADR3
                            - InputCatClustersDR3
                            - InputCatFiller
                            - VisualMorphologyDR3
                            - MGEPhotomUnregDR3

     and stored at ../data/ using the naming convention

        sami_InputCatGAMADR3.csv
        sami_InputCatClustersDR3.csv
        sami_InputCatFiller.csv
        sami_VisualMorphologyDR3.csv
        sami_CubeObs.csv
        sami_MGEPhotomUnregDR3.csv.

    """
    logger.info("creating metadata DataFrame...")

    # Determine number of threads
    if nthreads is None:
        nthreads = os.cpu_count()
        logger.warning(f"nthreads not specified: running make_sami_metadata_df() on {nthreads} threads...")

    ###########################################################################
    # Filenames
    df_fname = f"sami_dr3_metadata.hd5"
    gama_metadata_fname = "sami_InputCatGAMADR3.csv"
    cluster_metadata_fname = "sami_InputCatClustersDR3.csv"
    filler_metadata_fname = "sami_InputCatFiller.csv"
    morphologies_fname = "sami_VisualMorphologyDR3.csv"
    flag_metadata_fname = "sami_CubeObs.csv"
    mge_fits_metadata_fname = "sami_MGEPhotomUnregDR3.csv"

    # Get the data path
    data_path = Path(__file__.split("loaddata")[0]) / "data"
    for fname in [
            gama_metadata_fname, cluster_metadata_fname, filler_metadata_fname,
            morphologies_fname, flag_metadata_fname, mge_fits_metadata_fname
    ]:
        assert os.path.exists(data_path / fname),\
            f"File {data_path / fname} not found!"

    ###########################################################################
    # Read in galaxy metadata
    ###########################################################################
    df_metadata_gama = pd.read_csv(data_path / gama_metadata_fname)  # ALL possible GAMA targets
    df_metadata_cluster = pd.read_csv(data_path / cluster_metadata_fname)  # ALL possible cluster targets
    df_metadata_filler = pd.read_csv(data_path / filler_metadata_fname)  # ALL possible filler targets
    df_metadata = pd.concat([df_metadata_gama, df_metadata_cluster, df_metadata_filler], sort=True).drop(["Unnamed: 0"], axis=1)

    ###########################################################################
    # Append morphology data
    ###########################################################################
    df_morphologies = pd.read_csv(data_path / morphologies_fname).drop(["Unnamed: 0"], axis=1)
    df_morphologies = df_morphologies.rename(
        columns={"type": "Morphology (numeric)"})

    # Morphologies (numeric) - merge "?" and "no agreement" into a single category.
    df_morphologies.loc[df_morphologies["Morphology (numeric)"] == 5.0,
                        "Morphology (numeric)"] = -0.5
    df_morphologies.loc[df_morphologies["Morphology (numeric)"] == -9.0,
                        "Morphology (numeric)"] = -0.5
    df_morphologies.loc[df_morphologies["Morphology (numeric)"] == np.nan,
                        "Morphology (numeric)"] = -0.5

    # merge with metadata
    # Note: this step trims df_metadata to include only those objects with morphologies (9949 --> 3068)
    df_metadata = df_metadata.merge(
        df_morphologies[["catid", "Morphology (numeric)"]],
        on="catid")

    ###########################################################################
    # Read in flag metadata
    ###########################################################################
    df_flags = pd.read_csv(data_path / flag_metadata_fname).drop(
        ["Unnamed: 0"], axis=1)
    df_flags = df_flags.astype(
        {col: "int64"
         for col in df_flags.columns if col.startswith("warn")})
    df_flags = df_flags.astype({"isbest": bool})

    # Get rid of rows failing the following data quality criteria
    cond = df_flags["isbest"] == True
    cond &= df_flags["warnstar"] == 0
    cond &= df_flags[
        "warnmult"] < 2  # multiple objects overlapping with galaxy area
    cond &= df_flags["warnfcal"] == 0  # flux calibration issues
    cond &= df_flags["warnfcbr"] == 0  # flux calibration issues
    cond &= df_flags["warnskyb"] == 0  # bad sky subtraction residuals
    cond &= df_flags["warnskyr"] == 0  # bad sky subtraction residuals
    cond &= df_flags[
        "warnre"] == 0  # significant difference between standard & MGE Re. NOTE: there are actually no entries in this DataFrame with WARNRE = 1!
    df_flags_cut = df_flags[cond]

    for gal in df_flags_cut["catid"]:
        if df_flags_cut[df_flags_cut["catid"] == gal].shape[0] > 1:
            # If there are two "best" observations, drop the second one.
            drop_idxs = df_flags_cut.index[df_flags_cut["catid"] == gal][1:]
            df_flags_cut = df_flags_cut.drop(drop_idxs)

    assert df_flags_cut.shape[0] == len(df_flags_cut["catid"].unique())

    # Convert to int
    df_metadata["catid"] = df_metadata["catid"].astype(int)
    df_flags_cut["catid"] = df_flags_cut["catid"].astype(int)
    gal_ids_dq_cut = list(df_flags_cut["catid"])

    # Remove 9008500001 since it's a duplicate!
    gal_ids_dq_cut.pop(gal_ids_dq_cut.index(9008500001))

    # Add DQ cut column
    df_metadata["Good?"] = False
    df_metadata.loc[df_metadata["catid"].isin(gal_ids_dq_cut), "Good?"] = True

    # Reset index
    df_metadata = df_metadata.set_index(df_metadata["catid"])

    ###########################################################################
    # Add R_e and other parameters derived from MGE fits
    # Note: these are based on SDSS and VST photometry, not GAMA.
    ###########################################################################
    df_mge = pd.read_csv(data_path / mge_fits_metadata_fname).drop(["Unnamed: 0"], axis=1).set_index("catid")  # Data from multi-expansion fits

    # Drop duplicated rows: those with both SDSS and VST photometry
    df_mge_vst = df_mge[df_mge["photometry"] == "VST"]
    df_mge_sdss = df_mge[df_mge["photometry"] == "SDSS"]
    gals_vst_and_sdss = [
        g for g in df_mge_vst.index.values if g in df_mge_sdss.index.values
    ]
    df_mge = df_mge.sort_values(
        by="photometry"
    )  # Sort so that photometry values are alphabetically sorted
    bad_rows = df_mge.index.duplicated(
        keep="last"
    )  # Find duplicate rows - these will all be galaxies with both SDSS and VST data, in which case we keep the VST measurement
    df_mge = df_mge[~bad_rows]
    # Check that all of the galaxies with both VST and SDSS entries have VST photometry
    for gal in gals_vst_and_sdss:
        assert df_mge.loc[gal, "photometry"] == "VST"
    # Merge
    df_metadata = df_metadata.merge(df_mge,
                                    how="left",
                                    left_index=True,
                                    right_index=True)

    ###############################################################################
    """
    Drop and rename columns. The following columns are included in the input tables used here:
    a_g               g-band extinction - keep
    bad_class         Flag for bad or problem objects - 0, 5 and 8 are "good" - keep
    catid             SAMI Galaxy ID - keep
    dec_ifu           J2000 Declination of IFU - keep
    dec_obj           J2000 Declination of object - keep
    ellip             r-band ellipticity - ???
    fillflag          Flag for different filler classes - drop
    g_i               (g-i) colour - keep
    is_mem            Flag indicating cluster membership (1=member, 0=non-member) - keep
    m_r               Absolute r-band magnitude - keep
    mstar             Logarithm of stellar mass - keep
    mu_1re            r-band surface brightness at 1 effective radius - keep 
    mu_2re            r-band surface brightness at 2 effective radii - keep    
    mu_within_1re,    Mean r-band surface brightness within 1 effective radius - keep   
    pa                r-band position angle - ???
    r_auto            r-band SExtractor auto magnitude - drop
    r_e               r-band major axis effective radius - ????
    r_on_rtwo         Projected distance from cluster centre normalised by R200 - keep
    r_petro           Extinction-corrected r-band Petrosian mag - drop
    ra_ifu            J2000 Right Ascension of IFU - keep
    ra_obj            J2000 Right Ascension of object - keep
    surv_sami         Drop
    v_on_sigma,       Line-of-sight velocity relative to cluster redshift normalised by cluster velocity dispersion measured within R200 - keep
    z                 Spectroscopic redshift - keep 
    z_tonry           Flow-corrected redshift  
    Morphology (numeric)  - keep
    Good?             keep 
    photometry        Denotes which images were used. - keep
    remge             Circularised effective radius from MGE fit. - keep
    mmge              Total AB magnitude from the MGE fit. No corrections applied. - keep 
    rextinction,      Extinction from Schlafly+2011. - drop    
    pamge             Position Angle of the MGE model, from N to E is positive. - ??? 
    epsmge_re         Model isophotal ellipticity at one Re. - ???
    epsmge_lw         Light-weighted ellipticity of the model. - ???
    dist2nneigh,      Distance to nearest neighbour from SExtractor source extraction. - drop 
    chi2              Chi^2 from MGE fit. - drop
    
    Added columns:
    D_A (Mpc)         Angular diameter distance 
    D_L (Mpc)         Luminosity distance       
    kpc per arcsec    Angular scale            
    log(M/R_e)        stellar mass / R_e (proxy for gravitational potential)
    Inclination i (degrees)  Inclination (computed from ellpiticity)
    """
    ###############################################################################
    # Drop unnecessary columns (including object-type columns) & rename others for readability
    cols_to_remove = [
        "r_auto", "r_petro", "surv_sami", "rextinction", "dist2nneigh", "chi2",
        "fillflag", "MGE photometry"
    ]
    rename_dict = {
        "a_g": "A_g",
        "bad_class": "Bad class #",
        "catid": "ID",
        "dec_obj": "Dec (J2000)",
        "ra_obj": "RA (J2000)",
        "dec_ifu":
        "Dec (IFU) (J2000)",  # NOTE: some galaxies do not have these coordinates, for some reason.
        "ra_ifu":
        "RA (IFU) (J2000)",  # NOTE: some galaxies do not have these coordinates, for some reason.
        "g_i": "g - i colour",
        "is_mem": "Cluster member",
        "m_r": "M_r",
        "mstar": "log M_*",
        "mu_1re": "mu_r at 1R_e",
        "mu_2re": "mu_r at 2R_e",
        "mu_within_1re": "mu_r within 1R_e",
        "v_on_sigma": "v/sigma_cluster",
        "pa": "PA (degrees)",
        "r_e": "R_e (arcsec)",
        "ellip": "e",
        "r_on_rtwo": "r/R_200",
        "z_spec": "z (spectroscopic)",
        "z_tonry": "z (flow-corrected)",
        "photometry": "MGE photometry",
        "remge": "R_e (MGE) (arcsec)",
        "mmge": "m_AB (MGE)",
        "pamge": "PA (MGE) (degrees)",
        "epsmge_re": "e at 1R_e (MGE)",
        "epsmge_lw": "e, LW (MGE)",
    }
    df_metadata = df_metadata.rename(columns=rename_dict)
    df_metadata = df_metadata.drop(columns=cols_to_remove)

    ###########################################################################
    # Assign redshifts based on cluster membership.
    # For all galaxies, column "z" will contain the Tonry redshift for
    # non-cluster members and the cluster redshift for cluster members.
    ###########################################################################
    cond_has_no_Tonry_z = df_metadata["z (flow-corrected)"].isna()
    df_metadata.loc[cond_has_no_Tonry_z,
                    "z"] = df_metadata.loc[cond_has_no_Tonry_z,
                                           "z (spectroscopic)"]
    df_metadata.loc[~cond_has_no_Tonry_z,
                    "z"] = df_metadata.loc[~cond_has_no_Tonry_z,
                                           "z (flow-corrected)"]

    # Check that NO cluster members have flow-corrected redshifts
    assert not any((df_metadata["Cluster member"] == 1.0)
                   & ~df_metadata["z (flow-corrected)"].isna())

    ###########################################################################
    # Add angular scale info
    ###########################################################################
    logger.info(f"computing distances...")
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    for gal in gal_ids_dq_cut:
        D_A_Mpc = cosmo.angular_diameter_distance(df_metadata.loc[gal,
                                                                  "z"]).value
        D_L_Mpc = cosmo.luminosity_distance(df_metadata.loc[gal, "z"]).value
        df_metadata.loc[gal, "D_A (Mpc)"] = D_A_Mpc
        df_metadata.loc[gal, "D_L (Mpc)"] = D_L_Mpc
    df_metadata["kpc per arcsec"] = df_metadata[
        "D_A (Mpc)"] * 1e3 * np.pi / 180.0 / 3600.0
    df_metadata["R_e (kpc)"] = df_metadata["R_e (arcsec)"] * df_metadata[
        "kpc per arcsec"]
    df_metadata["R_e (MGE) (kpc)"] = df_metadata[
        "R_e (MGE) (arcsec)"] * df_metadata["kpc per arcsec"]
    df_metadata["log(M/R_e)"] = df_metadata["log M_*"] - np.log10(
        df_metadata["R_e (kpc)"])
    df_metadata["log(M/R_e^2)"] = df_metadata["log M_*"] - 2 * np.log10(
        df_metadata["R_e (kpc)"])
    df_metadata["log(M/R_e) (MGE)"] = df_metadata["log M_*"] - np.log10(
        df_metadata["R_e (MGE) (kpc)"])
    df_metadata["log(M/R_e^2) (MGE)"] = df_metadata["log M_*"] - 2 * np.log10(
        df_metadata["R_e (MGE) (kpc)"])

    ###########################################################################
    # Compute inclination
    ###########################################################################
    e = df_metadata["e"]
    PA = df_metadata["PA (degrees)"]
    beta_rad = np.deg2rad(PA - 90)
    b_over_a = 1 - e
    q0 = 0.2
    i_rad = np.arccos(np.sqrt(
        (b_over_a**2 - q0**2) / (1 - q0**2)))  # Want to store this!
    df_metadata["i (degrees)"] = np.rad2deg(i_rad)

    ###############################################################################
    # Compute continuum SNRs from the data cubes
    ###############################################################################
    logger.info("computing continuum SNRs...")
    if not recompute_continuum_SNRs and os.path.exists(output_path / "sami_dr3_aperture_snrs.hd5"):
        logger.warning(
            f"file {output_path / 'sami_dr3_aperture_snrs.hd5'} found; loading SNRs from existing DataFrame..."
        )
        df_snr = pd.read_hdf(output_path / "sami_dr3_aperture_snrs.hd5", key="SNR")
    else:
        args_list = [[gal, df_metadata] for gal in gal_ids_dq_cut]
        if nthreads > 1:
            logger.info(f"computing continuum SNRs on {nthreads} threads...")
            pool = multiprocessing.Pool(nthreads)
            res_list = np.array((pool.map(_compute_snr, args_list)))
            pool.close()
            pool.join()
        else:
            res_list = []
            logger.info("computing continuum SNRs sequentially...")
            for arg in args_list:
                res_list.append(_compute_snr(arg))

        ###########################################################################
        # Create DataFrame from results
        ###############################################################################
        df_snr = pd.DataFrame(
            np.vstack(res_list),
            columns=[
                "ID", "Median SNR (B, full field)",
                "Median SNR (R, full field)", "Median SNR (B, 1R_e)",
                "Median SNR (R, 1R_e)", "Median SNR (B, 1.5R_e)",
                "Median SNR (R, 1.5R_e)", "Median SNR (B, 2R_e)",
                "Median SNR (R, 2R_e)"
            ])
        df_snr["ID"] = df_snr["ID"].astype(int)
        df_snr.set_index("ID")

        # Save
        logger.info(
            "saving aperture SNR DataFrame to file..."
        )
        df_snr.to_hdf(output_path / "sami_dr3_aperture_snrs.hd5", key="SNR")

    ###############################################################################
    # Merge with the metadata DataFrame
    ###############################################################################
    df_snr = df_snr.set_index("ID")
    df_metadata = pd.concat([df_snr, df_metadata], axis=1)

    ###########################################################################
    # Save to file
    ###########################################################################
    logger.info(
        f"saving metadata DataFrame to file {output_path / df_fname}..."
    )
    df_metadata = df_metadata.sort_index()
    df_metadata.to_hdf(output_path / df_fname, key="metadata")

    logger.info(f"finished!")
    return


###############################################################################
def _process_gals(args):
    """
    DESCRIPTION
    ---------------------------------------------------------------------------
    Helper function used to multithread the processing of SAMI galaxies in 
    make_sami_df().

    INPUTS
    ---------------------------------------------------------------------------
    args:       list 
        List containing gal_idx, gal, ncomponents, bin_type, df_metadata, 
        use_lzifu_fits, lzifu_ncomponents.

    OUTPUTS
    ---------------------------------------------------------------------------
    DataFrame rows and corresponding columns corresponding to galaxy gal.

    """
    # Extract input arguments
    gal_idx, gal, ncomponents, bin_type, df_metadata, use_lzifu_fits, lzifu_ncomponents = args

    # List of filenames for SAMI data products
    fname_list = [
        f"stellar-velocity-dispersion_{bin_type}_two-moment",
        f"stellar-velocity_{bin_type}_two-moment",
        f"extinct-corr_{bin_type}_{ncomponents}-comp",
        f"sfr-dens_{bin_type}_{ncomponents}-comp",
        f"sfr_{bin_type}_{ncomponents}-comp"
    ]
    if not use_lzifu_fits:
        fname_list += [
            f"Halpha_{bin_type}_{ncomponents}-comp",
            f"Hbeta_{bin_type}_{ncomponents}-comp",
            f"NII6583_{bin_type}_{ncomponents}-comp",
            f"OI6300_{bin_type}_{ncomponents}-comp",
            f"OII3728_{bin_type}_{ncomponents}-comp",
            f"OIII5007_{bin_type}_{ncomponents}-comp",
            f"SII6716_{bin_type}_{ncomponents}-comp",
            f"SII6731_{bin_type}_{ncomponents}-comp",
            f"gas-vdisp_{bin_type}_{ncomponents}-comp",
            f"gas-velocity_{bin_type}_{ncomponents}-comp",
        ]
    fnames = [
        str(input_path / f"ifs/{gal}/{gal}_A_{f}.fits") for f in fname_list
    ]

    #######################################################################
    # Open the red & blue cubes.
    with fits.open(data_cube_path / f"ifs/{gal}/{gal}_A_cube_blue.fits.gz") as hdulist_B_cube:
        header_R = hdulist_B_cube[0].header
        data_cube_B = hdulist_B_cube[0].data
        var_cube_B = hdulist_B_cube[1].data
        hdulist_B_cube.close()

        # Wavelength values
        lambda_0_A = header_R[
            "CRVAL3"] - header_R["CRPIX3"] * header_R["CDELT3"]
        dlambda_A = header_R["CDELT3"]
        N_lambda = header_R["NAXIS3"]
        lambda_vals_B_A = np.array(range(N_lambda)) * dlambda_A + lambda_0_A
        lambda_vals_B_rest_A = lambda_vals_B_A / (
            1 + df_metadata.loc[gal, "z (spectroscopic)"]
        )  #NOTE: we use the spectroscopic redshift here, because when it comes to measuring e.g. continuum levels, it's important that the wavelength range we use is consistent between galaxies. For some galaxies the flow-corrected redshift is sufficiently different from the spectroscopic redshift that when we use it to define wavelength windows for computing the continuum level for instance we end up enclosing an emission line which throws the measurement way out of whack (e.g. for 572402)

    with fits.open(data_cube_path / f"ifs/{gal}/{gal}_A_cube_red.fits.gz") as hdulist_R_cube:
        header_R = hdulist_R_cube[0].header
        data_cube_R = hdulist_R_cube[0].data
        var_cube_R = hdulist_R_cube[1].data

        # Wavelength values
        lambda_0_A = header_R[
            "CRVAL3"] - header_R["CRPIX3"] * header_R["CDELT3"]
        dlambda_A = header_R["CDELT3"]
        N_lambda = header_R["NAXIS3"]
        lambda_vals_R_A = np.array(range(N_lambda)) * dlambda_A + lambda_0_A
        lambda_vals_R_rest_A = lambda_vals_R_A / (
            1 + df_metadata.loc[gal, "z (spectroscopic)"]
        )  #NOTE: we use the spectroscopic redshift here, because when it comes to measuring e.g. continuum levels, it's important that the wavelength range we use is consistent between galaxies. For some galaxies the flow-corrected redshift is sufficiently different from the spectroscopic redshift that when we use it to define wavelength windows for computing the continuum level for instance we end up enclosing an emission line which throws the measurement way out of whack (e.g. for 572402)

    #######################################################################
    # Compute continuum quantities

    # Load gas/stellar velocity maps so that we can window in around wavelength ranges accounting for velocity shifts
    with fits.open(input_path / f"ifs/{gal}/{gal}_A_stellar-velocity_{bin_type}_two-moment.fits") as hdulist_v_star:
        v_star_map = hdulist_v_star[0].data.astype(np.float64)
    if not use_lzifu_fits:
        with fits.open(input_path / f"ifs/{gal}/{gal}_A_gas-velocity_{bin_type}_{ncomponents}-comp.fits") as hdulist_v:
            v_map = hdulist_v[0].data.astype(np.float64)
    else:
        lzifu_fname = [
            f for f in os.listdir(__lzifu_products_path)
            if f.startswith(str(gal)) and f"{lzifu_ncomponents}_comp" in f
        ][0]
        with fits.open(__lzifu_products_path / lzifu_fname) as hdu_lzifu:
            v_map = hdu_lzifu["V"].data.astype(np.float64)

    # Compute the d4000 Angstrom break.
    d4000_map, d4000_map_err = compute_d4000(
        data_cube=data_cube_B,
        var_cube=var_cube_B,
        lambda_vals_rest_A=lambda_vals_B_rest_A,
        v_star_map=v_star_map)

    # Compute the continuum intensity so that we can compute the Halpha equivalent width. Units of 10**(-16) erg /s /cm**2 /angstrom /pixel
    # Continuum wavelength range taken from here: https://ui.adsabs.harvard.edu/abs/2019MNRAS.485.4024V/abstract
    cont_HALPHA_map, cont_HALPHA_map_std, cont_HALPHA_map_err = compute_continuum_intensity(
        data_cube=data_cube_R,
        var_cube=var_cube_R,
        lambda_vals_rest_A=lambda_vals_R_rest_A,
        start_A=6500,
        stop_A=6540,
        v_map=v_map[0])

    # Compute the approximate B-band continuum. Units of 10**(-16) erg /s /cm**2 /angstrom /pixel
    cont_B_map, cont_B_map_std, cont_B_map_err = compute_continuum_intensity(
        data_cube=data_cube_B,
        var_cube=var_cube_B,
        lambda_vals_rest_A=lambda_vals_B_rest_A,
        start_A=4000,
        stop_A=5000,
        v_map=v_star_map)

    # Compute v_grad using eqn. 1 of Zhou+2017
    v_grad = compute_v_grad(v_map)

    # Compute the HALPHA amplitude-to-noise. Store as "meas" to distinguish from A/N measurements for individual emission line components
    AN_HALPHA_map = compute_HALPHA_amplitude_to_noise(
        data_cube=data_cube_R,
        var_cube=var_cube_R,
        lambda_vals_rest_A=lambda_vals_R_rest_A,
        v_star_map=v_star_map,
        v_map=v_map[0],
        dv=300)

    #######################################################################
    #######################################################################
    # X, Y pixel coordinates
    # Compute the spaxel or bin coordinates, depending on the binning scheme
    im = np.nansum(data_cube_B, axis=0)
    ny, nx = im.shape

    if bin_type == "default":
        # Create an image from the datacube to figure out where are "good" spaxels
        if np.any(
                im.flatten() <= 0
        ):  # NaN out -ve spaxels. Most galaxies seem to have *some* -ve pixels
            im[im <= 0] = np.nan

        # Compute the coordinates of "good" spaxels, store in arrays
        y_c_list, x_c_list = np.argwhere(~np.isnan(im)).T
        ngood_bins = len(x_c_list)

        # List of bin sizes, in pixels
        bin_size_list_px = [1] * ngood_bins
        bin_number_list = np.arange(1, ngood_bins + 1)

    # Compute the light-weighted bin centres, based on the blue unbinned
    # data cube
    elif bin_type == "adaptive" or bin_type == "sectors":
        ys, xs = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
        # Open the binned blue cube. Get the bin mask extension.
        hdulist_binned_cube = fits.open(input_path / f"ifs/{gal}/{gal}_A_{bin_type}_blue.fits.gz")
        bin_map = hdulist_binned_cube[2].data.astype("float")
        bin_map[bin_map == 0] = np.nan

        bin_number_list = np.array(
            [nn for nn in np.unique(bin_map) if ~np.isnan(nn)])
        nbins = len(bin_number_list)
        x_c_list = np.full(nbins, np.nan)
        y_c_list = np.full(nbins, np.nan)
        bin_size_list_px = np.full(nbins, np.nan)
        for ii, nn in enumerate(bin_number_list):
            # generate a bin mask.
            bin_mask = bin_map == nn
            bin_size_list_px[ii] = len(bin_mask[bin_mask == True])
            # compute the centroid of the bin.
            x_c = np.nansum(xs * bin_mask * im) / np.nansum(bin_mask * im)
            y_c = np.nansum(ys * bin_mask * im) / np.nansum(bin_mask * im)
            # Don't add the centroids if they are out of bounds.
            if (x_c < 0 or x_c >= nx or y_c < 0 or y_c >= ny):
                x_c_list[ii] = np.nan
                y_c_list[ii] = np.nan
            else:
                x_c_list[ii] = x_c
                y_c_list[ii] = y_c

        #######################################################################
        # Bin numbers corresponding to bins actually present in the image
        good_bins = np.argwhere(~np.isnan(x_c_list)).flatten()
        ngood_bins = len(good_bins)
        x_c_list = x_c_list[good_bins]
        y_c_list = y_c_list[good_bins]
        bin_size_list_px = bin_size_list_px[good_bins]
        bin_number_list = bin_number_list[good_bins]

    #######################################################################
    # Compute deprojected pixel coordinates
    PA_deg = df_metadata.loc[gal, "PA (degrees)"]
    i_deg = 0 if np.isnan(
        df_metadata.loc[gal,
                        "i (degrees)"]) else df_metadata.loc[gal,
                                                             "i (degrees)"]
    x_prime_list, y_prime_list, r_prime_list = deproject_coordinates(
        x_c_list,
        y_c_list,
        settings["sami"]["x0_px"],
        settings["sami"]["y0_px"],
        PA_deg,
        i_deg,
    )

    #######################################################################
    # Open each FITS file, extract the values from the maps in each bin & append
    rows_list = []
    colnames = []

    def _2d_map_to_1d_list(colmap):
        """Returns a 1D array of values extracted from from spaxels in x_c_list and y_c_list in 2D array colmap."""
        if colmap.ndim != 2:
            raise ValueError(
                f"colmap must be a 2D array but has ndim = {colmap.ndim}!")
        row = np.full_like(x_c_list, np.nan, dtype="float")
        for jj, coords in enumerate(zip(x_c_list, y_c_list)):
            x_c, y_c = coords
            y, x = (int(np.round(y_c)), int(np.round(x_c)))
            if x > nx - 1 or y > ny - 1:
                x = min([x, nx - 1])
                y = min([y, ny - 1])
            row[jj] = colmap[y, x]
        return row

    # Tidy up column names
    colname_dict = {
        f"stellar-velocity-dispersion_{bin_type}_two-moment": "sigma_*",
        f"stellar-velocity_{bin_type}_two-moment": "v_*",
        f"extinct-corr_{bin_type}_{ncomponents}-comp":
        "HALPHA extinction correction",
        f"sfr-dens_{bin_type}_{ncomponents}-comp": "SFR surface density",
        f"sfr_{bin_type}_{ncomponents}-comp": "SFR",
        f"Halpha_{bin_type}_{ncomponents}-comp": "HALPHA",
        f"Hbeta_{bin_type}_{ncomponents}-comp": "HBETA",
        f"NII6583_{bin_type}_{ncomponents}-comp": "NII6583",
        f"OI6300_{bin_type}_{ncomponents}-comp": "OI6300",
        f"OII3728_{bin_type}_{ncomponents}-comp": "OII3726+OII3729",
        f"OIII5007_{bin_type}_{ncomponents}-comp": "OIII5007",
        f"SII6716_{bin_type}_{ncomponents}-comp": "SII6716",
        f"SII6731_{bin_type}_{ncomponents}-comp": "SII6731",
        f"gas-vdisp_{bin_type}_{ncomponents}-comp": "sigma_gas",
        f"gas-velocity_{bin_type}_{ncomponents}-comp": "v_gas",
    }

    for ff, fname in enumerate(fnames):
        hdu = fits.open(fname)
        data = hdu[0].data.astype(np.float64)
        data_err = hdu[1].data.astype(np.float64)
        hdu.close()

        # HALPHA, SFR quantities
        if data.ndim > 2:
            if "Halpha" in fname or "sfr" in fname:
                rows_list.append(_2d_map_to_1d_list(data[0]))
                colnames.append(f"{colname_dict[fname_list[ff]]} (total)")
                rows_list.append(_2d_map_to_1d_list(data_err[0]))
                colnames.append(
                    f"{colname_dict[fname_list[ff]]} error (total)")
                # Trim the 0th slice
                data = data[1:]
                data_err = data_err[1:]
            # Add individual components
            for nn in range(3 if ncomponents == "recom" else 1):
                rows_list.append(_2d_map_to_1d_list(data[nn]))
                colnames.append(
                    f"{colname_dict[fname_list[ff]]} (component {nn + 1})")
                rows_list.append(_2d_map_to_1d_list(data_err[nn]))
                colnames.append(
                    f"{colname_dict[fname_list[ff]]} error (component {nn + 1})"
                )

        # EXTINCTION, STELLAR KINEMATICS & EMISSION LINES EXCEPT FOR HALPHA
        else:
            rows_list.append(_2d_map_to_1d_list(data))
            rows_list.append(_2d_map_to_1d_list(data_err))
            # Column name
            # If adding the stellar kinematics, no point in adding "total" here
            if "stellar" in fname:
                colnames.append(f"{colname_dict[fname_list[ff]]}")
                colnames.append(f"{colname_dict[fname_list[ff]]} error")
            # Otherwise append "total" to signify total fluxes.
            else:
                colnames.append(f"{colname_dict[fname_list[ff]]} (total)")
                colnames.append(
                    f"{colname_dict[fname_list[ff]]} error (total)")

    # Load LZIFU files
    if use_lzifu_fits:
        # Open the FITS file
        lzifu_fname = [
            f for f in os.listdir(__lzifu_products_path)
            if f.startswith(str(gal)) and f"{lzifu_ncomponents}_comp" in f
        ][0]
        hdu_lzifu = fits.open(__lzifu_products_path / lzifu_fname)

        # Load emission line fluxes & kinematics (except for [OII])
        for quantity in ["HBETA", "OIII5007", "OI6300",
                         "HALPHA", "NII6583", "SII6716", "SII6731"] +\
                        ["V", "VDISP"]:

            # Load data from the FITS file
            data = hdu_lzifu[f"{quantity}"].data.astype(np.float64)
            data_err = hdu_lzifu[f"{quantity}_ERR"].data.astype(np.float64)

            # Total fluxes
            if quantity not in ["V", "VDISP"]:
                rows_list.append(_2d_map_to_1d_list(data[0]))
                rows_list.append(_2d_map_to_1d_list(data_err[0]))
                # Column name
                if quantity == "V":
                    quantity_colname = "v_gas"
                elif quantity == "VDISP":
                    quantity_colname = "sigma_gas"
                else:
                    quantity_colname = quantity
                colnames.append(f"{quantity_colname} (total)")
                colnames.append(f"{quantity_colname} error (total)")

            # Fluxes/kinematics in components 1, 2 and 3
            for nn in range(data.shape[0] - 1):
                rows_list.append(_2d_map_to_1d_list(data[nn + 1]))
                rows_list.append(_2d_map_to_1d_list(data_err[nn + 1]))
                # Column name
                if quantity == "V":
                    quantity_colname = "v_gas"
                elif quantity == "VDISP":
                    quantity_colname = "sigma_gas"
                else:
                    quantity_colname = quantity
                colnames.append(f"{quantity_colname} (component {nn + 1})")
                colnames.append(
                    f"{quantity_colname} error (component {nn + 1})")

        # OII doublet: these need to be combined to be consistent with the DR3 data products.
        # We will store combined fluxes in column "OII3726+OII3729"
        data_OII3726 = hdu_lzifu[f"OII3726"].data.astype(np.float64)
        data_OII3726_err = hdu_lzifu[f"OII3726_ERR"].data.astype(np.float64)
        data_OII3729 = hdu_lzifu[f"OII3729"].data.astype(np.float64)
        data_OII3729_err = hdu_lzifu[f"OII3729_ERR"].data.astype(np.float64)
        data = data_OII3726 + data_OII3729
        data_err = np.sqrt(data_OII3726_err**2 + data_OII3729_err**2)

        # Total fluxes
        rows_list.append(_2d_map_to_1d_list(data[0]))
        colnames.append(f"OII3726+OII3729 (total)")
        rows_list.append(_2d_map_to_1d_list(data_err[0]))
        colnames.append(f"OII3726+OII3729 error (total)")

        # Fluxes in components 1, 2 and 3
        for nn in range(data.shape[0] - 1):
            rows_list.append(_2d_map_to_1d_list(data[nn + 1]))
            colnames.append(f"OII3726+OII3729 (component {nn + 1})")
            rows_list.append(_2d_map_to_1d_list(data_err[nn + 1]))
            colnames.append(f"OII3726+OII3729 error (component {nn + 1})")

    # Add v_grad
    for nn in range(v_grad.shape[0]):
        rows_list.append(_2d_map_to_1d_list(v_grad[nn]))
        colnames.append(f"v_grad (component {nn + 1})")

    # Add HALPHA amplitude-to-noise
    rows_list.append(_2d_map_to_1d_list(AN_HALPHA_map))
    colnames.append(f"HALPHA A/N (measured)")

    # Add the continuum intensity for calculating the HALPHA EW
    rows_list.append(_2d_map_to_1d_list(cont_HALPHA_map))
    colnames.append(f"HALPHA continuum")
    rows_list.append(_2d_map_to_1d_list(cont_HALPHA_map_std))
    colnames.append(f"HALPHA continuum std. dev.")
    rows_list.append(_2d_map_to_1d_list(cont_HALPHA_map_err))
    colnames.append(f"HALPHA continuum error")

    # Add the B-band continuum intensity
    rows_list.append(_2d_map_to_1d_list(cont_B_map))
    colnames.append(f"B-band continuum")
    rows_list.append(_2d_map_to_1d_list(cont_B_map_std))
    colnames.append(f"B-band continuum std. dev.")
    rows_list.append(_2d_map_to_1d_list(cont_B_map_err))
    colnames.append(f"B-band continuum error")

    # Add the D4000Å break
    rows_list.append(_2d_map_to_1d_list(d4000_map))
    colnames.append(f"D4000")
    rows_list.append(_2d_map_to_1d_list(d4000_map_err))
    colnames.append(f"D4000 error")

    # Add pixel coordinates
    rows_list.append(
        np.array([settings["sami"]["x0_px"]] * ngood_bins) *
        settings["sami"]["as_per_px"])
    colnames.append("Galaxy centre x0_px (projected, arcsec)")
    rows_list.append(
        np.array([settings["sami"]["y0_px"]] * ngood_bins) *
        settings["sami"]["as_per_px"])
    colnames.append("Galaxy centre y0_px (projected, arcsec)")
    rows_list.append(
        np.array(x_c_list).flatten() * settings["sami"]["as_per_px"])
    colnames.append("x (projected, arcsec)")
    rows_list.append(
        np.array(y_c_list).flatten() * settings["sami"]["as_per_px"])
    colnames.append("y (projected, arcsec)")
    rows_list.append(
        np.array(x_prime_list).flatten() * settings["sami"]["as_per_px"])
    colnames.append("x (relative to galaxy centre, deprojected, arcsec)")
    rows_list.append(
        np.array(y_prime_list).flatten() * settings["sami"]["as_per_px"])
    colnames.append("y (relative to galaxy centre, deprojected, arcsec)")
    rows_list.append(
        np.array(r_prime_list).flatten() * settings["sami"]["as_per_px"])
    colnames.append("r (relative to galaxy centre, deprojected, arcsec)")
    rows_list.append(np.array(bin_number_list))
    colnames.append("Bin number")
    rows_list.append(np.array(bin_size_list_px))
    colnames.append("Bin size (pixels)")
    rows_list.append(
        np.array(bin_size_list_px) * settings["sami"]["as_per_px"]**2)
    colnames.append("Bin size (square arcsec)")
    rows_list.append(
        np.array(bin_size_list_px) * settings["sami"]["as_per_px"]**2 *
        df_metadata.loc[gal, "kpc per arcsec"]**2)
    colnames.append("Bin size (square kpc)")

    ##########################################################
    # Transpose so that each row represents a single pixel & each column a measured quantity.
    rows_arr = np.array(rows_list).T

    # Get rid of rows that are all NaNs
    bad_rows = np.all(np.isnan(rows_arr), axis=1)
    rows_good = rows_arr[~bad_rows]

    # Append a column with the galaxy ID, so we know which galaxy these rows belong to
    gal_metadata = np.tile(
        df_metadata.loc[df_metadata.loc[:, "ID"] == gal]["ID"].values,
        (ngood_bins, 1))
    rows_good = np.hstack((gal_metadata, rows_good))

    logger.info(f"finished processing {gal} ({gal_idx})")

    return rows_good, ["ID"] + colnames


###############################################################################
def make_sami_df(bin_type, 
                 ncomponents,
                 eline_SNR_min,
                 correct_extinction,
                 sigma_gas_SNR_min=3,
                 eline_list=settings["sami"]["eline_list"],
                 line_flux_SNR_cut=True,
                 missing_fluxes_cut=True,
                 line_amplitude_SNR_cut=True,
                 flux_fraction_cut=False,
                 sigma_gas_SNR_cut=True,
                 vgrad_cut=False,
                 stekin_cut=True,
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
                 nthreads=None,
                 debug=False,
                 __use_lzifu_fits=False,
                 __lzifu_ncomponents=None):
    """
    Make the SAMI DataFrame, where each row represents a single spaxel in a SAMI galaxy.

    DESCRIPTION
    ---------------------------------------------------------------------------
    This function is used to create a Pandas DataFrame containing emission line 
    fluxes & kinematics, stellar kinematics, extinction, star formation rates, 
    and other quantities for individual spaxels in SAMI galaxies as taken from 
    SAMI DR3.

    The output is stored in HDF format as a Pandas DataFrame in which each row 
    corresponds to a given spaxel (or Voronoi/sector bin) for every galaxy. 

    USAGE
    ---------------------------------------------------------------------------
    
        >>> from spaxelsleuth.loaddata.sami import make_sami_df()
        >>> make_sami_df(ncomponents="1", bin_type="default", correct_extinction=True, eline_SNR_min=5)

    will create a DataFrame using the data products from 1-component Gaussian 
    fits to the unbinned datacubes, and will adopt a minimum S/N threshold of 
    5 to mask out unreliable emission line fluxes and associated quantities.

    Other input arguments may be configured to control other aspects of the data 
    quality and S/N cuts made.

    Running this function on the full sample takes some time (~10-20 minutes 
    when threaded across 20 threads). Execution time can be sped up by tweaking 
    the nthreads parameter. 

    If you wish to run in debug mode, set the DEBUG flag to True: this will run 
    the script on a subset (by default 10) galaxies to speed up execution. 

    INPUTS
    ---------------------------------------------------------------------------
    ncomponents:                str
        Controls which data products are used, depending on the number of 
        Gaussian components fitted to the emission lines. 
        Options are "recom" (the recommended multi-component fits) or "1" 
        (1-component fits).
        
        NOTE: if __use_lzifu_fits is True, then ncomponents is ONLY used in  
        loading data products that are NOT contained in the output LZIFU files -
        i.e., SFRs/SFR surface densities and HALPHA extinction correction 
        factors. Use parameter __lzifu_ncomponents to control which data 
        derived from the LZIFU fits is loaded. 

    bin_type:                   str
        Spatial binning strategy. Options are "default" (unbinned), "adaptive"
        (Voronoi binning) or "sectors" (sector binning)

    eline_SNR_min:              int 
        Minimum emission line flux S/N to adopt when making S/N and data 
        quality cuts.

    correct_extinction:         bool
        If True, correct emission line fluxes for extinction. 

        Note that the Halpha equivalent widths are NOT corrected for extinction if 
        correct_extinction is True. This is because stellar continuum extinction 
        measurements are not available, and so applying the correction only to the 
        Halpha fluxes may over-estimate the true EW.

    sigma_gas_SNR_min:          float (optional)
        Minimum velocity dipersion S/N to accept. Defaults to 3.

    line_flux_SNR_cut:          bool (optional)
        Whether to NaN emission line components AND total fluxes 
        (corresponding to emission lines in eline_list) below a specified S/N 
        threshold, given by eline_SNR_min. The S/N is simply the flux dividied 
        by the formal 1sigma uncertainty on the flux. Default: True.

    missing_fluxes_cut:         bool (optional)
        Whether to NaN out "missing" fluxes - i.e., cells in which the flux
        of an emission line (total or per component) is NaN, but the error 
        is not for some reason. Default: True.

    line_amplitude_SNR_cut:     bool (optional)
        If True, removes components with Gaussian amplitudes < 3 * RMS of the 
        continuum in the vicinity of Halpha. By default this is set to True
        because this does well at removing shallow components which are most 
        likely due to errors in the stellar continuum fit. Default: True.

    flux_fraction_cut:          bool (optional)
        If True, and if ncomponents > 1, remove intermediate and broad 
        components with line amplitudes < 0.05 that of the narrow componet.
        Set to False by default b/c it's unclear whether this is needed to 
        reject unreliable components. Default: False.

    sigma_gas_SNR_cut:          bool (optional)
        If True, mask component velocity dispersions where the S/N on the 
        velocity dispersion measurement is below sigma_gas_SNR_min. 
        By default this is set to True as it's a robust way to account for 
        emission line widths < instrumental.  Default: True.

    vgrad_cut:                  bool (optional)     
        If True, mask component kinematics (velocity and velocity dispersion)
        that are likely to be affected by beam smearing.
        By default this is set to False because it tends to remove nuclear spaxels 
        which may be of interest to your science case, & because it doesn't 
        reliably remove spaxels with quite large beam smearing components.
        Default: False.

    stekin_cut:                 bool (optional)
        If True, mask stellar kinematic quantities that do not meet the DQ and 
        S/N requirements specified in Croom et al. (2021). Default: True.

    metallicity_diagnostics:    list of str (optional)
        List of strong-line metallicity diagnostics to compute. 
        Options:
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

    eline_list:                 list of str (optional)
        List of emission lines to use. Defaults to the full list of lines fitted 
        in SAMI DR3.

    nthreads:                   int (optional)           
        Maximum number of threads to use. Defaults to os.cpu_count().

    __use_lzifu_fits:           bool (optional)
        If True, load the DataFrame containing emission line quantities
        (including fluxes, kinematics, etc.) derived directly from the LZIFU
        output FITS files, rather than those included in DR3. Default: False.

    __lzifu_ncomponents:        str  (optional)
        Number of components corresponding to the LZIFU fit, if 
        __use_lzifu_fits is specified. May be '1', '2', '3' or 'recom'. Note 
        that this keyword ONLY affects emission line fluxes and gas kinematics;
        other quantities including SFR/SFR surface densities and HALPHA 
        extinction correction factors are loaded from DR3 data products as per
        the ncomponents keyword. Default: False.

    debug:                      bool (optional)
        If True, run on a subset of the entire sample (10 galaxies) and save
        the output with "_DEBUG" appended to the filename. This is useful for
        tweaking S/N and DQ cuts since running the function on the entire 
        sample is quite slow.  Default: False.

    OUTPUTS
    ---------------------------------------------------------------------------
    The resulting DataFrame will be stored as 

        settings["sami"]["output_path"]/sami_{bin_type}_{ncomponents}-comp_extcorr_minSNR={eline_SNR_min}.hd5

    if correct_extinction is True, or else

        settings["sami"]["output_path"]/sami_{bin_type}_{ncomponents}-comp_minSNR={eline_SNR_min}.hd5

    The DataFrame will be stored in CSV format in case saving in HDF format 
    fails for any reason.

    PREREQUISITES
    ---------------------------------------------------------------------------
    make_sami_metadata_df() must be run first.

    SAMI data products must be downloaded from DataCentral

        https://datacentral.org.au/services/download/

    and stored as follows: 

        settings["sami"]["input_path"]/ifs/<gal>/<gal>_<quantity>_<bin type>_<number of components>-comp.fits

    This is essentially the default file structure when data products are 
    downloaded from DataCentral and unzipped:

        sami/dr3/ifs/<gal>/<gal>_<quantity>_<bin type>_<number of components>-comp.fits

    The following data products are required to run this script:

        Halpha_{bin_type}_{ncomponents}-comp.fits,
        Hbeta_{bin_type}_{ncomponents}-comp.fits,
        NII6583_{bin_type}_{ncomponents}-comp.fits,
        OI6300_{bin_type}_{ncomponents}-comp.fits,
        OII3728_{bin_type}_{ncomponents}-comp.fits,
        OIII5007_{bin_type}_{ncomponents}-comp.fits,
        SII6716_{bin_type}_{ncomponents}-comp.fits,
        SII6731_{bin_type}_{ncomponents}-comp.fits,
        gas-vdisp_{bin_type}_{ncomponents}-comp.fits,
        gas-velocity_{bin_type}_{ncomponents}-comp.fits,
        stellar-velocity-dispersion_{bin_type}_two-moment.fits,
        stellar-velocity_{bin_type}_two-moment.fits,
        extinct-corr_{bin_type}_{ncomponents}-comp.fits,
        sfr-dens_{bin_type}_{ncomponents}-comp.fits,
        sfr_{bin_type}_{ncomponents}-comp.fits

    SAMI data cubes must also be downloaded from DataCentral and stored as follows: 

        settings["sami"]["data_cube_path"]/ifs/<gal>/<gal>_A_cube_<blue/red>.fits.gz

    settings["sami"]["data_cube_path"] can be the same as settings["sami"]["output_path"].
    """

    #######################################################################
    # Input checking
    #######################################################################
    assert (ncomponents == "recom") | (
        ncomponents == "1"), "ncomponents must be 'recom' or '1'!!"
    assert bin_type in [
        "default", "adaptive", "sectors"
    ], "bin_type must be 'default' or 'adaptive' or 'sectors'!!"
    if __use_lzifu_fits:
        assert __lzifu_ncomponents in [
            "recom", "1", "2", "3"
        ], "__lzifu_ncomponents must be 'recom', '1', '2' or '3'!!"
        assert os.path.exists(
            __lzifu_products_path
        ), f"lzifu_products_path directory {__lzifu_products_path} not found!!"
        logger.warning(
            "using LZIFU {__lzifu_ncomponents}-component fits to obtain emission line fluxes & kinematics, NOT DR3 data products!!",
            RuntimeWarning)

    logger.info(f"input parameters: bin_type={bin_type}, ncomponents={ncomponents}, debug={debug}, eline_SNR_min={eline_SNR_min}, correct_extinction={correct_extinction}")

    # Determine number of threads
    if nthreads is None:
        nthreads = os.cpu_count()
        logger.warning(f"nthreads not specified: running make_sami_metadata_df() on {nthreads} threads...")

    ###############################################################################
    # Filenames
    #######################################################################
    df_metadata_fname = "sami_dr3_metadata.hd5"

    # Output file names
    if correct_extinction:
        df_fname = f"sami_{bin_type}_{ncomponents}-comp_extcorr_minSNR={eline_SNR_min}"
    else:
        df_fname = f"sami_{bin_type}_{ncomponents}-comp_minSNR={eline_SNR_min}"
    if __use_lzifu_fits:
        df_fname += f"_lzifu_{__lzifu_ncomponents}-comp"
    if debug:
        df_fname += "_DEBUG"
    df_fname += ".hd5"

    logger.info(f"saving to file {df_fname}...")

    ###############################################################################
    # Read metadata
    ###############################################################################
    try:
        df_metadata = pd.read_hdf(output_path / df_metadata_fname,
                                  key="metadata")
    except FileNotFoundError:
        logger.error(
            f"metadata DataFrame file not found ({output_path / df_metadata_fname}). Please run make_sami_metadata_df.py first!"
        )

    # Only include galaxies flagged as "good" & for which we have data
    gal_ids_dq_cut = df_metadata[df_metadata["Good?"] == True].index.values
    gal_ids_dq_cut = [
        g for g in gal_ids_dq_cut
        if os.path.exists(input_path / f"ifs/{g}/")
    ]

    # If running in DEBUG mode, run on a subset to speed up execution time
    if debug:
        gal_ids_dq_cut_debug = gal_ids_dq_cut[:10]
        for gal in [
                572402, 209807
        ]:  # Add these two galaxies because they are very distinctive, making debugging a bit easier (for me at least...)
            if gal in gal_ids_dq_cut:
                gal_ids_dq_cut_debug += [gal]
        gal_ids_dq_cut = gal_ids_dq_cut_debug
        # Also only run on a subset of metallicity diagnostics to speed up execution time
        metallicity_diagnostics = ["N2Ha_PP04", "N2Ha_K19"]

    # Cast to flaot to avoid issues around Object data types
    df_metadata["Good?"] = df_metadata["Good?"].astype("float")

    ###############################################################################
    # Scrape measurements for each galaxy from FITS files
    ###############################################################################
    args_list = [[
        gg, gal, ncomponents, bin_type, df_metadata,
        __use_lzifu_fits, __lzifu_ncomponents
    ] for gg, gal in enumerate(gal_ids_dq_cut)]

    if len(gal_ids_dq_cut) == 1:
        res_list = [_process_gals(args_list[0])]
    else:
        if nthreads > 1:
            logger.info(f"Beginning pool...")
            pool = multiprocessing.Pool(
                min([nthreads, len(gal_ids_dq_cut)]))
            res_list = np.array((pool.map(_process_gals, args_list)), dtype=object)
            pool.close()
            pool.join()
        else:
            logger.info(f"Running sequentially...")
            res_list = []
            for args in args_list:
                res = _process_gals(args)
                res_list.append(res)

    ###############################################################################
    # Convert to a Pandas DataFrame
    ###############################################################################
    rows_list_all = [r[0] for r in res_list]
    colnames = res_list[0][1]
    df_spaxels = pd.DataFrame(np.vstack(tuple(rows_list_all)),
                              columns=colnames)

    # Merge with metadata
    df_spaxels = df_spaxels.merge(df_metadata, on="ID", how="left")

    ###############################################################################
    # Add extra columns
    ###############################################################################
    df_spaxels["r/R_e"] = df_spaxels[
        "r (relative to galaxy centre, deprojected, arcsec)"] / df_spaxels[
            "R_e (arcsec)"]

    ###############################################################################
    # Generic stuff: compute additional columns - extinction, metallicity, etc.
    ###############################################################################
    df_spaxels = add_columns(df_spaxels,
                             eline_SNR_min=eline_SNR_min,
                             sigma_gas_SNR_min=sigma_gas_SNR_min,
                             eline_list=eline_list,
                             line_flux_SNR_cut=line_flux_SNR_cut,
                             missing_fluxes_cut=missing_fluxes_cut,
                             line_amplitude_SNR_cut=line_amplitude_SNR_cut,
                             flux_fraction_cut=flux_fraction_cut,
                             sigma_gas_SNR_cut=sigma_gas_SNR_cut,
                             vgrad_cut=vgrad_cut,
                             stekin_cut=stekin_cut,
                             correct_extinction=correct_extinction,
                             metallicity_diagnostics=metallicity_diagnostics,
                             compute_sfr=False,
                             sigma_inst_kms=settings["sami"]["sigma_inst_kms"],
                             nthreads=nthreads,
                             __use_lzifu_fits=__use_lzifu_fits,
                             __lzifu_ncomponents=__lzifu_ncomponents)

    ###############################################################################
    # Save
    ###############################################################################
    logger.info(f"saving to file {df_fname}...")

    # Remove object-type columns
    bad_cols = [c for c in df_spaxels if df_spaxels[c].dtype == "object"]
    if len(bad_cols) > 0:
        logger.warning(f"The following object-type columns are present in the DataFrame: {','.join(bad_cols)}")

    try:
        df_spaxels.to_hdf(output_path / df_fname,
                          key=f"{bin_type}{ncomponents}comp")
    except:
        logger.error(
            f"unable to save to HDF file! Saving to .csv instead"
        )
        df_spaxels.to_csv(output_path / df_fname.split("hd5")[0] + "csv")
    
    logger.info("finished!")
    return


###############################################################################
def load_sami_df(ncomponents,
                 bin_type,
                 correct_extinction,
                 eline_SNR_min,
                 __use_lzifu_fits=False,
                 __lzifu_ncomponents='3',
                 debug=False):
    """
    DESCRIPTION
    ---------------------------------------------------------------------------
    Load and return the Pandas DataFrame containing spaxel-by-spaxel 
    information for all SAMI galaxies which was created using make_sami_df(),
    making a series of optional S/N and data quality cuts.

    INPUTS
    ---------------------------------------------------------------------------
    ncomponents:        str
        Number of components; may either be "1" (corresponding to the 
        1-component Gaussian fits) or "recom" (corresponding to the multi-
        component Gaussian fits).

    bin_type:           str
        Binning scheme used. Must be one of 'default' or 'adaptive' or 
        'sectors'.

    correct_extinction: bool
        If True, load the DataFrame in which the emission line fluxes (but not 
        EWs) have been corrected for intrinsic extinction.

    eline_SNR_min:      int 
        Minimum flux S/N to accept. Fluxes below the threshold (plus associated
        data products) are set to NaN.

    eline_list:                 list of str
        List of emission lines to which the flagging operations are applied
        for S/N cuts, etc. 

    __use_lzifu_fits:           bool (optional)
        If True, load the DataFrame containing emission line quantities
        (including fluxes, kinematics, etc.) derived directly from the LZIFU
        output FITS files, rather than those included in DR3. 

    __lzifu_ncomponents:        str  (optional)
        Number of components corresponding to the LZIFU fit, if 
        __use_lzifu_fits is specified. May be '1', '2', '3' or 'recom'. Note 
        that this keyword ONLY affects emission line fluxes and gas kinematics;
        other quantities including SFR/SFR surface densities and HALPHA 
        extinction correction factors are loaded from DR3 data products as per
        the ncomponents keyword. 

    debug:                      bool
        If True, load the "debug" version of the DataFrame created when 
        running make_sami_df() with debug=True.
    
    USAGE
    ---------------------------------------------------------------------------
    load_sami_df() is called as follows:

        >>> from spaxelsleuth.loaddata.sami import load_sami_df
        >>> df = load_sami_df(ncomponents, bin_type, correct_extinction, 
                              eline_SNR_min, debug)

    OUTPUTS
    ---------------------------------------------------------------------------
    The Dataframe.

    """
    #######################################################################
    # INPUT CHECKING
    #######################################################################
    assert (ncomponents == "recom") | (
        ncomponents == "1"), "ncomponents must be 'recom' or '1'!!"
    assert bin_type in settings["sami"][
        "bin_types"], f"bin_type must be one of {', '.join(settings['sami']['bin_types'])}"
    if __use_lzifu_fits:
        assert __lzifu_ncomponents in [
            "recom", "1", "2", "3"
        ], "__lzifu_ncomponents must be 'recom', '1', '2' or '3'!!"
        assert os.path.exists(
            __lzifu_products_path
        ), f"lzifu_products_path directory {__lzifu_products_path} not found!!"
        logger.warning(
            "using LZIFU {__lzifu_ncomponents}-component fits to obtain emission line fluxes & kinematics, NOT DR3 data products!!",
            RuntimeWarning)

    # Input file name
    df_fname = f"sami_{bin_type}_{ncomponents}-comp"
    if correct_extinction:
        df_fname += "_extcorr"
    df_fname += f"_minSNR={eline_SNR_min}"
    if __use_lzifu_fits:
        df_fname += f"_lzifu_{__lzifu_ncomponents}-comp"
    if debug:
        df_fname += "_DEBUG"
    df_fname += ".hd5"

    assert os.path.exists(output_path / df_fname),\
        f"File {output_path / df_fname} does does not exist!"

    # Load the data frame
    t = os.path.getmtime(output_path / df_fname)
    logger.info(
        f"Loading DataFrame from file {output_path / df_fname} [last modified {datetime.datetime.fromtimestamp(t)}]..."
    )
    df = pd.read_hdf(output_path / df_fname)

    # Add "metadata" columns to the DataFrame
    df["survey"] = "sami"
    df["as_per_px"] = settings["sami"]["as_per_px"]
    df["N_x"] = settings["sami"]["N_x"]
    df["N_y"] = settings["sami"]["N_y"]
    df["x0_px"] = settings["sami"]["x0_px"]
    df["y0_px"] = settings["sami"]["y0_px"]
    df["ncomponents"] = ncomponents
    df["bin_type"] = bin_type
    df["__use_lzifu_fits"] = __use_lzifu_fits
    df["__lzifu_ncomponents"] = __lzifu_ncomponents
    df["debug"] = debug
    df["flux units"] = "E-16 erg/cm^2/s"  # Units of continuum & emission line flux
    df["continuum units"] = "E-16 erg/cm^2/Å/s"  # Units of continuum & emission line flux

    # Add back in object-type columns
    df["x, y (pixels)"] = list(
    zip(df["x (projected, arcsec)"] / 0.5,
        df["y (projected, arcsec)"] / 0.5))
    df["Morphology"] = morph_num_to_str(df["Morphology (numeric)"])
    df["BPT (total)"] = bpt_num_to_str(df["BPT (numeric) (total)"])

    # Return
    logger.info("Finished!")
    return df.sort_index()


###############################################################################
def load_sami_metadata_df():
    """Load the SAMI metadata DataFrame, containing "metadata" for each galaxy."""
    if not os.path.exists(output_path / "sami_dr3_metadata.hd5"):
        raise FileNotFoundError(
            f"File {output_path / 'sami_dr3_metadata.hd5'} not found. Did you remember to run make_sami_metadata_df() first?"
        )
    df_metadata = pd.read_hdf(output_path / "sami_dr3_metadata.hd5")

    # Add back in object-type columns
    df_metadata["Morphology"] = morph_num_to_str(df_metadata["Morphology (numeric)"])

    return df_metadata
