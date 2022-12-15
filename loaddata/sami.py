"""
File:       sami.py
Author:     Henry Zovaro
Email:      henry.zovaro@anu.edu.au

DESCRIPTION
------------------------------------------------------------------------------
This script contains the following functions:

    make_sami_metadata_df():
        This function is used to create a DataFrame containing "metadata", including
        stellar masses, spectroscopic redshifts, morphologies and other information
        for each galaxy in SAMI.

        This script must be run before make_sami_df() as the resulting DataFrame
        is used there.

        The information used is from the catalogues are available at 
        https://datacentral.org.au/. See the function docstring for details
        on which catalogues are needed.
    
    make_sami_df():     
        This function is used to create a Pandas DataFrame containing emission line 
        fluxes & kinematics, stellar kinematics, extinction, star formation rates, 
        and other quantities for individual spaxels in SAMI galaxies as taken from 
        SAMI DR3.

        The output is stored in HDF format as a Pandas DataFrame in which each row 
        corresponds to a given spaxel (or Voronoi bin) for every galaxy. 

    _process_gals():
        A helper function used in make_sami_df() to multithread processing of
        individual galaxies.

    _compute_snr():
        A helper function used in make_sami_metadata_df() to multithread 
        continuum SNR computations.

    load_sami_df():
        load a DataFrame containing emission line fluxes, etc. that was created 
        using make_sami_df().

PREREQUISITES
------------------------------------------------------------------------------
SAMI_DIR and SAMI_DATACUBE_DIR must be defined as an environment variable.

See function docstrings for specific prerequisites.

------------------------------------------------------------------------------
Copyright (C) 2022 Henry Zovaro
"""
###############################################################################
# Imports
import datetime
import os
import pandas as pd
import numpy as np
from itertools import product
from scipy import constants
from astropy.cosmology import FlatLambdaCDM
from astropy.io import fits
from tqdm import tqdm
import multiprocessing

from spaxelsleuth.utils import dqcut, linefns, metallicity, extcorr

import matplotlib.pyplot as plt
plt.ion()
plt.close("all")

from IPython.core.debugger import Tracer

import warnings
warnings.filterwarnings(action="ignore", message="Mean of empty slice")
warnings.filterwarnings(action="ignore", message="invalid value encountered in sqrt")

###############################################################################
# Paths
sami_data_path = os.environ["SAMI_DIR"]
assert "SAMI_DIR" in os.environ, "Environment variable SAMI_DIR is not defined!"
sami_datacube_path = os.environ["SAMI_DATACUBE_DIR"]
assert "SAMI_DATACUBE_DIR" in os.environ, "Environment variable SAMI_DATACUBE_DIR is not defined!"

# Path for LZIFU data products
__lzifu_products_path = "/priv/sami/sami_data/Final_SAMI_data/LZIFU/lzifu_default_products_old"

###############################################################################
# For computing median continuum S/N values in make_sami_metadata_df()
def _compute_snr(args, plotit=False):
    gal, df_metadata = args

    # Load the red & blue data cubes.
    hdulist_R_cube = fits.open(os.path.join(sami_datacube_path, f"ifs/{gal}/{gal}_A_cube_red.fits.gz"))
    hdulist_B_cube = fits.open(os.path.join(sami_datacube_path, f"ifs/{gal}/{gal}_A_cube_blue.fits.gz"))
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
    x0_px = 25.5
    y0_px = 25.5
    as_per_px = 0.5
    ys, xs = np.meshgrid(np.arange(50), np.arange(50), indexing="ij")
    x_cc = xs - x0_px  # pixels
    y_cc = ys - y0_px  # pixels
    x_prime = x_cc * np.cos(beta_rad) + y_cc * np.sin(beta_rad)
    y_prime_projec = (- x_cc * np.sin(beta_rad) + y_cc * np.cos(beta_rad))
    y_prime = (- x_cc * np.sin(beta_rad) + y_cc * np.cos(beta_rad)) / np.cos(i_rad)
    r_prime = np.sqrt(x_prime**2 + y_prime**2)

    # Convert to arcsec
    r_prime_as = r_prime * as_per_px

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
    print(f"In make_sami_metadata_df(): Finished processing {gal}")
    return [gal, SNR_full_B, SNR_full_R, 
                 SNR_1Re_B, SNR_1Re_R, 
                 SNR_15Re_B, SNR_15Re_R, 
                 SNR_2Re_B, SNR_2Re_R]

###############################################################################
def make_sami_metadata_df(recompute_continuum_SNRs=False, nthreads=20):
    """
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

        SAMI_DIR/sami_dr3_metadata.hd5

    The DataFrame containing continuum S/N values is saved to 

        SAMI_DIR/sami_dr3_aperture_snrs.hd5

    PREREQUISITES
    ---------------------------------------------------------------------------
    SAMI_DIR must be defined as an environment variable.

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
    print("In make_sami_metadata_df(): Creating metadata DataFrame...")
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
    data_path = os.path.join(__file__.split("loaddata")[0], "data")
    for fname in [gama_metadata_fname, cluster_metadata_fname, 
                  filler_metadata_fname, morphologies_fname, 
                  flag_metadata_fname, mge_fits_metadata_fname]:
        assert os.path.exists(os.path.join(data_path, fname)),\
            f"File {os.path.join(data_path, fname)} not found!"

    ###########################################################################
    # Read in galaxy metadata
    ###########################################################################
    df_metadata_gama = pd.read_csv(os.path.join(data_path, gama_metadata_fname))  # ALL possible GAMA targets
    df_metadata_cluster = pd.read_csv(os.path.join(data_path, cluster_metadata_fname))  # ALL possible cluster targets
    df_metadata_filler = pd.read_csv(os.path.join(data_path, filler_metadata_fname))  # ALL possible filler targets
    df_metadata = pd.concat([df_metadata_gama, df_metadata_cluster, df_metadata_filler]).drop(["Unnamed: 0"], axis=1)
    gal_ids_metadata = list(np.sort(list(df_metadata["catid"])))

    ###########################################################################
    # Append morphology data
    ###########################################################################
    df_morphologies = pd.read_csv(os.path.join(data_path, morphologies_fname)).drop(["Unnamed: 0"], axis=1)
    df_morphologies = df_morphologies.rename(columns={"type": "Morphology (numeric)"})

    # Morphologies (numeric) - merge "?" and "no agreement" into a single category.
    df_morphologies.loc[df_morphologies["Morphology (numeric)"] == 5.0, "Morphology (numeric)"] = -0.5
    df_morphologies.loc[df_morphologies["Morphology (numeric)"] == -9.0, "Morphology (numeric)"] = -0.5
    df_morphologies.loc[df_morphologies["Morphology (numeric)"] == np.nan, "Morphology (numeric)"] = -0.5

    # Key: Morphological Type
    morph_dict = {
        "0.0": "E",
        "0.5": "E/S0",
        "1.0": "S0",
        "1.5": "S0/Early-spiral",
        "2.0": "Early-spiral",
        "2.5": "Early/Late spiral",
        "3.0": "Late spiral",
        "5.0": "?",
        "-9.0": "no agreement",
        "-0.5": "Unknown"
    }
    df_morphologies["Morphology"] = [morph_dict[str(m)] for m in df_morphologies["Morphology (numeric)"]]

    # merge with metadata, but do NOT include the morphology column as it 
    # causes all data to be cast to "object" type which is extremely slow!!!
    # Note: this step trims df_metadata to include only those objects with morphologies (9949 --> 3068)
    df_metadata = df_metadata.merge(df_morphologies[["catid", "Morphology (numeric)", "Morphology"]], on="catid")

    ###########################################################################
    # Read in flag metadata
    ###########################################################################
    df_flags = pd.read_csv(os.path.join(data_path, flag_metadata_fname)).drop(["Unnamed: 0"], axis=1)
    df_flags = df_flags.astype({col: "int64" for col in df_flags.columns if col.startswith("warn")})
    df_flags = df_flags.astype({"isbest": bool})

    # Get rid of rows failing the following data quality criteria
    cond = df_flags["isbest"] == True
    cond &= df_flags["warnstar"] == 0
    cond &= df_flags["warnmult"] < 2  # multiple objects overlapping with galaxy area
    cond &= df_flags["warnfcal"] == 0  # flux calibration issues
    cond &= df_flags["warnfcbr"] == 0  # flux calibration issues
    cond &= df_flags["warnskyb"] == 0  # bad sky subtraction residuals
    cond &= df_flags["warnskyr"] == 0  # bad sky subtraction residuals
    cond &= df_flags["warnre"] == 0  # significant difference between standard & MGE Re
    df_flags_cut = df_flags[cond]

    for gal in df_flags_cut["catid"]:
        if df_flags_cut[df_flags_cut["catid"] == gal].shape[0] > 1:
            # If there are two "best" observations, drop the second one.
            drop_idxs = df_flags_cut.index[df_flags_cut["catid"] == gal][1:]
            df_flags_cut = df_flags_cut.drop(drop_idxs)

    if df_flags_cut.shape[0] != len(df_flags_cut["catid"].unique()):
        Tracer()() 

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
    df_mge = pd.read_csv(os.path.join(data_path, mge_fits_metadata_fname)).drop(["Unnamed: 0"], axis=1).set_index("catid") # Data from multi-expansion fits

    # Drop duplicated rows: those with both SDSS and VST photometry
    df_mge_vst = df_mge[df_mge["photometry"] == "VST"]
    df_mge_sdss = df_mge[df_mge["photometry"] == "SDSS"]
    gals_vst_and_sdss = [g for g in df_mge_vst.index.values if g in df_mge_sdss.index.values]
    df_mge = df_mge.sort_values(by="photometry")    # Sort so that photometry values are alphabetically sorted
    bad_rows = df_mge.index.duplicated(keep="last") # Find duplicate rows - these will all be galaxies with both SDSS and VST data, in which case we keep the VST measurement
    df_mge = df_mge[~bad_rows]
    # Check that all of the galaxies with both VST and SDSS entries have VST photometry
    for gal in gals_vst_and_sdss:
        assert df_mge.loc[gal, "photometry"] == "VST"
    # Merge 
    df_metadata = df_metadata.merge(df_mge, how="left", left_index=True, right_index=True)

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
    Morphology        keep   
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
    # Drop unnecessary columns & rename others for readability
    cols_to_remove = ["r_auto", "r_petro", "surv_sami", "rextinction", "dist2nneigh", "chi2", "fillflag"]
    rename_dict = {
        "a_g": "A_g",
        "bad_class": "Bad class #",
        "catid": "ID",
        "dec_obj": "Dec (J2000)",
        "ra_obj": "RA (J2000)",
        "dec_ifu": "Dec (IFU) (J2000)",
        "ra_ifu": "RA (IFU) (J2000)",
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
        "z_spec": "z",
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
    # Add angular scale info
    ###########################################################################
    print(f"In make_sami_metadata_df(): Computing distances...")
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    for gal in gal_ids_dq_cut:
        D_A_Mpc = cosmo.angular_diameter_distance(df_metadata.loc[gal, "z"]).value
        D_L_Mpc = cosmo.luminosity_distance(df_metadata.loc[gal, "z"]).value
        df_metadata.loc[gal, "D_A (Mpc)"] = D_A_Mpc
        df_metadata.loc[gal, "D_L (Mpc)"] = D_L_Mpc
    df_metadata["kpc per arcsec"] = df_metadata["D_A (Mpc)"] * 1e3 * np.pi / 180.0 / 3600.0
    df_metadata["R_e (kpc)"] = df_metadata["R_e (arcsec)"] * df_metadata["kpc per arcsec"]
    df_metadata["R_e (MGE) (kpc)"] = df_metadata["R_e (MGE) (arcsec)"] * df_metadata["kpc per arcsec"]
    df_metadata["log(M/R_e)"] = df_metadata["log M_*"] - np.log10(df_metadata["R_e (kpc)"])
    df_metadata["log(M/R_e^2)"] = df_metadata["log M_*"] - 2 * np.log10(df_metadata["R_e (kpc)"])

    ###########################################################################
    # Compute inclination
    ###########################################################################
    e = df_metadata["e"]
    PA = df_metadata["PA (degrees)"]
    beta_rad = np.deg2rad(PA - 90)
    b_over_a = 1 - e
    q0 = 0.2
    i_rad = np.arccos(np.sqrt((b_over_a**2 - q0**2) / (1 - q0**2)))  # Want to store this!
    df_metadata["i (degrees)"] = np.rad2deg(i_rad)

    ###############################################################################
    # Compute continuum SNRs from the data cubes
    ###############################################################################
    print("In make_sami_metadata_df(): Computing continuum SNRs...")
    if not recompute_continuum_SNRs and os.path.exists(os.path.join(sami_data_path, "sami_dr3_aperture_snrs.hd5")):
        print(f"In make_sami_metadata_df(): WARNING: file {os.path.join(sami_data_path, 'sami_dr3_aperture_snrs.hd5')} found; loading SNRs from existing DataFrame...")
        df_snr = pd.read_hdf(os.path.join(sami_data_path, "sami_dr3_aperture_snrs.hd5"), key="SNR")
    else:
        print(f"In make_sami_metadata_df(): WARNING: computing continuum SNRs on {nthreads} threads...")
        print("In make_sami_metadata_df(): Beginning pool...")
        args_list = [[gal, df_metadata] for gal in gal_ids_dq_cut]
        pool = multiprocessing.Pool(nthreads)
        res_list = np.array((pool.map(_compute_snr, args_list)))
        pool.close()
        pool.join()

        ###########################################################################
        # Create DataFrame from results
        ###############################################################################
        df_snr = pd.DataFrame(np.vstack(res_list), columns=["ID",
                                                            "Median SNR (B, full field)",
                                                            "Median SNR (R, full field)",
                                                            "Median SNR (B, 1R_e)",
                                                            "Median SNR (R, 1R_e)",
                                                            "Median SNR (B, 1.5R_e)",
                                                            "Median SNR (R, 1.5R_e)",
                                                            "Median SNR (B, 2R_e)",
                                                            "Median SNR (R, 2R_e)"])
        df_snr["ID"] = df_snr["ID"].astype(int)
        df_snr.set_index("ID")

        # Save 
        print("In make_sami_metadata_df(): Saving aperture SNR DataFrame to file...")
        df_snr.to_hdf(os.path.join(sami_data_path, "sami_dr3_aperture_snrs.hd5"), key="SNR")

    ###############################################################################
    # Merge with the metadata DataFrame
    ###############################################################################
    common_cols = [c for c in df_metadata.columns if c not in df_snr.columns]
    df_snr = df_snr.set_index("ID")
    df_metadata = pd.concat([df_snr, df_metadata], axis=1)

    ###########################################################################
    # Save to file
    ###########################################################################
    print(f"In make_sami_metadata_df(): Saving metadata DataFrame to file {os.path.join(sami_data_path, df_fname)}...")
    df_metadata.to_hdf(os.path.join(sami_data_path, df_fname), key="metadata")

    print(f"In make_sami_metadata_df(): Finished!")
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
        status_str.

    OUTPUTS
    ---------------------------------------------------------------------------
    DataFrame rows and corresponding columns corresponding to galaxy gal.

    """
    # Extract input arguments
    gal_idx, gal, ncomponents, bin_type, df_metadata, status_str, use_lzifu_fits, lzifu_ncomponents = args

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
    fnames = [os.path.join(sami_data_path, f"ifs/{gal}/{gal}_A_{f}.fits") for f in fname_list]

    # X, Y pixel coordinates
    ys, xs = np.meshgrid(np.arange(50), np.arange(50), indexing="ij")
    as_per_px = 0.5
    ys_as = ys * as_per_px
    xs_as = xs * as_per_px

    # Centre galaxy coordinates (see p16 of Croom+2021)
    x0_px = 25.5
    y0_px = 25.5

    # If True, plot the bin coordinates before and after de-projection
    # Should not be used if multithreading is enabled!
    plotit = False

    #######################################################################
    # Open the red & blue cubes.
    hdulist_B_cube = fits.open(os.path.join(sami_datacube_path, f"ifs/{gal}/{gal}_A_cube_blue.fits.gz"))
    hdulist_R_cube = fits.open(os.path.join(sami_datacube_path, f"ifs/{gal}/{gal}_A_cube_red.fits.gz"))

    #######################################################################
    # Compute the d4000 Angstrom break.
    header = hdulist_B_cube[0].header
    data_cube_B = hdulist_B_cube[0].data
    var_cube_B = hdulist_B_cube[1].data
    hdulist_B_cube.close()

    # Wavelength values
    lambda_0_A = header["CRVAL3"] - header["CRPIX3"] * header["CDELT3"]
    dlambda_A = header["CDELT3"]
    N_lambda = header["NAXIS3"]
    lambda_vals_B_A = np.array(range(N_lambda)) * dlambda_A + lambda_0_A 

    # Compute the D4000Å break
    # Definition from Balogh+1999 (see here: https://arxiv.org/pdf/1611.07050.pdf, page 3)
    start_b_idx = np.nanargmin(np.abs(lambda_vals_B_A / (1 + df_metadata.loc[gal, "z"]) - 3850))
    stop_b_idx = np.nanargmin(np.abs(lambda_vals_B_A / (1 + df_metadata.loc[gal, "z"]) - 3950))
    start_r_idx = np.nanargmin(np.abs(lambda_vals_B_A / (1 + df_metadata.loc[gal, "z"]) - 4000))
    stop_r_idx = np.nanargmin(np.abs(lambda_vals_B_A / (1 + df_metadata.loc[gal, "z"]) - 4100))
    N_b = stop_b_idx - start_b_idx
    N_r = stop_r_idx - start_r_idx

    # Convert datacube & variance cubes to units of F_nu
    data_cube_B_Hz = data_cube_B * lambda_vals_B_A[:, None, None]**2 / (constants.c * 1e10)
    var_cube_B_Hz2 = var_cube_B * (lambda_vals_B_A[:, None, None]**2 / (constants.c * 1e10))**2

    num = np.nanmean(data_cube_B_Hz[start_r_idx:stop_r_idx], axis=0)
    denom = np.nanmean(data_cube_B_Hz[start_b_idx:stop_b_idx], axis=0)
    err_num = 1 / N_r * np.sqrt(np.nansum(var_cube_B_Hz2[start_r_idx:stop_r_idx], axis=0))
    err_denom = 1 / N_b * np.sqrt(np.nansum(var_cube_B_Hz2[start_b_idx:stop_b_idx], axis=0))

    d4000_map = num / denom
    d4000_map_err = d4000_map * np.sqrt((err_num / num)**2 + (err_denom / denom)**2)

    #######################################################################
    # Use the red cube to calculate the continuum intensity so 
    # that we can compute the Halpha equivalent width.
    # Units of 10**(-16) erg /s /cm**2 /angstrom /pixel
    # Continuum wavelength range taken from here: https://ui.adsabs.harvard.edu/abs/2019MNRAS.485.4024V/abstract
    header = hdulist_R_cube[0].header
    data_cube_R = hdulist_R_cube[0].data 
    var_cube_R = hdulist_R_cube[1].data  

    # Wavelength values
    lambda_0_A = header["CRVAL3"] - header["CRPIX3"] * header["CDELT3"]
    dlambda_A = header["CDELT3"]
    N_lambda = header["NAXIS3"]
    lambda_vals_R_A = np.array(range(N_lambda)) * dlambda_A + lambda_0_A 

    # Compute continuum intensity
    start_idx = np.nanargmin(np.abs(lambda_vals_R_A / (1 + df_metadata.loc[gal, "z"]) - 6500))
    stop_idx = np.nanargmin(np.abs(lambda_vals_R_A / (1 + df_metadata.loc[gal, "z"]) - 6540))
    cont_HALPHA_map = np.nanmean(data_cube_R[start_idx:stop_idx], axis=0)
    cont_HALPHA_map_std = np.nanstd(data_cube_R[start_idx:stop_idx], axis=0)
    cont_HALPHA_map_err = 1 / (stop_idx - start_idx) * np.sqrt(np.nansum(var_cube_R[start_idx:stop_idx], axis=0))
    hdulist_R_cube.close() 

    #######################################################################
    # Use the blue cube to calculate the approximate B-band continuum.
    # Units of 10**(-16) erg /s /cm**2 /angstrom /pixel
    header = hdulist_B_cube[0].header
    data_cube_B = hdulist_B_cube[0].data 
    var_cube_B = hdulist_B_cube[1].data  

    # Wavelength values
    lambda_0_A = header["CRVAL3"] - header["CRPIX3"] * header["CDELT3"]
    dlambda_A = header["CDELT3"]
    N_lambda = header["NAXIS3"]
    lambda_vals_B_A = np.array(range(N_lambda)) * dlambda_A + lambda_0_A 

    # Compute continuum intensity
    start_idx = np.nanargmin(np.abs(lambda_vals_B_A / (1 + df_metadata.loc[gal, "z"]) - 4000))
    stop_idx = np.nanargmin(np.abs(lambda_vals_B_A / (1 + df_metadata.loc[gal, "z"]) - 5000))
    cont_B_map = np.nanmean(data_cube_B[start_idx:stop_idx], axis=0)
    cont_B_map_std = np.nanstd(data_cube_B[start_idx:stop_idx], axis=0)
    cont_B_map_err = 1 / (stop_idx - start_idx) * np.sqrt(np.nansum(var_cube_B[start_idx:stop_idx], axis=0))
    hdulist_B_cube.close() 

    #######################################################################
    # Compute v_grad using eqn. 1 of Zhou+2017
    if not use_lzifu_fits:
        hdulist_v = fits.open(os.path.join(sami_data_path, f"ifs/{gal}/{gal}_A_gas-velocity_{bin_type}_{ncomponents}-comp.fits"))
        v = hdulist_v[0].data.astype(np.float64)
        hdulist_v.close()
    else:
        lzifu_fname = [f for f in os.listdir(__lzifu_products_path) if f.startswith(str(gal)) and f"{lzifu_ncomponents}_comp" in f][0]
        hdu_lzifu = fits.open(os.path.join(__lzifu_products_path, lzifu_fname))
        v = hdu_lzifu["V"].data.astype(np.float64)
        hdu_lzifu.close()

    # Compute v_grad for each spaxel in each component
    # in units of km/s/pixel
    v_grad = np.full_like(v, np.nan)
    for yy, xx in product(range(1, 49), range(1, 49)):
        v_grad[:, yy, xx] = np.sqrt(((v[:, yy, xx + 1] - v[:, yy, xx - 1]) / 2)**2 +\
                                    ((v[:, yy + 1, xx] - v[:, yy - 1, xx]) / 2)**2)
    
    #######################################################################
    # Measure the HALPHA amplitude-to-noise
    # We measure this as 
    #       (peak spectral value in window around Ha - mean R continuum flux density) / standard deviation in R continuum flux density 
    # As such, this value can be negative. 
    lambda_vals_rest_R_A = lambda_vals_R_A / (1 + df_metadata.loc[gal, "z"])
    lambda_vals_rest_R_A_cube = np.zeros(data_cube_R.shape)
    lambda_vals_rest_R_A_cube[:] = lambda_vals_rest_R_A[:, None, None]

    dv = 300
    lambda_c_A = dqcut.get_wavelength_from_velocity(6562.8, v[0], units="km/s")
    lambda_max_A = dqcut.get_wavelength_from_velocity(6562.8, v[0] + dv, units="km/s")
    lambda_min_A = dqcut.get_wavelength_from_velocity(6562.8, v[0] - dv, units="km/s")

    # Measure HALPHA amplitude-to-noise 
    # Store as "meas" to distinguish from A/N measurements for individual
    # emission line components 
    A_HALPHA_mask = (lambda_vals_rest_R_A_cube > lambda_min_A) & (lambda_vals_rest_R_A_cube < lambda_max_A)
    data_cube_masked_R = np.copy(data_cube_R)
    data_cube_masked_R[~A_HALPHA_mask] = np.nan
    A_HALPHA_map = np.nanmax(data_cube_masked_R, axis=0)
    AN_HALPHA_map = (A_HALPHA_map - cont_HALPHA_map) / cont_HALPHA_map_std

    #######################################################################
    # Compute the spaxel or bin coordinates, depending on the binning scheme 
    im = np.nansum(data_cube_B, axis=0)
    if bin_type == "default":
        # Create an image from the datacube to figure out where are "good" spaxels
        if np.any(im.flatten() <= 0): # NaN out -ve spaxels. Most galaxies seem to have *some* -ve pixels
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
        # Open the binned blue cube. Get the bin mask extension.
        hdulist_binned_cube = fits.open(os.path.join(sami_data_path, f"ifs/{gal}/{gal}_A_{bin_type}_blue.fits.gz"))
        bin_map = hdulist_binned_cube[2].data.astype("float")
        bin_map[bin_map==0] = np.nan

        bin_number_list = np.array([nn for nn in np.unique(bin_map) if ~np.isnan(nn)])
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
            if (x_c < 0 or x_c >= 50 or y_c < 0 or y_c >= 50):
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
    # Calculate the inclination
    # I think beta = 90 - PA...
    # Transform coordinates into the galaxy plane
    e = df_metadata.loc[gal, "e"]
    PA = df_metadata.loc[gal, "PA (degrees)"]
    i_rad = np.deg2rad(df_metadata.loc[gal, "i (degrees)"])
    i_rad = 0 if np.isnan(i_rad) else i_rad
    beta_rad = np.deg2rad(PA - 90)

    #######################################################################
    # De-project the centroids to the coordinate system of the galaxy plane
    x_prime_list = np.full_like(x_c_list, np.nan, dtype="float")
    y_prime_list = np.full_like(x_c_list, np.nan, dtype="float")
    y_prime_projec_list = np.full_like(x_c_list, np.nan, dtype="float")
    r_prime_list = np.full_like(x_c_list, np.nan, dtype="float")
    for jj, coords in enumerate(zip(x_c_list, y_c_list)):
        x_c, y_c = coords
        # De-shift, de-rotate & de-incline
        x_cc = x_c - x0_px
        y_cc = y_c - y0_px
        x_prime = x_cc * np.cos(beta_rad) + y_cc * np.sin(beta_rad)
        y_prime_projec = (- x_cc * np.sin(beta_rad) + y_cc * np.cos(beta_rad))
        y_prime = (- x_cc * np.sin(beta_rad) + y_cc * np.cos(beta_rad)) / np.cos(i_rad)
        r_prime = np.sqrt(x_prime**2 + y_prime**2)

        # Add to list
        x_prime_list[jj] = x_prime
        y_prime_list[jj] = y_prime
        y_prime_projec_list[jj] = y_prime_projec
        r_prime_list[jj] = r_prime

    #######################################################################
    # For plotting
    if plotit:
        fig, axs = plt.subplots(ncols=2, figsize=(10, 5))
        axs[0].imshow(im, origin="lower")
        axs[1].axhline(0)
        axs[1].axvline(0)
        axs[0].scatter(x_c_list, y_c_list, color="k")
        axs[0].scatter(x0_px, y0_px, color="white")
        axs[1].scatter(x_prime_list, y_prime_list, color="r")
        axs[1].scatter(x_prime_list, y_prime_projec_list, color="r", alpha=0.3)
        axs[1].axis("equal")
        fig.canvas.draw()

    #######################################################################
    # Open each FITS file, extract the values from the maps in each bin & append
    rows_list = []
    colnames = []

    # Tidy up column names
    colname_dict = {
        f"stellar-velocity-dispersion_{bin_type}_two-moment" : "sigma_*",
        f"stellar-velocity_{bin_type}_two-moment" : "v_*",
        f"extinct-corr_{bin_type}_{ncomponents}-comp" : "HALPHA extinction correction",
        f"sfr-dens_{bin_type}_{ncomponents}-comp" : "SFR surface density",
        f"sfr_{bin_type}_{ncomponents}-comp" : "SFR",
        f"Halpha_{bin_type}_{ncomponents}-comp" : "HALPHA",
        f"Hbeta_{bin_type}_{ncomponents}-comp" : "HBETA",
        f"NII6583_{bin_type}_{ncomponents}-comp" : "NII6583",
        f"OI6300_{bin_type}_{ncomponents}-comp" : "OI6300",
        f"OII3728_{bin_type}_{ncomponents}-comp" : "OII3726+OII3729",
        f"OIII5007_{bin_type}_{ncomponents}-comp" : "OIII5007",
        f"SII6716_{bin_type}_{ncomponents}-comp" : "SII6716",
        f"SII6731_{bin_type}_{ncomponents}-comp" : "SII6731",
        f"gas-vdisp_{bin_type}_{ncomponents}-comp" : "sigma_gas",
        f"gas-velocity_{bin_type}_{ncomponents}-comp" : "v_gas",    
    } 

    for ff, fname in enumerate(fnames):
        hdu = fits.open(fname)
        data = hdu[0].data.astype(np.float64)
        data_err = hdu[1].data.astype(np.float64)
        hdu.close()

        #//////////////////////////////////////////////////////////////////////
        # HALPHA, SFR quantities
        if data.ndim > 2:
            if "Halpha" in fname or "sfr" in fname:
                # Add total fluxees
                thisrow = np.full_like(x_c_list, np.nan, dtype="float")
                thisrow_err = np.full_like(x_c_list, np.nan, dtype="float")
                for jj, coords in enumerate(zip(x_c_list, y_c_list)):
                    x_c, y_c = coords
                    y, x = (int(np.round(y_c)), int(np.round(x_c)))
                    if x > 49 or y > 49:
                        x = min([x, 49])
                        y = min([y, 49])
                    thisrow[jj] = data[0, y, x]
                    thisrow_err[jj] = data_err[0, y, x]
                rows_list.append(thisrow)
                rows_list.append(thisrow_err)
                colnames.append(f"{colname_dict[fname_list[ff]]} (total)")
                colnames.append(f"{colname_dict[fname_list[ff]]} error (total)")

                # Trim the 0th slice 
                data = data[1:]
                data_err = data_err[1:]

            # Add individual components 
            for nn in range(3 if ncomponents == "recom" else 1):
                thisrow = np.full_like(x_c_list, np.nan, dtype="float")
                thisrow_err = np.full_like(x_c_list, np.nan, dtype="float")
                for jj, coords in enumerate(zip(x_c_list, y_c_list)):
                    x_c, y_c = coords
                    y, x = (int(np.round(y_c)), int(np.round(x_c)))
                    if x > 49 or y > 49:
                        x = min([x, 49])
                        y = min([y, 49])
                    thisrow[jj] = data[nn, y, x]
                    thisrow_err[jj] = data_err[nn, y, x]
                rows_list.append(thisrow)
                rows_list.append(thisrow_err)
                colnames.append(f"{colname_dict[fname_list[ff]]} (component {nn + 1})")
                colnames.append(f"{colname_dict[fname_list[ff]]} error (component {nn + 1})")
        
        #//////////////////////////////////////////////////////////////////////
        # EXTINCTION, STELLAR KINEMATICS & EMISSION LINES EXCEPT FOR HALPHA 
        else:
            thisrow = np.full_like(x_c_list, np.nan, dtype="float")
            thisrow_err = np.full_like(x_c_list, np.nan, dtype="float")
            for jj, coords in enumerate(zip(x_c_list, y_c_list)):
                x_c, y_c = coords
                y, x = (int(np.round(y_c)), int(np.round(x_c)))
                if x > 49 or y > 49:
                    x = min([x, 49])
                    y = min([y, 49])
                thisrow[jj] = data[y, x]
                thisrow_err[jj] = data_err[y, x]
            rows_list.append(thisrow)
            rows_list.append(thisrow_err)

            # If adding the stellar kinematics, no point in adding "total" here
            if "stellar" in fname:  
                colnames.append(f"{colname_dict[fname_list[ff]]}")
                colnames.append(f"{colname_dict[fname_list[ff]]} error")
            # Otherwise append "total" to signify total fluxes.
            else:  
                colnames.append(f"{colname_dict[fname_list[ff]]} (total)")
                colnames.append(f"{colname_dict[fname_list[ff]]} error (total)")

    #######################################################################
    # Load LZIFU files
    if use_lzifu_fits:
        # Open the FITS file 
        lzifu_fname = [f for f in os.listdir(__lzifu_products_path) if f.startswith(str(gal)) and f"{lzifu_ncomponents}_comp" in f][0]
        hdu_lzifu = fits.open(os.path.join(__lzifu_products_path, lzifu_fname))

        #//////////////////////////////////////////////////////////////////////
        # Load emission line fluxes & kinematics (except for [OII])
        for quantity in ["HBETA", "OIII5007", "OI6300", 
                         "HALPHA", "NII6583", "SII6716", "SII6731"] +\
                        ["V", "VDISP"]:

            # Load data from the FITS file
            data = hdu_lzifu[f"{quantity}"].data.astype(np.float64)
            data_err = hdu_lzifu[f"{quantity}_ERR"].data.astype(np.float64)

            # Total fluxes 
            if quantity not in ["V", "VDISP"]:
                thisrow = np.full_like(x_c_list, np.nan, dtype="float")
                thisrow_err = np.full_like(x_c_list, np.nan, dtype="float")
                for jj, coords in enumerate(zip(x_c_list, y_c_list)):
                    x_c, y_c = coords
                    y, x = (int(np.round(y_c)), int(np.round(x_c)))
                    if x > 49 or y > 49:
                        x = min([x, 49])
                        y = min([y, 49])
                    thisrow[jj] = data[0, y, x]
                    thisrow_err[jj] = data_err[0, y, x]
                rows_list.append(thisrow)
                rows_list.append(thisrow_err)

                # Edit column name 
                if quantity == "V":
                    quantity_colname = "v_gas"
                elif quantity == "VDISP":
                    quantity_colname = "sigma_gas"
                else:
                    quantity_colname = quantity
                colnames.append(f"{quantity_colname} (total)")
                colnames.append(f"{quantity_colname} error (total)")

                # data = data[1:]
                # data_err = data_err[1:]

            # Fluxes/kinematics in components 1, 2 and 3
            for nn in range(data.shape[0] - 1):
                thisrow = np.full_like(x_c_list, np.nan, dtype="float")
                thisrow_err = np.full_like(x_c_list, np.nan, dtype="float")
                for jj, coords in enumerate(zip(x_c_list, y_c_list)):
                    x_c, y_c = coords
                    y, x = (int(np.round(y_c)), int(np.round(x_c)))
                    if x > 49 or y > 49:
                        x = min([x, 49])
                        y = min([y, 49])
                    thisrow[jj] = data[nn + 1, y, x]
                    thisrow_err[jj] = data_err[nn + 1, y, x]
                rows_list.append(thisrow)
                rows_list.append(thisrow_err)

                # Edit column name 
                if quantity == "V":
                    quantity_colname = "v_gas"
                elif quantity == "VDISP":
                    quantity_colname = "sigma_gas"
                else:
                    quantity_colname = quantity
                colnames.append(f"{quantity_colname} (component {nn + 1})")
                colnames.append(f"{quantity_colname} error (component {nn + 1})")

        #//////////////////////////////////////////////////////////////////////
        # OII doublet: these need to be combined to be consistent with the DR3 data products. 
        # We will store combined fluxes in column "OII3726+OII3729"
        data_OII3726 = hdu_lzifu[f"OII3726"].data.astype(np.float64)
        data_OII3726_err = hdu_lzifu[f"OII3726_ERR"].data.astype(np.float64)
        data_OII3729 = hdu_lzifu[f"OII3729"].data.astype(np.float64)
        data_OII3729_err = hdu_lzifu[f"OII3729_ERR"].data.astype(np.float64)
        data = data_OII3726 + data_OII3729
        data_err = np.sqrt(data_OII3726_err**2 + data_OII3729_err**2)

        # Total fluxes 
        thisrow = np.full_like(x_c_list, np.nan, dtype="float")
        thisrow_err = np.full_like(x_c_list, np.nan, dtype="float")
        for jj, coords in enumerate(zip(x_c_list, y_c_list)):
            x_c, y_c = coords
            y, x = (int(np.round(y_c)), int(np.round(x_c)))
            if x > 49 or y > 49:
                x = min([x, 49])
                y = min([y, 49])
            thisrow[jj] = data[0, y, x]
            thisrow_err[jj] = data_err[0, y, x]
        rows_list.append(thisrow)
        rows_list.append(thisrow_err)
        colnames.append(f"OII3726+OII3729 (total)")
        colnames.append(f"OII3726+OII3729 error (total)")

        # Fluxes in components 1, 2 and 3
        for nn in range(data.shape[0] - 1):
            thisrow = np.full_like(x_c_list, np.nan, dtype="float")
            thisrow_err = np.full_like(x_c_list, np.nan, dtype="float")
            for jj, coords in enumerate(zip(x_c_list, y_c_list)):
                x_c, y_c = coords
                y, x = (int(np.round(y_c)), int(np.round(x_c)))
                if x > 49 or y > 49:
                    x = min([x, 49])
                    y = min([y, 49])
                thisrow[jj] = data[nn + 1, y, x]
                thisrow_err[jj] = data_err[nn + 1, y, x]
            rows_list.append(thisrow)
            rows_list.append(thisrow_err)
            colnames.append(f"OII3726+OII3729 (component {nn + 1})")
            colnames.append(f"OII3726+OII3729 error (component {nn + 1})")

    ####################################################################### 
    # Do the same but with v_grad
    for nn in range(v_grad.shape[0]):
        thisrow = np.full_like(x_c_list, np.nan, dtype="float")
        thisrow_err = np.full_like(x_c_list, np.nan, dtype="float")
        for jj, coords in enumerate(zip(x_c_list, y_c_list)):
            x_c, y_c = coords
            y, x = (int(np.round(y_c)), int(np.round(x_c)))
            if x > 49 or y > 49:
                x = min([x, 49])
                y = min([y, 49])
            thisrow[jj] = v_grad[nn, y, x]
        rows_list.append(thisrow)
        colnames.append(f"v_grad (component {nn + 1})")       

    ####################################################################### 
    # Do the same but with HALPHA amplitude-to-noise
    thisrow = np.full_like(x_c_list, np.nan, dtype="float")
    thisrow_err = np.full_like(x_c_list, np.nan, dtype="float")
    for jj, coords in enumerate(zip(x_c_list, y_c_list)):
        x_c, y_c = coords
        y, x = (int(np.round(y_c)), int(np.round(x_c)))
        if x > 49 or y > 49:
            x = min([x, 49])
            y = min([y, 49])
        thisrow[jj] = AN_HALPHA_map[y, x]
    rows_list.append(thisrow)
    colnames.append(f"HALPHA A/N (measured)")       

    #######################################################################
    # Do the same but with the continuum intensity for calculating the HALPHA EW
    thisrow = np.full_like(x_c_list, np.nan, dtype="float")
    thisrow_std = np.full_like(x_c_list, np.nan, dtype="float")
    thisrow_err = np.full_like(x_c_list, np.nan, dtype="float")
    for jj, coords in enumerate(zip(x_c_list, y_c_list)):
        x_c, y_c = coords
        y, x = (int(np.round(y_c)), int(np.round(x_c)))
        if x > 49 or y > 49:
            x = min([x, 49])
            y = min([y, 49])
        thisrow[jj] = cont_HALPHA_map[y, x]
        thisrow_std[jj] = cont_HALPHA_map_std[y, x]
        thisrow_err[jj] = cont_HALPHA_map_err[y, x]
    rows_list.append(thisrow)
    rows_list.append(thisrow_std)
    rows_list.append(thisrow_err)
    colnames.append("HALPHA continuum")
    colnames.append("HALPHA continuum std. dev.")
    colnames.append("HALPHA continuum error")        

    #######################################################################
    # Do the same but with the continuum intensity for calculating the HALPHA EW
    thisrow = np.full_like(x_c_list, np.nan, dtype="float")
    thisrow_std = np.full_like(x_c_list, np.nan, dtype="float")
    thisrow_err = np.full_like(x_c_list, np.nan, dtype="float")
    for jj, coords in enumerate(zip(x_c_list, y_c_list)):
        x_c, y_c = coords
        y, x = (int(np.round(y_c)), int(np.round(x_c)))
        if x > 49 or y > 49:
            x = min([x, 49])
            y = min([y, 49])
        thisrow[jj] = cont_B_map[y, x]
        thisrow_std[jj] = cont_B_map_std[y, x]
        thisrow_err[jj] = cont_B_map_err[y, x]
    rows_list.append(thisrow)
    rows_list.append(thisrow_std)
    rows_list.append(thisrow_err)
    colnames.append("B-band continuum")
    colnames.append("B-band continuum std. dev.")
    colnames.append("B-band continuum error")        

    #######################################################################
    # Do the same but with the D4000Å break
    thisrow = np.full_like(x_c_list, np.nan, dtype="float")
    thisrow_err = np.full_like(x_c_list, np.nan, dtype="float")
    for jj, coords in enumerate(zip(x_c_list, y_c_list)):
        x_c, y_c = coords
        y, x = (int(np.round(y_c)), int(np.round(x_c)))
        if x > 49 or y > 49:
            x = min([x, 49])
            y = min([y, 49])
        thisrow[jj] = d4000_map[y, x]
        thisrow_err[jj] = d4000_map_err[y, x]
    rows_list.append(thisrow)
    rows_list.append(thisrow_err)
    colnames.append("D4000")
    colnames.append("D4000 error")          

    #######################################################################
    # Add pixel coordinates
    rows_list.append(np.array([x0_px] * ngood_bins) * as_per_px)
    rows_list.append(np.array([y0_px] * ngood_bins) * as_per_px)
    rows_list.append(np.array(x_c_list).flatten() * as_per_px)
    rows_list.append(np.array(y_c_list).flatten() * as_per_px)
    rows_list.append(np.array(x_prime_list).flatten() * as_per_px)
    rows_list.append(np.array(y_prime_list).flatten() * as_per_px)
    rows_list.append(np.array(r_prime_list).flatten() * as_per_px)
    rows_list.append(np.array(bin_number_list))
    rows_list.append(np.array(bin_size_list_px))
    rows_list.append(np.array(bin_size_list_px) * as_per_px**2)
    rows_list.append(np.array(bin_size_list_px) * as_per_px**2 * df_metadata.loc[gal, "kpc per arcsec"]**2)
    colnames.append("Galaxy centre x0_px (projected, arcsec)")
    colnames.append("Galaxy centre y0_px (projected, arcsec)")
    colnames.append("x (projected, arcsec)")
    colnames.append("y (projected, arcsec)")
    colnames.append("x (relative to galaxy centre, deprojected, arcsec)")
    colnames.append("y (relative to galaxy centre, deprojected, arcsec)")
    colnames.append("r (relative to galaxy centre, deprojected, arcsec)")
    colnames.append("Bin number")
    colnames.append("Bin size (pixels)")
    colnames.append("Bin size (square arcsec)")
    colnames.append("Bin size (square kpc)")

    # Transpose so that each row represents a single pixel & each column a measured quantity.
    rows_arr = np.array(rows_list).T

    # Get rid of rows that are all NaNs
    bad_rows = np.all(np.isnan(rows_arr), axis=1)
    rows_good = rows_arr[~bad_rows]
    ngood = rows_good.shape[0]

    # Append a column with the galaxy ID & other properties
    safe_cols = [c for c in df_metadata.columns if c != "Morphology" and c != "MGE photometry"]
    gal_metadata = np.tile(df_metadata.loc[df_metadata.loc[:, "ID"] == gal][safe_cols].values, (ngood_bins, 1))
    rows_good = np.hstack((gal_metadata, rows_good))

    print(f"{status_str}: Finished processing {gal} ({gal_idx})")

    return rows_good, colnames 

###############################################################################
def make_sami_df(bin_type="default", ncomponents="recom", 
                 eline_SNR_min=5, sigma_gas_SNR_min=3,
                 eline_list=["HALPHA", "HBETA", "NII6583", "OI6300", 
                             "OII3726+OII3729", "OIII5007", 
                             "SII6716", "SII6731"],
                 line_flux_SNR_cut=True,
                 missing_fluxes_cut=True,
                 line_amplitude_SNR_cut=True,
                 flux_fraction_cut=False,
                 sigma_gas_SNR_cut=True, 
                 vgrad_cut=False,
                 stekin_cut=True,
                 correct_extinction=True,
                 nthreads_max=20, debug=False,
                 __use_lzifu_fits=False, __lzifu_ncomponents=None):
    """
    DESCRIPTION
    ---------------------------------------------------------------------------
    This function is used to create a Pandas DataFrame containing emission line 
    fluxes & kinematics, stellar kinematics, extinction, star formation rates, 
    and other quantities for individual spaxels in SAMI galaxies as taken from 
    SAMI DR3.

    The output is stored in HDF format as a Pandas DataFrame in which each row 
    corresponds to a given spaxel (or Voronoi bin) for every galaxy. 

    USAGE
    ---------------------------------------------------------------------------
    
        >>> from spaxelsleuth.loaddata.sami import make_sami_df()
        >>> make_sami_df(ncomponents="1", bin_type="default", eline_SNR_min=5)

    will create a DataFrame comprising the 1-component Gaussian fits to the 
    unbinned datacubes, and will use a minimum S/N threshold of 5 to mask out 
    unreliable emission line fluxes and associated quantities.

    Other input arguments may be set in the script to control other aspects
    of the data quality and S/N cuts made, however the default values can be
    left as-is.

    Running this function on the full sample takes some time (~10-20 minutes 
    when threaded across 20 threads). Execution time can be sped up by tweaking 
    the NTHREADS_MAX parameter. 

    If you wish to run in debug mode, set the DEBUG flag to True: this will run 
    the script on a subset (by default 10) galaxies to speed up execution. 

    INPUTS
    ---------------------------------------------------------------------------
    ncomponents:        str
        Which number of Gaussian components to assume. Options are "recom" (the
        recommended multi-component fits) or "1" (1-component fits).
        
        NOTE: if __use_lzifu_fits is True, then ncomponents is ONLY used in  
        loading data products that are NOT contained in the output LZIFU files -
        i.e., SFRs/SFR surface densities and HALPHA extinction orrection 
        factors. Use parameter __lzifu_ncomponents to control which data 
        derived from the LZIFU fits is loaded. 

    bin_type:           str
        Spatial binning strategy. Options are "default" (unbinned), "adaptive"
        (Voronoi binning) or "sectors" (sector binning)

    eline_SNR_min:      int 
        Minimum emission line flux S/N to assume.

    sigma_gas_SNR_min:          int
        Minimum velocity dipersion S/N to accept.

    correct_extinction:         bool
        If True, correct emission line fluxes for extinction. 
        NOTE: metallicities are ONLY computed if correct_extinction is True,
        due to the large spacing in wavelength between the emission lines 
        used in some diagnostics.

    line_flux_SNR_cut:          bool
        Whether to NaN emission line components AND total fluxes 
        (corresponding to emission lines in eline_list) below a specified S/N 
        threshold, given by eline_SNR_min. The S/N is simply the flux dividied 
        by the formal 1sigma uncertainty on the flux. 

    missing_fluxes_cut:         bool
        Whether to NaN out "missing" fluxes - i.e., cells in which the flux
        of an emission line (total or per component) is NaN, but the error 
        is not for some reason.

    line_amplitude_SNR_cut:     bool
        If True, removes components with Gaussian amplitudes < 3 * RMS of the 
        continuum in the vicinity of Halpha. By default this is set to True
        because this does well at removing shallow components which are most 
        likely due to errors in the stellar continuum fit.

    flux_fraction_cut:          bool
        If True, and if ncomponents > 1, remove intermediate and broad 
        components with line amplitudes < 0.05 that of the narrow componet.
        Set to False by default b/c it's unclear whether this is needed to 
        reject unreliable components.

    sigma_gas_SNR_cut:          bool
        If True, mask component velocity dispersions where the S/N on the 
        velocity dispersion measurement is below sigma_gas_SNR_min. 
        By default this is set to True b/c it's a robust way to account for 
        emission line widths < instrumental.

    vgrad_cut:                  bool         
        If True, mask component kinematics (velocity and velocity dispersion)
        that are likely to be affected by beam smearing.
        By default this is set to False b/c it tends to remove nuclear spaxels 
        which may be of interest to your science case, & because it doesn't 
        reliably remove spaxels with quite large beam smearing components

    stekin_cut:                 bool
        If True, mask stellar kinematic quantities that do not meet the DQ and 
        S/N requirements specified in Croom et al. (2021). True by default.

    eline_list:                 list of str
        Default SAMI emission lines - don't change this!

    nthreads_max:               int            
        Maximum number of threds to use. 

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

    debug:                      bool (optional)
        If True, run on a subset of the entire sample (10 galaxies) and save
        the output with "_DEBUG" appended to the filename. This is useful for
        tweaking S/N and DQ cuts since running the function on the entire 
        sample is quite slow.

    OUTPUTS
    ---------------------------------------------------------------------------
    Each time the function is run, TWO DataFrames are produced - with and without 
    extinction correction applied to the emission line fluxes. 

    The resulting DataFrame will be stored as 

        SAMI_DIR/sami_{bin_type}_{ncomponents}-comp_extcorr_minSNR={eline_SNR_min}.hd5

    if correct_extinction is True, or else

        SAMI_DIR/sami_{bin_type}_{ncomponents}-comp_minSNR={eline_SNR_min}.hd5

    The DataFrame will be stored in CSV format in case saving in HDF format 
    fails     for any reason.

    Note that the Halpha equivalent widths are NOT corrected for extinction if 
    correct_extinction is True. This is because stellar continuum extinction 
    measurements are not available, and so applying the correction only to the 
    Halpha fluxes may over-estimate the true EW.

    PREREQUISITES
    ---------------------------------------------------------------------------
    SAMI_DIR and SAMI_DATACUBE_DIR must be defined as an environment variable.

    make_sami_metadata_df() must be run first.

    SAMI data products must be downloaded from DataCentral

        https://datacentral.org.au/services/download/

    and stored as follows: 

        SAMI_DIR/ifs/<gal>/<gal>_<quantity>_<bin type>_<number of components>-comp.fits

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

        SAMI_DATACUBE_DIR/ifs/<gal>/<gal>_A_cube_<blue/red>.fits.gz

    SAMI_DATACUBE_DIR can be the same as SAMI_DIR (I just have them differently
    in my setup due to storage space limitations).
    """

    #######################################################################
    # INPUT CHECKING
    #######################################################################
    assert (ncomponents == "recom") | (ncomponents == "1"), "ncomponents must be 'recom' or '1'!!"
    assert bin_type in ["default", "adaptive", "sectors"], "bin_type must be 'default' or 'adaptive' or 'sectors'!!"
    if __use_lzifu_fits:
        assert __lzifu_ncomponents in ["recom", "1", "2", "3"], "__lzifu_ncomponents must be 'recom', '1', '2' or '3'!!"
        assert os.path.exists(__lzifu_products_path), f"lzifu_products_path directory {__lzifu_products_path} not found!!"
        print(f"WARNING: using LZIFU {__lzifu_ncomponents}-component fits to obtain emission line fluxes & kinematics, NOT DR3 data products!!")

    # Component indices for emission line-derived quantities
    range_ncomponents_elines =\
        range(3 if ncomponents == "recom" else 1) if not __use_lzifu_fits \
        else range(3 if __lzifu_ncomponents == "recom" else int(__lzifu_ncomponents))        

    # For printing to stdout
    status_str = f"In sami.make_df_sami() [bin_type={bin_type}, ncomponents={ncomponents}, debug={debug}, eline_SNR_min={eline_SNR_min}]"

    ###############################################################################
    # FILENAMES
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

    print(f"{status_str}: saving to files {df_fname}...")

    ###############################################################################
    # READ IN THE METADATA
    ###############################################################################
    try:
        df_metadata = pd.read_hdf(os.path.join(sami_data_path, df_metadata_fname), key="metadata")
    except FileNotFoundError:
        print(f"ERROR: metadata DataFrame file not found ({os.path.join(sami_data_path, df_metadata_fname)}). Please run make_sami_metadata_df.py first!")

    # Only include galaxies flagged as "good"
    gal_ids_dq_cut = df_metadata[df_metadata["Good?"] == True].index.values
    
    # Only include galaxies for which we have data 
    gal_ids_dq_cut = [g for g in gal_ids_dq_cut if os.path.exists(os.path.join(sami_data_path, f"ifs/{g}/"))]

    # If running in DEBUG mode, run on a subset to speed up execution time
    if debug: 
        gal_ids_dq_cut = gal_ids_dq_cut[:10] + [572402, 209807]
    df_metadata["Good?"] = df_metadata["Good?"].astype("float")

    # Turn off plotting if more than 1 galaxy is to be run
    if len(gal_ids_dq_cut) > 1:
        plotit = False
    else:
        plotit = True

    ###############################################################################
    # Run in parallel
    ###############################################################################
    args_list = [[gg, gal, ncomponents, bin_type, df_metadata, status_str, 
                  __use_lzifu_fits, __lzifu_ncomponents] for gg, gal in enumerate(gal_ids_dq_cut)]

    if len(gal_ids_dq_cut) == 1:
        res_list = [_process_gals(args_list[0])]
    else:
        if nthreads_max > 1:
            print(f"{status_str}: Beginning pool...")
            pool = multiprocessing.Pool(min([nthreads_max, len(gal_ids_dq_cut)]))
            res_list = np.array((pool.map(_process_gals, args_list)))
            pool.close()
            pool.join()
        else:
            print(f"{status_str}: Running sequentially...")
            res_list = []
            for args in args_list:
                res = _process_gals(args)
                res_list.append(res)

    ###############################################################################
    # Convert to a Pandas DataFrame
    ###############################################################################
    rows_list_all = [r[0] for r in res_list]
    colnames = res_list[0][1]
    safe_cols = [c for c in df_metadata.columns if c != "Morphology" and c != "MGE photometry"]
    df_spaxels = pd.DataFrame(np.vstack(tuple(rows_list_all)), columns=safe_cols + colnames)

    ######################################################################
    # Add extra columns
    ######################################################################
    df_spaxels["r/R_e"] = df_spaxels["r (relative to galaxy centre, deprojected, arcsec)"] / df_spaxels["R_e (arcsec)"]
    
    # Add pixel coordinates
    df_spaxels["x, y (pixels)"] = list(zip(df_spaxels["x (projected, arcsec)"] / 0.5, df_spaxels["y (projected, arcsec)"] / 0.5))

    # Add the morphology column back in
    morph_dict = {
        "0.0": "E",
        "0.5": "E/S0",
        "1.0": "S0",
        "1.5": "S0/Early-spiral",
        "2.0": "Early-spiral",
        "2.5": "Early/Late spiral",
        "3.0": "Late spiral",
        "5.0": "?",
        "-9.0": "no agreement",
        "-0.5": "Unknown"
    }
    df_spaxels["Morphology"] = [morph_dict[str(m)] for m in df_spaxels["Morphology (numeric)"]]

    ###############################################################################
    # Compute the ORIGINAL number of components: define these as those in which sigma_gas is defined
    ###############################################################################
    ncomponents_original = (~df_spaxels[f"sigma_gas (component 1)"].isna()).astype(int)
    for nn in range_ncomponents_elines[1:]:
        ncomponents_original += (~df_spaxels[f"sigma_gas (component {nn + 1})"].isna()).astype(int)
    df_spaxels["Number of components (original)"] = ncomponents_original

    # Leave this here for debugging!
    # for col in df_spaxels.columns:
    #     if df_spaxels[col].dtypes != "float64":
    #         print(f"{col}: {df_spaxels[col].dtypes}")
    # Tracer()()

    ###############################################################################
    # Calculate equivalent widths
    ###############################################################################
    df_spaxels.loc[df_spaxels["HALPHA continuum"] < 0, "HALPHA continuum"] = 0
    
    # Compute EW in each component
    for nn in range_ncomponents_elines:
        df_spaxels[f"HALPHA EW (component {nn + 1})"] = df_spaxels[f"HALPHA (component {nn + 1})"] / df_spaxels["HALPHA continuum"]
        df_spaxels[f"HALPHA EW error (component {nn + 1})"] = df_spaxels[f"HALPHA EW (component {nn + 1})"] *\
            np.sqrt((df_spaxels[f"HALPHA error (component {nn + 1})"] / df_spaxels[f"HALPHA (component {nn + 1})"])**2 +\
                    (df_spaxels[f"HALPHA continuum error"] / df_spaxels[f"HALPHA continuum"])**2) 
        
        # If the continuum level <= 0, then the EW is undefined, so set to NaN.
        df_spaxels.loc[df_spaxels["HALPHA continuum"] <= 0, 
                       [f"HALPHA EW (component {nn + 1})", 
                        f"HALPHA EW error (component {nn + 1})"]] = np.nan  

    # Calculate total EW
    df_spaxels[f"HALPHA EW (total)"] = df_spaxels[f"HALPHA (total)"] / df_spaxels["HALPHA continuum"]
    df_spaxels[f"HALPHA EW error (total)"] = df_spaxels[f"HALPHA EW (total)"] *\
        np.sqrt((df_spaxels[f"HALPHA error (total)"] / df_spaxels[f"HALPHA (total)"])**2 +\
                (df_spaxels[f"HALPHA continuum error"] / df_spaxels[f"HALPHA continuum"])**2) 
    
    # If the continuum level <= 0, then the EW is undefined, so set to NaN.
    df_spaxels.loc[df_spaxels["HALPHA continuum"] <= 0, 
                   [f"HALPHA EW (total)", 
                    f"HALPHA EW error (total)"]] = np.nan  

    ######################################################################
    # Compute S/N in all lines
    ######################################################################
    for eline in eline_list:
        # Compute S/N 
        for nn in range_ncomponents_elines:
            if f"{eline} (component {nn + 1})" in df_spaxels.columns:
                df_spaxels[f"{eline} S/N (component {nn + 1})"] = df_spaxels[f"{eline} (component {nn + 1})"] / df_spaxels[f"{eline} error (component {nn + 1})"]
        
        # Compute the S/N in the TOTAL line flux
        df_spaxels[f"{eline} S/N (total)"] = df_spaxels[f"{eline} (total)"] / df_spaxels[f"{eline} error (total)"]

    ######################################################################
    # Fix SFR columns
    ######################################################################
    # NaN the SFR surface density if the inclination is undefined
    cond_NaN_inclination = np.isnan(df_spaxels["i (degrees)"])
    cols = [c for c in df_spaxels.columns if "SFR surface density" in c]
    df_spaxels.loc[cond_NaN_inclination, cols] = np.nan

    # NaN the SFR if the SFR == 0
    # Note: I'm not entirely sure why there are spaxels with SFR == 0
    # in the first place.
    cond_zero_SFR = df_spaxels["SFR (total)"]  == 0
    cols = [c for c in df_spaxels.columns if "SFR" in c]
    df_spaxels.loc[cond_zero_SFR, cols] = np.nan

    ######################################################################
    # DQ and S/N CUTS
    ######################################################################
    df_spaxels = dqcut.set_flags(df=df_spaxels, 
                  eline_SNR_min=eline_SNR_min, eline_list=eline_list,
                  sigma_gas_SNR_min=sigma_gas_SNR_min,
                  sigma_inst_kms=29.6)

    # Apply S/N and DQ cuts
    df_spaxels = dqcut.apply_flags(df=df_spaxels, 
                                   line_flux_SNR_cut=line_flux_SNR_cut,
                                   missing_fluxes_cut=missing_fluxes_cut,
                                   line_amplitude_SNR_cut=line_amplitude_SNR_cut,
                                   flux_fraction_cut=flux_fraction_cut,
                                   vgrad_cut=vgrad_cut,
                                   sigma_gas_SNR_cut=sigma_gas_SNR_cut,
                                   stekin_cut=stekin_cut,
                                   eline_list=eline_list)    

    ######################################################################
    # NaN out SFR quantities if the HALPHA flux is NaN
    ###################################################################### 
    cond_Ha_isnan = df_spaxels["HALPHA (total)"].isna()
    cols_sfr = [c for c in df_spaxels.columns if "SFR" in c]
    for col in cols_sfr:
        df_spaxels.loc[cond_Ha_isnan, col] = np.nan
    
    ######################################################################
    # Make a copy of the DataFrame with EXTINCTION CORRECTION
    # Correct emission line fluxes (but not EWs!)
    # NOTE: extinction.fm07 assumes R_V = 3.1 so do not change R_V from 
    # this value!!!
    ######################################################################
    if correct_extinction:
        print(f"{status_str}: Correcting emission line fluxes (but not EWs) for extinction...")
        # Compute A_V using total Halpha and Hbeta emission line fluxes
        df_spaxels = extcorr.compute_A_V(df_spaxels,
                                         reddening_curve="fm07", 
                                         balmer_SNR_min=5,
                                         s=f" (total)")

        # Apply the extinction correction to total emission line fluxes
        df_spaxels = extcorr.apply_extinction_correction(df_spaxels, 
                                        reddening_curve="fm07", 
                                        eline_list=[e for e in eline_list if f"{e} (total)" in df_spaxels],
                                        a_v_col_name="A_V (total)",
                                        nthreads=nthreads_max,
                                        s=f" (total)")
        
        # Apply the extinction correction to fluxes of  individual components
        for nn in range_ncomponents_elines:
            df_spaxels = extcorr.apply_extinction_correction(df_spaxels, 
                                            reddening_curve="fm07", 
                                            eline_list=[e for e in eline_list if f"{e} (component {nn + 1})" in df_spaxels],
                                            a_v_col_name="A_V (total)",
                                            nthreads=nthreads_max,
                                            s=f" (component {nn + 1})")

        df_spaxels["Corrected for extinction?"] = True
    else:
        df_spaxels["Corrected for extinction?"] = False
    df_spaxels = df_spaxels.sort_index()

    ######################################################################
    # EVALUATE LINE RATIOS & SPECTRAL CLASSIFICATIONS
    ######################################################################
    df_spaxels = linefns.ratio_fn(df_spaxels, s=f" (total)")
    df_spaxels = linefns.bpt_fn(df_spaxels, s=f" (total)")

    ######################################################################
    # EVALUATE ADDITIONAL COLUMNS - log quantites, etc.
    ######################################################################
    df_spaxels = dqcut.compute_extra_columns(df_spaxels)

    ######################################################################
    # EVALUATE METALLICITY (only for spaxels with extinction correction)
    ######################################################################
    if not debug:
        df_spaxels = metallicity.calculate_metallicity(met_diagnostic="N2Ha_PP04", compute_errors=True, niters=1000, df=df_spaxels, s=" (total)")
        df_spaxels = metallicity.calculate_metallicity(met_diagnostic="N2Ha_M13", compute_errors=True, niters=1000, df=df_spaxels, s=" (total)")
        df_spaxels = metallicity.calculate_metallicity(met_diagnostic="O3N2_PP04", compute_errors=True, niters=1000, df=df_spaxels, s=" (total)")
        df_spaxels = metallicity.calculate_metallicity(met_diagnostic="O3N2_M13", compute_errors=True, niters=1000, df=df_spaxels, s=" (total)")
        df_spaxels = metallicity.calculate_metallicity(met_diagnostic="N2S2Ha_D16", compute_errors=True, niters=1000, df=df_spaxels, s=" (total)")
        df_spaxels = metallicity.calculate_metallicity(met_diagnostic="N2O2_KD02", compute_errors=True, niters=1000, df=df_spaxels, s=" (total)")
        df_spaxels = metallicity.calculate_metallicity(met_diagnostic="Rcal_PG16", compute_errors=True, niters=1000, df=df_spaxels, s=" (total)")
        df_spaxels = metallicity.calculate_metallicity(met_diagnostic="Scal_PG16", compute_errors=True, niters=1000, df=df_spaxels, s=" (total)")
        df_spaxels = metallicity.calculate_metallicity(met_diagnostic="ON_P10", compute_errors=True, niters=1000, df=df_spaxels, s=" (total)")
        df_spaxels = metallicity.calculate_metallicity(met_diagnostic="ONS_P10", compute_errors=True, niters=1000, df=df_spaxels, s=" (total)")
        df_spaxels = metallicity.calculate_metallicity(met_diagnostic="N2Ha_K19", compute_logU=True, ion_diagnostic="O3O2_K19", compute_errors=True, niters=1000, df=df_spaxels, s=" (total)")
        df_spaxels = metallicity.calculate_metallicity(met_diagnostic="O3N2_K19", compute_logU=True, ion_diagnostic="O3O2_K19", compute_errors=True, niters=1000, df=df_spaxels, s=" (total)")
        df_spaxels = metallicity.calculate_metallicity(met_diagnostic="N2O2_K19", compute_logU=True, ion_diagnostic="O3O2_K19", compute_errors=True, niters=1000, df=df_spaxels, s=" (total)")
        df_spaxels = metallicity.calculate_metallicity(met_diagnostic="R23_KK04", compute_logU=True, ion_diagnostic="O3O2_KK04", compute_errors=True, niters=1000, df=df_spaxels, s=" (total)")
    else:
        df_spaxels = metallicity.calculate_metallicity(met_diagnostic="N2Ha_PP04", compute_errors=True, niters=1000, df=df_spaxels, s=" (total)")
        df_spaxels = metallicity.calculate_metallicity(met_diagnostic="N2Ha_K19", compute_logU=True, ion_diagnostic="O3O2_K19", compute_errors=True, niters=1000, df=df_spaxels, s=" (total)")

    ###############################################################################
    # Save input flags to the DataFrame so that we can keep track
    ###############################################################################
    df_spaxels["eline_SNR_min"] = eline_SNR_min
    df_spaxels["sigma_gas_SNR_min"] = sigma_gas_SNR_min
    df_spaxels["Extinction correction applied"] = correct_extinction
    df_spaxels["line_flux_SNR_cut"] = line_flux_SNR_cut
    df_spaxels["missing_fluxes_cut"] = missing_fluxes_cut
    df_spaxels["line_amplitude_SNR_cut"] = line_amplitude_SNR_cut
    df_spaxels["flux_fraction_cut"] = flux_fraction_cut
    df_spaxels["vgrad_cut"] = vgrad_cut
    df_spaxels["sigma_gas_SNR_cut"] = sigma_gas_SNR_cut
    df_spaxels["stekin_cut"] = stekin_cut
    
    for col in [c for c in df_spaxels.columns if "SFR" in c and "error" not in c]:
        assert f"{col}" in df_spaxels.columns
        assert all(df_spaxels.loc[df_spaxels["HALPHA (total)"].isna(), f"{col}"].isna())

    ###############################################################################
    # Save to .hd5 & .csv
    ###############################################################################
    print(f"{status_str}: Saving to file...")

    # No extinction correction
    df_spaxels.to_csv(os.path.join(sami_data_path, df_fname.split("hd5")[0] + "csv"))
    try:
        df_spaxels.to_hdf(os.path.join(sami_data_path, df_fname), key=f"{bin_type}, {ncomponents}-comp")
    except:
        print(f"{status_str}: Unable to save to HDF file... sigh...")

    return

###############################################################################
def load_sami_df(ncomponents, bin_type, correct_extinction, eline_SNR_min,
                 __use_lzifu_fits=False, __lzifu_ncomponents='3',
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
    assert (ncomponents == "recom") | (ncomponents == "1"), "ncomponents must be 'recom' or '1'!!"
    assert bin_type in ["default", "adaptive", "sectors"], "bin_type must be 'default' or 'adaptive' or 'sectors'!!"
    if __use_lzifu_fits:
        assert __lzifu_ncomponents in ["recom", "1", "2", "3"], "__lzifu_ncomponents must be 'recom', '1', '2' or '3'!!"
        assert os.path.exists(__lzifu_products_path), f"lzifu_products_path directory {__lzifu_products_path} not found!!"
        print(f"WARNING: using LZIFU {__lzifu_ncomponents}-component fits to obtain emission line fluxes & kinematics, NOT DR3 data products!!")

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

    assert os.path.exists(os.path.join(sami_data_path, df_fname)),\
        f"File {os.path.join(sami_data_path, df_fname)} does does not exist!"

    # Load the data frame
    t = os.path.getmtime(os.path.join(sami_data_path, df_fname))
    print(f"In load_sami_df(): Loading DataFrame from file {os.path.join(sami_data_path, df_fname)} [last modified {datetime.datetime.fromtimestamp(t)}]...")
    df = pd.read_hdf(os.path.join(sami_data_path, df_fname))

    # Return
    print("In load_sami_df(): Finished!")
    return df.sort_index()

###############################################################################
def make_sami_aperture_df(eline_SNR_min, 
                          line_flux_SNR_cut=True,
                          missing_fluxes_cut=True,
                          sigma_gas_SNR_cut=True, sigma_gas_SNR_min=3,
                          nthreads_max=20, correct_extinction=True):
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
        By default this is set to True b/c it's a robust way to account for 
        emission line widths < instrumental.

    sigma_gas_SNR_min:          int
        Minimum velocity dipersion S/N to accept.

    correct_extinction:         bool 
        If True, correct emission line fluxes for extinction.
        NOTE: if False, then metallicities are NOT computed. 

    nthreads_max:               int            
        Maximum number of threds to use during extinction correction 
        calculation.    

    OUTPUTS
    ---------------------------------------------------------------------------
    The DataFrames (one with extinction correction, and one without) are saved 
    to 

        SAMI_DIR/sami_apertures_<extcorr>_minSNR={eline_SNR_min}.hd5
    and SAMI_DIR/sami_apertures_<extcorr>_minSNR={eline_SNR_min}.hd5

    """
    #######################################################################
    # INPUT CHECKING
    #######################################################################
    # For printing to stdout
    status_str = f"In sami.make_sami_aperture_df():"

    ###############################################################################
    # FILENAMES
    #######################################################################
    df_metadata_fname = "sami_dr3_metadata.hd5"

    # Output file names
    if correct_extinction:
        df_fname = f"sami_apertures_extcorr_minSNR={eline_SNR_min}.hd5"
    else:
        df_fname = f"sami_apertures_minSNR={eline_SNR_min}.hd5"
    print(f"{status_str}: saving to files {df_fname}...")

    ###########################################################################
    # Open the .csv file containing the table 
    ###########################################################################
    data_path = os.path.join(__file__.split("loaddata")[0], "data")
    
    # Emission line info
    df_ap_elines = pd.read_csv(os.path.join(data_path, "sami_EmissionLine1compDR3.csv"))
    df_ap_elines = df_ap_elines.set_index("catid").drop("Unnamed: 0", axis=1)
    df_ap_elines = df_ap_elines.rename(columns={"catid": "ID"})
    
    # SSP info
    df_ap_ssp = pd.read_csv(os.path.join(data_path, "sami_SSPAperturesDR3.csv"))
    df_ap_ssp = df_ap_ssp.set_index("catid").drop("Unnamed: 0", axis=1)
    df_ap_ssp = df_ap_ssp.rename(columns={"catid": "ID"})

    # Stellar indices
    df_ap_indices = pd.read_csv(os.path.join(data_path, "sami_IndexAperturesDR3.csv"))
    df_ap_indices = df_ap_indices.set_index("catid").drop("Unnamed: 0", axis=1)
    df_ap_indices = df_ap_indices.rename(columns={"catid": "ID"})

    # Merge 
    df_ap = df_ap_elines.merge(df_ap_ssp, left_index=True, right_index=True).drop(["cubeid_x", "cubeid_y"], axis=1)
    stellar_idx_cols = [c for c in df_ap_indices if c not in df_ap]
    df_ap = df_ap.merge(df_ap_indices[stellar_idx_cols], left_index=True, right_index=True)

    # Drop duplicate rows 
    df_ap = df_ap[~df_ap.index.duplicated(keep="first")]

    ###########################################################################
    # Merge with metadata DataFrame
    ###########################################################################
    df_metadata = pd.read_hdf(os.path.join(sami_data_path, df_metadata_fname), key="metadata")
    df_ap = df_ap.merge(df_metadata, left_index=True, right_index=True)

    ###########################################################################
    # Rename columns 
    ###########################################################################
    for ap in ["_1_4_arcsecond", "_2_arcsecond", "_3_arcsecond", "_4_arcsecond", "_re_mge", "_re", "_3kpc_round"]:
        cols_ap = [col for col in df_ap.columns if ap in col]
        rename_dict = {}
        for old_col in cols_ap:
            if old_col.endswith("_err"):
                new_col = old_col.split(ap)[0].upper() + f" error ({ap.replace('_', ' ')[1:]})"
            else:
                new_col = old_col.split(ap)[0].upper() + f" ({ap.replace('_', ' ')[1:]})"
            rename_dict[old_col] = new_col
            print(f"{old_col} --> {new_col}")
        df_ap = df_ap.rename(columns=rename_dict)

    old_cols_v = [col for col in df_ap.columns if "V_GAS" in col]
    new_cols_v = [col.replace("V_GAS", "v_gas") for col in old_cols_v]
    old_cols_sigma = [col for col in df_ap.columns if "VDISP_GAS" in col]
    new_cols_sigma = [col.replace("VDISP_GAS", "sigma_gas") for col in old_cols_sigma]
    rename_dict = dict(zip(old_cols_v + old_cols_sigma, new_cols_v + new_cols_sigma))
    df_ap = df_ap.rename(columns=rename_dict)

    old_cols_re = [col for col in df_ap.columns if "(re)" in col]
    new_cols_re = [col.replace("(re)", "(R_e)") for col in old_cols_re]
    old_cols_mge = [col for col in df_ap.columns if "mge)" in col]
    new_cols_mge = [col.replace("(re mge)", "(R_e (MGE))") for col in old_cols_mge]
    rename_dict = dict(zip(old_cols_re + old_cols_mge, new_cols_re + new_cols_mge))
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
    ######################################################################
    for ap in ["1.4 arcsecond", "2 arcsecond", "3 arcsecond", "4 arcsecond", "R_e", "R_e (MGE)", "3kpc round"]:
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
        df_ap[f"SFR surface density error ({ap})"] = df_ap[f"SFR error ({ap})"] / A_kpc2
        
        # Log SFR surface density
        df_ap[f"log SFR surface density ({ap})"] = np.log10(df_ap[f"SFR surface density ({ap})"])
        df_ap[f"log SFR surface density error (upper) ({ap})"] = np.log10(df_ap[f"SFR surface density ({ap})"] + df_ap[f"SFR surface density error ({ap})"]) - df_ap[f"log SFR surface density ({ap})"]
        df_ap[f"log SFR surface density error (lower) ({ap})"] = df_ap[f"log SFR surface density ({ap})"] - np.log10(df_ap[f"SFR surface density ({ap})"] - df_ap[f"SFR surface density error ({ap})"])
        
        # log SFR
        df_ap[f"log SFR ({ap})"] = np.log10(df_ap[f"SFR ({ap})"])
        df_ap[f"log SFR error (upper) ({ap})"] = np.log10(df_ap[f"SFR ({ap})"] + df_ap[f"SFR error ({ap})"]) - df_ap[f"log SFR ({ap})"]
        df_ap[f"log SFR error (lower) ({ap})"] = df_ap[f"log SFR ({ap})"] - np.log10(df_ap[f"SFR ({ap})"] - df_ap[f"SFR error ({ap})"])

    ######################################################################
    # Compute specific SFRs 
    ######################################################################
    for ap in ["1.4 arcsecond", "2 arcsecond", "3 arcsecond", "4 arcsecond", "R_e", "R_e (MGE)", "3kpc round"]:
        df_ap[f"log sSFR ({ap})"] = df_ap[f"log SFR ({ap})"] - df_ap["log M_*"]

    ######################################################################
    # Compute S/N for each emission line in each aperture
    ######################################################################
    aps = ["1.4 arcsecond", "2 arcsecond", "3 arcsecond", "4 arcsecond", "R_e", "R_e (MGE)", "3kpc round"]
    eline_list = ["OII3726", "OII3729", "NEIII3869", "HEPSILON", "HDELTA", 
                  "HGAMMA", "OIII4363", "HBETA", "OIII5007", "OI6300", 
                  "HALPHA", "NII6583", "SII6716", "SII6731"]

    for eline in eline_list:
        # Compute S/N 
        for ap in aps:
            if f"{eline} ({ap})" in df_ap.columns:
                df_ap[f"{eline} S/N ({ap})"] = df_ap[f"{eline} ({ap})"] / df_ap[f"{eline} error ({ap})"]
    
    ######################################################################
    # Initialise DQ and S/N flags
    ######################################################################
    for ap in aps:
        df_ap[f"Low sigma_gas S/N flag ({ap})"] = False
        for eline in eline_list:
            df_ap[f"Missing flux flag - {eline} ({ap})"] = False
            df_ap[f"Low flux S/N flag - {eline} ({ap})"] = False

    ######################################################################
    # Fix SFR columns
    ######################################################################
    # NaN the SFR if the SFR == 0
    for ap in aps:
        cond_zero_SFR = df_ap[f"SFR ({ap})"]  == 0
        cols = [c for c in df_ap.columns if "SFR" in c and f"({ap})" in c]
        df_ap.loc[cond_zero_SFR, cols] = np.nan

    ######################################################################
    # DQ and S/N CUTS
    ######################################################################
    # Flag low S/N emission lines
    ######################################################################
    print(f"{status_str}: Flagging low S/N components and spaxels...")
    for eline in eline_list:
        # Fluxes in individual components
        for ap in aps:
            if f"{eline} ({ap})" in df_ap.columns:
                cond_low_SN = df_ap[f"{eline} S/N ({ap})"] < eline_SNR_min
                df_ap.loc[cond_low_SN, f"Low flux S/N flag - {eline} ({ap})"] = True

    ######################################################################
    # Flag emission lines with "missing" (i.e. NaN) fluxes in which the 
    # ERROR on the flux is not NaN
    ######################################################################
    print(f"{status_str}: Flagging components and galaxies with NaN fluxes and finite errors...")
    for eline in eline_list:
        # Fluxes in individual components
        for ap in aps:
            if f"{eline} ({ap})" in df_ap.columns:
                cond_missing_flux = df_ap[f"{eline} ({ap})"].isna() & ~df_ap[f"{eline} error ({ap})"].isna()
                df_ap.loc[cond_missing_flux, f"Missing flux flag - {eline} ({ap})"] = True
                print(f"{eline} ({ap}): {df_ap[cond_missing_flux].shape[0]:d} galaxies have missing fluxes in this component")

    ######################################################################
    # Flag rows with insufficient S/N in sigma_gas
    ######################################################################
    print(f"{status_str}: Flagging components with low sigma_gas S/N...")
    sigma_inst_kms = 29.6  # for SAMI
    sigma_gas_SNR_min = 3
    # Gas kinematics: NaN out cells w/ sigma_gas S/N ratio < sigma_gas_SNR_min 
    # (For SAMI, the red arm resolution is 29.6 km/s - see p6 of Croom+2021)
    for ap in aps:
        # 1. Define sigma_obs = sqrt(sigma_gas**2 + sigma_inst_kms**2).
        df_ap[f"sigma_obs ({ap})"] = np.sqrt(df_ap[f"sigma_gas ({ap})"]**2 + sigma_inst_kms**2)

        # 2. Define the S/N ratio of sigma_obs.
        # NOTE: here we assume that sigma_gas error (as output by LZIFU) 
        # really refers to the error on sigma_obs.
        df_ap[f"sigma_obs S/N ({ap})"] = df_ap[f"sigma_obs ({ap})"] / df_ap[f"sigma_gas error ({ap})"]

        # 3. Given our target SNR_gas, compute the target SNR_obs,
        # using the method in section 2.2.2 of Zhou+2017.
        df_ap[f"sigma_obs target S/N ({ap})"] = sigma_gas_SNR_min * (1 + sigma_inst_kms**2 / df_ap[f"sigma_gas ({ap})"]**2)
        cond_bad_sigma = df_ap[f"sigma_obs S/N ({ap})"] < df_ap[f"sigma_obs target S/N ({ap})"]
        df_ap.loc[cond_bad_sigma, f"Low sigma_gas S/N flag ({ap})"] = True

    ######################################################################
    # NaN out offending cells
    ######################################################################
    if line_flux_SNR_cut:
        print(f"{status_str}: Masking components that don't meet the S/N requirements...")
        for eline in eline_list:
            # Individual fluxes
            for ap in aps:
                cond_low_SN = df_ap[f"Low flux S/N flag - {eline} ({ap})"]

                # Cells to NaN
                if eline == "HALPHA":
                    # Then NaN out EVERYTHING associated with this component - if we can't trust HALPHA then we probably can't trust anything else either!
                    cols_low_SN = [c for c in df_ap.columns if f"({ap})" in c and "flag" not in c]
                else:
                    cols_low_SN = [c for c in df_ap.columns if eline in c and f"({ap})" in c and "flag" not in c]
                df_ap.loc[cond_low_SN, cols_low_SN] = np.nan


    if missing_fluxes_cut:
        print(f"{status_str}: Masking components with missing fluxes...")
        for eline in eline_list:
            # Individual fluxes
            for ap in aps:
                cond_missing_flux = df_ap[f"Missing flux flag - {eline} ({ap})"]

                # Cells to NaN
                if eline == "HALPHA":
                    # Then NaN out EVERYTHING associated with this component - if we can't trust HALPHA then we probably can't trust anything else either!
                    cols_missing_fluxes = [c for c in df_ap.columns if f"({ap})" in c and "flag" not in c]
                else:
                    cols_missing_fluxes = [c for c in df_ap.columns if eline in c and f"({ap})" in c and "flag" not in c]
                df_ap.loc[cond_missing_flux, cols_missing_fluxes] = np.nan

    if sigma_gas_SNR_cut:
        print(f"{status_str}: Masking components with insufficient S/N in sigma_gas...")
        for ap in aps:
            cond_bad_sigma = df_ap[f"Low sigma_gas S/N flag ({ap})"]
            
            # Cells to NaN
            cols_sigma_gas_SNR_cut = [c for c in df_ap.columns if f"({ap})" in c and "sigma_gas" in c and "flag" not in c]
            cols_sigma_gas_SNR_cut += [c for c in df_ap.columns if "delta" in c and str(nn + 1) in c]
            df_ap.loc[cond_bad_sigma, cols_sigma_gas_SNR_cut] = np.nan

    ######################################################################
    # Make a copy of the DataFrame with EXTINCTION CORRECTION
    # Correct emission line fluxes (but not EWs!)
    # NOTE: extinction.fm07 assumes R_V = 3.1 so do not change R_V from 
    # this value!!!
    ######################################################################
    if correct_extinction:
        print(f"{status_str}: Correcting emission line fluxes (but not EWs) for extinction...")
        for ap in aps:
            # Compute A_V using total Halpha and Hbeta emission line fluxes
            df_ap = extcorr.compute_A_V(df_ap,
                                             reddening_curve="fm07", 
                                             balmer_SNR_min=5,
                                             s=f" ({ap})")

            # Apply the extinction correction to emission line fluxes
            df_ap = extcorr.apply_extinction_correction(df_ap, 
                                            reddening_curve="fm07", 
                                            eline_list=[e for e in eline_list if f"{e} ({ap})" in df_ap],
                                            a_v_col_name=f"A_V ({ap})",
                                            nthreads=nthreads_max,
                                            s=f" ({ap})")
    df_ap["Corrected for extinction?"] = correct_extinction
    df_ap = df_ap.sort_index()

    ######################################################################
    # EVALUATE LINE RATIOS & SPECTRAL CLASSIFICATIONS
    ######################################################################
    for ap in aps:
        df_ap = linefns.ratio_fn(df_ap, s=f" ({ap})")
        df_ap = linefns.bpt_fn(df_ap, s=f" ({ap})")

    ######################################################################
    # EVALUATE METALLICITY
    ######################################################################
    for ap in aps:
        df_ap = metallicity.calculate_metallicity(met_diagnostic="N2Ha_PP04", compute_errors=True, niters=1000, df=df_ap, s=f" ({ap})")
        df_ap = metallicity.calculate_metallicity(met_diagnostic="N2Ha_M13", compute_errors=True, niters=1000, df=df_ap, s=f" ({ap})")
        df_ap = metallicity.calculate_metallicity(met_diagnostic="O3N2_PP04", compute_errors=True, niters=1000, df=df_ap, s=f" ({ap})")
        df_ap = metallicity.calculate_metallicity(met_diagnostic="O3N2_M13", compute_errors=True, niters=1000, df=df_ap, s=f" ({ap})")
        df_ap = metallicity.calculate_metallicity(met_diagnostic="N2S2Ha_D16", compute_errors=True, niters=1000, df=df_ap, s=f" ({ap})")
        df_ap = metallicity.calculate_metallicity(met_diagnostic="N2O2_KD02", compute_errors=True, niters=1000, df=df_ap, s=f" ({ap})")
        df_ap = metallicity.calculate_metallicity(met_diagnostic="Rcal_PG16", compute_errors=True, niters=1000, df=df_ap, s=f" ({ap})")
        df_ap = metallicity.calculate_metallicity(met_diagnostic="Scal_PG16", compute_errors=True, niters=1000, df=df_ap, s=f" ({ap})")
        df_ap = metallicity.calculate_metallicity(met_diagnostic="ON_P10", compute_errors=True, niters=1000, df=df_ap, s=f" ({ap})")
        df_ap = metallicity.calculate_metallicity(met_diagnostic="ONS_P10", compute_errors=True, niters=1000, df=df_ap, s=f" ({ap})")
        df_ap = metallicity.calculate_metallicity(met_diagnostic="N2Ha_K19", compute_logU=True, ion_diagnostic="O3O2_K19", compute_errors=True, niters=1000, df=df_ap, s=f" ({ap})")
        df_ap = metallicity.calculate_metallicity(met_diagnostic="O3N2_K19", compute_logU=True, ion_diagnostic="O3O2_K19", compute_errors=True, niters=1000, df=df_ap, s=f" ({ap})")
        df_ap = metallicity.calculate_metallicity(met_diagnostic="N2O2_K19", compute_logU=True, ion_diagnostic="O3O2_K19", compute_errors=True, niters=1000, df=df_ap, s=f" ({ap})")
        df_ap = metallicity.calculate_metallicity(met_diagnostic="R23_KK04", compute_logU=True, ion_diagnostic="O3O2_KK04", compute_errors=True, niters=1000, df=df_ap, s=f" ({ap})")

    ###############################################################################
    # Save input flags to the DataFrame so that we can keep track
    ###############################################################################
    df_ap["Extinction correction applied"] = correct_extinction
    df_ap["line_flux_SNR_cut"] = line_flux_SNR_cut
    df_ap["eline_SNR_min"] = eline_SNR_min
    df_ap["missing_fluxes_cut"] = missing_fluxes_cut
    df_ap["sigma_gas_SNR_min"] = sigma_gas_SNR_min
    df_ap["sigma_gas_SNR_cut"] = sigma_gas_SNR_cut

    ###############################################################################
    # Save to .hd5 & .csv
    ###############################################################################
    print(f"{status_str}: Saving to file...")

    # No extinction correction
    df_ap.to_csv(os.path.join(sami_data_path, df_fname.split("hd5")[0] + "csv"))
    try:
        df_ap.to_hdf(os.path.join(sami_data_path, df_fname), key=f"1-comp aperture fit")
    except:
        print(f"{status_str}: Unable to save to HDF file... sigh...")
    print(f"{status_str}: Finished!")
    return

