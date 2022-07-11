"""
File:       make_dfs_lzifu.py
Author:     Henry Zovaro
Email:      henry.zovaro@anu.edu.au

DESCRIPTION
------------------------------------------------------------------------------
This script contains the following functions:
    
    make_lzifu_df():     
        stores emission line fluxes & other measurements from LZIFU in
        DataFrames for individual SAMI galaxies that were re-processed using
        LZIFU. This essentially produces the same output as make_df_sami.py, 
        except where we use our own LZIFU fits rather than those in the 
        official data release so that we can obtain emission line fluxes 
        for individual emission line components for lines other than Halpha.

    merge_datacubes():
        Merge the results from my own LZIFU with the SAMI component maps to 
        determine the optimal number of components in each spaxel.

    load_lzifu_df():
        load a DataFrame containing emission line fluxes, etc. for a galaxy 
        (or galaxies) which were created using make_lzifu_df().

PREREQUISITES
------------------------------------------------------------------------------
SAMI_DIR, SAMI_DATACUBE_DIR, LZIFU_DIR and LZIFU_PRODUCTS_PATH must be defined 
as environment variables. The output .hd5 files containing the DataFrame
for each LZIFU galaxy are stored in LZIFU_DIR, whilst LZIFU_PRODUCTS_PATH is
the location of the files output by LZIFU (e.g., <gal>_1_comp.fits).

LZIFU must have been run for all galaxies specified.

------------------------------------------------------------------------------
Copyright (C) 2022 Henry Zovaro 
"""
import os, sys
import numpy as np
from itertools import product
from astropy.io import fits
import pandas as pd
from scipy import constants
from tqdm import tqdm

from spaxelsleuth.loaddata.sami import load_sami_galaxies
from spaxelsleuth.loaddata import dqcut, linefns, metallicity, extcorr

import matplotlib.pyplot as plt
plt.ion()
plt.close("all")

from IPython.core.debugger import Tracer

import warnings
warnings.filterwarnings(action="ignore", message="Mean of empty slice")
warnings.filterwarnings(action="ignore", message="invalid value encountered in sqrt")

###############################################################################
# Paths
assert "SAMI_DIR" in os.environ, "Environment variable SAMI_DIR is not defined!"
sami_data_path = os.environ["SAMI_DIR"]
assert "SAMI_DATACUBE_DIR" in os.environ, "Environment variable SAMI_DATACUBE_DIR is not defined!"
sami_datacube_path = os.environ["SAMI_DATACUBE_DIR"]
assert "LZIFU_DIR" in os.environ, "Environment variable LZIFU_DIR is not defined!"
lzifu_data_path = os.environ["LZIFU_DIR"]
assert "LZIFU_PRODUCTS_PATH" in os.environ, "Environment variable LZIFU_PRODUCTS_PATH is not defined!"
lzifu_products_path = os.environ["LZIFU_PRODUCTS_PATH"]

###############################################################################
def merge_datacubes(gal=None, plotit=False):
    """
    Merge the results from my own LZIFU with the SAMI component maps to 
    determine the optimal number of components in each spaxel.

    INPUTS
    ---------------------------------------------------------------------------
    gal:    int
        GAMA ID of the galaxy.

    plotit: bool
        If True, show diagnostic plots. Useful for debugging.

    OUTPUTS
    ---------------------------------------------------------------------------
    Nothing.

    """
    if gal is None:
        gals = [int(f.split("_merge_comp.fits")[0]) for f in os.listdir(lzifu_products_path) if f.endswith("merge_comp.fits") and not f.startswith("._")]
    else:
        if type(gal) == list:
            gals = gal
        else:
            gals = [gal]
        for gal in gals:
            assert type(gal) == int, "gal must be an integer!"
            fname = os.path.join(lzifu_products_path, f"{gal}_merge_comp.fits")
            assert os.path.exists(fname), f"File {fname} not found!"

    for gal in tqdm(gals):
        ###############################################################################
        # Step 1: Create a map from the SAMI data showing how many components there 
        # should be in each spaxel.
        ###############################################################################
        # Open the SAMI data.
        hdulist_sami = fits.open(os.path.join(sami_data_path, f"{gal}/{gal}_A_Halpha_default_recom-comp.fits"))
        halpha_map = np.copy(hdulist_sami[0].data[1:])

        # Figure out how many components there are in each spaxel.
        halpha_map[~np.isnan(halpha_map)] = 1
        halpha_map[np.isnan(halpha_map)] = 0
        ncomponents_map = np.nansum(halpha_map, axis=0)

        ###############################################################################
        # Step 2: Go through each of the 1, 2, 3 component-fit LZIFU cubes and re-
        # construct the data saved in the final FITS file
        # Simply open the FITS file saved in merge.pro and over-write the contents.
        # Add an extra FITS header string to show that it's been edited.
        ###############################################################################

        # Open the FITS file saved in merge.pro.
        hdulist_merged = fits.open(os.path.join(lzifu_products_path, f"{gal}_merge_comp.fits"))
        hdulist_1 = fits.open(os.path.join(lzifu_products_path, f"{gal}_1_comp.fits"))
        hdulist_2 = fits.open(os.path.join(lzifu_products_path, f"{gal}_2_comp.fits"))
        hdulist_3 = fits.open(os.path.join(lzifu_products_path, f"{gal}_3_comp.fits"))

        # For checking
        halpha_map_old = np.copy(hdulist_merged["HALPHA"].data[1:])

        ###############################################################################
        # Replace the data in the FITS file with that from the appropriate LZIFU fit
        ###############################################################################
        for ncomponents, hdulist in zip([1, 2, 3], [hdulist_1, hdulist_2, hdulist_3]):
            mask = ncomponents_map == ncomponents
            # Quantities defined for each component
            for ext in ["V", "VDISP", "HALPHA", "HBETA", "OIII5007", "OI6300", "NII6583", "SII6716", "SII6731"]:
                hdulist_merged[ext].data[:ncomponents + 1, mask] = hdulist[ext].data[:, mask]
                hdulist_merged[ext].data[ncomponents + 1:, mask] = np.nan
                hdulist_merged[f"{ext}_ERR"].data[:ncomponents + 1, mask] = hdulist[f"{ext}_ERR"].data[:, mask]
                hdulist_merged[f"{ext}_ERR"].data[ncomponents + 1:, mask] = np.nan

            # Quantities defined as 2D maps
            for ext in ["CHI2", "DOF"]:
                hdulist_merged[ext].data[mask] = hdulist[ext].data[mask]

            # Quantities defined over the whole data cube 
            for ext in ["CONTINUUM", "LINE"] + [f"LINE_COMP{n + 1}" for n in range(ncomponents)]:
                hdulist_merged[f"B_{ext}"].data[:, mask] = hdulist[f"B_{ext}"].data[:, mask]
                hdulist_merged[f"R_{ext}"].data[:, mask] = hdulist[f"R_{ext}"].data[:, mask]

        ###############################################################################
        # Where there are 0 components, set everything to NaN
        ###############################################################################
        mask = ncomponents_map == 0
        # Quantities defined for each component
        for ext in ["V", "VDISP", "HALPHA", "HBETA", "OIII5007", "OI6300", "NII6583", "SII6716", "SII6731"]:
            hdulist_merged[ext].data[:, mask] = np.nan
            hdulist_merged[f"{ext}_ERR"].data[:, mask] = np.nan

        # Quantities defined as 2D maps
        for ext in ["CHI2", "DOF"]:
            hdulist_merged[ext].data[mask] = np.nan

        # Quantities defined over the whole data cube 
        for ext in ["CONTINUUM", "LINE"] + [f"LINE_COMP{n + 1}" for n in range(ncomponents)]:
            hdulist_merged[f"B_{ext}"].data[:, mask] = np.nan
            hdulist_merged[f"R_{ext}"].data[:, mask] = np.nan

        # Add the component map to the FITS file
        ncomponents_map[mask] = np.nan
        hdulist_merged["COMP_MAP"].data = ncomponents_map

        ###############################################################################
        # Save
        ###############################################################################
        # Add an extra FITS header string to show that it's been edited.
        hdulist_merged[0].header["NOTE"] = "Number of components determined from SAMI DR3 data"

        # Save
        hdulist_merged.writeto(os.path.join(lzifu_products_path, f"{gal}_merge_lzcomp.fits"), overwrite=True, output_verify="ignore")

        ###############################################################################
        # If desired, plot the Halpha maps in each component for both the 
        # LRT and LZCOMP-derived component maps
        ###############################################################################
        if plotit:
            halpha_map_new = np.copy(hdulist_merged["HALPHA"].data[1:])

            # Figure out how many components there are in each spaxel.
            halpha_map_new[~np.isnan(halpha_map_new)] = 1
            halpha_map_new[np.isnan(halpha_map_new)] = 0
            ncomponents_map_check = np.nansum(halpha_map_new, axis=0)
            ncomponents_map_check[ncomponents_map_check == 0] = np.nan

            fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
            axs[0].imshow(ncomponents_map)
            axs[1].imshow(ncomponents_map_check)

            # Show the old & new Halpha maps side-by-side
            fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(10, 15))
            fig.suptitle(r"%s - H$\alpha$ maps" % gal)
            axs[0][0].set_title("LZIFU LRT result")
            axs[0][1].set_title("SAMI DR3 LZCOMP result")
            axs[0][0].imshow(halpha_map_old[0])
            axs[0][1].imshow(hdulist_merged["HALPHA"].data[1])
            axs[1][0].imshow(halpha_map_old[1])
            axs[1][1].imshow(hdulist_merged["HALPHA"].data[2])
            axs[2][0].imshow(halpha_map_old[2])
            axs[2][1].imshow(hdulist_merged["HALPHA"].data[3])

        ###############################################################################
        # Assertion checks 
        ###############################################################################
        # Check that the 1st slice of the V, VDISP extensions are all NaN
        assert np.all(np.isnan(hdulist_merged["V"].data[0]))
        assert np.all(np.isnan(hdulist_merged["V_ERR"].data[0]))
        assert np.all(np.isnan(hdulist_merged["VDISP"].data[0]))
        assert np.all(np.isnan(hdulist_merged["VDISP_ERR"].data[0]))

        for ext in ["V", "VDISP", "HALPHA", "HBETA", "OIII5007", "OI6300", "NII6583", "SII6716", "SII6731"]:
            for ncomponents, hdulist in zip([1, 2, 3], [hdulist_1, hdulist_2, hdulist_3]):
                # Check that there are ONLY N data values in all N-component spaxels
                mask = ncomponents_map == ncomponents
                assert np.all(np.isnan(hdulist_merged[ext].data[ncomponents + 1:, mask]))
                assert np.all(np.isnan(hdulist_merged[f"{ext}_ERR"].data[ncomponents + 1:, mask]))

                # Check that the right data has been added 
                for ii in range(ncomponents):
                    diff = np.abs((hdulist_merged[ext].data[ii + 1, mask] - hdulist[ext].data[ii + 1, mask]) /hdulist_merged[ext].data[ii + 1, mask])
                    diff[np.isnan(diff)] = 0
                    assert np.all(diff < 1e-6)
                    diff = np.abs((hdulist_merged[f"{ext}_ERR"].data[ii + 1, mask] - hdulist[f"{ext}_ERR"].data[ii + 1, mask]) / hdulist_merged[f"{ext}_ERR"].data[ii + 1, mask])
                    diff[np.isnan(diff)] = 0
                    assert np.all(diff < 1e-6)

    return

#######################################################################
def load_lzifu_df(ncomponents, bin_type, correct_extinction, eline_SNR_min,
                        gal=None):
    """
    load a DataFrame containing emission line fluxes, etc. for a galaxy 
    (or galaxies) re-processed using LZIFU.

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

    gal:                int
        GAMA ID of the galaxy to load. If not specified, the DataFrame 
        containing all galaxies meeting the S/N criterion in make_dfs_lzifu.py 
        is returned.

    USAGE
    ---------------------------------------------------------------------------
    load_lzifu_df() is called as follows:

        >>> from spaxelsleuth.loaddata.lzifu import load_lzifu_df
        >>> df = load_lzifu_df(ncomponents, bin_type, correct_extinction, 
                                     eline_SNR_min, gal)

    If gal is not specified then the DataFrame containing the subset of "high-S/N"
    galaxies as defined in make_dfs_lzifu.py are returned.

    OUTPUTS
    ---------------------------------------------------------------------------
    The Dataframe.

    """
    #######################################################################
    # INPUT CHECKING
    #######################################################################
    assert (ncomponents == "recom") | (ncomponents == "1"), "ncomponents must be 'recom' or '1'!!"
    assert bin_type in ["default", "adaptive", "sectors"], "bin_type must be 'default' or 'adaptive' or 'sectors'!!"

    # Input file name
    if gal is None:
        print(f"Loading LZIFU DataFrame for all galaxies in the LZIFU subsample...")
        df_fname = f"lzifu_subsample_{bin_type}_{ncomponents}-comp"
    else:
        print(f"Loading LZIFU DataFrame for galaxy {gal}...")
        df_fname = f"lzifu_{gal}_{bin_type}_{ncomponents}-comp"
    if correct_extinction:
        df_fname += "_extcorr"
    df_fname += f"_minSNR={eline_SNR_min}"
    df_fname += ".hd5"

    assert os.path.exists(os.path.join(lzifu_data_path, df_fname)),\
        f"File {os.path.join(lzifu_data_path, df_fname)} does does not exist!"
    
    #######################################################################
    # LOAD THE DATAFRAME FROM MEMORY
    #######################################################################
    df = pd.read_hdf(os.path.join(lzifu_data_path, df_fname))

    return df

###############################################################################
def make_lzifu_df(gals=None, make_master_df=False,
                  bin_type="default", ncomponents="recom", 
                  line_flux_SNR_cut=True, eline_SNR_min=5,
                  vgrad_cut=False ,
                  sigma_gas_SNR_cut=True, sigma_gas_SNR_min=3,
                  line_amplitude_SNR_cut=True,
                  flux_fraction_cut=False,
                  stekin_cut=True,
                  met_diagnostic_list=["Dopita+2016"], logU = -3.0,
                  eline_list=["HALPHA", "HBETA", "NII6583", "OI6300", "OII3726+OII3729", "OIII5007", "SII6716", "SII6731"],
                  nthreads_max=20, plotit=False,):
    """
    DESCRIPTION
    ---------------------------------------------------------------------------
    This function stores emission line fluxes & other measurements from LZIFU 
    in a DataFrame. This essentially produces the same output as make_df_sami.py, except where we use our own LZIFU fits rather than 
    those in the official data release so that we can obtain emission line fluxes 
    for individual emission line components for lines other than Halpha.

    INPUTS
    ---------------------------------------------------------------------------
    gals:               int or list of ints
        Galaxy ID or IDs for which to generate DataFrames. A single ID can be 
        supplied or a list of IDs. IDs are GAMA IDs expressed as integers.
        If gals is not specified then DataFrames are run for ALL LZIFU galaxies 
        that meet the minimum red S/N requiremnt of

            Median SNR (R, 2R_e) >= 10.0.

    make_master_df:             bool
        If True, create DataFrames for ALL LZIFU galaxies that meet the minimum 
        red S/N requiremnt of

            Median SNR (R, 2R_e) >= 10.0

        and store the resullt in a 
    
    ncomponents:                str
        Which number of Gaussian components to assume. For now this must be 
        "recom" (the recommended multi-component fits).

    bin_type:                   str
        Spatial binning strategy. For now this must be "default" (unbinned).

    eline_SNR_min:              int 
        Minimum emission line flux S/N to accept.

    line_flux_SNR_cut:          bool
        If True, make a S/N cut on all emission line components and associated
        quantities with flux S/N below eline_SNR_min.

    vgrad_cut:                  bool         
        If True, mask component kinematics (velocity and velocity dispersion)
        that are likely to be affected by beam smearing.
        By default this is set to False b/c it tends to remove nuclear spaxels 
        which may be of interest to your science case, & because it doesn't 
        reliably remove spaxels with quite large beam smearing components

    sigma_gas_SNR_cut:          bool
        If True, mask component velocity dispersions where the S/N on the 
        velocity dispersion measurement is below sigma_gas_SNR_min. 
        By default this is set to True b/c it's a robust way to account for 
        emission line widths < instrumental.

    sigma_gas_SNR_min:          int
        Minimum velocity dipersion S/N to accept.

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

    stekin_cut:
        If True, mask stellar kinematic quantities that do not meet the DQ and 
        S/N requirements specified in Croom et al. (2021). True by default.

    eline_list:                 list of str
        Default SAMI emission lines - don't change this!

    met_diagnostic_list:        list of str
        Which metallicity diagnostics to compute. Good options are "Dopita+2016"
        and "N2O2".

    logU:                       float            
        Constant ionisation parameter to assume in metallicity calculation.
        Default value is -3.0.

    nthreads_max:               int            
        Maximum number of threds to use.  

    OUTPUTS
    ------------------------------------------------------------------------------
    Separate DataFrames will be saved for each individual galaxy, following the
    name format 

        f"lzifu_<gal>_<bin_type>_<ncomponents>-comp_minSNR=<eline_SNR_min>.hd5"
    and f"lzifu_{gal}_{bin_type}_{ncomponents}-comp_extcorr_minSNR={eline_SNR_min}.hd5"

    in the directory specified by lzifu_products_path. If no galaxies are specified in
    the input arguments, then a single DataFrame containing all galaxies in the 
    subsample is created with name 

        f"lzifu_subsample_{bin_type}_{ncomponents}-comp_minSNR={eline_SNR_min}.hd5"
    and f"lzifu_subsample_{bin_type}_{ncomponents}-comp_extcorr_minSNR={eline_SNR_min}.hd5"

    PREREQUISITES
    ------------------------------------------------------------------------------
    SAMI_DIR, SAMI_DATACUBE_DIR, LZIFU_DIR and LZIFU_PRODUCTS_PATH must be defined 
    as environment variables. The output .hd5 files containing the DataFrame
    for each LZIFU galaxy are stored in LZIFU_DIR, whilst LZIFU_PRODUCTS_PATH is
    the location of the files output by LZIFU (e.g., <gal>_1_comp.fits).

    LZIFU must have been run for all galaxies specified, and merge_datacubes()
    must also have been run first.

    If you want to run this function on all LZIFU galaxies, then 
    make_sami_metadata_df_extended.py must also have been run first in order to 
    obtain continuum S/N values used in deciding which targets are included in
    the high-S/N sample.

    """

    # For printing to stdout
    status_str = f"In sami.make_dfs_lzifu() [eline_SNR_min={eline_SNR_min}]"

    ###############################################################################
    # READ IN THE METADATA
    ###############################################################################
    df_metadata_fname = "sami_dr3_metadata.hd5"
    df_metadata = pd.read_hdf(os.path.join(sami_data_path, df_metadata_fname))
    df_metadata["Good?"] = df_metadata["Good?"].astype("float")

    ###############################################################################
    # Input checking 
    if gals is not None:
        assert make_master_df == False,\
            "make_master_df can only be set to True if no gals are specified!"
        assert type(gals) == list or type(gals) == int,\
            "gals must either be a list of galaxy IDs or an integer!"
        if type(gals) == int:
            gals = [gals]
        for gal in gals:
            Tracer()()
            assert type(gal) == int, "gal must be an integer!"
            fname = os.path.join(lzifu_products_path, f"{gal}_merge_lzcomp.fits")
            assert os.path.exists(fname), f"File {fname} not found!"
        print(f"{status_str}: WARNING: creating DFs for provided galaxies")
    else:
        assert make_master_df,\
            "If no gals are specified then make_master_df must be set to True!"

    if make_master_df:
        print(f"{status_str}: WARNING: creating DFs for all galaxies in the subsample")

        ###############################################################################
        # Make DataFrames for all galaxies in our high-S/N sample
        ###############################################################################
        # List of ALL galaxies we have LZIFU data for
        gals_merge_lzcomp = [int(f.split("_merge_lzcomp.fits")[0]) for f in os.listdir(lzifu_products_path) if f.endswith("merge_lzcomp.fits") and not f.startswith("._")]    
        gals_merge_comp = [int(f.split("_merge_comp.fits")[0]) for f in os.listdir(lzifu_products_path) if f.endswith("merge_comp.fits") and not f.startswith("._")]    
        gals_not_merged = [g for g in gals_merge_comp if g not in gals_merge_lzcomp]
        if len(gals_not_merged) > 0:
            print("WARNING: there are galaxies with LZIFU data that has not been re-component-ized using merge_datacubes()!")

        # Load the DataFrame that gives us the continuum S/N, so we can select a subset
        df_info = pd.read_hdf(os.path.join(sami_data_path, "sami_dr3_metadata_extended.hd5"))

        # Shortlist: median R S/N in 2R_e > 10
        cond_subsample = df_info["Median SNR (R, 2R_e)"] >= 10
        cond_subsample &= df_info["Good?"] == True
        gals_subsample = df_info[cond_subsample].index.values

        # List of galaxies with only 1 component, which we will get from the SAMI data instead
        gals_good_1comp = [g for g in gals_subsample if df_info.loc[g, "Maximum number of components"] == 1]
        gals_good_0comp = [g for g in gals_subsample if df_info.loc[g, "Maximum number of components"] == 0]

        # List of galaxies with LZIFU 
        gals_good_lzifu = [g for g in gals_subsample if g in gals_merge_lzcomp and g not in gals_good_1comp and g not in gals_good_0comp]

        # List of galaxies WITHOUT LZIFU in our sample
        gals_good_missing_lzifu = [g for g in gals_subsample if (g not in gals_merge_lzcomp) and (g not in gals_good_1comp) and (g not in gals_good_0comp)]
        assert len(gals_good_missing_lzifu) == 0,\
            f"{len(gals_good_missing_lzifu)} are missing LZIFU data!"

        # Want to run this script on the gals in the subsample that have LZIFU data.
        gals = gals_good_lzifu

        # Save a separate data frame for each galaxy      
        df_all = None  # "big" DF to store all LZIFU galaxies
        df_all_extcorr = None  # "big" DF to store all LZIFU galaxies

    for gal in tqdm(gals):
        # Filenames
        df_fname = f"lzifu_{gal}_{bin_type}_{ncomponents}-comp_minSNR={eline_SNR_min}.hd5"
        df_fname_extcorr = f"lzifu_{gal}_{bin_type}_{ncomponents}-comp_extcorr_minSNR={eline_SNR_min}.hd5"

        ###############################################################################
        # STORING IFS DATA
        ###############################################################################
        # X, Y pixel coordinates
        ys, xs = np.meshgrid(np.arange(50), np.arange(50), indexing="ij")
        as_per_px = 0.5
        ys_as = ys * as_per_px
        xs_as = xs * as_per_px

        # Centre galaxy coordinates (see p16 of Croom+2021)
        x0_px = 25.5
        y0_px = 25.5

        #######################################################################
        # Open the required FITS files.
        hdulist_B_cube = fits.open(os.path.join(sami_datacube_path, f"ifs/{gal}/{gal}_A_cube_blue.fits.gz"))
        hdulist_R_cube = fits.open(os.path.join(sami_datacube_path, f"ifs/{gal}/{gal}_A_cube_red.fits.gz"))
        hdulist_lzifu = fits.open(os.path.join(lzifu_products_path, f"{gal}_merge_lzcomp.fits"))

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
        lambda_vals_A = np.array(range(N_lambda)) * dlambda_A + lambda_0_A 

        # Compute the D4000Å break
        # Definition from Balogh+1999 (see here: https://arxiv.org/pdf/1611.07050.pdf, page 3)
        start_b_idx = np.nanargmin(np.abs(lambda_vals_A / (1 + df_metadata.loc[gal, "z_spec"]) - 3850))
        stop_b_idx = np.nanargmin(np.abs(lambda_vals_A / (1 + df_metadata.loc[gal, "z_spec"]) - 3950))
        start_r_idx = np.nanargmin(np.abs(lambda_vals_A / (1 + df_metadata.loc[gal, "z_spec"]) - 4000))
        stop_r_idx = np.nanargmin(np.abs(lambda_vals_A / (1 + df_metadata.loc[gal, "z_spec"]) - 4100))
        N_b = stop_b_idx - start_b_idx
        N_r = stop_r_idx - start_r_idx

        # Convert datacube & variance cubes to units of F_nu
        data_cube_B_Hz = data_cube_B * lambda_vals_A[:, None, None]**2 / (constants.c * 1e10)
        var_cube_B_Hz2 = var_cube_B * (lambda_vals_A[:, None, None]**2 / (constants.c * 1e10))**2

        num = np.nanmean(data_cube_B_Hz[start_r_idx:stop_r_idx], axis=0)
        denom = np.nanmean(data_cube_B_Hz[start_b_idx:stop_b_idx], axis=0)
        err_num = 1 / N_r * np.sqrt(np.nansum(var_cube_B_Hz2[start_r_idx:stop_r_idx], axis=0))
        err_denom = 1 / N_b * np.sqrt(np.nansum(var_cube_B_Hz2[start_b_idx:stop_b_idx], axis=0))

        d4000_map = num / denom
        d4000_map_err = d4000_map * np.sqrt((err_num / num)**2 + (err_denom / denom)**2)

        #######################################################################
        # Use the red cube to calculate the continuum intensity so 
        # that we can compute the HALPHA equivalent width.
        # Units of 10**(-16) erg /s /cm**2 /angstrom /pixel
        # Continuum wavelength range taken from here: https://ui.adsabs.harvard.edu/abs/2019MNRAS.485.4024V/abstract
        header = hdulist_R_cube[0].header
        data_cube_R = hdulist_R_cube[0].data 
        var_cube_R = hdulist_R_cube[1].data  

        # Wavelength values
        lambda_0_A = header["CRVAL3"] - header["CRPIX3"] * header["CDELT3"]
        dlambda_A = header["CDELT3"]
        N_lambda = header["NAXIS3"]
        lambda_vals_A = np.array(range(N_lambda)) * dlambda_A + lambda_0_A 

        # Compute continuum intensity
        start_idx = np.nanargmin(np.abs(lambda_vals_A / (1 + df_metadata.loc[gal, "z_spec"]) - 6500))
        stop_idx = np.nanargmin(np.abs(lambda_vals_A / (1 + df_metadata.loc[gal, "z_spec"]) - 6540))
        cont_map = np.nanmean(data_cube_R[start_idx:stop_idx], axis=0)
        cont_map_std = np.nanstd(data_cube_R[start_idx:stop_idx], axis=0)
        cont_map_err = 1 / (stop_idx - start_idx) * np.sqrt(np.nansum(var_cube_R[start_idx:stop_idx], axis=0))
        hdulist_R_cube.close() 

        #######################################################################
        # Compute v_grad using eqn. 1 of Zhou+2017
        # Open the velocity & velocity dispersion FITS files
        v = hdulist_lzifu["V"].data.astype(np.float64)[1:]
        v_grad = np.full_like(v, np.nan)

        # Compute v_grad for each spaxel in each component
        # in units of km/s/pixel
        for yy, xx in product(range(1, 49), range(1, 49)):
            v_grad[:, yy, xx] = np.sqrt(((v[:, yy, xx + 1] - v[:, yy, xx - 1]) / 2)**2 +\
                                        ((v[:, yy + 1, xx] - v[:, yy - 1, xx]) / 2)**2)

        #######################################################################
        # Create an image from the datacube to figure out where are "good" spaxels
        im = np.nansum(data_cube_B, axis=0)
        if np.any(im.flatten() < 0): # NaN out -ve spaxels. Most galaxies seem to have *some* -ve pixels
            im[im <= 0] = np.nan

        # Compute the coordinates of "good" spaxels, store in arrays
        y_c_list, x_c_list = np.argwhere(~np.isnan(im)).T
        ngood_bins = len(x_c_list)

        #######################################################################
        # Calculate the inclination
        # I think beta = 90 - PA...
        # Transform coordinates into the galaxy plane
        e = df_metadata.loc[gal, "ellip"]
        PA = df_metadata.loc[gal, "pa"]
        beta_rad = np.deg2rad(PA - 90)
        b_over_a = 1 - e
        q0 = 0.2
        # How to deal with scenario in which b_over_a**2 - q0**2 < 0?
        # This looks kind of dodgy for many galaxies - but the fraction we end up throwing away is about the same as Henry's DR2 work. So leave it for now.
        i_rad = np.arccos(np.sqrt((b_over_a**2 - q0**2) / (1 - q0**2)))  # Want to store this!

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
            for ax in axs:
                ax.clear()
            fig.suptitle(gal)
            axs[0].imshow(im, origin="lower")
            axs[1].axhline(0)
            axs[1].axvline(0)
            axs[0].scatter(x_c_list, y_c_list, color="k")
            axs[0].scatter(x0_px, y0_px, color="white")
            axs[1].scatter(x_prime_list, y_prime_list, color="r")
            axs[1].scatter(x_prime_list, y_prime_projec_list, color="r", alpha=0.3)
            # Plot circles showing 
            axs[1].axis("equal")
            fig.canvas.draw()

        #######################################################################
        # Open the LZIFU FITS file, extract the values from the maps in each 
        # extension & append
        rows_list = []
        colnames = []

        eline_list = ["HALPHA", "HBETA", "OIII5007", "OI6300", "NII6583", "SII6716", "SII6731"]
        extnames = eline_list + ["V", "VDISP"]

        # Number of kinematic components in each bin
        ncomponent_map = hdulist_lzifu["COMP_MAP"].data

        for ext in extnames:
            data = hdulist_lzifu[ext].data.astype(np.float64)[1:]
            data_err = hdulist_lzifu[f"{ext}_ERR"].data.astype(np.float64)[1:]

            # Extract values from maps in each spaxel, store in a row
            for nn in range(data.shape[0]):
                thisrow = np.full_like(x_c_list, np.nan, dtype="float")
                thisrow_err = np.full_like(x_c_list, np.nan, dtype="float")
                for jj, coords in enumerate(zip(x_c_list, y_c_list)):
                    x_c, y_c = coords
                    y, x = (int(np.round(y_c)), int(np.round(x_c)))
                    thisrow[jj] = data[nn, y, x]
                    thisrow_err[jj] = data_err[nn, y, x]
                rows_list.append(thisrow)
                rows_list.append(thisrow_err)
                colnames.append(f"{ext} (component {nn})")
                colnames.append(f"{ext}_ERR (component {nn})")

        hdulist_lzifu.close()

        #######################################################################
        # Do the same but with the stellar kinematics
        hdulist_v_star = fits.open(os.path.join(sami_data_path, f"ifs/{gal}/{gal}_A_stellar-velocity_default_two-moment.fits"))
        v_star = hdulist_v_star[0].data.astype(np.float64)
        v_star_err = hdulist_v_star[1].data.astype(np.float64)
        hdulist_v_star.close()

        hdulist_vdisp_star = fits.open(os.path.join(sami_data_path, f"ifs/{gal}/{gal}_A_stellar-velocity-dispersion_default_two-moment.fits"))
        vdisp_star = hdulist_vdisp_star[0].data.astype(np.float64)
        vdisp_star_err = hdulist_vdisp_star[1].data.astype(np.float64)
        hdulist_vdisp_star.close()

        thisrow = np.full_like(x_c_list, np.nan, dtype="float")
        thisrow_err = np.full_like(x_c_list, np.nan, dtype="float")
        for jj, coords in enumerate(zip(x_c_list, y_c_list)):
            x_c, y_c = coords
            y, x = (int(np.round(y_c)), int(np.round(x_c)))
            thisrow[jj] = v_star[y, x]
            thisrow_err[jj] = v_star_err[y, x]
        rows_list.append(thisrow)
        rows_list.append(thisrow_err)
        colnames.append("v_*")
        colnames.append("v_* error")        

        thisrow = np.full_like(x_c_list, np.nan, dtype="float")
        thisrow_err = np.full_like(x_c_list, np.nan, dtype="float")
        for jj, coords in enumerate(zip(x_c_list, y_c_list)):
            x_c, y_c = coords
            y, x = (int(np.round(y_c)), int(np.round(x_c)))
            thisrow[jj] = vdisp_star[y, x]
            thisrow_err[jj] = vdisp_star_err[y, x]
        rows_list.append(thisrow)
        rows_list.append(thisrow_err)
        colnames.append("sigma_*")
        colnames.append("sigma_* error")  

        #######################################################################
        # Do the same but with the HALPHA extinction correction factor
        hdulist_ext_corr = fits.open(os.path.join(sami_data_path, f"ifs/{gal}/{gal}_A_extinct-corr_default_recom-comp.fits"))
        ext_corr = hdulist_ext_corr[0].data.astype(np.float64)
        ext_corr_err = hdulist_ext_corr[1].data.astype(np.float64)
        hdulist_ext_corr.close()

        thisrow = np.full_like(x_c_list, np.nan, dtype="float")
        thisrow_err = np.full_like(x_c_list, np.nan, dtype="float")
        for jj, coords in enumerate(zip(x_c_list, y_c_list)):
            x_c, y_c = coords
            y, x = (int(np.round(y_c)), int(np.round(x_c)))
            thisrow[jj] = ext_corr[y, x]
            thisrow_err[jj] = ext_corr_err[y, x]
        rows_list.append(thisrow)
        rows_list.append(thisrow_err)
        colnames.append("HALPHA extinction correction")
        colnames.append("HALPHA extinction correction error")        

        #######################################################################
        # Do the same but with the SFR and SFR surface density
        hdulist_sfr = fits.open(os.path.join(sami_data_path, f"ifs/{gal}/{gal}_A_sfr_default_recom-comp.fits"))
        sfr = hdulist_sfr[0].data.astype(np.float64)[1]
        sfr_err = hdulist_sfr[1].data.astype(np.float64)[1]
        hdulist_sfr.close()

        thisrow = np.full_like(x_c_list, np.nan, dtype="float")
        thisrow_err = np.full_like(x_c_list, np.nan, dtype="float")
        for jj, coords in enumerate(zip(x_c_list, y_c_list)):
            x_c, y_c = coords
            y, x = (int(np.round(y_c)), int(np.round(x_c)))
            thisrow[jj] = sfr[y, x]
            thisrow_err[jj] = sfr_err[y, x]
        rows_list.append(thisrow)
        rows_list.append(thisrow_err)
        colnames.append("SFR (total)")
        colnames.append("SFR error (total)")   

        #######################################################################
        # Do the same but with the SFR and SFR surface density
        hdulist_sfrdens = fits.open(os.path.join(sami_data_path, f"ifs/{gal}/{gal}_A_sfr-dens_default_recom-comp.fits"))
        sfrdens = hdulist_sfrdens[0].data.astype(np.float64)[1]
        sfrdens_err = hdulist_sfrdens[1].data.astype(np.float64)[1]
        hdulist_sfrdens.close()

        thisrow = np.full_like(x_c_list, np.nan, dtype="float")
        thisrow_err = np.full_like(x_c_list, np.nan, dtype="float")
        for jj, coords in enumerate(zip(x_c_list, y_c_list)):
            x_c, y_c = coords
            y, x = (int(np.round(y_c)), int(np.round(x_c)))
            thisrow[jj] = sfrdens[y, x]
            thisrow_err[jj] = sfrdens_err[y, x]
        rows_list.append(thisrow)
        rows_list.append(thisrow_err)
        colnames.append("SFR surface density (total)")
        colnames.append("SFR surface density error (total)")

        ####################################################################### 
        # Do the same but with v_grad
        for nn in range(v_grad.shape[0]):
            thisrow = np.full_like(x_c_list, np.nan, dtype="float")
            thisrow_err = np.full_like(x_c_list, np.nan, dtype="float")
            for jj, coords in enumerate(zip(x_c_list, y_c_list)):
                x_c, y_c = coords
                y, x = (int(np.round(y_c)), int(np.round(x_c)))
                thisrow[jj] = v_grad[nn, y, x]
            rows_list.append(thisrow)
            colnames.append(f"v_grad (component {nn})")       

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
            thisrow[jj] = cont_map[y, x]
            thisrow_std[jj] = cont_map_std[y, x]
            thisrow_err[jj] = cont_map_err[y, x]
        rows_list.append(thisrow)
        rows_list.append(thisrow_std)
        rows_list.append(thisrow_err)
        colnames.append("HALPHA continuum")
        colnames.append("HALPHA continuum std. dev.")
        colnames.append("HALPHA continuum error")        

        #######################################################################
        # Do the same but with the D4000Å break
        thisrow = np.full_like(x_c_list, np.nan, dtype="float")
        thisrow_err = np.full_like(x_c_list, np.nan, dtype="float")
        for jj, coords in enumerate(zip(x_c_list, y_c_list)):
            x_c, y_c = coords
            y, x = (int(np.round(y_c)), int(np.round(x_c)))
            thisrow[jj] = d4000_map[y, x]
            thisrow_err[jj] = d4000_map_err[y, x]
        rows_list.append(thisrow)
        rows_list.append(thisrow_err)
        colnames.append("D4000")
        colnames.append("D4000 error")        

        #######################################################################
        # Do the same but with the number of kinematic components
        thisrow = np.full_like(x_c_list, np.nan, dtype="float")
        thisrow_err = np.full_like(x_c_list, np.nan, dtype="float")
        for jj, coords in enumerate(zip(x_c_list, y_c_list)):
            x_c, y_c = coords
            y, x = (int(np.round(y_c)), int(np.round(x_c)))
            thisrow[jj] = ncomponent_map[y, x]
        rows_list.append(thisrow)
        colnames.append("Number of components")   

        #######################################################################
        # Add pixel coordinates
        rows_list.append(np.array([np.rad2deg(i_rad)] * ngood_bins))
        rows_list.append(np.array([x0_px] * ngood_bins) * as_per_px)
        rows_list.append(np.array([y0_px] * ngood_bins) * as_per_px)
        rows_list.append(np.array(x_c_list).flatten() * as_per_px)
        rows_list.append(np.array(y_c_list).flatten() * as_per_px)
        rows_list.append(np.array(x_prime_list).flatten() * as_per_px)
        rows_list.append(np.array(y_prime_list).flatten() * as_per_px)
        rows_list.append(np.array(r_prime_list).flatten() * as_per_px)
        rows_list.append(np.array([1] * ngood_bins))
        rows_list.append(np.array([as_per_px**2] * ngood_bins))
        rows_list.append(np.array([as_per_px**2 * df_metadata.loc[gal, "kpc per arcsec"]**2] * ngood_bins))
        colnames.append("Inclination i (degrees)")
        colnames.append("Galaxy centre x0_px (projected, arcsec)")
        colnames.append("Galaxy centre y0_px (projected, arcsec)")
        colnames.append("x (projected, arcsec)")
        colnames.append("y (projected, arcsec)")
        colnames.append("x (relative to galaxy centre, deprojected, arcsec)")
        colnames.append("y (relative to galaxy centre, deprojected, arcsec)")
        colnames.append("r (relative to galaxy centre, deprojected, arcsec)")
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
        safe_cols = [c for c in df_metadata.columns if c != "Morphology"]
        gal_metadata = np.tile(df_metadata[df_metadata.loc[:, "catid"] == gal][safe_cols].values, (ngood_bins, 1))
        rows_good = np.hstack((gal_metadata, rows_good))

        ###############################################################################
        # Convert to a Pandas DataFrame
        ###############################################################################
        df_spaxels = pd.DataFrame(rows_good, columns=safe_cols + colnames)

        ###############################################################################
        # Add the morphology column back in
        ###############################################################################
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
        # Rename some columns
        ###############################################################################
        rename_dict = {}
        for eline in eline_list:
            for ii in range(3):
                rename_dict[f"{eline}_ERR (component {ii})"] = f"{eline} error (component {ii})"
        for ii in range(3):
            rename_dict[f"V (component {ii})"] = f"v_gas (component {ii})"
            rename_dict[f"V_ERR (component {ii})"] = f"v_gas error (component {ii})"
        for ii in range(3):
            rename_dict[f"VDISP (component {ii})"] = f"sigma_gas (component {ii})"
            rename_dict[f"VDISP_ERR (component {ii})"] = f"sigma_gas error (component {ii})"

        # R_e
        rename_dict["r_e"] = "R_e (arcsec)"

        # Rename columns
        df_spaxels = df_spaxels.rename(columns=rename_dict)

        ###############################################################################
        # Compute the ORIGINAL number of components
        ###############################################################################
        if ncomponents == "recom":
            df_spaxels["Number of components (original)"] =\
                (~df_spaxels["sigma_gas (component 0)"].isna()).astype(int) +\
                (~df_spaxels["sigma_gas (component 1)"].isna()).astype(int) +\
                (~df_spaxels["sigma_gas (component 2)"].isna()).astype(int)
        elif ncomponents == "1":
            df_spaxels["Number of components (original)"] =\
                (~df_spaxels["sigma_gas (component 0)"].isna()).astype(int)

        ###############################################################################
        # Calculate equivalent widths
        ###############################################################################
        df_spaxels.loc[df_spaxels["HALPHA continuum"] < 0, "HALPHA continuum"] = 0
        for nn in range(3):
            # Cast to float
            df_spaxels[f"HALPHA (component {nn})"] = pd.to_numeric(df_spaxels[f"HALPHA (component {nn})"])
            df_spaxels[f"HALPHA error (component {nn})"] = pd.to_numeric(df_spaxels[f"HALPHA error (component {nn})"])

            # Compute EWs
            df_spaxels[f"HALPHA EW (component {nn})"] = df_spaxels[f"HALPHA (component {nn})"] / df_spaxels["HALPHA continuum"]
            df_spaxels.loc[np.isinf(df_spaxels[f"HALPHA EW (component {nn})"].astype(float)), f"HALPHA EW (component {nn})"] = np.nan  # If the continuum level == 0, then the EW is undefined, so set to NaN.
            df_spaxels[f"HALPHA EW error (component {nn})"] = df_spaxels[f"HALPHA EW (component {nn})"] *\
                np.sqrt((df_spaxels[f"HALPHA error (component {nn})"] / df_spaxels[f"HALPHA (component {nn})"])**2 +\
                        (df_spaxels[f"HALPHA continuum error"] / df_spaxels[f"HALPHA continuum"])**2) 

            # If the continuum level <= 0, then the EW is undefined, so set to NaN.
            df_spaxels.loc[df_spaxels["HALPHA continuum"] <= 0, 
                       [f"HALPHA EW (component {nn})", 
                        f"HALPHA EW error (component {nn})"]] = np.nan  

        # Calculate total EWs
        df_spaxels["HALPHA EW (total)"] = np.nansum([df_spaxels[f"HALPHA EW (component {ii})"] for ii in range(3 if ncomponents == "recom" else 1)], axis=0)
        df_spaxels["HALPHA EW error (total)"] = np.sqrt(np.nansum([df_spaxels[f"HALPHA EW error (component {ii})"]**2 for ii in range(3 if ncomponents == "recom" else 1)], axis=0))

        # If all HALPHA EWs are NaN, then make the total HALPHA EW NaN too
        if ncomponents == "recom":
            df_spaxels.loc[df_spaxels["HALPHA EW (component 0)"].isna() &\
                           df_spaxels["HALPHA EW (component 1)"].isna() &\
                           df_spaxels["HALPHA EW (component 2)"].isna(), 
                           ["HALPHA EW (total)", "HALPHA EW error (total)"]] = np.nan
        elif ncomponents == "1":
            df_spaxels.loc[df_spaxels["HALPHA EW (component 0)"].isna(),
                           ["HALPHA EW (total)", "HALPHA EW error (total)"]] = np.nan
        
        ######################################################################
        # SFR and SFR surface density
        ######################################################################
        # Rename SFR (compnent 0) to SFR (total)
        rename_dict = {}
        rename_dict["SFR (component 0)"] = "SFR (total)"
        rename_dict["SFR error (component 0)"] = "SFR error (total)"
        rename_dict["SFR surface density (component 0)"] = "SFR surface density (total)"
        rename_dict["SFR surface density error (component 0)"] = "SFR surface density error (total)"

        df_spaxels = df_spaxels.rename(columns=rename_dict)

        ######################################################################
        # Add radius-derived value columns
        ######################################################################
        df_spaxels["r/R_e"] = df_spaxels["r (relative to galaxy centre, deprojected, arcsec)"] / df_spaxels["R_e (arcsec)"]
        df_spaxels["R_e (kpc)"] = df_spaxels["R_e (arcsec)"] * df_spaxels["kpc per arcsec"]
        df_spaxels["log(M/R_e)"] = df_spaxels["mstar"] - np.log10(df_spaxels["R_e (kpc)"])

        ######################################################################
        # Compute S/N in all lines, in all components
        # Compute TOTAL line fluxes
        ######################################################################
        for eline in ["HALPHA", "HBETA", "NII6583", "OI6300", "OIII5007", "SII6716", "SII6731"]:
            # Compute S/N 
            for ii in range(3):
                if f"{eline} (component {ii})" in df_spaxels.columns:
                    df_spaxels[f"{eline} S/N (component {ii})"] = df_spaxels[f"{eline} (component {ii})"] / df_spaxels[f"{eline} error (component {ii})"]
            
            # Compute total line fluxes, if the total fluxes are not given
            if f"{eline} (total)" not in df_spaxels.columns:
                df_spaxels[f"{eline} (total)"] = np.nansum([df_spaxels[f"{eline} (component {ii})"] for ii in range(3)], axis=0)
                df_spaxels[f"{eline} error (total)"] = np.sqrt(np.nansum([df_spaxels[f"{eline} error (component {ii})"]**2 for ii in range(3)], axis=0))

            # Compute the S/N in the TOTAL line flux
            df_spaxels[f"{eline} S/N (total)"] = df_spaxels[f"{eline} (total)"] / df_spaxels[f"{eline} error (total)"]

        ######################################################################
        # Fix SFR columns
        ######################################################################
        # Compute the SFR and SFR surface density from the 0th component ONLY
        df_spaxels["SFR surface density (component 0)"] = df_spaxels["SFR surface density (total)"] * df_spaxels["HALPHA (component 0)"] / df_spaxels["HALPHA (total)"]
        df_spaxels["SFR surface density error (component 0)"] = df_spaxels["SFR surface density error (total)"] * df_spaxels["HALPHA (component 0)"] / df_spaxels["HALPHA (total)"]
        df_spaxels["SFR (component 0)"] = df_spaxels["SFR (total)"] * df_spaxels["HALPHA (component 0)"] / df_spaxels["HALPHA (total)"]
        df_spaxels["SFR error (component 0)"] = df_spaxels["SFR error (total)"] * df_spaxels["HALPHA (component 0)"] / df_spaxels["HALPHA (total)"]

        # NaN the SFR surface density if the inclination is undefined
        cond_NaN_inclination = np.isnan(df_spaxels["Inclination i (degrees)"])
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
        df_spaxels = dqcut.dqcut(df=df_spaxels, 
                      ncomponents=3 if ncomponents == "recom" else 1,
                      line_flux_SNR_cut=line_flux_SNR_cut,
                      eline_SNR_min=eline_SNR_min, eline_list=eline_list,
                      sigma_gas_SNR_cut=sigma_gas_SNR_cut,
                      sigma_gas_SNR_min=sigma_gas_SNR_min,
                      sigma_inst_kms=29.6,
                      vgrad_cut=vgrad_cut,
                      line_amplitude_SNR_cut=line_amplitude_SNR_cut,
                      flux_fraction_cut=flux_fraction_cut,
                      stekin_cut=stekin_cut)

        ######################################################################
        # Make a copy of the DataFrame with EXTINCTION CORRECTION
        # Correct emission line fluxes (but not EWs!)
        # NOTE: extinction.fm07 assumes R_V = 3.1 so do not change R_V from 
        # this value!!!
        ######################################################################
        print(f"{status_str}: Correcting emission line fluxes (but not EWs) for extinction...")
        df_spaxels_extcorr = df_spaxels.copy()
        df_spaxels_extcorr = extcorr.extinction_corr_fn(df_spaxels_extcorr, 
                                        eline_list=eline_list,
                                        reddening_curve="fm07", 
                                        balmer_SNR_min=5, nthreads=nthreads_max,
                                        s=f" (total)")
        df_spaxels_extcorr["Corrected for extinction?"] = True
        df_spaxels["Corrected for extinction?"] = False

        # Sort so that both DataFrames have the same order
        df_spaxels_extcorr = df_spaxels_extcorr.sort_index()
        df_spaxels = df_spaxels.sort_index()

        ######################################################################
        # EVALUATE LINE RATIOS & SPECTRAL CLASSIFICATIONS
        ######################################################################
        df_spaxels = linefns.ratio_fn(df_spaxels, s=f" (total)")
        df_spaxels = linefns.bpt_fn(df_spaxels, s=f" (total)")
        df_spaxels_extcorr = linefns.ratio_fn(df_spaxels_extcorr, s=f" (total)")
        df_spaxels_extcorr = linefns.bpt_fn(df_spaxels_extcorr, s=f" (total)")

        ######################################################################
        # EVALUATE ADDITIONAL COLUMNS - log quantites, etc.
        ######################################################################
        df_spaxels = dqcut.compute_extra_columns(df_spaxels, ncomponents=3 if ncomponents=="recom" else 1)
        df_spaxels_extcorr = dqcut.compute_extra_columns(df_spaxels_extcorr, ncomponents=3 if ncomponents=="recom" else 1)

        ######################################################################
        # EVALUATE METALLICITY
        ######################################################################
        for met_diagnostic in met_diagnostic_list:
            df_spaxels = metallicity.metallicity_fn(df_spaxels, met_diagnostic, logU, s=" (total)")
            df_spaxels_extcorr = metallicity.metallicity_fn(df_spaxels_extcorr, met_diagnostic, logU, s=" (total)")

        ###############################################################################
        # Save input flags to the DataFrame so that we can keep track
        ###############################################################################
        df_spaxels["Extinction correction applied"] = False
        df_spaxels["line_flux_SNR_cut"] = line_flux_SNR_cut
        df_spaxels["eline_SNR_min"] = eline_SNR_min
        df_spaxels["sigma_gas_SNR_min"] = sigma_gas_SNR_min
        df_spaxels["vgrad_cut"] = vgrad_cut
        df_spaxels["sigma_gas_SNR_cut"] = sigma_gas_SNR_cut
        df_spaxels["sigma_gas_SNR_min"] = sigma_gas_SNR_min
        df_spaxels["line_amplitude_SNR_cut"] = line_amplitude_SNR_cut
        df_spaxels["flux_fraction_cut"] = flux_fraction_cut
        df_spaxels["stekin_cut"] = stekin_cut
        df_spaxels["log(U) (const.)"] = logU

        df_spaxels_extcorr["Extinction correction applied"] = True
        df_spaxels_extcorr["line_flux_SNR_cut"] = line_flux_SNR_cut
        df_spaxels_extcorr["eline_SNR_min"] = eline_SNR_min
        df_spaxels_extcorr["sigma_gas_SNR_min"] = sigma_gas_SNR_min
        df_spaxels_extcorr["vgrad_cut"] = vgrad_cut
        df_spaxels_extcorr["sigma_gas_SNR_cut"] = sigma_gas_SNR_cut
        df_spaxels_extcorr["sigma_gas_SNR_min"] = sigma_gas_SNR_min
        df_spaxels_extcorr["line_amplitude_SNR_cut"] = line_amplitude_SNR_cut
        df_spaxels_extcorr["flux_fraction_cut"] = flux_fraction_cut
        df_spaxels_extcorr["stekin_cut"] = stekin_cut
        df_spaxels_extcorr["log(U) (const.)"] = logU

        ######################################################################
        # Save to .hd5
        ######################################################################
        # Add catid column
        df_spaxels["catid"] = gal
        df_spaxels_extcorr["catid"] = gal
        df_spaxels.to_hdf(os.path.join(lzifu_data_path, df_fname), key=f"LZIFU")
        df_spaxels_extcorr.to_hdf(os.path.join(lzifu_data_path, df_fname_extcorr), key=f"LZIFU")

        ######################################################################
        # Concatenate 
        ######################################################################
        if make_master_df:
            if df_all is not None:
                df_all = pd.concat([df_all, df_spaxels], ignore_index=True)
                df_all_extcorr = pd.concat([df_all_extcorr, df_spaxels_extcorr], ignore_index=True)
            else:
                df_all = df_spaxels
                df_all_extcorr = df_spaxels_extcorr

    ######################################################################
    # Tack on the 1 and 0-component galaxies to the master DataFrame
    ######################################################################
    if make_master_df:
        #/////////////////////////////////////////////////////////////////
        # Load the SAMI sample (with extinction correction)
        df_sami_extcorr_fname = f"sami_{bin_type}_{ncomponents}-comp_extcorr_minSNR={eline_SNR_min}.hd5"
        df_sami_extcorr = pd.read_hdf(os.path.join(sami_data_path, df_sami_extcorr_fname), key=f"{bin_type}, {ncomponents}-comp")

        # Add the information for the 0- and 1-component galaxies
        for gal in gals_good_1comp + gals_good_0comp:
            df_gal = df_sami_extcorr[df_sami_extcorr["catid"] == gal] 
            df_all_extcorr = pd.concat([df_all_extcorr, df_gal], ignore_index=True)

        #/////////////////////////////////////////////////////////////////
        # Load the SAMI sample (no extinction correction)
        df_sami_fname = f"sami_{bin_type}_{ncomponents}-comp_minSNR={eline_SNR_min}.hd5"
        df_sami = pd.read_hdf(os.path.join(sami_data_path, df_sami_fname), key=f"{bin_type}, {ncomponents}-comp")

        # Add the information for the 0- and 1-component galaxies
        for gal in gals_good_1comp + gals_good_0comp:
            df_gal = df_sami[df_sami["catid"] == gal] 
            df_all = pd.concat([df_all, df_gal], ignore_index=True)

        ######################################################################
        # Save to file 
        ######################################################################
        df_all.to_hdf(os.path.join(lzifu_data_path, f"lzifu_subsample_{bin_type}_{ncomponents}-comp_minSNR={eline_SNR_min}.hd5"), key="LZIFU")
        df_all_extcorr.to_hdf(os.path.join(lzifu_data_path, f"lzifu_subsample_{bin_type}_{ncomponents}-comp_extcorr_minSNR={eline_SNR_min}.hd5"), key="LZIFU")

    return

