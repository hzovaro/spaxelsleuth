"""
File:       lzifu.py
Author:     Henry Zovaro
Email:      henry.zovaro@anu.edu.au

DESCRIPTION
------------------------------------------------------------------------------
This script contains the functions 

load_lzifu_galaxies():
    Equivalent to load_sami_galaxies() in sami.py. This function returns the 
    Pandas DataFrames containing spaxel-by-spaxel information for SAMI galaxies 
    that were created in make_dfs_lzifu.py. The DataFrame corresponding to a 
    single galaxy can be returned, or the entire subset of high-S/N galaxies,
    which includes 1- or 0-component-only galaxies from the original SAMI
    dataset.

merge_datacubes():
    By default, LZIFU uses the "likelihood ratio test (LRT)" to determine the 
    optimal number of components in each spaxel. However, this technique often
    produces results very inconsistent with those derived using LZCOMP, the 
    ANN used in SAMI DR3. 

    This function takes the component maps from SAMI DR3, and applies them
    to the LZIFU 1, 2 and 3-component fits to determine the optimal number of 
    components in each spaxel. 

    The results are saved in file <gal>_merge_lzcomp.fits, with identical
    extensions and data formats as those in <gal>_merge_comp.fits, which is 
    the default output of LZIFU.   


PREREQUISITES
------------------------------------------------------------------------------
SAMI_DIR and LZIFU_PRODUCTS_PATH must be defined as an environment variable,
where LZIFU_PRODUCTS_PATH points to the location of the LZIFU output files 
(e.g. <gal>_1_comp.fits, etc.)

LZIFU must have been run for all galaxies specified. 

Both make_dfs_lzifu.py and make_sami_metadata_df.py must be run first.

------------------------------------------------------------------------------
Copyright (C) 2022 Henry Zovaro
"""
#######################################################################
import os
import pandas as pd
import numpy as np
from astropy.io import fits
from tqdm import tqdm

assert "SAMI_DIR" in os.environ, "Environment variable SAMI_DIR is not defined!"
sami_data_path = os.environ["SAMI_DIR"]
assert "LZIFU_PRODUCTS_PATH" in os.environ, "Environment variable LZIFU_PRODUCTS_PATH is not defined!"
lzifu_products_path = os.environ["LZIFU_PRODUCTS_PATH"]
assert "LZIFU_PRODUCTS_PATH" in os.environ, "Environment variable LZIFU_PRODUCTS_PATH is not defined!"
lzifu_products_path = os.environ["LZIFU_PRODUCTS_PATH"]

#######################################################################
def load_lzifu_galaxies(ncomponents, bin_type, correct_extinction, eline_SNR_min,
                        gal=None, debug=False):
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
    load_lzifu_galaxies() is called as follows:

        >>> from spaxelsleuth.loaddata.lzifu import load_lzifu_galaxies
        >>> df = load_lzifu_galaxies(ncomponents, bin_type, correct_extinction, 
                                     eline_SNR_min, gal, debug)

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
    if debug:
        df_fname += "_DEBUG"
    df_fname += ".hd5"

    assert os.path.exists(os.path.join(sami_data_path, df_fname)),\
        f"File {os.path.join(sami_data_path, df_fname)} does does not exist!"
    
    #######################################################################
    # LOAD THE DATAFRAME FROM MEMORY
    #######################################################################
    df = pd.read_hdf(os.path.join(sami_data_path, df_fname))

    return df

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

###############################################################################
if __name__ == "__main__":

    import sys
    import numpy as np
    import matplotlib.pyplot as plt
    plt.close("all")
    plt.ion()

    gal = 209807
    eline_SNR_min = 5
    vgrad_cut = False

    df = load_lzifu_galaxies(gal=gal,
                            ncomponents="recom", bin_type="default")

    # CHECK: proper S/N cuts have been made
    for eline in ["HALPHA", "HBETA", "OIII5007", "OI6300", "NII6583", "SII6716", "SII6731"]:
        plt.figure()
        plt.scatter(x=df[f"{eline} (total)"], y=df[f"{eline} S/N (total)"], c="k", s=5)
        plt.ylabel(f"{eline} (total) S/N")
        plt.axhline(eline_SNR_min, color="r")
        plt.xlabel(f"{eline} (total) flux")
    for ii in range(3):
        plt.figure()
        plt.scatter(x=df[f"HALPHA (component {ii})"], y=df[f"HALPHA S/N (component {ii})"], c="k", s=5)
        plt.ylabel(f"HALPHA (component {ii}) S/N")
        plt.axhline(eline_SNR_min, color="r")
        plt.xlabel(f"HALPHA (component {ii}) flux")
    for ii in range(3):
        plt.figure()
        plt.scatter(x=df[f"HALPHA EW (component {ii})"], y=df[f"HALPHA S/N (component {ii})"], c="k", s=5)
        plt.ylabel(f"HALPHA (component {ii}) S/N")
        plt.axhline(eline_SNR_min, color="r")
        plt.xlabel(f"HALPHA EW (component {ii})")

    # CHECK: gas kinematics
    for ii in range(3):
        plt.figure()
        plt.scatter(x=df[f"sigma_obs S/N (component {ii})"], y=df[f"sigma_obs target S/N (component {ii})"], c="k", s=5)
        plt.ylabel(f"sigma_obs target S/N (component {ii})")
        plt.axhline(3, color="b")
        plt.xlabel(f"sigma_obs S/N (component {ii})")

    # CHECK: "Number of components" is 0, 1, 2 or 3
    unique_components = df["Number of components"].unique()
    unique_components.sort()
    assert np.all(unique_components == np.array([0., 1., 2., 3.]))

    # CHECK: all emission line fluxes in spaxels with 0 components are NaN
    for eline in ["HALPHA", "HBETA", "NII6583", "OI6300", "OIII5007", "SII6716", "SII6731"]:
        assert np.all(np.isnan(df.loc[df["Number of components"] == 0, f"{eline} (total)"]))
        assert np.all(np.isnan(df.loc[df["Number of components"] == 0, f"{eline} error (total)"]))

    # CHECK: all HALPHA-derived columns in spaxels with 0 components are NaN
    col = "HALPHA EW"
    assert np.all(np.isnan(df.loc[df["Number of components"] == 0, f"{col} (total)"]))
    assert np.all(np.isnan(df.loc[df["Number of components"] == 0, f"{col} error (total)"]))
    for ii in range(3):
        assert np.all(np.isnan(df.loc[df["Number of components"] == 0, f"{col} (component {ii})"]))
        assert np.all(np.isnan(df.loc[df["Number of components"] == 0, f"{col} error (component {ii})"]))

    # CHECK: all HALPHA-derived columns in spaxels with 0 components are NaN
    col = "log HALPHA EW"
    assert np.all(np.isnan(df.loc[df["Number of components"] == 0, f"{col} (total)"]))
    assert np.all(np.isnan(df.loc[df["Number of components"] == 0, f"{col} error (upper) (total)"]))
    assert np.all(np.isnan(df.loc[df["Number of components"] == 0, f"{col} error (lower) (total)"]))
    for ii in range(3):
        assert np.all(np.isnan(df.loc[df["Number of components"] == 0, f"{col} (component {ii})"]))
        assert np.all(np.isnan(df.loc[df["Number of components"] == 0, f"{col} error (upper) (component {ii})"]))
        assert np.all(np.isnan(df.loc[df["Number of components"] == 0, f"{col} error (lower) (component {ii})"]))

    # CHECK: all kinematic quantities in spaxels with 0 components are NaN
    for col in ["sigma_gas", "v_gas"]:
        for ii in range(3):
            assert np.all(np.isnan(df.loc[df["Number of components"] == 0, f"{col} (component {ii})"]))
            assert np.all(np.isnan(df.loc[df["Number of components"] == 0, f"{col} error (component {ii})"]))

    # CHECK: all emission line fluxes with S/N < SNR_min are NaN
    for eline in ["HALPHA", "HBETA", "OIII5007", "OI6300", "NII6583", "SII6716", "SII6731"]:
        assert np.all(np.isnan(df.loc[df[f"{eline} S/N (total)"] < eline_SNR_min, f"{eline} (total)"]))

        # CHECK: all eline components below S/N limit are NaN
        for ii in range(3):
            assert np.all(np.isnan(df.loc[df[f"{eline} S/N (component {ii})"] < eline_SNR_min, f"{eline} (component {ii})"]))
            assert np.all(np.isnan(df.loc[df[f"{eline} S/N (component {ii})"] < eline_SNR_min, f"{eline} error (component {ii})"]))
    
    # CHECK: all HALPHA EW components where the HALPHA flux is below S/N limit are NaN
    assert np.all(np.isnan(df.loc[df[f"HALPHA S/N (component {ii})"] < eline_SNR_min, f"HALPHA EW (component {ii})"]))
    assert np.all(np.isnan(df.loc[df[f"HALPHA S/N (component {ii})"] < eline_SNR_min, f"HALPHA EW error (component {ii})"]))
    assert np.all(np.isnan(df.loc[df[f"HALPHA S/N (component {ii})"] < eline_SNR_min, f"log HALPHA EW (component {ii})"]))
    assert np.all(np.isnan(df.loc[df[f"HALPHA S/N (component {ii})"] < eline_SNR_min, f"log HALPHA EW error (upper) (component {ii})"]))
    assert np.all(np.isnan(df.loc[df[f"HALPHA S/N (component {ii})"] < eline_SNR_min, f"log HALPHA EW error (lower) (component {ii})"]))

    # CHECK: all sigma_gas components with S/N < S/N target are NaN
    for ii in range(3):
        assert np.all(np.isnan(df.loc[df[f"sigma_obs S/N (component {ii})"] < df[f"sigma_obs target S/N (component {ii})"], f"sigma_gas (component {ii})"]))
        assert np.all(np.isnan(df.loc[df[f"sigma_obs S/N (component {ii})"] < df[f"sigma_obs target S/N (component {ii})"], f"sigma_gas error (component {ii})"]))

    # CHECK: all sigma_gas components that don't meet the v_grad requirement are NaN
    if vgrad_cut:
        for ii in range(3):
            assert np.all(np.isnan(df.loc[df[f"v_grad (component {ii})"] > 2 * df[f"sigma_gas (component {ii})"], f"sigma_gas (component {ii})"]))
            assert np.all(np.isnan(df.loc[df[f"v_grad (component {ii})"] > 2 * df[f"sigma_gas (component {ii})"], f"sigma_gas error (component {ii})"]))

    # CHECK: stellar kinematics
    assert np.all(df.loc[~np.isnan(df["sigma_*"]), "sigma_*"] > 35)
    assert np.all(df.loc[~np.isnan(df["v_* error"]), "v_* error"] < 30)
    assert np.all(df.loc[~np.isnan(df["sigma_* error"]), "sigma_* error"] < df.loc[~np.isnan(df["sigma_* error"]), "sigma_*"] * 0.1 + 25)

    # CHECK: no "orphan" components
    assert df[np.isnan(df["HALPHA (component 0)"]) & ~np.isnan(df["HALPHA (component 1)"])].shape[0] == 0
    assert df[np.isnan(df["HALPHA (component 0)"]) & ~np.isnan(df["HALPHA (component 2)"])].shape[0] == 0
    assert df[np.isnan(df["HALPHA (component 1)"]) & ~np.isnan(df["HALPHA (component 2)"])].shape[0] == 0
    assert df[np.isnan(df["sigma_gas (component 0)"]) & ~np.isnan(df["sigma_gas (component 1)"])].shape[0] == 0
    assert df[np.isnan(df["sigma_gas (component 0)"]) & ~np.isnan(df["sigma_gas (component 2)"])].shape[0] == 0
    assert df[np.isnan(df["sigma_gas (component 1)"]) & ~np.isnan(df["sigma_gas (component 2)"])].shape[0] == 0
