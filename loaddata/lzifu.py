import os
import pandas as pd
import numpy as np
from astropy.io import fits
from tqdm import tqdm

from spaxelsleuth.loaddata import dqcut, linefns

sami_data_path = "/priv/meggs3/u5708159/SAMI/sami_dr3/"
sami_datacube_path = "/priv/myrtle1/sami/sami_data/Final_SAMI_data/cube/sami/dr3/"
lzifu_data_path = "/priv/meggs3/u5708159/LZIFU/products/"

def load_lzifu_galaxy(gal, ncomponents, bin_type,
                      eline_SNR_min,
                      eline_list=["HALPHA", "HBETA", "OIII5007", "OI6300", "NII6583", "SII6716", "SII6731"],
                      sigma_gas_SNR_cut=True, sigma_gas_SNR_min=3, stekin_cut=True,
                      vgrad_cut=False, correct_extinction=False):

    #######################################################################
    # INPUT CHECKING
    #######################################################################
    assert ncomponents == "recom", "ncomponents must be 'recom'!!"
    assert bin_type == "default", "bin_type must be 'default' for now!!"
    assert eline_SNR_min >= 0, "eline_SNR_min must be positive!"
    assert sigma_gas_SNR_min >= 0, "sigma_gas_SNR_min must be positive!"

    df_fname = f"lzifu_{gal}.hd5"
    assert os.path.exists(os.path.join(sami_data_path, df_fname)),\
        f"File {os.path.join(sami_data_path, df_fname)} does does not exist!"
    
    #######################################################################
    # LOAD THE DATAFRAME FROM MEMORY
    #######################################################################
    df = pd.read_hdf(os.path.join(sami_data_path, df_fname))

    ######################################################################
    # DQ and S/N CUTS
    ######################################################################
    df = dqcut.dqcut(df=df, ncomponents=3,
                  eline_SNR_min=eline_SNR_min, eline_list=eline_list,
                  sigma_gas_SNR_cut=sigma_gas_SNR_cut, sigma_gas_SNR_min=sigma_gas_SNR_min, 
                  sigma_inst_kms=29.6,
                  vgrad_cut=vgrad_cut,
                  stekin_cut=stekin_cut)
    
    ######################################################################
    # EVALUATE LINE RATIOS & SPECTRAL CLASSIFICATIONS
    ######################################################################
    df = linefns.ratio_fn(df, s=f" (total)")
    df = linefns.bpt_fn(df, s=f" (total)")
    df = linefns.law2021_fn(df, s=f" (total)")
    for ii in range(3):
        df = linefns.ratio_fn(df, s=f" (component {ii})")
        df = linefns.bpt_fn(df, s=f" (component {ii})")
        df = linefns.law2021_fn(df, s=f" (component {ii})")

    ######################################################################
    # EVALUATE ADDITIONAL COLUMNS - log quantites, etc.
    ######################################################################
    df = dqcut.compute_extra_columns(df, ncomponents=3 if ncomponents=="recom" else 1)

    ######################################################################
    # CORRECT HALPHA FLUX AND EW FOR EXTINCTION
    # Note that we only have enough information from the provided extinction
    # maps to correct the Halpha line. We therefore apply this correction 
    # AFTER we compute line ratios, etc.
    ######################################################################
    if correct_extinction:
        print("WARNING: in load_lzifu_galaxy: correcting Halpha and HALPHA EW for extinction!")
        # Use the provided extinction correction map to correct Halpha 
        # fluxes & EWs
        df["HALPHA (total)"] *= df["HALPHA extinction correction"]
        df["HALPHA error (total)"] *= df["HALPHA extinction correction"]
        df["HALPHA EW (total)"] *= df["HALPHA extinction correction"]
        df["HALPHA EW error (total)"] *= df["HALPHA extinction correction"]
        df["log HALPHA EW (total)"] += np.log10(df["HALPHA extinction correction"])
        # Re-compute log errors
        df["log HALPHA EW error (lower) (total)"] = df["log HALPHA EW (total)"] - np.log10(df["HALPHA EW (total)"] - df["HALPHA EW error (total)"])
        df["log HALPHA EW error (upper) (total)"] = np.log10(df["HALPHA EW (total)"] + df["HALPHA EW error (total)"]) -  df["log HALPHA EW (total)"]

        for component in range(3 if ncomponents == "recom" else 1):
            df[f"HALPHA (component {component})"] *= df["HALPHA extinction correction"]
            df[f"HALPHA error (component {component})"] *= df["HALPHA extinction correction"]
            df[f"HALPHA EW (component {component})"] *= df["HALPHA extinction correction"]
            df[f"HALPHA EW error (component {component})"] *= df["HALPHA extinction correction"]
            df[f"log HALPHA EW (component {component})"] += np.log10(df["HALPHA extinction correction"])
            # Re-compute log errors
            df[f"log HALPHA EW error (lower) (component {component})"] = df[f"log HALPHA EW (component {component})"] - np.log10(df[f"HALPHA EW (component {component})"] - df[f"HALPHA EW error (component {component})"])
            df[f"log HALPHA EW error (upper) (component {component})"] = np.log10(df[f"HALPHA EW (component {component})"] + df[f"HALPHA EW error (component {component})"]) -  df[f"log HALPHA EW (component {component})"]
    else:
        print("WARNING: in load_lzifu_galaxy: NOT correcting Halpha and HALPHA EW for extinction!")
    return df

###############################################################################
def merge_datacubes(gal=None, plotit=False):
    """
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

    """
    if gal is None:
        gals = [int(f.split("_merge_comp.fits")[0]) for f in os.listdir(lzifu_data_path) if f.endswith("merge_comp.fits") and not f.startswith("._")]
    else:
        if type(gal) == list:
            gals = gal
        else:
            gals = [gal]
        for gal in gals:
            assert type(gal) == int, "gal must be an integer!"
            fname = os.path.join(lzifu_data_path, f"{gal}_merge_comp.fits")
            assert os.path.exists(fname), f"File {fname} not found!"

    for gal in tqdm(gals):
        ###############################################################################
        # Step 1: Create a map from the SAMI data showing how many components there 
        # should be in each spaxel.
        ###############################################################################
        # Open the SAMI data.
        hdulist_sami = fits.open(os.path.join(sami_data_path, f"ifs/{gal}/{gal}_A_Halpha_default_recom-comp.fits"))
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
        hdulist_merged = fits.open(os.path.join(lzifu_data_path, f"{gal}_merge_comp.fits"))
        hdulist_1 = fits.open(os.path.join(lzifu_data_path, f"{gal}_1_comp.fits"))
        hdulist_2 = fits.open(os.path.join(lzifu_data_path, f"{gal}_2_comp.fits"))
        hdulist_3 = fits.open(os.path.join(lzifu_data_path, f"{gal}_3_comp.fits"))

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
        hdulist_merged.writeto(os.path.join(lzifu_data_path, f"{gal}_merge_lzcomp.fits"), overwrite=True, output_verify="ignore")

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
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        import numpy as np
        import matplotlib.pyplot as plt
        plt.close("all")
        plt.ion()

        gal = 209807
        eline_SNR_min = 5
        vgrad_cut = False

        df = load_lzifu_galaxy(gal,
                                ncomponents="recom", bin_type="default",
                                eline_SNR_min=eline_SNR_min, vgrad_cut=vgrad_cut)
        df_nocut = load_lzifu_galaxy(gal,
                                ncomponents="recom", bin_type="default",
                                eline_SNR_min=0, vgrad_cut=False)

        # CHECK: proper S/N cuts have been made
        for eline in ["HALPHA", "HBETA", "OIII5007", "OI6300", "NII6583", "SII6716", "SII6731"]:
            plt.figure()
            plt.scatter(x=df_nocut[f"{eline} (total)"], y=df_nocut[f"{eline} S/N (total)"], c="grey", alpha=0.1, s=5)
            plt.scatter(x=df[f"{eline} (total)"], y=df[f"{eline} S/N (total)"], c="k", s=5)
            plt.ylabel(f"{eline} (total) S/N")
            plt.axhline(eline_SNR_min, color="r")
            plt.xlabel(f"{eline} (total) flux")
        for ii in range(3):
            plt.figure()
            plt.scatter(x=df_nocut[f"HALPHA (component {ii})"], y=df_nocut[f"HALPHA S/N (component {ii})"], c="grey", alpha=0.1, s=5)
            plt.scatter(x=df[f"HALPHA (component {ii})"], y=df[f"HALPHA S/N (component {ii})"], c="k", s=5)
            plt.ylabel(f"HALPHA (component {ii}) S/N")
            plt.axhline(eline_SNR_min, color="r")
            plt.xlabel(f"HALPHA (component {ii}) flux")
        for ii in range(3):
            plt.figure()
            plt.scatter(x=df_nocut[f"HALPHA EW (component {ii})"], y=df_nocut[f"HALPHA S/N (component {ii})"], c="grey", alpha=0.1, s=5)
            plt.scatter(x=df[f"HALPHA EW (component {ii})"], y=df[f"HALPHA S/N (component {ii})"], c="k", s=5)
            plt.ylabel(f"HALPHA (component {ii}) S/N")
            plt.axhline(eline_SNR_min, color="r")
            plt.xlabel(f"HALPHA EW (component {ii})")

        # CHECK: gas kinematics
        for ii in range(3):
            plt.figure()
            plt.scatter(x=df_nocut[f"sigma_obs S/N (component {ii})"], y=df_nocut[f"sigma_obs target S/N (component {ii})"], c="grey", alpha=0.1, s=5)
            plt.scatter(x=df[f"sigma_obs S/N (component {ii})"], y=df[f"sigma_obs target S/N (component {ii})"], c="k", s=5)
            plt.ylabel(f"sigma_obs target S/N (component {ii})")
            plt.axhline(3, color="b")
            plt.xlabel(f"sigma_obs S/N (component {ii})")

        # CHECK: "Number of components" is 0, 1, 2 or 3
        assert np.all(df["Number of components"].unique() == [0, 1, 2, 3])

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
        
    else:
        # Imports
        import sys
        import os 
        import numpy as np
        import pandas as pd
        from astropy.visualization import hist

        from spaxelsleuth.loaddata.lzifu import load_lzifu_galaxy
        from spaxelsleuth.loaddata.sami import load_sami_galaxies
        from spaxelsleuth.plotting.plot2dmap import plot2dmap
        from spaxelsleuth.plotting.sdssimg import plot_sdss_image
        from spaxelsleuth.plotting.plottools import plot_empty_BPT_diagram, plot_BPT_lines
        from spaxelsleuth.plotting.plottools import vmin_fn, vmax_fn, label_fn, cmap_fn, component_colours
        from spaxelsleuth.plotting.plotgalaxies import plot2dscatter, plot2dhistcontours

        import matplotlib
        from matplotlib import rc, rcParams
        from matplotlib.backends.backend_pdf import PdfPages
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D

        from IPython.core.debugger import Tracer

        rc("text", usetex=False)
        rc("font",**{"family": "serif", "size": 12})
        rcParams["savefig.bbox"] = "tight"
        rcParams["savefig.format"] = "pdf"
        plt.ion()
        plt.close("all")

        ###########################################################################
        # Options
        ###########################################################################
        fig_path = "/priv/meggs3/u5708159/SAMI/figs/individual_plots/"
        savefigs = True
        bin_type = "default"    # Options: "default" or "adaptive" for Voronoi binning
        ncomponents = "recom"   # Options: "1" or "recom"
        eline_SNR_min = 5       # Minimum S/N of emission lines to accept

        ###########################################################################
        # Load the SAMI sample
        ###########################################################################
        df_sami = load_sami_galaxies(ncomponents="recom",
                                     bin_type="default",
                                     eline_SNR_min=eline_SNR_min, 
                                     vgrad_cut=False,
                                     correct_extinction=False,
                                     sigma_gas_SNR_cut=True)

        ###########################################################################
        # Make summary plots
        ###########################################################################
        if len(sys.argv) > 1:
            gals = sys.argv[1:]
            for gal in gals:
                assert gal.isdigit(), "each gal given must be an integer!"
                assert os.path.exists(os.path.join(lzifu_data_path, f"{gal}_merge_lzcomp.fits"))
        else:
            gals = [int(f.split("_merge_lzcomp.fits")[0]) for f in os.listdir(lzifu_data_path) if f.endswith("merge_lzcomp.fits") and not f.startswith("._")]

        for gal in tqdm(gals):

            # Load the DataFrame
            df_gal = load_lzifu_galaxy(gal=gal, bin_type=bin_type, ncomponents=ncomponents,
                                   eline_SNR_min=eline_SNR_min,
                                   sigma_gas_SNR_cut=True,
                                   vgrad_cut=False,
                                   stekin_cut=True)
            df_gal.loc[df_gal["Number of components"] == 0, "Number of components"] = np.nan

            ###########################################################################
            # Collage figure 1: coloured by number of components
            ###########################################################################
            markers = ["o", ">", "D"]
            l = 0.05
            b = 0.05
            dw = 0.1
            dh = 0.1
            w = (1 - 2 * l - dw) / 4
            h = (1 - 2 * b - dh) / 2

            ###########################################################################
            # Collage figure 2: coloured by r/R_e
            ###########################################################################
            for col_z in ["Number of components", "r/R_e"]:

                # Create the figure
                fig_collage = plt.figure(figsize=(15, 7))
                ax_sdss = fig_collage.add_axes([l, b, w, h])
                ax_im = fig_collage.add_axes([l, b + h + dh, w, h])
                bbox = ax_im.get_position()
                cax_im = fig_collage.add_axes([bbox.x0 + bbox.width * 0.035, bbox.y0 + bbox.height, bbox.width * 0.93, 0.025])
                axs_bpt = []
                axs_bpt.append(fig_collage.add_axes([l + w + dw, b + h + dh, w, h]))
                axs_bpt.append(fig_collage.add_axes([l + w + dw + w, b + h + dh, w, h]))
                axs_bpt.append(fig_collage.add_axes([l + w + dw + 2 * w, b + h + dh, w, h]))
                # cax_bpt = fig_collage.add_axes(([l + w + dw + 3 * w, b + h + dh, 0.025, h])) if col_z != "Number of components" else None
                axs_whav = []
                axs_whav.append(fig_collage.add_axes([l + w + dw, b, w, h]))
                axs_whav.append(fig_collage.add_axes([l + w + dw + w, b, w, h]))
                axs_whav.append(fig_collage.add_axes([l + w + dw + 2 * w, b, w, h]))
                # cax_whav = fig_collage.add_axes(([l + w + dw + 3 * w, b, 0.025, h])) if col_z != "Number of components" else None

                # SDSS image
                plot_sdss_image(df_gal, ax=ax_sdss)

                # Plot the number of components fitted.
                plot2dmap(df_gal=df_gal, bin_type="default", survey="sami",
                          PA_deg=0,
                          col_z=col_z, 
                          ax=ax_im, cax=cax_im, cax_orientation="horizontal", show_title=False)

                # Plot BPT diagram
                col_y = "log O3"
                axs_bpt[0].text(s=gal, x=0.1, y=0.9, transform=axs_bpt[0].transAxes)
                for cc, col_x in enumerate(["log N2", "log S2", "log O1"]):
                    # Plot full SAMI sample
                    plot2dhistcontours(df=df_sami, 
                                       col_x=f"{col_x} (total)",
                                       col_y=f"{col_y} (total)", col_z="count", log_z=True,
                                       alpha=0.5, cmap="gray_r",
                                       ax=axs_bpt[cc], plot_colorbar=False)

                    # Add BPT functions
                    plot_BPT_lines(ax=axs_bpt[cc], col_x=col_x)    

                    # Plot LZIFU measurements
                    for ii in range(3):
                        plot2dscatter(df=df_gal,
                                      col_x=f"{col_x} (component {ii})",
                                      col_y=f"{col_y} (component {ii})",
                                      col_z=None if col_z == "Number of components" else col_z,
                                      marker=markers[ii], ax=axs_bpt[cc], 
                                      cax=None,
                                      markersize=20, 
                                      markerfacecolour=component_colours[ii] if col_z == "Number of components" else None, 
                                      edgecolors="black",
                                      plot_colorbar=False)

                # Decorations
                [ax.grid() for ax in axs_bpt]
                [ax.set_ylabel("") for ax in axs_bpt[1:]]
                [ax.set_yticklabels([]) for ax in axs_bpt[1:]]
                [ax.set_xticks(ax.get_xticks()[:-1]) for ax in axs_bpt[:-1]]
                [ax.collections[0].set_rasterized(True) for ax in axs_bpt]

                ###########################################################################
                # Plot WHAN, WHAV and WHAV* diagrams.
                ###########################################################################
                col_y = "log HALPHA EW"
                # Plot LZIFU measurements
                for cc, col_x in enumerate(["log N2", "sigma_gas - sigma_*", "v_gas - v_*"]):
                    # Plot full SAMI sample
                    plot2dhistcontours(df=df_sami, col_x=f"{col_x} (total)" if col_x == "log N2" else f"{col_x}",
                                       col_y=f"{col_y} (total)" if col_x == "log N2" else f"{col_y}",
                                       col_z="count", log_z=True,
                                       alpha=0.5, cmap="gray_r", ax=axs_whav[cc],
                                       plot_colorbar=False)
                    # Plot the S7 data
                    for ii in range(3):
                        plot2dscatter(df=df_gal,
                                      col_x=f"{col_x} (component {ii})",
                                      col_y=f"{col_y} (component {ii})",
                                      col_z=None if col_z == "Number of components" else col_z,
                                      marker=markers[ii], ax=axs_whav[cc], 
                                      cax=None,
                                      markersize=20, 
                                      markerfacecolour=component_colours[ii] if col_z == "Number of components" else None, 
                                      edgecolors="black",
                                      plot_colorbar=False)

                # Decorations
                [ax.grid() for ax in axs_whav]
                [ax.set_ylabel("") for ax in axs_whav[1:]]
                [ax.set_yticklabels([]) for ax in axs_whav[1:]]
                [ax.set_xticks(ax.get_xticks()[:-1]) for ax in axs_whav[:-1]]
                [ax.axvline(0, ls="--", color="k") for ax in axs_whav[1:]]
                
                # Legend
                legend_elements = [Line2D([0], [0], marker=markers[ii], 
                                          color="none", markeredgecolor="black",
                                          label=f"Component {ii}",
                                          markerfacecolor=component_colours[ii], markersize=5) for ii in range(3)]
                axs_bpt[-1].legend(handles=legend_elements, fontsize="x-small", loc="upper right")

                if savefigs:
                    fname = "ncomponents" if col_z == "Number of components" else "radius"
                    fig_collage.savefig(os.path.join(fig_path, f"{gal}_LZIFU_summary_{fname}.pdf"), format="pdf", bbox_inches="tight")

            ###########################################################################
            # Scatter plot: line ratios vs. velocity dispersion
            ###########################################################################
            col_y_list = ["log N2", "log S2", "log O1", "log O3"]
            fig_line_ratios, axs = plt.subplots(nrows=len(col_y_list), ncols=2, figsize=(7, 6 * len(col_y_list)), sharex="col", sharey="row")
            fig_line_ratios.subplots_adjust(wspace=0, hspace=0)
            bbox = axs[0][0].get_position()
            cax = fig_line_ratios.add_axes([bbox.x0, bbox.y0 + bbox.height, 2 * bbox.width, 0.03])

            # log N2, S2, O1 vs. velocity dispersion
            axs[0][0].text(s=f"{gal} ($i = {df_gal['Inclination i (degrees)'].unique()[0]:.2f}^\circ$)", x=0.1, y=0.9, transform=axs[0][0].transAxes)
            for rr, col_y in enumerate(col_y_list):
                for ii in range(3):
                    plot2dscatter(df=df_gal, 
                                  col_x=f"sigma_gas (component {ii})",
                                  col_y=f"{col_y} (component {ii})",
                                  col_z=f"HALPHA EW (component {ii})",
                                  marker=markers[ii], markerfacecolour=component_colours[ii], edgecolors="black",
                                  ax=axs[rr][0], plot_colorbar=False)
                    plot2dscatter(df=df_gal, 
                                  col_x=f"sigma_gas - sigma_* (component {ii})",
                                  col_y=f"{col_y} (component {ii})",
                                  col_z=f"HALPHA EW (component {ii})",
                                  marker=markers[ii], markerfacecolour=component_colours[ii], edgecolors="black",
                                  ax=axs[rr][1], plot_colorbar=True if ii == 2 else False, cax_orientation="horizontal",
                                  cax=cax) 
                    axs[rr][1].set_ylabel("")
                    # axs[rr][1].set_yticklabels([])
            [ax.grid() for ax in axs.flat]

            ###########################################################################
            # Maps showing HALPHA EW for each component
            ###########################################################################
            col_z_list = ["HALPHA EW", "sigma_gas", "sigma_gas - sigma_*"]
            fig_maps, axs = plt.subplots(nrows=len(col_z_list), ncols=3, figsize=(10, 10))
            fig_maps.subplots_adjust(wspace=0)

            for cc, col_z in enumerate(col_z_list):
                bbox = axs[cc][-1].get_position()
                cax = fig_maps.add_axes([bbox.x0 + bbox.width, bbox.y0, 0.03, bbox.height])

                for ii in range(3):
                    plot2dmap(df_gal=df_gal, bin_type="default", survey="sami", 
                              col_z=f"{col_z} (component {ii})",
                              vmax=np.nanmax(df_gal["HALPHA (total)"]) if col_z == "HALPHA" else None, 
                              ax=axs[cc][ii], show_title=False, plot_colorbar=True if ii == 2 else False, cax=cax)
                    
                    # Decorations
                    if ii > 0:
                        # Turn off axis labels
                        lat = plt.gca().coords[1]
                        lat.set_ticks_visible(False)
                        lat.set_ticklabel_visible(False)
                    if cc < len(col_z_list) - 1:
                        lon = plt.gca().coords[0]
                        lon.set_ticks_visible(False)
                        lon.set_ticklabel_visible(False)


            # Save 
            if savefigs:
                fig_maps.savefig(os.path.join(fig_path, f"{gal}_LZIFU_maps.pdf"), format="pdf", bbox_inches="tight")
                fig_line_ratios.savefig(os.path.join(fig_path, f"{gal}_LZIFU_line_ratios_vs_sigma_gas.pdf"), format="pdf", bbox_inches="tight")

            plt.close(fig_collage)
            plt.close(fig_line_ratios)
            plt.close(fig_maps)

