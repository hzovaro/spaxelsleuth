import os
import pandas as pd
import numpy as np
from scipy import constants

from spaxelsleuth.loaddata import dqcut, linefns

from IPython.core.debugger import Tracer

"""
In this file:
- a function or class to load a dataframe containing the entire SAMI sample.

"""
s7_data_path = "/priv/meggs3/u5708159/S7/"
sami_datacube_path = "/priv/myrtle1/sami/sami_data/Final_SAMI_data/cube/sami/dr3/"

def load_s7_galaxies(eline_SNR_min, eline_list=["HALPHA", "HBETA", "NII6583", "OI6300", "OII3726", "OII3729", "OIII5007", "SII6716", "SII6731"],
                     sigma_gas_SNR_cut=True, sigma_gas_SNR_min=3,
                     vgrad_cut=False, correct_extinction=False,
                     stekin_cut=True):

    #######################################################################
    # INPUT CHECKING
    #######################################################################
    assert eline_SNR_min >= 0, "eline_SNR_min must be positive!"
    assert sigma_gas_SNR_min >= 0, "sigma_gas_SNR_min must be positive!"
    
    df_fname = f"s7_spaxels.hd5"
    assert os.path.exists(os.path.join(s7_data_path, df_fname)),\
        f"File {os.path.join(s7_data_path, df_fname)} does does not exist!"

    #######################################################################
    # LOAD THE DATAFRAME FROM MEMORY
    #######################################################################    
    df = pd.read_hdf(os.path.join(s7_data_path, df_fname))

    ######################################################################
    # DQ and S/N CUTS
    ######################################################################
    # For WiFes
    FWHM_inst_A = 0.9 # Based on skyline at 6950 A
    sigma_inst_A = FWHM_inst_A / ( 2 * np.sqrt( 2 * np.log(2) ))
    sigma_inst_km_s = sigma_inst_A * constants.c / 1e3 / 6562.8  # Defined at Halpha
    print("WARNING: in load_s7_galaxies: estimating instrumental dispersion from my own WiFeS observations - may not be consistent with assumed value in LZIFU!")

    df = dqcut.dqcut(df=df, ncomponents=3,
                  eline_SNR_min=eline_SNR_min, eline_list=eline_list,
                  sigma_gas_SNR_cut=sigma_gas_SNR_cut,
                  sigma_gas_SNR_min=sigma_gas_SNR_min,
                  sigma_inst_kms=sigma_inst_km_s,
                  vgrad_cut=vgrad_cut,
                  stekin_cut=stekin_cut)
    
    ######################################################################
    # EVALUATE ADDITIONAL COLUMNS - log quantites, etc.
    ######################################################################
    df = dqcut.compute_extra_columns(df, ncomponents=3)

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
    # CORRECT HALPHA FLUX AND EW FOR EXTINCTION
    # We use the same A_V for all 3 components because the Hbeta fluxes
    # for individual components aren't always reliable.
    ######################################################################
    if correct_extinction:
        cond_SF = df["BPT (total)"] == "SF"
        df_SF = df[cond_SF]
        df_not_SF = df[~cond_SF]

        print("WARNING: in load_s7_galaxies: correcting for extinction!")
        # Use the provided extinction correction map to correct Halpha 
        # fluxes & EWs
        for rr in rainge(df_SF.shape[0]):
            for eline in ["HALPHA", "HBETA", "NII6583", "OI6300", "OII3726", "OII3729", "OIII5007", "SII6716", "SII6731"]:
                A = extinction.fm07(wave=np.array([eline_lambdas_A[eline]]), 
                                    a_v=df.iloc[rr]["A_V (total)"])[0] if ~np.isnan(df.iloc[rr]["A_V (total)"]) else 0
                if A > 0:
                    corr_factor = 10**(0.4 * A)

                    # Total fluxes
                    df.iloc[rr][f"{eline} (total)"] *= corr_factor
                    df.iloc[rr][f"{eline} error (total)"] *= corr_factor
                    if eline == "HALPHA":
                        df.iloc[rr]["HALPHA (total)"] *= corr_factor
                        df.iloc[rr]["HALPHA error (total)"] *= corr_factor
                        df.iloc[rr]["HALPHA EW (total)"] *= corr_factor
                        df.iloc[rr]["HALPHA EW error (total)"] *= corr_factor                
                        df.iloc[rr]["log HALPHA EW (total)"] += np.log10(corr_factor)
                        # Re-compute log errors
                        df.iloc[rr]["log HALPHA EW error (lower) (total)"] = df.iloc[rr]["log HALPHA EW (total)"] - np.log10(df.iloc[rr]["HALPHA EW (total)"] - df.iloc[rr]["HALPHA EW error (total)"])
                        df.iloc[rr]["log HALPHA EW error (upper) (total)"] = np.log10(df.iloc[rr]["HALPHA EW (total)"] + df.iloc[rr]["HALPHA EW error (total)"]) -  df.iloc[rr]["log HALPHA EW (total)"]

                    # Individual components
                    for ii in range(3):
                        df.iloc[rr][f"{eline} (component {ii})"] *= corr_factor
                        df.iloc[rr][f"{eline} error (component {ii})"] *= corr_factor
                        if eline == "HALPHA":
                            df.iloc[rr][f"HALPHA (component {ii})"] *= corr_factor
                            df.iloc[rr][f"HALPHA error (component {ii})"] *= corr_factor
                            df.iloc[rr][f"HALPHA EW (component {ii})"] *= corr_factor
                            df.iloc[rr][f"HALPHA EW error (component {ii})"] *= corr_factor
                            df.iloc[rr][f"log HALPHA EW (component {ii})"] += np.log10(corr_factor)
                            # Re-compute log errors
                            df.iloc[rr][f"log HALPHA EW error (lower) (component {ii})"] = df.iloc[rr][f"log HALPHA EW (component {ii})"] - np.log10(df.iloc[rr][f"HALPHA EW (component {ii})"] - df.iloc[rr][f"HALPHA EW error (component {ii})"])
                            df.iloc[rr][f"log HALPHA EW error (upper) (component {ii})"] = np.log10(df.iloc[rr][f"HALPHA EW (component {ii})"] + df.iloc[rr][f"HALPHA EW error (component {ii})"]) -  df.iloc[rr][f"log HALPHA EW (component {ii})"]
    else:
        print("WARNING: in load_s7_galaxies: NOT correcting for extinction!")

    return df

###############################################################################
if __name__ == "__main__":

    import numpy as np
    import matplotlib.pyplot as plt
    plt.close("all")
    plt.ion()

    ncomponents = "recom"
    bin_type = "default"
    eline_SNR_min = 5
    vgrad_cut = False
    stekin_cut = True

    df = load_s7_galaxies(eline_SNR_min=eline_SNR_min, vgrad_cut=vgrad_cut)
    df_nocut = load_s7_galaxies(eline_SNR_min=0, vgrad_cut=False)

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
    for eline in ["HALPHA", "HBETA", "NII6583", "OI6300", "OII3726", "OII3729", "OIII5007", "SII6716", "SII6731"]:
        assert np.all(np.isnan(df.loc[df["Number of components"] == 0, f"{eline} (total)"]))
        assert np.all(np.isnan(df.loc[df["Number of components"] == 0, f"{eline} error (total)"]))

    # CHECK: all emission line columns in spaxels with 0 components are NaN
    for eline in ["HALPHA", "HBETA", "NII6583", "OI6300", "OII3726", "OII3729", "OIII5007", "SII6716", "SII6731"]:
        assert np.all(np.isnan(df.loc[df["Number of components"] == 0, f"{eline} (total)"]))
        assert np.all(np.isnan(df.loc[df["Number of components"] == 0, f"{eline} error (total)"]))
        if eline != "OII3726" and eline != "OII3729":
            for ii in range(3):
                assert np.all(np.isnan(df.loc[df["Number of components"] == 0, f"{eline} (component {ii})"]))
                assert np.all(np.isnan(df.loc[df["Number of components"] == 0, f"{eline} error (component {ii})"]))

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
    for eline in ["HALPHA", "HBETA", "NII6583", "OI6300", "OII3726", "OII3729", "OIII5007", "SII6716", "SII6731"]:
        assert np.all(np.isnan(df.loc[df[f"{eline} S/N (total)"] < eline_SNR_min, f"{eline} (total)"]))

    # CHECK: all Halpha components below S/N limit are NaN
    for ii in range(3):
        assert np.all(np.isnan(df.loc[df[f"HALPHA S/N (component {ii})"] < eline_SNR_min, f"HALPHA (component {ii})"]))
        assert np.all(np.isnan(df.loc[df[f"HALPHA S/N (component {ii})"] < eline_SNR_min, f"HALPHA error (component {ii})"]))
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
    if stekin_cut:
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
    
###############################################################################
# Make "collage" plots for the paper
###############################################################################
if __name__ == "__main__":

    # Imports
    import sys
    import os 
    import numpy as np
    import pandas as pd
    from astropy.visualization import hist

    from spaxelsleuth.loaddata.s7 import load_s7_galaxies
    from spaxelsleuth.loaddata.sami import load_sami_galaxies
    from spaxelsleuth.plotting.plot2dmap import plot2dmap
    from spaxelsleuth.plotting.sdssimg import plot_sdss_image
    from spaxelsleuth.plotting.plottools import plot_empty_BPT_diagram, plot_BPT_lines
    from spaxelsleuth.plotting.plottools import vmin_fn, vmax_fn, label_fn, cmap_fn
    from spaxelsleuth.plotting.plotgalaxies import plot2dscatter, plot2dhistcontours

    import matplotlib
    from matplotlib import rc, rcParams
    import matplotlib.pyplot as plt

    rc("text", usetex=False)
    rc("font",**{"family": "serif", "size": 14})
    rcParams["savefig.bbox"] = "tight"
    rcParams["savefig.format"] = "pdf"
    plt.ion()
    plt.close("all")

    ###########################################################################
    # Options
    ###########################################################################
    fig_path = "/priv/meggs3/u5708159/S7/figs/"
    savefigs = True
    eline_SNR_min = 3       # Minimum S/N of emission lines to accept

    ###########################################################################
    # Load the S7 sample
    ###########################################################################
    df = load_s7_galaxies(eline_SNR_min=eline_SNR_min,
                           sigma_gas_SNR_cut=True,
                          stekin_cut=True,
                           vgrad_cut=False)

    ###########################################################################
    # Load the SAMI sample
    ###########################################################################
    df_sami = load_sami_galaxies(ncomponents="recom",
                                 bin_type="default",
                                 eline_SNR_min=5, 
                                 vgrad_cut=False,
                                 correct_extinction=False,
                                 sigma_gas_SNR_cut=True)

    ###########################################################################
    # Set up the figure
    ###########################################################################
    markers = ["o", ">", "D"]
    l = 0.05
    b = 0.05
    dw = 0.1
    dh = 0.1
    w = (1 - 2 * l - dw) / 4
    h = (1 - 2 * b - dh) / 2

    gals = [g for g in df["catid"].unique() if g != "PKS0056-572" and g != "3C278"]
    for gal in gals[72:]:
        df_gal = df[df["catid"] == gal]
        
        ###############################################################################
        # Create the figure
        ###############################################################################
        fig = plt.figure(figsize=(15, 7))
        ax_im = fig.add_axes([l, b + 0.8 * h / 2, w, h * 1.6])
        bbox = ax_im.get_position()
        cax_im = fig.add_axes([bbox.x0, bbox.y0 + bbox.height, bbox.width, 0.025])
        axs_bpt = []
        axs_bpt.append(fig.add_axes([l + w + dw, b + h + dh, w, h]))
        axs_bpt.append(fig.add_axes([l + w + dw + w, b + h + dh, w, h]))
        axs_bpt.append(fig.add_axes([l + w + dw + 2 * w, b + h + dh, w, h]))
        cax_bpt = fig.add_axes(([l + w + dw + 3 * w, b + h + dh, 0.025, h]))
        axs_whav = []
        axs_whav.append(fig.add_axes([l + w + dw, b, w, h]))
        axs_whav.append(fig.add_axes([l + w + dw + w, b, w, h]))
        axs_whav.append(fig.add_axes([l + w + dw + 2 * w, b, w, h]))
        cax_whav = fig.add_axes(([l + w + dw + 3 * w, b, 0.025, h]))

        ###############################################################################
        # Plot the number of components fitted.
        ###############################################################################
        plot2dmap(df_gal=df_gal, bin_type="default", survey="s7",
                  PA_deg=df_gal["pa"].unique()[0],
                  col_z="Number of components", 
                  ax=ax_im, cax=cax_im, cax_orientation="horizontal", show_title=False)

        ###############################################################################
        # Plot BPT diagrams.
        ###############################################################################
        col_z = "Number of components"
        col_y = "log O3"
        for cc, col_x in enumerate(["log N2", "log S2", "log O1"]):
            # Plot full SAMI sample
            plot2dhistcontours(df=df_sami, 
                               col_x=f"{col_x} (total)",
                               col_y=f"{col_y} (total)", col_z="count", log_z=True,
                               alpha=0.5, cmap="gray_r",
                               ax=axs_bpt[cc], plot_colorbar=False)

            # Add BPT functions
            plot_BPT_lines(ax=axs_bpt[cc], col_x=col_x)    

            # Plot S7 measurements
            for ii in range(3):
                plot2dscatter(df=df_gal,
                              col_x=f"{col_x} (component {ii})",
                              col_y=f"{col_y} (component {ii})",
                              col_z=f"{col_z} (component {ii})" if f"{col_z} (component {ii})"  in df_gal else col_z,
                              marker=markers[ii], ax=axs_bpt[cc], cax=cax_bpt,
                              alpha=1.0 if ii == 1 else 0.2,
                              plot_colorbar=True if ii == 1 and cc == 2 else False)

        # Decorations
        [ax.grid() for ax in axs_bpt]
        [ax.set_ylabel("") for ax in axs_bpt[1:]]
        [ax.set_yticklabels([]) for ax in axs_bpt[1:]]
        [ax.set_xticks(ax.get_xticks()[:-1]) for ax in axs_bpt[:-1]]
        [ax.collections[0].set_rasterized(True) for ax in axs_bpt]

        ###############################################################################
        # Plot WHAN, WHAV and WHAV* diagrams.
        ###############################################################################
        col_x = "sigma_gas - sigma_*"
        col_y = "log HALPHA EW"
        col_z = "Number of components"

        # Plot LZIFU measurements
        for cc, col_x in enumerate(["log sigma_gas", "sigma_gas - sigma_*", "v_gas - v_*"]):
            # Plot full SAMI sample
            plot2dhistcontours(df=df_sami, col_x=f"{col_x}",
                               col_y=f"{col_y}",
                               col_z="count", log_z=True,
                               alpha=0.5, cmap="gray_r", ax=axs_whav[cc],
                               plot_colorbar=False)
            # Plot the S7 data
            for ii in range(3):
                plot2dscatter(df=df_gal,
                              col_x=f"{col_x} (component {ii})",
                              col_y=f"{col_y} (component {ii})",
                              col_z=f"{col_z} (component {ii})" if f"{col_z} (component {ii})"  in df_gal else col_z,
                              marker=markers[ii], ax=axs_whav[cc], cax=cax_whav,
                              alpha=1.0 if ii == 1 else 0.2,
                              plot_colorbar=True if ii == 1 else False)

        # Decorations
        [ax.grid() for ax in axs_whav]
        [ax.set_ylabel("") for ax in axs_whav[1:]]
        [ax.set_yticklabels([]) for ax in axs_whav[1:]]
        [ax.set_xticks(ax.get_xticks()[:-1]) for ax in axs_whav[:-1]]
        [ax.axvline(0, ls="--", color="k") for ax in axs_whav]

        Tracer()()

        # Save 
        if savefigs:
            fig.savefig(os.path.join(fig_path, f"{gal}_summary"))

        plt.close(fig)
