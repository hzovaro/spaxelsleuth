import matplotlib.pyplot as plt
from matplotlib import rcParams

from spaxelsleuth import load_user_config, configure_logger

# load_user_config("../integration_tests/test_config.json")
load_user_config("/home/u5708159/.spaxelsleuthconfig.json")
configure_logger(level="INFO")
from spaxelsleuth.loaddata.sami import load_sami_df
from spaxelsleuth.utils import metallicity
from spaxelsleuth.plotting.plotgalaxies import plot2dscatter, plot2dhistcontours

# Testing philosophy: run script in terminal to check "manual" things. Place this code in __main__.
# Place stuff we want to automatically test with pytest into functions & also call these in __main__.

if __name__ == "__main__":
    """Test metallicity calculations."""
    plt.ion()
    plt.close("all")
    rcParams["font.size"] = 8

    #TODO test this using a library of HII regions instead loaded at runtime
    #TODO over-plot grids showing the polynomial fits?
    #NOTE this stuff is important for Alex's work so it's important to get it right!!! 
    #TODO double-check ratios used in the diagnostics as well
    # Load the DataFrame.
    df = load_sami_df(
        ncomponents="recom",
        bin_type="default",
        eline_SNR_min=5,
        eline_ANR_min=3,
        correct_extinction=True, 
        debug=False
    )

    # Start off by testing _compute_logOH12 since this is the lowest-level function in the module.
    # For automated unit testing, should just check the basic stuff -
    #    - only SF-like rows have metallicity calculations
    #    - NaNs in the input --> NaNs in the output, etc.
    # Not really necessary to do these because we already test for this in the integration testing.

    # For "manual" inspection, plot the metallicity vs. the line ratio & compare against paper.

    #///////////////////////////////////////////////////////////////////////
    # Kewley 2019 diagnostics
    #TODO: these should be defined in metallicity.py somewhere, probably... 
    k19_ratio_colnames = {
        "N2Ha_K19": "log N2",
        "S2Ha_K19": "log S2",
        "N2S2_K19": "N2S2",
        "S23_K19": "S23",
        "O3N2_K19": "O3N2",
        "O2S2_K19": "O2S2",
        "O2Hb_K19": "O2",
        "N2O2_K19": "N2O2",
        "R23_K19": "R23",
        "O3O2_K19": "O3O2",
        "S32_K19": "S32",
    }

    k19_kwargs = {
        "cmap": "Spectral_r",
        "xmin": 7.5, "xmax": 9.3,
        "ymin": -2.2, "ymax": 4.0,
        "vmin": -3.98, "vmax": -1.98,
        "figsize": (7, 3),
    }
    k19_met_diags = metallicity.met_coeffs_K19.keys()
    for diag in k19_met_diags:
        try:
            fig = plot2dhistcontours(
                df,
                col_x=f"log(O/H) + 12 ({diag}/O3O2_K19) (total)",
                col_y=f"{k19_ratio_colnames[diag]} (total)",
                col_z=f"log(U) ({diag}/O3O2_K19) (total)",
                **k19_kwargs,
            )
            fig.get_axes()[0].axvline(metallicity.met_coeffs_K19[diag]["Zmin"], color="k")
            fig.get_axes()[0].axvline(metallicity.met_coeffs_K19[diag]["Zmax"], color="k")
            fig.suptitle(diag)
        except KeyError:
            print(f"Cannot plot diagnostic {diag} because it does not exist in the DataFrame!")

    # Ionisation parameter diagnostics
    k19_kwargs = {
    "cmap": "Spectral_r",
    "xmin": -4.0, "xmax": -1.8,
    "ymin": -2.0, "ymax": 2.1,
    "vmin": 7.63, "vmax": 9.23,
    "figsize": (7, 4),
    }
    k19_ion_diags = metallicity.ion_coeffs_K19.keys()
    for diag in ["O3O2_K19"]:
        try:
            fig = plot2dhistcontours(
                df,
                col_x=f"log(U) (N2Ha_K19/{diag}) (total)",
                col_y=f"{k19_ratio_colnames[diag]} (total)",
                col_z=f"log(O/H) + 12 (N2Ha_K19/{diag}) (total)",
                **k19_kwargs,
            )
            fig.get_axes()[0].axvline(metallicity.ion_coeffs_K19[diag]["Umin"], color="k")
            fig.get_axes()[0].axvline(metallicity.ion_coeffs_K19[diag]["Umax"], color="k")
            fig.suptitle(diag)
        except KeyError:
            print(f"Cannot plot diagnostic {diag} because it does not exist in the DataFrame!")

    #///////////////////////////////////////////////////////////////////////
    # N2Ha_PP04 (fig. 1)
    fig = plot2dhistcontours(
        df,
        col_x=f"log N2 (total)",
        col_y=f"log(O/H) + 12 (N2Ha_PP04) (total)",
        col_z="count", log_z=True,
        figsize=(7, 4),
        xmin=-2.7, xmax=0.0, 
        ymin=6.8, ymax=9.5,
    )
    fig.get_axes()[0].axvline(-2.5, color="k")
    fig.get_axes()[0].axvline(-0.3, color="k")

    # O3N2_PP04 (fig. 2)
    fig = plot2dhistcontours(
        df,
        col_x=f"O3N2 (total)",
        col_y=f"log(O/H) + 12 (O3N2_PP04) (total)",
        col_z="count", log_z=True,
        figsize=(7, 4),
        xmin=-2.7, xmax=0.0, 
        ymin=6.8, ymax=9.5,
    )
    fig.get_axes()[0].axvline(-1, color="k")
    fig.get_axes()[0].axvline(1.9, color="k")


    #///////////////////////////////////////////////////////////////////////
    #TODO: there are metallicity points beyond the validity limits of these calculations - why?
    # N2Ha_M13 (fig. 5)
    fig = plot2dhistcontours(
        df,
        col_x=f"log N2 (total)",
        col_y=f"log(O/H) + 12 (N2Ha_M13) (total)",
        col_z="count", log_z=True,
        figsize=(7, 4),
        xmin=-1.7, xmax=0.2, 
        ymin=7.5, ymax=9.0,
    )
    fig.get_axes()[0].axvline(-1.6, color="k")
    fig.get_axes()[0].axvline(-0.2, color="k")

    # O3N2_M13 (fig. 3)
    fig = plot2dhistcontours(
        df,
        col_x=f"O3N2 (total)",
        col_y=f"log(O/H) + 12 (O3N2_M13) (total)",
        col_z="count", log_z=True,
        figsize=(7, 4),
        xmin=-1.2, xmax=2.0, 
        ymin=7.4, ymax=9.0,
    )
    fig.get_axes()[0].axvline(-1.1, color="k")
    fig.get_axes()[0].axvline(1.7, color="k")


    #///////////////////////////////////////////////////////////////////////
    #TODO: there are pts w/ R23 > 1.0 with non-NaN metallicity measurements... these are MEANT to be NaN'd (see line 483) but clearly aren't!!!
    #TODO: double-check O3O2 definition!!
    # R23_KK04
    fig = plot2dhistcontours(
        df,
        col_x=f"log(O/H) + 12 (R23_KK04/O3O2_KK04) (total)",
        col_y=f"R23 (total)",
        col_z=f"log(U) (R23_KK04/O3O2_KK04) (total)",
        figsize=(7, 4),
        xmin=7.5, xmax=9.5, 
        ymin=-0.7, ymax=1.5,
        vmin=-4.0, vmax=-2.0,
        cmap="Spectral_r",
    )
    fig.suptitle("R23_KK04/O3O2_KK04") 

    #///////////////////////////////////////////////////////////////////////
    # R23_C17
    #TODO this actually hasn't been implemented in the code

    #///////////////////////////////////////////////////////////////////////
    # N2S2Ha_D16
    fig = plot2dhistcontours(
        df,
        col_x=f"log(O/H) + 12 (N2S2Ha_D16) (total)",
        col_y=f"Dopita+2016 (total)",
        col_z=f"count", log_z=True,
        figsize=(7, 4),
        xmin=7.4, xmax=9.5, 
        ymin=-1.1, ymax=0.6,
    )
    fig.suptitle("N2S2Ha_D16") 
    
    #///////////////////////////////////////////////////////////////////////
    # N2O2_KD02
    #TODO: fix cloud of points at N2O2 < -1.0
    fig = plot2dhistcontours(
        df,
        col_x=f"log(O/H) + 12 (N2O2_KD02) (total)",
        col_y=f"N2O2 (total)",
        col_z=f"count", log_z=True,
        figsize=(7, 4),
        xmin=7.5, xmax=9.5, 
        ymin=-2.0, ymax=1.5,
    )
    fig.suptitle("N2O2_KD02") 

    #///////////////////////////////////////////////////////////////////////
    # Rcal_PG16 (fig. 8c)
    #TODO: fix bug in implementation of upper/lower branches
    fig = plot2dhistcontours(
        df,
        col_x=f"R23 (total)",
        col_y=f"log(O/H) + 12 (Rcal_PG16) (total)",
        col_z=f"count", log_z=True,
        figsize=(7, 4),
        xmin=-1.0, xmax=2.0,
        ymin=6.5, ymax=9.5,
    )
    fig.suptitle("Rcal_PG16") 

    # Scal_PG16 (fig. 8a - comparison between R and S cals)
    fig = plot2dhistcontours(
        df,
        col_x=f"log(O/H) + 12 (Rcal_PG16) (total)",
        col_y=f"log(O/H) + 12 (Scal_PG16) (total)",
        col_z=f"count", log_z=True,
        figsize=(7, 4),
        xmin=7.0, xmax=9.0,
        ymin=7.0, ymax=9.0,
    )
    fig.suptitle("S/Rcal_PG16") 

    #///////////////////////////////////////////////////////////////////////
    # ONS_P10
    # ON_P10


    #///////////////////////////////////////////////////////////////////////
    # Also, check repeatability! Run calculations twice & plot retrieved values against one another.
    # This is important...
