import matplotlib.pyplot as plt
from matplotlib import rcParams
import sys

from spaxelsleuth import load_user_config, configure_logger

load_user_config("/home/u5708159/.spaxelsleuthconfig.json")
configure_logger(level="ERROR")
from spaxelsleuth.loaddata.sami import load_sami_df
from spaxelsleuth.utils import metallicity
from spaxelsleuth.plotting.plotgalaxies import plot2dhistcontours

# Testing philosophy: run script in terminal to check "manual" things. Place this code in __main__.
# Place stuff we want to automatically test with pytest into functions & also call these in __main__.


def test_metallicity():
    """Placeholder test so that pytest doesn't return an error code"""
    assert True


if __name__ == "__main__":
    """Test metallicity calculations."""
    plt.ion()
    plt.close("all")
    rcParams["font.size"] = 8

    # Load the DataFrame.
    df = load_sami_df(
        ncomponents="recom",
        bin_type="default",
        eline_SNR_min=5,
        eline_ANR_min=3,
        correct_extinction=True,
        debug=False,
    )

    # Subset of star-forming galaxies to speed up execution
    gb = df.loc[df["BPT (total)"] == "SF"].groupby("ID")
    counts = gb["x, y (pixels)"].count().sort_values(ascending=False)
    gals_SF = counts.index.values[:10]
    df_SF = df.loc[df["ID"].isin(gals_SF)]
    df_SF_updated = df_SF.copy()
    cols_to_drop = [c for c in df_SF_updated if "log(O/H)" in c or "log(U)" in c]
    df_SF_updated = df_SF_updated.drop(columns=cols_to_drop)

    # Re-calculate metallicities and check
    niters = 100
    diagnostics = [
        "N2O2_KD02",
        "R23_KK04",
        "N2Ha_PP04",
        "O3N2_PP04",
        "N2Ha_M13",
        "O3N2_M13",
        "Rcal_PG16",
        "Scal_PG16",
        "N2S2Ha_D16",
        "N2Ha_K19",
        "S2Ha_K19",
        "N2S2_K19",
        "O3N2_K19",
        "O2S2_K19",
        "O2Hb_K19",
        "N2O2_K19",
        "R23_K19",
    ]
    for diagnostic in diagnostics:
        if diagnostic.endswith("K19"):
            df_SF_updated = metallicity.calculate_metallicity(
                met_diagnostic=diagnostic,
                compute_logU=True,
                ion_diagnostic="O3O2_K19",
                compute_errors=True,
                niters=niters,
                df=df_SF_updated,
                s=" (total)",
            )
        elif diagnostic.endswith("KK04"):
            df_SF_updated = metallicity.calculate_metallicity(
                met_diagnostic=diagnostic,
                compute_logU=True,
                ion_diagnostic="O3O2_KK04",
                compute_errors=True,
                niters=niters,
                df=df_SF_updated,
                s=" (total)",
            )
        else:
            df_SF_updated = metallicity.calculate_metallicity(
                met_diagnostic=diagnostic,
                compute_errors=True,
                niters=niters,
                df=df_SF_updated,
                s=" (total)",
            )

    # Reset the index to aid comparison
    df_SF_updated = df_SF_updated.sort_index()
    df_SF = df_SF.sort_index()

    # ///////////////////////////////////////////////////////////////////////
    def compare_before_after(diag, ax):
        """Plot metallicity measurements before and after to compare changes"""
        ax.scatter(df_SF[f"log(O/H) + 12 ({diag}) (total)"], df_SF_updated[f"log(O/H) + 12 ({diag}) (total)"])
        ax.plot([ax.get_xlim()[0], ax.get_xlim()[1]], [ax.get_xlim()[0], ax.get_xlim()[1]], "k")
 

    # ///////////////////////////////////////////////////////////////////////
    # Rcal/Scal_PG16
    fig, axs = plt.subplots(ncols=3, figsize=(14, 4))
    fig.subplots_adjust(wspace=0.5)
    plot2dhistcontours(
        df_SF,
        col_x=f"R23 (total)",
        col_y=f"log(O/H) + 12 (Rcal_PG16) (total)",
        col_z=f"count",
        log_z=True,
        figsize=(7, 4),
        xmin=-1.0,
        xmax=2.0,
        ymin=6.5,
        ymax=9.5,
        ax=axs[0],
    )
    plot2dhistcontours(
        df_SF_updated,
        col_x=f"R23 (total)",
        col_y=f"log(O/H) + 12 (Rcal_PG16) (total)",
        col_z=f"count",
        log_z=True,
        figsize=(7, 4),
        xmin=-1.0,
        xmax=2.0,
        ymin=6.5,
        ymax=9.5,
        ax=axs[1],
    )
    fig.suptitle("Rcal_PG16")
    compare_before_after("Rcal_PG16", ax=axs[2])

   # Scal_PG16 (fig. 8a - comparison between R and S cals)
    fig, axs = plt.subplots(ncols=3, figsize=(14, 4))
    fig.subplots_adjust(wspace=0.5)
    plot2dhistcontours(
        df_SF,
        col_x=f"log(O/H) + 12 (Rcal_PG16) (total)",
        col_y=f"log(O/H) + 12 (Scal_PG16) (total)",
        col_z=f"count",
        log_z=True,
        figsize=(7, 4),
        xmin=7.0,
        xmax=9.0,
        ymin=7.0,
        ymax=9.0,
        ax=axs[0],
    )
    plot2dhistcontours(
        df_SF_updated,
        col_x=f"log(O/H) + 12 (Rcal_PG16) (total)",
        col_y=f"log(O/H) + 12 (Scal_PG16) (total)",
        col_z=f"count",
        log_z=True,
        figsize=(7, 4),
        xmin=7.0,
        xmax=9.0,
        ymin=7.0,
        ymax=9.0,
        ax=axs[1],
    )
    fig.suptitle("S/Rcal_PG16")
    compare_before_after("Scal_PG16", ax=axs[2])

    # ///////////////////////////////////////////////////////////////////////
    # N2Ha_M13
    fig, axs = plt.subplots(ncols=3, figsize=(14, 4))
    fig.subplots_adjust(wspace=0.5)
    plot2dhistcontours(
        df_SF,
        col_x=f"log N2 (total)",
        col_y=f"log(O/H) + 12 (N2Ha_M13) (total)",
        col_z="count",
        log_z=True,
        figsize=(7, 4),
        xmin=-1.7,
        xmax=0.2,
        ymin=7.5,
        ymax=9.0,
        ax=axs[0],
    )
    axs[0].axvline(-1.6, color="k")
    axs[0].axvline(-0.2, color="k")
    plot2dhistcontours(
        df_SF_updated,
        col_x=f"log N2 (total)",
        col_y=f"log(O/H) + 12 (N2Ha_M13) (total)",
        col_z="count",
        log_z=True,
        figsize=(7, 4),
        xmin=-1.7,
        xmax=0.2,
        ymin=7.5,
        ymax=9.0,
        ax=axs[1],
    )
    axs[1].axvline(-1.6, color="k")
    axs[1].axvline(-0.2, color="k")
    fig.suptitle("N2Ha_M13")
    compare_before_after("N2Ha_M13", ax=axs[2])

    # ///////////////////////////////////////////////////////////////////////
    # N2O2_KD02
    fig, axs = plt.subplots(ncols=3, figsize=(14, 4))
    fig.subplots_adjust(wspace=0.5)
    plot2dhistcontours(
        df_SF,
        col_x=f"log(O/H) + 12 (N2O2_KD02) (total)",
        col_y=f"N2O2 (total)",
        col_z=f"count",
        log_z=True,
        figsize=(7, 4),
        xmin=7.5,
        xmax=9.5,
        ymin=-2.0,
        ymax=1.5,
        ax=axs[0],
    )
    axs[0].axvline(-0.2, color="k")
    plot2dhistcontours(
        df_SF_updated,
        col_x=f"log(O/H) + 12 (N2O2_KD02) (total)",
        col_y=f"N2O2 (total)",
        col_z=f"count",
        log_z=True,
        figsize=(7, 4),
        xmin=7.5,
        xmax=9.5,
        ymin=-2.0,
        ymax=1.5,
        ax=axs[1],
    )
    axs[1].axvline(-0.2, color="k")
    fig.suptitle("N2O2_KD02")
    compare_before_after("N2O2_KD02", ax=axs[2])

    # ///////////////////////////////////////////////////////////////////////
    # Repeat with R23
    fig, axs = plt.subplots(ncols=3, figsize=(14, 4))
    fig.subplots_adjust(wspace=0.5)
    plot2dhistcontours(
        df_SF,
        col_x=f"log(O/H) + 12 (R23_KK04/O3O2_KK04) (total)",
        col_y=f"R23 (total)",
        col_z=f"log(U) (R23_KK04/O3O2_KK04) (total)",
        figsize=(7, 4),
        xmin=7.5,
        xmax=9.5,
        ymin=-0.7,
        ymax=1.5,
        vmin=-4.0,
        vmax=-2.0,
        cmap="Spectral_r",
        ax=axs[0],
    )
    plot2dhistcontours(
        df_SF_updated,
        col_x=f"log(O/H) + 12 (R23_KK04/O3O2_KK04) (total)",
        col_y=f"R23 (total)",
        col_z=f"log(U) (R23_KK04/O3O2_KK04) (total)",
        figsize=(7, 4),
        xmin=7.5,
        xmax=9.5,
        ymin=-0.7,
        ymax=1.5,
        vmin=-4.0,
        vmax=-2.0,
        cmap="Spectral_r",
        ax=axs[1],
    )
    fig.suptitle("R23_KK04/O3O2_KK04")
    compare_before_after("R23_KK04/O3O2_KK04", ax=axs[2])

    # ///////////////////////////////////////////////////////////////////////
    # N2Ha_PP04 (fig. 1)
    fig, axs = plt.subplots(ncols=3, figsize=(14, 4))
    fig.subplots_adjust(wspace=0.5)
    plot2dhistcontours(
        df_SF,
        col_x=f"log N2 (total)",
        col_y=f"log(O/H) + 12 (N2Ha_PP04) (total)",
        col_z="count",
        log_z=True,
        figsize=(7, 4),
        xmin=-2.7,
        xmax=0.0,
        ymin=6.8,
        ymax=9.5,
        ax=axs[0],
    )
    axs[0].axvline(-2.5, color="k")
    axs[0].axvline(-0.3, color="k")
    plot2dhistcontours(
        df_SF_updated,
        col_x=f"log N2 (total)",
        col_y=f"log(O/H) + 12 (N2Ha_PP04) (total)",
        col_z="count",
        log_z=True,
        figsize=(7, 4),
        xmin=-2.7,
        xmax=0.0,
        ymin=6.8,
        ymax=9.5,
        ax=axs[1],
    )
    axs[1].axvline(-2.5, color="k")
    axs[1].axvline(-0.3, color="k")
    compare_before_after("N2Ha_PP04", ax=axs[2])

    # ///////////////////////////////////////////////////////////////////////
    # O3N2_PP04 (fig. 2)
    fig, axs = plt.subplots(ncols=3, figsize=(14, 4))
    fig.subplots_adjust(wspace=0.5)
    plot2dhistcontours(
        df_SF,
        col_x=f"O3N2 (total)",
        col_y=f"log(O/H) + 12 (O3N2_PP04) (total)",
        col_z="count",
        log_z=True,
        figsize=(7, 4),
        xmin=-2.7,
        xmax=0.0,
        ymin=6.8,
        ymax=9.5,
        ax=axs[0],
    )
    axs[0].axvline(-1, color="k")
    axs[0].axvline(1.9, color="k")
    plot2dhistcontours(
        df_SF_updated,
        col_x=f"O3N2 (total)",
        col_y=f"log(O/H) + 12 (O3N2_PP04) (total)",
        col_z="count",
        log_z=True,
        figsize=(7, 4),
        xmin=-2.7,
        xmax=0.0,
        ymin=6.8,
        ymax=9.5,
        ax=axs[1],
    )
    axs[1].axvline(-1, color="k")
    axs[1].axvline(1.9, color="k")
    compare_before_after("O3N2_PP04", ax=axs[2])

    # ///////////////////////////////////////////////////////////////////////
    # O3N2_M13 (fig. 3)
    fig, axs = plt.subplots(ncols=3, figsize=(14, 4))
    fig.subplots_adjust(wspace=0.5)
    plot2dhistcontours(
        df_SF,
        col_x=f"O3N2 (total)",
        col_y=f"log(O/H) + 12 (O3N2_M13) (total)",
        col_z="count",
        log_z=True,
        figsize=(7, 4),
        xmin=-1.2,
        xmax=2.0,
        ymin=7.4,
        ymax=9.0,
        ax=axs[0],
    )
    axs[0].axvline(-1.1, color="k")
    axs[0].axvline(1.7, color="k")
    plot2dhistcontours(
        df_SF_updated,
        col_x=f"O3N2 (total)",
        col_y=f"log(O/H) + 12 (O3N2_M13) (total)",
        col_z="count",
        log_z=True,
        figsize=(7, 4),
        xmin=-1.2,
        xmax=2.0,
        ymin=7.4,
        ymax=9.0,
        ax=axs[1],
    )
    axs[1].axvline(-1.1, color="k")
    axs[1].axvline(1.7, color="k")
    compare_before_after("O3N2_M13", ax=axs[2])

    # ///////////////////////////////////////////////////////////////////////
    # N2S2Ha_D16
    fig, axs = plt.subplots(ncols=3, figsize=(14, 4))
    fig.subplots_adjust(wspace=0.5)
    plot2dhistcontours(
        df_SF,
        col_x=f"log(O/H) + 12 (N2S2Ha_D16) (total)",
        col_y=f"Dopita+2016 (total)",
        col_z=f"count",
        log_z=True,
        figsize=(7, 4),
        xmin=7.4,
        xmax=9.5,
        ymin=-1.1,
        ymax=0.6,
        ax=axs[0],
    )
    plot2dhistcontours(
        df_SF_updated,
        col_x=f"log(O/H) + 12 (N2S2Ha_D16) (total)",
        col_y=f"Dopita+2016 (total)",
        col_z=f"count",
        log_z=True,
        figsize=(7, 4),
        xmin=7.4,
        xmax=9.5,
        ymin=-1.1,
        ymax=0.6,
        ax=axs[1],
    )
    fig.suptitle("N2S2Ha_D16")
    compare_before_after("N2S2Ha_D16", ax=axs[2])

    # ///////////////////////////////////////////////////////////////////////
    # Kewley 2019 diagnostics
    # TODO: these should be defined in metallicity.py somewhere, probably...
    k19_ratio_colnames = {
        "N2Ha_K19": "log N2",
        "S2Ha_K19": "log S2",
        "N2S2_K19": "N2S2",
        "O3N2_K19": "O3N2",
        # "O2S2_K19": "O2S2",  # Skipping b/c line ratios not in DataFrame
        # "O2Hb_K19": "O2",  # Skipping b/c line ratios not in DataFrame
        "N2O2_K19": "N2O2",
        "R23_K19": "R23",
        "O3O2_K19": "O3O2",
    }
    k19_kwargs = {
        "cmap": "Spectral_r",
        "xmin": 7.5,
        "xmax": 9.3,
        "ymin": -2.2,
        "ymax": 4.0,
        "vmin": -3.98,
        "vmax": -1.98,
        "figsize": (7, 3),
    }
    k19_met_diags = [diag for diag in metallicity.met_coeffs_K19.keys() if f"log(O/H) + 12 ({diag}/O3O2_K19) (total)" in df_SF_updated and diag != "O2S2_K19" and diag != "O2Hb_K19"]
    for diag in k19_met_diags:
        try:
            fig, axs = plt.subplots(ncols=3, figsize=(14, 4))
            fig.subplots_adjust(wspace=0.5)
            if f"log(O/H) + 12 ({diag}/O3O2_K19) (total)" in df_SF:
                axs[0].scatter(x=df_SF[f"log(O/H) + 12 ({diag}/O3O2_K19) (total)"], y=df_SF[f"{k19_ratio_colnames[diag]} (total)"], c=df_SF[f"log(U) ({diag}/O3O2_K19) (total)"], cmap="Spectral_r")
                axs[0].axvline(metallicity.met_coeffs_K19[diag]["Zmin"], color="k")
                axs[0].axvline(metallicity.met_coeffs_K19[diag]["Zmax"], color="k")
            axs[1].scatter(x=df_SF_updated[f"log(O/H) + 12 ({diag}/O3O2_K19) (total)"], y=df_SF_updated[f"{k19_ratio_colnames[diag]} (total)"], c=df_SF_updated[f"log(U) ({diag}/O3O2_K19) (total)"], cmap="Spectral_r",)
            axs[1].axvline(metallicity.met_coeffs_K19[diag]["Zmin"], color="k")
            axs[1].axvline(metallicity.met_coeffs_K19[diag]["Zmax"], color="k")
            fig.suptitle(diag)
            if f"log(O/H) + 12 ({diag}/O3O2_K19) (total)" in df_SF:
                compare_before_after(f"{diag}/O3O2_K19", ax=axs[2])
        except ValueError:
            print(f"ValueError occurred for {diag}")

    # Ionisation parameter diagnostics
    k19_kwargs = {
        "cmap": "Spectral_r",
        "xmin": -4.0,
        "xmax": -1.8,
        "ymin": -2.0,
        "ymax": 2.1,
        "vmin": 7.63,
        "vmax": 9.23,
        "figsize": (7, 4),
    }
    for diag in ["O3O2_K19"]:
        if f"log(U) (N2Ha_K19/{diag}) (total)" in df_SF:
            fig, axs = plt.subplots(ncols=3, figsize=(14, 4))
            fig.subplots_adjust(wspace=0.5)
            plot2dhistcontours(
                df_SF,
                col_x=f"log(U) (N2Ha_K19/{diag}) (total)",
                col_y=f"{k19_ratio_colnames[diag]} (total)",
                col_z=f"log(O/H) + 12 (N2Ha_K19/{diag}) (total)",
                ax=axs[0],
                **k19_kwargs,
            )
            axs[0].axvline(metallicity.ion_coeffs_K19[diag]["Umin"], color="k")
            axs[0].axvline(metallicity.ion_coeffs_K19[diag]["Umax"], color="k")
        plot2dhistcontours(
            df_SF_updated,
            col_x=f"log(U) (N2Ha_K19/{diag}) (total)",
            col_y=f"{k19_ratio_colnames[diag]} (total)",
            col_z=f"log(O/H) + 12 (N2Ha_K19/{diag}) (total)",
            ax=axs[1],
            **k19_kwargs,
        )
        axs[1].axvline(metallicity.ion_coeffs_K19[diag]["Umin"], color="k")
        axs[1].axvline(metallicity.ion_coeffs_K19[diag]["Umax"], color="k")
        fig.suptitle(diag)


