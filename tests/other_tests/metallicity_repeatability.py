import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import rcParams
import numpy as np

from spaxelsleuth import load_user_config, configure_logger
load_user_config("/home/u5708159/.spaxelsleuthconfig.json")
configure_logger(level="INFO")
from spaxelsleuth.io.sami import load_sami_df
from spaxelsleuth.utils import metallicity


if __name__ == "__main__":
    """Test the repeatability metallicity calculations."""
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
    counts = gb["x (pixels)"].count().sort_values(ascending=False)
    gals_SF = counts.index.values[:10]
    df_SF = df.loc[df["ID"].isin(gals_SF)]

    diags = [
        "Rcal_PG16",
        "Scal_PG16",
        "N2Ha_M13",
        "N2O2_KD02",
        "R23_KK04",
        "N2Ha_K19",
        "S2Ha_K19",
        "N2S2_K19",
        "O3N2_K19",
        "O2S2_K19",
        "O2Hb_K19",
        "N2O2_K19",
        "R23_K19",
        "O3N2_PP04",
        "O3N2_M13",
        "N2S2Ha_D16",
    ]

    # Test repeatability of iterative calculations
    niters = 1000 
    df_SF_1 = df_SF.copy()
    df_SF_2 = df_SF.copy()
    for diag in diags:
        df_SF_1 = metallicity.calculate_metallicity(
            met_diagnostic=diag,
            compute_logU=True if diag.endswith("K19") or diag.endswith("KK04") else False,
            ion_diagnostic="O3O2_K19" if diag.endswith("K19") else "O3O2_KK04",
            compute_errors=True,
            niters=niters,
            seed=1,
            df=df_SF_1,
            s=" (total)",
        )
        df_SF_2 = metallicity.calculate_metallicity(
            met_diagnostic=diag,
            compute_logU=True if diag.endswith("K19") or diag.endswith("KK04") else False,
            ion_diagnostic="O3O2_K19" if diag.endswith("K19") else "O3O2_KK04",
            compute_errors=True,
            niters=niters,
            seed=2,
            df=df_SF_2,
            s=" (total)",
        )

    # Save to .pdf 
    pp = PdfPages("metallicity_repeatability.pdf")
    for diag in diags:
        # Plot to check
        fig, axs = plt.subplots(ncols=3, figsize=(15, 4))
        if diag.endswith("K19"):
            met_str = f"{diag}/O3O2_K19"
        elif diag.endswith("KK04"):
            met_str = f"{diag}/O3O2_KK04"
        else:
            met_str = diag
        axs[0].scatter(
            df_SF_1[f"log(O/H) + 12 ({met_str}) (total)"],
            df_SF_2[f"log(O/H) + 12 ({met_str}) (total)"],
            c="b",
        )
        axs[0].plot([axs[0].get_xlim()[0], axs[0].get_xlim()[1]], [axs[0].get_xlim()[0], axs[0].get_xlim()[1]], "k")
        axs[2].hist(df_SF_1[f"log(O/H) + 12 ({met_str}) (total)"] - df_SF_2[f"log(O/H) + 12 ({met_str}) (total)"], histtype="step", color="b", range=(-0.02, +0.02), bins=30)
        std_logOH12 = np.nanstd(df_SF_1[f"log(O/H) + 12 ({met_str}) (total)"] - df_SF_2[f"log(O/H) + 12 ({met_str}) (total)"])
        axs[2].text(s=r"$\sigma_{\rm log(O/H) + 12} = %.2g$" % std_logOH12, x=0.1, y=0.9, va="top", ha="left", transform=axs[2].transAxes)
        axs[2].axvline(0, color="k")
        
        if f"log(U) ({met_str}) (total)" in df_SF_1:
            axs[1].scatter(
                df_SF_1[f"log(U) ({met_str}) (total)"],
                df_SF_2[f"log(U) ({met_str}) (total)"],
                c="r",
            )
            std_logU = np.nanstd(df_SF_1[f"log(U) ({met_str}) (total)"] - df_SF_2[f"log(U) ({met_str}) (total)"])
            axs[2].hist(df_SF_1[f"log(U) ({met_str}) (total)"] - df_SF_2[f"log(U) ({met_str}) (total)"], histtype="step", color="r", range=(-0.02, +0.02), bins=30)
            axs[2].text(s=r"$\sigma_{\rm log(U)} = %.2g$" % std_logU, x=0.1, y=0.8, va="top", ha="left", transform=axs[2].transAxes)
            axs[1].plot([axs[1].get_xlim()[0], axs[1].get_xlim()[1]], [axs[1].get_xlim()[0], axs[1].get_xlim()[1]], "k")
        
        fig.suptitle(f"{diag} ({niters} iterations)")

        # Check NaNs
        cond_nans_disagree = ~df_SF_1[f"log(O/H) + 12 ({met_str}) (total)"].isna() & df_SF_2[f"log(O/H) + 12 ({met_str}) (total)"].isna()
        cond_nans_disagree |= df_SF_1[f"log(O/H) + 12 ({met_str}) (total)"].isna() & ~df_SF_2[f"log(O/H) + 12 ({met_str}) (total)"].isna()
        N_tot = df_SF_1.shape[0]
        N_disagree = df_SF_1[cond_nans_disagree].shape[0]
        print(f"Diagnostic {diag}: {N_disagree} of {N_tot} spaxels have NaNs that disagree between runs")
        cond_both_finite = ~df_SF_1[f"log(O/H) + 12 ({met_str}) (total)"].isna() & ~df_SF_2[f"log(O/H) + 12 ({met_str}) (total)"].isna()
        np.isclose(df_SF_1.loc[cond_both_finite, f"log(O/H) + 12 ({met_str}) (total)"],  df_SF_2.loc[cond_both_finite, f"log(O/H) + 12 ({met_str}) (total)"],)

        pp.savefig(fig, bbox_inches="tight")
    pp.close()
