if __name__ == "__main__":
    import os
    import matplotlib.pyplot as plt
    from astropy.visualization import hist

    from spaxelsleuth import load_user_config
    try:
        load_user_config("/Users/u5708159/Desktop/spaxelsleuth_test/.myconfig.json")
    except FileNotFoundError:
        load_user_config("/home/u5708159/.spaxelsleuthconfig.json")
    from spaxelsleuth.loaddata.lzifu import make_lzifu_df, load_lzifu_df
    from spaxelsleuth.config import settings
    from spaxelsleuth.loaddata.sami import make_sami_metadata_df, make_sami_df, load_sami_metadata_df, load_sami_df
    from spaxelsleuth.plotting.plottools import plot_empty_BPT_diagram, plot_BPT_lines
    from spaxelsleuth.plotting.plotgalaxies import plot2dhistcontours, plot2dscatter

    nthreads = 4
    eline_SNR_min = 3
    DEBUG = True
    
    gals = [int(f.split("_")[0]) for f in os.listdir(settings["lzifu"]["input_path"]) if f.endswith("1_comp.fits") and not f.startswith(".")]
    print(gals)

    # Create the DataFrame
    make_lzifu_df(gals=gals,
                ncomponents=1,
                eline_SNR_min=3,
                eline_ANR_min=3,
                sigma_gas_SNR_min=1,
                line_flux_SNR_cut=False,
                missing_fluxes_cut=False,
                line_amplitude_SNR_cut=False,
                flux_fraction_cut=False,
                sigma_gas_SNR_cut=False,
                vgrad_cut=False,
                correct_extinction=False,
                metallicity_diagnostics=[
                    "N2Ha_PP04",
                ],
                sigma_inst_kms=29.6,
                nthreads=nthreads)

    # Load the DataFrames
    df = load_lzifu_df(
        ncomponents=1,
        correct_extinction=False,
        eline_SNR_min=3,
    )

    # Histograms showing the distribution in velocity dispersion
    fig, ax = plt.subplots(nrows=1, ncols=1)
    hist(df[f"sigma_gas (component 1)"].values, bins="scott", ax=ax, range=(0, 500), density=True, histtype="step", label=f"Component 1")
    ax.legend()
    ax.set_xlabel(r"\sigma_{\rm gas}")
    ax.set_ylabel(r"N (normalised)")


    # Plot a 2D histogram showing the distribution of SAMI spaxels in the WHAN diagram
    plot2dhistcontours(df=df,
                col_x=f"log N2 (total)",
                col_y=f"log HALPHA EW (total)",
                col_z="count", log_z=True,
                plot_colorbar=True)

    plt.show()