if __name__ == "__main__":
    import os
    import matplotlib.pyplot as plt
    from astropy.visualization import hist

    from spaxelsleuth import load_user_config
    try:
        load_user_config("/Users/u5708159/Desktop/spaxelsleuth_test/.myconfig.json")
    except FileNotFoundError:
        load_user_config("/home/u5708159/.spaxelsleuthconfig.json")
    from spaxelsleuth.config import settings
    from spaxelsleuth.io.io import make_df, load_df
    from spaxelsleuth.plotting.plottools import plot_empty_BPT_diagram, plot_BPT_lines
    from spaxelsleuth.plotting.plotgalaxies import plot2dhistcontours, plot2dscatter

    nthreads = 4
    ncomponents = 1
    eline_SNR_min = 1
    eline_ANR_min = 1
    correct_extinction = True
    
    # List of galaxies with LZIFU data
    gals = [int(f.split("_")[0]) for f in os.listdir(settings["lzifu"]["input_path"]) if f.endswith("1_comp.fits") and not f.startswith(".")]

    # Create the DataFrame
    make_df(survey="lzifu",
            gals=gals,
            ncomponents=ncomponents,
            eline_SNR_min=eline_SNR_min,
            eline_ANR_min=eline_ANR_min,
            correct_extinction=correct_extinction,
            metallicity_diagnostics=[
                "N2Ha_PP04",
            ],
            sigma_inst_kms=29.6,
            nthreads=nthreads)

    # Load the DataFrames
    df = load_df(
         survey="lzifu",
        ncomponents=ncomponents,
        eline_SNR_min=eline_SNR_min,
        eline_ANR_min=eline_ANR_min,
        correct_extinction=correct_extinction,
    )

    # Histograms showing the distribution in velocity dispersion
    fig, ax = plt.subplots(nrows=1, ncols=1)
    hist(df[f"sigma_gas (component 1)"].values, bins="scott", ax=ax, range=(0, 500), density=True, histtype="step", label=f"Component 1")
    ax.legend()
    ax.set_xlabel(r"\sigma_{\rm gas}")
    ax.set_ylabel(r"N (normalised)")

    # Plot a 2D histogram showing the distribution of spaxels in the WHAN diagram
    plot2dhistcontours(df=df,
                col_x=f"log N2 (total)",
                col_y=f"log HALPHA EW (total)",
                col_z="count", log_z=True,
                plot_colorbar=True)

    plt.show()