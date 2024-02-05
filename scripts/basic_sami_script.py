if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from astropy.visualization import hist

    from spaxelsleuth import load_user_config
    try:
        load_user_config("/Users/u5708159/Desktop/spaxelsleuth_test/.myconfig.json")
    except FileNotFoundError:
        load_user_config("/home/u5708159/.spaxelsleuthconfig.json")
    from spaxelsleuth.io.sami import make_sami_metadata_df, make_sami_df, load_sami_metadata_df, load_sami_df
    from spaxelsleuth.plotting.plottools import plot_empty_BPT_diagram, plot_BPT_lines
    from spaxelsleuth.plotting.plotgalaxies import plot2dhistcontours, plot2dscatter

    nthreads = 4
    DEBUG = True

    # Create the DataFrames
    make_sami_metadata_df(recompute_continuum_SNRs=False, nthreads=nthreads)
    make_sami_df(bin_type="default", 
                ncomponents="recom", 
                eline_SNR_min=5, 
                eline_ANR_min=3, 
                correct_extinction=True,
                metallicity_diagnostics=["R23_KK04"],
                nthreads=nthreads,
                debug=DEBUG)

    # Load the DataFrames
    df_metadata = load_sami_metadata_df()
    df = load_sami_df(ncomponents="recom",
                    bin_type="default",
                    eline_SNR_min=5,
                    eline_ANR_min=3,
                    correct_extinction=True,
                    debug=DEBUG)
    
    # Histograms showing the distribution in velocity dispersion
    fig, ax = plt.subplots(nrows=1, ncols=1)
    for nn in range(1, 4):
        hist(df[f"sigma_gas (component {nn})"].values, bins="scott", ax=ax, range=(0, 500), density=True, histtype="step", label=f"Component {nn}")
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