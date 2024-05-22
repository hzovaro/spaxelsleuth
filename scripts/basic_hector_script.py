if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from astropy.visualization import hist

    from spaxelsleuth import load_user_config
    try:
        load_user_config("/Users/u5708159/Desktop/spaxelsleuth_test/.myconfig.json")
    except FileNotFoundError:
        load_user_config("/home/u5708159/.spaxelsleuthconfig.json")
    from spaxelsleuth.io.io import make_metadata_df, make_df, load_metadata_df, load_df
    from spaxelsleuth.plotting.plotgalaxies import plot2dhistcontours
    from spaxelsleuth.plotting.plot2dmap import plot2dmap

    nthreads = 4

    # Create the DataFrames
    make_metadata_df(survey="hector")
    df_metadata = load_metadata_df(survey="hector")

    # Load the DataFrames
    make_df(survey="hector", 
            bin_type="default",
            ncomponents="rec", 
            eline_SNR_min=3, 
            eline_ANR_min=3, 
            correct_extinction=True,
            metallicity_diagnostics=["N2Ha_K19"],
            nthreads=nthreads)

    df, ss_params = load_df(survey="hector", 
                 bin_type="default",
                 ncomponents="rec",
                 eline_SNR_min=3,
                 eline_ANR_min=3,
                 correct_extinction=True)

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
    
    # Plot BPT diagram of a galaxy 
    gal = df["ID"].values[0]
    plot2dmap(df, gal=gal, col_z="HALPHA EW (total)")
    plot2dmap(df, gal=gal, col_z="BPT (total)")

    plt.show()