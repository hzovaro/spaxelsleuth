if __name__ == "__main__":
    from spaxelsleuth import load_user_config
    try:
        load_user_config("/Users/u5708159/Desktop/spaxelsleuth_test/.myconfig.json")
    except FileNotFoundError:
        load_user_config("/home/u5708159/.spaxelsleuthconfig.json")
    from spaxelsleuth.io.io import make_metadata_df, make_df, load_metadata_df, load_df

    nthreads = 10
    eline_SNR_min = 3
    eline_ANR_min = 3

    # Create the DataFrames
    make_metadata_df(survey="s7")
    make_df(survey="s7",
            bin_type="default",
            ncomponents="merge",
            eline_SNR_min=eline_SNR_min,
            eline_ANR_min=eline_ANR_min,
            correct_extinction=True,
            metallicity_diagnostics=["N2Ha_PP04",],
            nthreads=nthreads)

    # Load the DataFrames
    df_metadata = load_metadata_df(survey="s7")
    df = load_df(survey="s7",
                 bin_type="default",
                 ncomponents="merge",
                 eline_SNR_min=eline_SNR_min,
                 eline_ANR_min=eline_ANR_min,
                 correct_extinction=True)