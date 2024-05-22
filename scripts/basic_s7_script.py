if __name__ == "__main__":
    import sys
    from spaxelsleuth import load_user_config
    load_user_config(sys.argv[1])
    from spaxelsleuth.io.io import make_metadata_df, make_df, load_metadata_df, load_df

    nthreads = 1
    eline_SNR_min = 3
    eline_ANR_min = 3

    # Create the DataFrames
    make_metadata_df(survey="s7")
    make_df(survey="s7",
            bin_type="default",
            ncomponents="merge",
            gals=["NGC1068", "NGC6300",],
            eline_SNR_min=eline_SNR_min,
            eline_ANR_min=eline_ANR_min,
            correct_extinction=False,
            metallicity_diagnostics=["N2Ha_PP04",],
            nthreads=nthreads)

    # Load the DataFrames
    df_metadata = load_metadata_df(survey="s7")
    df, ss_params = load_df(survey="s7",
                 bin_type="default",
                 ncomponents="merge",
                 gals=["NGC1068", "NGC6300",],
                 eline_SNR_min=eline_SNR_min,
                 eline_ANR_min=eline_ANR_min,
                 correct_extinction=False)