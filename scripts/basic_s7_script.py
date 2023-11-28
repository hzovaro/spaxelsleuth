if __name__ == "__main__":
    from spaxelsleuth import load_user_config
    try:
        load_user_config("/Users/u5708159/Desktop/spaxelsleuth_test/.myconfig.json")
    except FileNotFoundError:
        load_user_config("/home/u5708159/.spaxelsleuthconfig.json")
    from spaxelsleuth.loaddata.s7 import make_s7_metadata_df, make_s7_df, load_s7_metadata_df, load_s7_df

    nthreads = 10
    eline_SNR_min = 3
    eline_ANR_min = 3

    # Create the DataFrames
    make_s7_metadata_df()
    make_s7_df(
                eline_SNR_min=eline_SNR_min,
                eline_ANR_min=eline_ANR_min,
                correct_extinction=True,
                metallicity_diagnostics=["N2Ha_PP04",],
                nthreads=nthreads)

    # Load the DataFrames
    df_metadata = load_s7_metadata_df()
    df = load_s7_df(eline_SNR_min=eline_SNR_min,
                    eline_ANR_min=eline_ANR_min,
                    correct_extinction=True)