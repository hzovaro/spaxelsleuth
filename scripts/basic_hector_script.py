if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from astropy.visualization import hist

    from spaxelsleuth import load_user_config
    try:
        load_user_config("/Users/u5708159/Desktop/spaxelsleuth_test/.myconfig.json")
    except FileNotFoundError:
        load_user_config("/home/u5708159/.spaxelsleuthconfig.json")
    from spaxelsleuth.io.io import make_metadata_df, make_df, load_metadata_df, load_df
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

