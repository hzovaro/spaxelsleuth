if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from astropy.visualization import hist

    from spaxelsleuth import load_user_config
    try:
        load_user_config("/Users/u5708159/Desktop/spaxelsleuth_test/.myconfig.json")
    except FileNotFoundError:
        load_user_config("/home/u5708159/.spaxelsleuthconfig.json")
    from spaxelsleuth.io.hector import make_hector_metadata_df, make_hector_df, load_hector_metadata_df, load_hector_df
    from spaxelsleuth.plotting.plot2dmap import plot2dmap

    nthreads = 1

    # Create the DataFrames
    make_hector_metadata_df()
    df_metadata = load_hector_metadata_df()
    
    gals = df_metadata.index.values[:10]

    # Load the DataFrames
    make_hector_df(ncomponents="rec", 
                eline_SNR_min=5, 
                eline_ANR_min=3, 
                correct_extinction=True,
                metallicity_diagnostics=["N2Ha_K19"],
                gals=gals,
                nthreads=nthreads)

    df = load_hector_df(ncomponents="rec",
                    eline_SNR_min=5,
                    eline_ANR_min=3,
                    correct_extinction=True)

