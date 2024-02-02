if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from astropy.visualization import hist

    from spaxelsleuth import load_user_config
    try:
        load_user_config("/Users/u5708159/Desktop/spaxelsleuth_test/.myconfig.json")
    except FileNotFoundError:
        load_user_config("/home/u5708159/.spaxelsleuthconfig.json")
    from spaxelsleuth.loaddata.hector import make_hector_metadata_df, make_hector_df, load_hector_metadata_df, load_hector_df

    nthreads = 1

    # Create the DataFrames
    make_hector_metadata_df()
    # df_metadata = load_hector_metadata_df()
    
    # # Load the DataFrames
    # make_hector_df(ncomponents="rec", 
    #             eline_SNR_min=5, 
    #             eline_ANR_min=3, 
    #             correct_extinction=True,
    #             metallicity_diagnostics=["R23_KK04"],
    #             nthreads=nthreads)
    # df = load_hector_df(ncomponents="recom",
    #                 eline_SNR_min=5,
    #                 eline_ANR_min=3,
    #                 correct_extinction=True)