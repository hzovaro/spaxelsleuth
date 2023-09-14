if __name__ == "__main__":
    from spaxelsleuth import load_user_config
    load_user_config("/Users/u5708159/Desktop/spaxelsleuth_test/.myconfig.json")
    from spaxelsleuth.loaddata.sami import make_sami_metadata_df, make_sami_df, load_sami_metadata_df, load_sami_df

    nthreads = 4
    DEBUG = True

    # Create the DataFrames
    make_sami_metadata_df(recompute_continuum_SNRs=True, nthreads=nthreads)
    make_sami_df(bin_type="default", 
                ncomponents="recom", 
                eline_SNR_min=5, 
                correct_extinction=True,
                metallicity_diagnostics=["R23_KK04"],
                nthreads=nthreads,
                debug=DEBUG)

    # Load the DataFrames
    df_metadata = load_sami_metadata_df()
    df = load_sami_df(ncomponents="recom",
                    bin_type="default",
                    eline_SNR_min=5,
                    correct_extinction=True,
                    debug=DEBUG)