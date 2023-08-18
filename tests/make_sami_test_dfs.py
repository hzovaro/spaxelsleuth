if __name__ == "__main__":
    from itertools import product
    import sys

    from spaxelsleuth import load_user_config
    load_user_config("/home/u5708159/.spaxelsleuthconfig.json")
    from spaxelsleuth.loaddata.sami import make_sami_metadata_df, make_sami_df

    ###########################################################################
    # Create the metadata DataFrame
    ###########################################################################
    make_sami_metadata_df(recompute_continuum_SNRs=False)

    ###########################################################################
    # Create the DataFrames
    ###########################################################################
    # Make test data
    for ncomponents, bin_type, correct_extinction in product(
        ["recom", "1"], ["default", "adaptive", "sectors"], [True, False]):
        try:
            make_sami_df(ncomponents=ncomponents,
                         bin_type=bin_type,
                         eline_SNR_min=5,
                         correct_extinction=correct_extinction,
                         nthreads_max=10,
                         debug=True)
        except:
            print(
                f"ERROR: failed to make DataFrame with ncomponents={ncomponents}, bin_type={bin_type}, correct_extinction={correct_extinction}"
            )
            sys.exit(1)
    sys.exit(0)
