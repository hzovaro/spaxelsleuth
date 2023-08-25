if __name__ == "__main__":
    from itertools import product
    import sys

    from spaxelsleuth import load_user_config, configure_logger
    load_user_config("/home/u5708159/.spaxelsleuthconfig.json")
    configure_logger(level="DEBUG")
    from spaxelsleuth.loaddata.sami import make_sami_metadata_df, make_sami_df

    ###########################################################################
    # Create the metadata DataFrame
    ###########################################################################
    make_sami_metadata_df(recompute_continuum_SNRs=False, nthreads=10)

    ###########################################################################
    # Create the DataFrames
    ###########################################################################
    # Make test data
    for ncomponents, bin_type, correct_extinction in product(
        ["recom", "1"], ["default", "adaptive", "sectors"], [True, False]):
        make_sami_df(ncomponents=ncomponents,
                        bin_type=bin_type,
                        eline_SNR_min=5,
                        correct_extinction=correct_extinction,
                        nthreads=10,
                        debug=True)
    sys.exit(0)
