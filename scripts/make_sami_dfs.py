from spaxelsleuth.loaddata.sami import make_sami_metadata_df, make_sami_metadata_df_extended, make_sami_df, load_sami_df

###########################################################################
# Create the metadata DataFrame
###########################################################################
# make_sami_metadata_df()

###########################################################################
# Create the DataFrames
###########################################################################
# Make test data 
# make_sami_df(ncomponents="recom", bin_type="default", eline_SNR_min=5, debug=True)
# make_sami_df(ncomponents="1", bin_type="default", eline_SNR_min=5, debug=True)
# make_sami_df(ncomponents="recom", bin_type="adaptive", eline_SNR_min=5, debug=True)
# make_sami_df(ncomponents="1", bin_type="adaptive", eline_SNR_min=5, debug=True)
make_sami_df(ncomponents="recom", bin_type="sectors", eline_SNR_min=5, debug=True)
make_sami_df(ncomponents="1", bin_type="sectors", eline_SNR_min=5, debug=True)

# Make full data set
# make_sami_df(ncomponents="recom", bin_type="default", eline_SNR_min=5, debug=False)
# make_sami_df(ncomponents="1", bin_type="default", eline_SNR_min=5, debug=False)
# make_sami_df(ncomponents="recom", bin_type="adaptive", eline_SNR_min=5, debug=False)
# make_sami_df(ncomponents="1", bin_type="adaptive", eline_SNR_min=5, debug=False)
make_sami_df(ncomponents="recom", bin_type="sectors", eline_SNR_min=5, debug=False)
make_sami_df(ncomponents="1", bin_type="sectors", eline_SNR_min=5, debug=False)


###########################################################################
# Create the extended metadata DataFrame
###########################################################################
# make_sami_metadata_df_extended()
