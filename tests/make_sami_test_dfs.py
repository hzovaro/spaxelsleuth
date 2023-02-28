from spaxelsleuth.loaddata.sami import make_sami_metadata_df, make_sami_df, make_sami_aperture_df

###########################################################################
# Create the metadata DataFrame
###########################################################################
make_sami_metadata_df()

###########################################################################
# Create the DataFrames
###########################################################################
# Make test data 
make_sami_df(ncomponents="recom", bin_type="default", eline_SNR_min=5, correct_extinction=True, debug=True)
make_sami_df(ncomponents="1", bin_type="default", eline_SNR_min=5, correct_extinction=True, debug=True)
make_sami_df(ncomponents="recom", bin_type="adaptive", eline_SNR_min=5, correct_extinction=True, debug=True)
make_sami_df(ncomponents="1", bin_type="adaptive", eline_SNR_min=5, correct_extinction=True, debug=True)
make_sami_df(ncomponents="recom", bin_type="sectors", eline_SNR_min=5, correct_extinction=True, debug=True)
make_sami_df(ncomponents="1", bin_type="sectors", eline_SNR_min=5, correct_extinction=True, debug=True)

# w/o extinction correction
make_sami_df(ncomponents="recom", bin_type="default", eline_SNR_min=5, correct_extinction=False, debug=True)
make_sami_df(ncomponents="1", bin_type="default", eline_SNR_min=5, correct_extinction=False, debug=True)
make_sami_df(ncomponents="recom", bin_type="adaptive", eline_SNR_min=5, correct_extinction=False, debug=True)
make_sami_df(ncomponents="1", bin_type="adaptive", eline_SNR_min=5, correct_extinction=False, debug=True)
make_sami_df(ncomponents="recom", bin_type="sectors", eline_SNR_min=5, correct_extinction=False, debug=True)
make_sami_df(ncomponents="1", bin_type="sectors", eline_SNR_min=5, correct_extinction=False, debug=True)