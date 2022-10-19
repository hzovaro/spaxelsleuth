"""
File:       make_sami_dfs.py
Author:     Henry Zovaro
Email:      henry.zovaro@anu.edu.au

DESCRIPTION
------------------------------------------------------------------------------
When run, this script will
    1) create the SAMI metadata DataFrame,
    2) create SAMI DataFrames for all binning and emission line fitting 
       combinations, with and without debug turned on, and 
    3) create the "metadata" DataFrame.

Input parameters such as correct_extinction and eline_SNR_min can be 
changed as required.

USAGE 
------------------------------------------------------------------------------
    
    >>> python make_sami_dfs.py

"""

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

# Make full data set
make_sami_df(ncomponents="recom", bin_type="default", eline_SNR_min=5, correct_extinction=True, debug=False)
make_sami_df(ncomponents="1", bin_type="default", eline_SNR_min=5, correct_extinction=True, debug=False)
make_sami_df(ncomponents="recom", bin_type="adaptive", eline_SNR_min=5, correct_extinction=True, debug=False)
make_sami_df(ncomponents="1", bin_type="adaptive", eline_SNR_min=5, correct_extinction=True, debug=False)
make_sami_df(ncomponents="recom", bin_type="sectors", eline_SNR_min=5, correct_extinction=True, debug=False)
make_sami_df(ncomponents="1", bin_type="sectors", eline_SNR_min=5, correct_extinction=True, debug=False)

###########################################################################
# Create the aperture DataFrame
###########################################################################
make_sami_aperture_df(eline_SNR_min=5, 
                      line_flux_SNR_cut=True,
                      missing_fluxes_cut=True,
                      sigma_gas_SNR_cut=True, sigma_gas_SNR_min=3,
                      nthreads_max=20, correct_extinction=True)