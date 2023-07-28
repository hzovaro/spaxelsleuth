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

from spaxelsleuth import load_user_config
load_user_config("/home/u5708159/.spaxelsleuthconfig.json")
from spaxelsleuth.loaddata.sami import make_sami_metadata_df, make_sami_df
from spaxelsleuth.loaddata.sami_apertures import make_sami_aperture_df

###########################################################################
# Create the metadata DataFrame
###########################################################################
# make_sami_metadata_df()

###########################################################################
# Create the DataFrames
###########################################################################
# Make test data 
# make_sami_df(ncomponents="recom", bin_type="default", eline_SNR_min=5, correct_extinction=True, debug=True)
# make_sami_df(ncomponents="1", bin_type="default", eline_SNR_min=5, correct_extinction=True, debug=True)
# make_sami_df(ncomponents="recom", bin_type="adaptive", eline_SNR_min=5, correct_extinction=True, debug=True)
# make_sami_df(ncomponents="1", bin_type="adaptive", eline_SNR_min=5, correct_extinction=True, debug=True)
# make_sami_df(ncomponents="recom", bin_type="sectors", eline_SNR_min=5, correct_extinction=True, debug=True)
# make_sami_df(ncomponents="1", bin_type="sectors", eline_SNR_min=5, correct_extinction=True, debug=True)

# Make full data set
# make_sami_df(ncomponents="recom", bin_type="default", eline_SNR_min=5, correct_extinction=True, debug=False)
# make_sami_df(ncomponents="1", bin_type="default", eline_SNR_min=5, correct_extinction=True, debug=False)
# make_sami_df(ncomponents="recom", bin_type="adaptive", eline_SNR_min=5, correct_extinction=True, debug=False)
# make_sami_df(ncomponents="1", bin_type="adaptive", eline_SNR_min=5, correct_extinction=True, debug=False)
# make_sami_df(ncomponents="recom", bin_type="sectors", eline_SNR_min=5, correct_extinction=True, debug=False)
# make_sami_df(ncomponents="1", bin_type="sectors", eline_SNR_min=5, correct_extinction=True, debug=False)

# Make LZIFU data set 
make_sami_df(ncomponents="recom", bin_type="default", eline_SNR_min=5, correct_extinction=True, __use_lzifu_fits=True, __lzifu_ncomponents="2", debug=True)
make_sami_df(ncomponents="recom", bin_type="default", eline_SNR_min=5, correct_extinction=True, __use_lzifu_fits=True, __lzifu_ncomponents="1", debug=True)
make_sami_df(ncomponents="recom", bin_type="default", eline_SNR_min=5, correct_extinction=True, __use_lzifu_fits=True, __lzifu_ncomponents="3", debug=True)
make_sami_df(ncomponents="recom", bin_type="default", eline_SNR_min=5, correct_extinction=True, __use_lzifu_fits=True, __lzifu_ncomponents="recom", debug=True)
make_sami_df(ncomponents="recom", bin_type="default", eline_SNR_min=5, correct_extinction=True, __use_lzifu_fits=True, __lzifu_ncomponents="2", debug=False)
make_sami_df(ncomponents="recom", bin_type="default", eline_SNR_min=5, correct_extinction=True, __use_lzifu_fits=True, __lzifu_ncomponents="1", debug=False)
make_sami_df(ncomponents="recom", bin_type="default", eline_SNR_min=5, correct_extinction=True, __use_lzifu_fits=True, __lzifu_ncomponents="3", debug=False)
make_sami_df(ncomponents="recom", bin_type="default", eline_SNR_min=5, correct_extinction=True, __use_lzifu_fits=True, __lzifu_ncomponents="recom", debug=False)

# Make LZIFU data set WITHOUT S/N or DQ cuts 
make_sami_df(ncomponents="recom", bin_type="default", eline_SNR_min=5, correct_extinction=True, __use_lzifu_fits=True, __lzifu_ncomponents="2", debug=False,
             line_flux_SNR_cut=False,
             missing_fluxes_cut=False,
             line_amplitude_SNR_cut=False,
             flux_fraction_cut=False,
             sigma_gas_SNR_cut=False, 
             vgrad_cut=False,
             stekin_cut=False,
             nthreads_max=20)
make_sami_df(ncomponents="recom", bin_type="default", eline_SNR_min=5, correct_extinction=False, __use_lzifu_fits=True, __lzifu_ncomponents="2", debug=False,
             line_flux_SNR_cut=False,
             missing_fluxes_cut=False,
             line_amplitude_SNR_cut=False,
             flux_fraction_cut=False,
             sigma_gas_SNR_cut=False, 
             vgrad_cut=False,
             stekin_cut=False,
             nthreads_max=20)


###########################################################################
# Create the aperture DataFrame
###########################################################################
# make_sami_aperture_df(eline_SNR_min=5, 
#                       line_flux_SNR_cut=True,
#                       missing_fluxes_cut=True,
#                       sigma_gas_SNR_cut=True, sigma_gas_SNR_min=3,
#                       nthreads_max=8, correct_extinction=True)