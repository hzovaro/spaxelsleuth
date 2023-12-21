import sys
nthreads = int(sys.argv[1])

from spaxelsleuth import load_user_config
load_user_config("/home/u5708159/.spaxelsleuthconfig.json")
from spaxelsleuth.loaddata.sami import make_sami_metadata_df, make_sami_df
from spaxelsleuth.loaddata.sami_apertures import make_sami_aperture_df

###########################################################################
# Create the metadata DataFrame
###########################################################################
make_sami_metadata_df(recompute_continuum_SNRs=False, nthreads=nthreads)

###########################################################################
# Create the DataFrames
###########################################################################
# Make test data 
make_sami_df(ncomponents="recom", bin_type="default", eline_SNR_min=5, eline_ANR_min=3, correct_extinction=True, debug=True, nthreads=nthreads)
make_sami_df(ncomponents="1", bin_type="default", eline_SNR_min=5, eline_ANR_min=3, correct_extinction=True, debug=True, nthreads=nthreads)
make_sami_df(ncomponents="recom", bin_type="adaptive", eline_SNR_min=5, eline_ANR_min=3, correct_extinction=True, debug=True, nthreads=nthreads)
make_sami_df(ncomponents="1", bin_type="adaptive", eline_SNR_min=5, eline_ANR_min=3, correct_extinction=True, debug=True, nthreads=nthreads)
make_sami_df(ncomponents="recom", bin_type="sectors", eline_SNR_min=5, eline_ANR_min=3, correct_extinction=True, debug=True, nthreads=nthreads)
make_sami_df(ncomponents="1", bin_type="sectors", eline_SNR_min=5, eline_ANR_min=3, correct_extinction=True, debug=True, nthreads=nthreads)

# Make full data set
make_sami_df(ncomponents="recom", bin_type="default", eline_SNR_min=5, eline_ANR_min=3, correct_extinction=True, debug=False, nthreads=nthreads)
make_sami_df(ncomponents="1", bin_type="default", eline_SNR_min=5, eline_ANR_min=3, correct_extinction=True, debug=False, nthreads=nthreads)
make_sami_df(ncomponents="recom", bin_type="adaptive", eline_SNR_min=5, eline_ANR_min=3, correct_extinction=True, debug=False, nthreads=nthreads)
make_sami_df(ncomponents="1", bin_type="adaptive", eline_SNR_min=5, eline_ANR_min=3, correct_extinction=True, debug=False, nthreads=nthreads)
make_sami_df(ncomponents="recom", bin_type="sectors", eline_SNR_min=5, eline_ANR_min=3, correct_extinction=True, debug=False, nthreads=nthreads)
make_sami_df(ncomponents="1", bin_type="sectors", eline_SNR_min=5, eline_ANR_min=3, correct_extinction=True, debug=False, nthreads=nthreads)

# Make LZIFU data set 
make_sami_df(ncomponents="recom", bin_type="default", eline_SNR_min=5, eline_ANR_min=3, correct_extinction=True, __use_lzifu_fits=True, __lzifu_ncomponents="2", debug=True, nthreads=nthreads)
make_sami_df(ncomponents="recom", bin_type="default", eline_SNR_min=5, eline_ANR_min=3, correct_extinction=True, __use_lzifu_fits=True, __lzifu_ncomponents="1", debug=True, nthreads=nthreads)
make_sami_df(ncomponents="recom", bin_type="default", eline_SNR_min=5, eline_ANR_min=3, correct_extinction=True, __use_lzifu_fits=True, __lzifu_ncomponents="3", debug=True, nthreads=nthreads)
make_sami_df(ncomponents="recom", bin_type="default", eline_SNR_min=5, eline_ANR_min=3, correct_extinction=True, __use_lzifu_fits=True, __lzifu_ncomponents="recom", debug=True, nthreads=nthreads)
make_sami_df(ncomponents="recom", bin_type="default", eline_SNR_min=5, eline_ANR_min=3, correct_extinction=True, __use_lzifu_fits=True, __lzifu_ncomponents="2", debug=False, nthreads=nthreads)
make_sami_df(ncomponents="recom", bin_type="default", eline_SNR_min=5, eline_ANR_min=3, correct_extinction=True, __use_lzifu_fits=True, __lzifu_ncomponents="1", debug=False, nthreads=nthreads)
make_sami_df(ncomponents="recom", bin_type="default", eline_SNR_min=5, eline_ANR_min=3, correct_extinction=True, __use_lzifu_fits=True, __lzifu_ncomponents="3", debug=False, nthreads=nthreads)
make_sami_df(ncomponents="recom", bin_type="default", eline_SNR_min=5, eline_ANR_min=3, correct_extinction=True, __use_lzifu_fits=True, __lzifu_ncomponents="recom", debug=False, nthreads=nthreads)

# Make LZIFU data set WITHOUT S/N or DQ cuts 
make_sami_df(ncomponents="recom", bin_type="default", eline_SNR_min=5, eline_ANR_min=3, correct_extinction=True, __use_lzifu_fits=True, __lzifu_ncomponents="2", debug=False,
             line_flux_SNR_cut=False,
             missing_fluxes_cut=False,
             line_amplitude_SNR_cut=False,
             flux_fraction_cut=False,
             sigma_gas_SNR_cut=False, 
             vgrad_cut=False,
             stekin_cut=False,
             nthreads=nthreads)
make_sami_df(ncomponents="recom", bin_type="default", eline_SNR_min=5, eline_ANR_min=3, correct_extinction=False, __use_lzifu_fits=True, __lzifu_ncomponents="2", debug=False,
             line_flux_SNR_cut=False,
             missing_fluxes_cut=False,
             line_amplitude_SNR_cut=False,
             flux_fraction_cut=False,
             sigma_gas_SNR_cut=False, 
             vgrad_cut=False,
             stekin_cut=False,
             nthreads=nthreads)