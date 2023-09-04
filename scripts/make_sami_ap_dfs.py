import sys
nthreads = int(sys.argv[1])

from spaxelsleuth import load_user_config
load_user_config("/home/u5708159/.spaxelsleuthconfig.json")
from spaxelsleuth.loaddata.sami_apertures import make_sami_aperture_df

###########################################################################
# Create the aperture DataFrame
###########################################################################
make_sami_aperture_df(eline_SNR_min=5, 
                      line_flux_SNR_cut=True,
                      missing_fluxes_cut=True,
                      sigma_gas_SNR_cut=True, sigma_gas_SNR_min=3,
                      nthreads=nthreads, correct_extinction=True)
make_sami_aperture_df(eline_SNR_min=0, 
                      line_flux_SNR_cut=True,
                      missing_fluxes_cut=True,
                      sigma_gas_SNR_cut=True, sigma_gas_SNR_min=3,
                      nthreads=nthreads, correct_extinction=False)