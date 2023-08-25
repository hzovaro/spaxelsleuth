import os

from spaxelsleuth import load_user_config, configure_logger
load_user_config("/home/u5708159/.spaxelsleuthconfig.json")
configure_logger(level="DEBUG")
from spaxelsleuth.loaddata.lzifu import make_lzifu_df
from spaxelsleuth.config import settings

gals = [
    g.rstrip("_R.fits.gz")
    for g in os.listdir(settings["lzifu"]["data_cube_path"])
    if g.endswith("_R.fits.gz") and not g.startswith("._")
][:3]
kwargs = {
    "eline_SNR_min": 1,
    "sigma_gas_SNR_min": 1,
    "line_flux_SNR_cut": True,
    "missing_fluxes_cut": True,
    "line_amplitude_SNR_cut": True,
    "flux_fraction_cut": False,
    "sigma_gas_SNR_cut": True,
    "vgrad_cut": False,
    "correct_extinction": False,
}
make_lzifu_df(gals=gals,
              ncomponents=1,
              sigma_inst_kms=29.6,
              metallicity_diagnostics=[
                  "N2Ha_PP04",
              ],
              nthreads=3,
              df_fname="test_lzifu.hd5",
              **kwargs)
