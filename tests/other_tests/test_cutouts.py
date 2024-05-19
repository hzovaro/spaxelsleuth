# Imports
import matplotlib.pyplot as plt

from spaxelsleuth import load_user_config, configure_logger
load_user_config("../integration_tests/test_config.json")
configure_logger(level="INFO")
from spaxelsleuth.config import settings
from spaxelsleuth.plotting.cutouts import download_image, plot_cutout_image

plt.ion()
plt.close("all")

# Galaxy in DECaLS but not in SDSS footprint
gal = 901005167908374  # Hector ID
ra = 303.941552
dec = -57.82656

plot_cutout_image(
    df=None, gal=gal, ra_deg=ra, dec_deg=dec, source="decals", as_per_px=0.262, sz_px=250,
    show_scale_bar=False
)

# Galaxy in SDSS
gal = 572402  # Hector ID
ra = 134.447158
dec = -0.199961

plot_cutout_image(
    df=None, gal=gal, ra_deg=ra, dec_deg=dec, source="sdss", as_per_px=0.25, sz_px=250,
    show_scale_bar=False
)
