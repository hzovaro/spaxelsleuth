"""
PROBLEM: in SOME galaxies, the fits contained in 

    /priv/sami/sami_data/Final_SAMI_data/LZIFU/lzifu_default_products_old/

are DIFFERENT from those in 

    /priv/sami/sami_data/Final_SAMI_data/LZIFU/lzifu_default_products/

which are the same as those used for DR3.

Want to open BOTH sets of fits & compare them

"""
from astropy.io import fits
from itertools import product
import sys 
import os
import numpy as np

import matplotlib.pyplot as plt
plt.ion()
plt.close("all")

from IPython.core.debugger import Tracer

lzifu_products_path_old = "/priv/sami/sami_data/Final_SAMI_data/LZIFU/lzifu_default_products_old"
lzifu_products_path_new = "/priv/sami/sami_data/Final_SAMI_data/LZIFU/lzifu_default_products"

gal = int(sys.argv[1])

fname_old = [f for f in os.listdir(lzifu_products_path_old) if f.startswith(str(gal)) and "_recom_comp.fits.gz" in f and "SF" not in f and "extinction" not in f][0]
fname_new = [f for f in os.listdir(lzifu_products_path_new) if f.startswith(str(gal)) and "_recom_comp.fits.gz" in f and "SF" not in f and "extinction" not in f][0]

# open
eline = "HALPHA"

hdulist_old = fits.open(os.path.join(lzifu_products_path_old, fname_old))
hdulist_new = fits.open(os.path.join(lzifu_products_path_new, fname_new))

flux_old = hdulist_old[eline].data[0]
flux_new = hdulist_new[eline].data[0]
flux_diff = flux_old - flux_new
flux_diff[flux_diff == 0] = np.nan

# mask_same = flux_old == flux_new
# mask_unfitted_in_old = np.isnan(flux_old) & ~np.isnan(flux_new)
# mask_both_finite = ~np.isnan(flux_new) & ~np.isnan(flux_old)
# mask_both_nan = np.isnan(flux_new) & np.isnan(flux_old)
# mask_is_different = flux_new != flux_old

# mask_map = np.full_like(flux_old, np.nan)
# mask_map[mask_both_nan] = -1
# mask_map[mask_same] = 0
# mask_map[mask_unfitted_in_old] = 1

mask_map = np.full_like(flux_old, np.nan)
for xx, yy in product(range(50), range(50)):
    if ~np.isnan(flux_new[yy, xx]) and ~np.isnan(flux_old[yy, xx]):
        if flux_old[yy, xx] == flux_new[yy, xx]:
            mask_map[yy, xx] = 1  # Identical fluxes
        else:
            mask_map[yy, xx] = 2  # Different fluxes
    elif np.isnan(flux_old[yy, xx]) and np.isnan(flux_new[yy, xx]):
        mask_map[yy, xx] = -1  # NaN in both
    elif np.isnan(flux_new[yy, xx]) and ~np.isnan(flux_old[yy, xx]):
        mask_map[yy, xx] = -2  # Fitted in old, but not in new 


fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))
vmax = np.nanmax([np.nanmax(flux_old), np.nanmax(flux_new)])
axs[0].imshow(flux_old, cmap="jet", vmin=0, vmax=vmax); axs[0].set_title("Old")
axs[1].imshow(flux_new, cmap="jet", vmin=0, vmax=vmax); axs[1].set_title("New")
m = axs[2].imshow(mask_map, vmin=-2.5, vmax=2.5, cmap="coolwarm")
# plt.gca().set_xticklabels(["Fitted in old, but not in new", "NaN in both", "Different fluxes", "Identical fluxes"])
plt.colorbar(mappable=m)
fig.suptitle(gal)