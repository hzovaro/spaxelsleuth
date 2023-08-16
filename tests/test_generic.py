import numpy as np
import os, sys
import matplotlib.pyplot as plt
plt.ion()
plt.close("all")

from astropy.io import fits

from spaxelsleuth import load_user_config
load_user_config("/home/u5708159/.spaxelsleuthconfig.json")
from spaxelsleuth.config import settings
from spaxelsleuth.loaddata.sami import load_sami_metadata_df
from spaxelsleuth.loaddata.generic import get_slices_in_velocity_range, compute_d4000, compute_d4000_old, compute_continuum_intensity, compute_continuum_intensity_old, compute_HALPHA_amplitude_to_noise, compute_HALPHA_amplitude_to_noise_old

"""
Test functions in generic.py
"""

#/////////////////////////////////////////////////////////////////////////
def plot_stuff_to_check(data_cube, data_cube_masked, lambda_vals_rest_A, v_map):
    """Plot a 3x3 grid of spectra indicating the selected regions."""
    fig, axs = plt.subplots(nrows=3, ncols=3, sharex=True)
    x0, y0 = 24, 24
    for rr in [-1, 0, 1]:
        for cc in [-1, 0, 1]:
            y = y0 + rr * 10
            x = x0 + cc * 10
            axs[rr][cc].plot(lambda_vals_rest_A, data_cube[:, y, x], "k")
            axs[rr][cc].plot(lambda_vals_rest_A, data_cube_masked[:, y, x], "r")
            axs[rr][cc].set_title(f"v = {v_map[y, x]:.2f} km/s")

######################################################################
# Test functions that compute continuum quantities from the datacubes
######################################################################
df_metadata = load_sami_metadata_df()

# Load a galaxy in the same way as in sami.py
gal = int(sys.argv[1])
bin_type = "default"
ncomponents = "recom"

# Open the red & blue cubes.
data_cube_path = settings["sami"]["data_cube_path"]
input_path = settings["sami"]["input_path"]
with fits.open(os.path.join(data_cube_path, f"ifs/{gal}/{gal}_A_cube_blue.fits.gz")) as hdulist_B_cube:
    header_R = hdulist_B_cube[0].header
    data_cube_B = hdulist_B_cube[0].data
    var_cube_B = hdulist_B_cube[1].data
    hdulist_B_cube.close()

    # Wavelength values
    lambda_0_A = header_R["CRVAL3"] - header_R["CRPIX3"] * header_R["CDELT3"]
    dlambda_A = header_R["CDELT3"]
    N_lambda = header_R["NAXIS3"]
    lambda_vals_B_A = np.array(range(N_lambda)) * dlambda_A + lambda_0_A
    lambda_vals_B_rest_A = lambda_vals_B_A / (1 + df_metadata.loc[gal, "z (spectroscopic)"]) #NOTE: we use the spectroscopic redshift here, because when it comes to measuring e.g. continuum levels, it's important that the wavelength range we use is consistent between galaxies. For some galaxies the flow-corrected redshift is sufficiently different from the spectroscopic redshift that when we use it to define wavelength windows for computing the continuum level for instance we end up enclosing an emission line which throws the measurement way out of whack (e.g. for 572402)

with fits.open(os.path.join(data_cube_path, f"ifs/{gal}/{gal}_A_cube_red.fits.gz")) as hdulist_R_cube:
    header_R = hdulist_R_cube[0].header
    data_cube_R = hdulist_R_cube[0].data
    var_cube_R = hdulist_R_cube[1].data

    # Wavelength values
    lambda_0_A = header_R["CRVAL3"] - header_R["CRPIX3"] * header_R["CDELT3"]
    dlambda_A = header_R["CDELT3"]
    N_lambda = header_R["NAXIS3"]
    lambda_vals_R_A = np.array(range(N_lambda)) * dlambda_A + lambda_0_A
    lambda_vals_R_rest_A = lambda_vals_R_A / (1 + df_metadata.loc[gal, "z (spectroscopic)"]) #NOTE: we use the spectroscopic redshift here, because when it comes to measuring e.g. continuum levels, it's important that the wavelength range we use is consistent between galaxies. For some galaxies the flow-corrected redshift is sufficiently different from the spectroscopic redshift that when we use it to define wavelength windows for computing the continuum level for instance we end up enclosing an emission line which throws the measurement way out of whack (e.g. for 572402)

with fits.open(os.path.join(input_path, f"ifs/{gal}/{gal}_A_gas-velocity_{bin_type}_{ncomponents}-comp.fits")) as hdulist_v:
    v_map = hdulist_v[0].data.astype(np.float64)
with fits.open(os.path.join(input_path, f"ifs/{gal}/{gal}_A_stellar-velocity_{bin_type}_two-moment.fits")) as hdulist_v:
    v_star_map = hdulist_v[0].data.astype(np.float64)

# TEST: get_slices_in_velocity_range() on the Ca H & K lines, using the stellar velocity map
data_cube_masked_B, var_cube_masked_B = get_slices_in_velocity_range(data_cube_B, var_cube_B, lambda_vals_B_rest_A, 3933, 3969, v_star_map)
plot_stuff_to_check(data_cube_B, data_cube_masked_B, lambda_vals_B_rest_A, v_star_map)

# TEST: get_slices_in_velocity_range() assuming no velocity shift (i.e. what we were doing before)
data_cube_masked_B, var_cube_masked_B = get_slices_in_velocity_range(data_cube_B, var_cube_B, lambda_vals_B_rest_A, 3933, 3969, np.zeros(v_star_map.shape))
plot_stuff_to_check(data_cube_B, data_cube_masked_B, lambda_vals_B_rest_A, np.zeros(v_star_map.shape))

"""
VERDICT: doesn't make that much difference whether we account for the velocity or not. But, seems to work just fine.
"""

# TEST: compute_d4000_old()
d4000_map_old, d4000_map_err_old = compute_d4000_old(data_cube_B, var_cube_B, lambda_vals_B_rest_A)
d4000_map_new, d4000_map_err_new = compute_d4000(data_cube_B, var_cube_B, lambda_vals_B_rest_A, v_star_map)
fig, axs = plt.subplots(nrows=2, ncols=3)
axs[0][0].imshow(d4000_map_old, vmin=1.1, vmax=3.5, cmap="afmhot", origin="lower")
axs[0][1].imshow(d4000_map_new, vmin=1.1, vmax=3.5, cmap="afmhot", origin="lower")
axs[0][2].imshow(d4000_map_new - d4000_map_old, vmin=-0.1, vmax=+0.1, cmap="copper", origin="lower")
axs[1][0].imshow(d4000_map_err_old, vmin=0, vmax=1, cmap="jet", origin="lower")
axs[1][1].imshow(d4000_map_err_new, vmin=0, vmax=1, cmap="jet", origin="lower")
axs[1][2].imshow(d4000_map_err_new - d4000_map_err_old, vmin=-0.1, vmax=+0.1, cmap="copper", origin="lower")

"""
VERDICT: looks OK
"""

# TEST: compute_continuum_intensity()
start_A = 4500
stop_A = 4700
cont_map_old, cont_map_std_old, cont_map_err_old = compute_continuum_intensity_old(data_cube_B, var_cube_B, lambda_vals_B_rest_A, start_A, stop_A)
cont_map_new, cont_map_std_new, cont_map_err_new = compute_continuum_intensity(data_cube_B, var_cube_B, lambda_vals_B_rest_A, start_A, stop_A, v_star_map)
fig, axs = plt.subplots(nrows=2, ncols=3)
axs[0][0].imshow(cont_map_old, vmin=0, vmax=0.1, cmap="afmhot", origin="lower")
axs[0][1].imshow(cont_map_new, vmin=0, vmax=0.1, cmap="afmhot", origin="lower")
axs[0][2].imshow(cont_map_new - cont_map_old, vmin=-0.1, vmax=+0.1, cmap="copper", origin="lower")
axs[1][0].imshow(cont_map_err_old, vmin=0, vmax=0.01, cmap="jet", origin="lower")
axs[1][1].imshow(cont_map_err_new, vmin=0, vmax=0.01, cmap="jet", origin="lower")
axs[1][2].imshow(cont_map_err_new - cont_map_err_old, vmin=-0.1, vmax=+0.1, cmap="copper", origin="lower")

"""
VERDICT: looks OK
"""

# TEST: compute_HALPHA_amplitude_to_noise()
AN_HALPHA_map_old = compute_HALPHA_amplitude_to_noise_old(data_cube_R, var_cube_R, lambda_vals_R_rest_A, v_map[0], dv=300)
AN_HALPHA_map_new = compute_HALPHA_amplitude_to_noise(data_cube_R, var_cube_R, lambda_vals_R_rest_A, v_star_map, v_map[0], dv=300)
fig, axs = plt.subplots(nrows=1, ncols=3)
axs[0].imshow(AN_HALPHA_map_old, vmin=0, vmax=500, cmap="afmhot", origin="lower")
axs[1].imshow(AN_HALPHA_map_new, vmin=0, vmax=500, cmap="afmhot", origin="lower")
axs[2].imshow(AN_HALPHA_map_old - AN_HALPHA_map_new, vmin=-0.1, vmax=+0.1, cmap="copper", origin="lower")

"""
VERDICT: looks OK
"""
