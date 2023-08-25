from itertools import product
import numpy as np
from scipy import constants

import logging 
logger = logging.getLogger(__name__)

###############################################################################
def get_wavelength_from_velocity(lambda_rest, v, units):
    """Compute the Doppler-shifted wavelength given a velocity and a rest-frame wavelength."""
    if units not in ['m/s', 'km/s',]:
        raise ValueError("units must be m/s or km/s!")
    if units == 'm/s':
        v_m_s = v
    elif units == 'km/s':
        v_m_s = v * 1e3
    lambda_obs = lambda_rest * np.sqrt((1 + v_m_s / constants.c) /
                                       (1 - v_m_s / constants.c))
    return lambda_obs


###############################################################################
def get_slices_in_velocity_range(data_cube, var_cube, lambda_vals_rest_A, lambda_rest_start_A, lambda_rest_stop_A, v_map):
    """Returns a copy of the data/variance cubes with slices outside those in a specified wavelength range masked out."""
    # 3D array containing wavelength values in each spaxel
    lambda_vals_rest_A_cube = np.zeros(data_cube.shape)
    lambda_vals_rest_A_cube[:] = lambda_vals_rest_A[:, None, None]

    # For indices where the velocity is NaN - assume that it's zero
    v_map[np.isnan(v_map)] = 0

    # Min/max wavelength values taking into account the velocities in each spaxel
    lambda_min_A = get_wavelength_from_velocity(lambda_rest_start_A, v_map, units="km/s")
    lambda_max_A = get_wavelength_from_velocity(lambda_rest_stop_A, v_map, units="km/s")

    # Indices within the desired wavelength window, after accounting for velocities in each spaxel
    slice_mask = (lambda_vals_rest_A_cube > lambda_min_A) & (lambda_vals_rest_A_cube < lambda_max_A)

    # Copies of datacubes with slices other than those in the wavelength window NaN'd out 
    data_cube_masked = np.copy(data_cube)
    data_cube_masked[~slice_mask] = np.nan
    var_cube_masked = np.copy(var_cube)
    var_cube_masked[~slice_mask] = np.nan

    return data_cube_masked, var_cube_masked

###############################################################################
def compute_v_grad(v_map):
    """Compute v_grad using eqn. 1 of Zhou+2017."""
    logger.debug("computing velocity gradients...")
    v_grad = np.full_like(v_map, np.nan)
    if v_map.ndim == 2:
        ny, nx = v_map.shape
        for yy, xx in product(range(1, ny - 1), range(1, nx - 1)):
            v_grad[yy, xx] = np.sqrt(((v_map[yy, xx + 1] - v_map[yy, xx - 1]) / 2)**2 +\
                                        ((v_map[yy + 1, xx] - v_map[yy - 1, xx]) / 2)**2)
    elif v_map.ndim == 3:
        ny, nx = v_map.shape[1:]
        for yy, xx in product(range(1, ny - 1), range(1, nx - 1)):
            v_grad[:, yy, xx] = np.sqrt(((v_map[:, yy, xx + 1] - v_map[:, yy, xx - 1]) / 2)**2 +\
                                        ((v_map[:, yy + 1, xx] - v_map[:, yy - 1, xx]) / 2)**2)

    return v_grad
