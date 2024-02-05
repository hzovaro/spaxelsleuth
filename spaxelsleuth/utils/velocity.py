from itertools import product
import numpy as np
from scipy import constants

import logging
logger = logging.getLogger(__name__)


###############################################################################
def get_wavelength_from_velocity(lambda_rest, v, units):
    """Compute the Doppler-shifted wavelength corresponding to a velocity and a rest-frame wavelength.

    Note that the input velocity value MUST be relative to the systemic velocity of the galaxy,
    i.e. must not include the cz term.

    The returned wavelength value is determined using the Doppler formula, i.e.

                    nu_0^2 - nu^2
        velocity =  ______________

                    nu_0^2 + nu^2

    where nu_0 is the rest-frame frequency and nu is the observer-frame frequency.

    See https://science.nrao.edu/facilities/vla/docs/manuals/obsguide/modes/line.

    Parameters:
    - lambda_rest (float): Rest wavelength of the spectral line.
    - v (float): Velocity of the source along the line of sight.
    - units (str): Units of velocity, must be 'm/s' or 'km/s'.

    Returns:
    - lambda_obs (float): Observed wavelength of the spectral line, in the same units as lambda_rest.

    NOTE: docstring written with help from ChatGPT 3.5.
    """
    if units not in [
        "m/s",
        "km/s",
    ]:
        raise ValueError("units must be m/s or km/s!")
    if units == "m/s":
        v_m_s = v
    elif units == "km/s":
        v_m_s = v * 1e3
    lambda_obs = lambda_rest * np.sqrt(
        (1 + v_m_s / constants.c) / (1 - v_m_s / constants.c)
    )
    return lambda_obs


###############################################################################
def get_slices_in_velocity_range(
    data_cube,
    var_cube,
    lambda_vals_rest_A,
    lambda_rest_start_A,
    lambda_rest_stop_A,
    v_map,
):
    """Returns a copy of the data and variance cubes with slices outside a specified rest-frame wavelength range masked out based on the velocity map.

    Parameters:
    - data_cube (numpy.ndarray): 3D array representing the data cube.
    - var_cube (numpy.ndarray): 3D array representing the variance cube.
    - lambda_vals_rest_A (numpy.ndarray): 1D array containing rest wavelength values.
    - lambda_rest_start_A (float): Start wavelength of the desired range in Angstroms.
    - lambda_rest_stop_A (float): Stop wavelength of the desired range in Angstroms.
    - v_map (numpy.ndarray): 2D array representing the velocity map in km/s.

    Returns:
    - data_cube_masked, var_cube_masked (numpy.ndarray): copies of the input data
      and variance cubes with slices outside the specified velocity range NaN'd out.

    Note:
    The velocity map is used to calculate the corresponding velocity-dependent wavelength range.
    For indices where the velocity is NaN, it is assumed to be zero.

    NOTE: docstring written with help from ChatGPT 3.5.
    """
    # 3D array containing wavelength values in each spaxel
    lambda_vals_rest_A_cube = np.zeros(data_cube.shape)
    lambda_vals_rest_A_cube[:] = lambda_vals_rest_A[:, None, None]

    # For indices where the velocity is NaN - assume that it's zero
    v_map = v_map.copy()  # Make a copy so that we don't accidentally overwrite the original velocity field
    v_map[np.isnan(v_map)] = 0

    # Min/max wavelength values taking into account the velocities in each spaxel
    lambda_min_A = get_wavelength_from_velocity(
        lambda_rest_start_A, v_map, units="km/s"
    )
    lambda_max_A = get_wavelength_from_velocity(
        lambda_rest_stop_A, v_map, units="km/s"
    )

    # Indices within the desired wavelength window Doppler-shifted according to the velocities in each spaxel
    slice_mask = (lambda_vals_rest_A_cube > lambda_min_A) & (
        lambda_vals_rest_A_cube < lambda_max_A
    )

    # Copies of datacubes with slices other than those in the wavelength window NaN'd out
    data_cube_masked = np.copy(data_cube)
    data_cube_masked[~slice_mask] = np.nan
    var_cube_masked = np.copy(var_cube)
    var_cube_masked[~slice_mask] = np.nan

    return data_cube_masked, var_cube_masked


###############################################################################
def compute_v_grad(v_map):
    """Computes the gradient of the velocity map using central differencing as per eqn. 1 of Zhou+2017.

    Parameters:
    - v_map (numpy.ndarray): 2D or 3D array representing the velocity map.

    Returns:
    - v_grad (numpy.ndarray): Array containing the magnitude of the velocity gradient at each point.

    Note:
    - For a 2D velocity map, central differencing is applied in both x and y directions.
    - For a 3D velocity map, central differencing is applied along the spatial dimensions.

    NOTE: docstring written with help from ChatGPT 3.5.
    """
    logger.debug("computing velocity gradients...")
    v_grad = np.full_like(v_map, np.nan)
    if v_map.ndim == 2:
        ny, nx = v_map.shape
        for yy, xx in product(range(1, ny - 1), range(1, nx - 1)):
            v_grad[yy, xx] = np.sqrt(
                ((v_map[yy, xx + 1] - v_map[yy, xx - 1]) / 2) ** 2
                + ((v_map[yy + 1, xx] - v_map[yy - 1, xx]) / 2) ** 2
            )
    elif v_map.ndim == 3:
        ny, nx = v_map.shape[1:]
        for yy, xx in product(range(1, ny - 1), range(1, nx - 1)):
            v_grad[:, yy, xx] = np.sqrt(
                ((v_map[:, yy, xx + 1] - v_map[:, yy, xx - 1]) / 2) ** 2
                + ((v_map[:, yy + 1, xx] - v_map[:, yy - 1, xx]) / 2) ** 2
            )

    return v_grad
