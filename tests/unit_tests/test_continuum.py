import pandas as pd 
import numpy as np
from scipy import constants

from spaxelsleuth.config import configure_logger
configure_logger(level="INFO")
from spaxelsleuth.utils import continuum, velocity

import logging
logger = logging.getLogger(__name__)


def test_compute_d4000():
    """Test continuum.compute_d4000()."""

    # Dummy data for testing 
    lambda_vals_rest_A = np.arange(3500, 4500, 10)
    N_lambda = len(lambda_vals_rest_A)
    N_x, N_y = (3, 3)
    data_cube_A = 10 * np.ones((N_lambda, N_y, N_x))
    var_cube_A2 = np.ones((N_lambda, N_y, N_x))
    v_map = np.zeros((N_x, N_y))

    # Convert from F_lambda to F_nu
    # i.e. from erg/s/cm2/A --> erg/s/cm2/Hz, so need to multiply by A . s
    data_cube_Hz = data_cube_A * lambda_vals_rest_A[:, None, None] * (lambda_vals_rest_A[:, None, None] * 1e-10) / (constants.c)
    var_cube_Hz2 = var_cube_A2 * (lambda_vals_rest_A[:, None, None] * (lambda_vals_rest_A[:, None, None] * 1e-10) / (constants.c))**2

    # From manually inspecting the wavelength array
    b_start_idx = 36
    b_stop_idx = 45 
    N_b = b_stop_idx - b_start_idx
    r_start_idx = 51
    r_stop_idx = 60
    N_r = r_stop_idx - r_start_idx
    num = np.nanmean(data_cube_Hz[r_start_idx:r_stop_idx], axis=0)
    denom = np.nanmean(data_cube_Hz[b_start_idx:b_stop_idx], axis=0)
    err_num = 1 / N_r * np.sqrt(np.nansum(var_cube_Hz2[r_start_idx:r_stop_idx], axis=0))
    err_denom = 1 / N_b * np.sqrt(np.nansum(var_cube_Hz2[b_start_idx:b_stop_idx], axis=0))

    expected_d4000_map = num / denom
    expected_d4000_map_err = expected_d4000_map * np.sqrt((err_num / num)**2 + (err_denom / denom)**2)

    d4000_map, d4000_map_err = continuum.compute_d4000(data_cube_A, var_cube_A2, lambda_vals_rest_A, v_map)

    assert np.all(np.isclose(expected_d4000_map, d4000_map))
    assert np.all(np.isclose(expected_d4000_map_err, d4000_map_err))

    logger.info("All test cases passed!")


def test_compute_continuum_intensity():
    """Test continuum.compute_continuum_luminosity()."""

    # Dummy data for testing 
    N_lambda = 101
    N_x, N_y = (3, 3)
    data_cube = 10 * np.ones((N_lambda, N_y, N_x))
    var_cube = np.ones((N_lambda, N_y, N_x))
    lambda_vals_rest_A = np.linspace(4000, 5000, N_lambda)
    lambda_rest_start_A = 4400
    lambda_rest_stop_A = 4500
    N_in_range = 9
    v_map = np.zeros((N_x, N_y))

    # This is independently tested so we don't need to check the output here
    data_cube_masked, var_cube_masked = velocity.get_slices_in_velocity_range(data_cube, var_cube, lambda_vals_rest_A, lambda_rest_start_A, lambda_rest_stop_A, v_map)

    expected_cont_map = 10 * np.ones((N_x, N_y))
    expected_cont_map_std = np.zeros((N_x, N_y))
    expected_cont_map_err = 1 / N_in_range * np.sqrt(np.nansum(var_cube[41:50], axis=0))
    
    cont_map, cont_map_std, cont_map_err = continuum.compute_continuum_intensity(data_cube, var_cube, lambda_vals_rest_A, lambda_rest_start_A, lambda_rest_stop_A, v_map)

    assert np.all(np.isclose(expected_cont_map, cont_map))
    assert np.all(np.isclose(expected_cont_map_std, cont_map_std))
    assert np.all(np.isclose(expected_cont_map_err, cont_map_err))

    logger.info("All test cases passed!")


def test_compute_continuum_luminosity():
    """Test continuum.compute_continuum_luminosity()."""

    # Dummy data for testing 
    df_test = pd.DataFrame({
        "D_L (Mpc)": [120],
        "Bin size (square kpc)": [0.025],
        "HALPHA continuum": [10.0],
        "HALPHA continuum error": [0.1],
    })

    # Correct answer: 
    D_cm = 120 * 1e6 * 3.086e18
    Ha_cont_luminosity = 10.0 * 1e-16 * (4 * np.pi * D_cm**2) / 0.025
    Ha_cont_luminosity_err = Ha_cont_luminosity * 0.1 / 10.0

    df = continuum.compute_continuum_luminosity(df_test)

    assert np.isclose(Ha_cont_luminosity, df["HALPHA continuum luminosity"])
    assert np.isclose(Ha_cont_luminosity_err, df["HALPHA continuum luminosity error"])
    
    logger.info("All test cases passed!")


def test_compute_EW():
    """Test continuum.compute_EW()."""

    # Dummy data for testing
    df_test = pd.DataFrame({
        "HALPHA continuum":           [1.00,   -0.01,  0.00,  1.00,  1.00],
        "HALPHA continuum error":     [0.05,    0.05,  0.05,  0.05,  0.05],
        "HALPHA (total)":             [2.00,    1.50,  1.50, -1.50,  np.nan],
        "HALPHA error (total)":       [0.20,    0.15,  0.15,  0.15,  0.15],
        "HALPHA (component 1)":       [2.00,    1.00,  1.00,  1.00,  1.00],
        "HALPHA error (component 1)": [0.20,    0.10,  0.10,  0.10,  0.10],
        "HALPHA (component 2)":       [np.nan,  0.50,  0.50,  0.50,  0.50],
        "HALPHA error (component 2)": [np.nan,  0.10,  0.10,  0.10,  0.10],
    })

    # Compute EWs
    df = continuum.compute_EW(df_test, ncomponents_max=3, eline_list=["HALPHA"])

    # Test case 1: all valid numbers 
    assert "HALPHA EW (component 1)" in df
    assert "HALPHA EW error (component 1)" in df
    assert df.loc[0, "HALPHA EW (component 1)"] == 2.0
    assert df.loc[0, "HALPHA EW error (component 1)"] == 2.0 * np.sqrt((0.20 / 2.00)**2 + (0.05 / 1.00)**2)
    assert df.loc[0, "HALPHA EW (total)"] == 2.0
    assert df.loc[0, "HALPHA EW error (total)"] == 2.0 * np.sqrt((0.20 / 2.00)**2 + (0.05 / 1.00)**2)

    # Test case 2: -ve continuum level 
    assert np.isnan(df.loc[1, "HALPHA EW (component 1)"])
    assert np.isnan(df.loc[1, "HALPHA EW error (component 1)"])
    assert np.isnan(df.loc[1, "HALPHA EW (total)"])
    assert np.isnan(df.loc[1, "HALPHA EW error (total)"])

    #TODO what about when the continuum error is -ve? OR when the emission line error is -ve? 

    # Test case 3: continuum level = 0
    assert np.isnan(df.loc[2, "HALPHA EW (component 1)"])
    assert np.isnan(df.loc[2, "HALPHA EW error (component 1)"])
    assert np.isnan(df.loc[2, "HALPHA EW (total)"])
    assert np.isnan(df.loc[2, "HALPHA EW error (total)"])

    # Test case 3: emission line flux < 0
    assert np.isnan(df.loc[3, "HALPHA EW (total)"])
    assert np.isnan(df.loc[3, "HALPHA EW error (total)"])

    # Test case 4: emission line flux is NaN
    assert np.isnan(df.loc[4, "HALPHA EW (total)"])
    assert np.isnan(df.loc[4, "HALPHA EW error (total)"])

    logger.info("All test cases passed!")


if __name__ == "__main__":
    test_compute_d4000()
    test_compute_continuum_intensity()
    test_compute_continuum_luminosity()
    test_compute_EW()
    



    

