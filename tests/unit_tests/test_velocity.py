import pandas as pd
import numpy as np

from spaxelsleuth.utils import velocity

import logging
logger = logging.getLogger(__name__)


def test_get_wavelength_from_velocity():
    """Test function for get_wavelength_from_velocity.

    NOTE: written with help from ChatGPT 3.5.
    """
    # Test case 1: Check if the function handles valid input correctly
    lambda_rest = 6562.8  # H-alpha rest wavelength in Angstroms
    v = 200.0  # Velocity in km/s
    units = "km/s"
    expected_result = 6567.179580067013  # Expected observed wavelength in Angstroms calculated using http://hyperphysics.phy-astr.gsu.edu/hbase/Relativ/reldop3.html#c3
    result = velocity.get_wavelength_from_velocity(lambda_rest, v, units)
    assert np.isclose(
        result, expected_result
    ), f"Test case 1 failed: {result} != {expected_result}"

    # Test case 2: Check if the function raises ValueError for invalid units
    lambda_rest = 656.28
    v = 200.0
    units = "invalid_units"
    try:
        velocity.get_wavelength_from_velocity(lambda_rest, v, units)
    except ValueError as e:
        assert str(e) == "units must be m/s or km/s!", f"Test case 2 failed: {e}"
    else:
        raise AssertionError(
            "Test case 2 failed: ValueError not raised for invalid units"
        )

    # Test case 3: Check if the function handles zero velocity correctly
    lambda_rest = 656.28
    v = 0.0
    units = "km/s"
    expected_result = lambda_rest
    result = velocity.get_wavelength_from_velocity(lambda_rest, v, units)
    assert np.isclose(
        result, expected_result
    ), f"Test case 3 failed: {result} != {expected_result}"

    logger.info("All test cases passed!")


def test_get_slices_in_velocity_range():
    """Test function for get_slices_in_velocity_range.

    NOTE: written with help from ChatGPT 3.5.
    """
    # Mock input data
    N_lambda = 100
    N_x, N_y = (3, 3)
    data_cube = np.random.rand(N_lambda, N_y, N_x)
    var_cube = np.random.rand(N_lambda, N_y, N_x)
    lambda_vals_rest_A = np.linspace(4000, 5000, N_lambda)
    lambda_rest_start_A = 4400
    lambda_rest_stop_A = 4500

    # Test case 1: all zeros - wavelength range should simply be the input wavelength ranges
    v_map = np.zeros((3, 3))
    result_data_cube, result_var_cube = velocity.get_slices_in_velocity_range(
        data_cube,
        var_cube,
        lambda_vals_rest_A,
        lambda_rest_start_A,
        lambda_rest_stop_A,
        v_map,
    )

    # Check if the shape of the result matches the input
    assert result_data_cube.shape == data_cube.shape, "Result data cube shape mismatch!"
    assert (
        result_var_cube.shape == var_cube.shape
    ), "Result variance cube shape mismatch!"
    velocity_mask = (lambda_vals_rest_A > lambda_rest_start_A) & (
        lambda_vals_rest_A < lambda_rest_stop_A
    )  # Adjust the velocity range as needed
    assert np.all(
        np.isnan(result_data_cube[~velocity_mask])
    ), "Non-NaN values outside the velocity range in data cube!"
    assert np.all(
        np.isnan(result_var_cube[~velocity_mask])
    ), "Non-NaN values outside the velocity range in variance cube!"

    # Test case 2: non-zero velocity values
    v_map = np.full_like(v_map, 300)
    result_data_cube, result_var_cube = velocity.get_slices_in_velocity_range(
        data_cube,
        var_cube,
        lambda_vals_rest_A,
        lambda_rest_start_A,
        lambda_rest_stop_A,
        v_map,
    )

    # These numbers from http://hyperphysics.phy-astr.gsu.edu/hbase/Relativ/reldop3.html#c3
    lambda_shifted_start_A = 0.4404405140432525e-6 * 1e10
    lambda_shifted_stop_A = 0.4504505257260537e-6 * 1e10
    velocity_mask = (lambda_vals_rest_A > lambda_shifted_start_A) & (
        lambda_vals_rest_A < lambda_shifted_stop_A
    )  # Adjust the velocity range as needed
    assert np.all(
        np.isnan(result_data_cube[~velocity_mask])
    ), "Non-NaN values outside the velocity range in data cube!"
    assert np.all(
        np.isnan(result_var_cube[~velocity_mask])
    ), "Non-NaN values outside the velocity range in variance cube!"

    logger.info("All test cases passed!")


def test_compute_v_grad():
    """Test function for compute_v_grad.

    NOTE: written with help from ChatGPT 3.5.
    """
    # Test case 1: Check if the function handles a 2D velocity map correctly
    v_map_2d = np.array([[1.0, 2.0, 3.0],
                         [4.0, 5.0, 6.0],
                         [7.0, 8.0, 9.0]])

    # Result should be 3.1622776602
    expected_result_2d = np.array(
        [
            [np.nan, np.nan, np.nan],
            [np.nan, 3.1622776602, np.nan],
            [np.nan, np.nan, np.nan],
        ]
    )
    result_2d = velocity.compute_v_grad(v_map_2d)
    assert np.allclose(
        result_2d, expected_result_2d, equal_nan=True
    ), f"Test case 1 failed: {result_2d}"

    # Test case 2: Check if the function handles a 3D velocity map correctly
    v_map_3d = np.array(
        [
            [[1.0, 2.0, 3.0],
             [4.0, 5.0, 6.0],
             [7.0, 8.0, 9.0]],
            [[1.0, 2.0, 3.0],
             [4.0, 3.0, 4.0],
             [7.0, 2.0, 9.0]],
        ]
    )

    # Result should be 3.1622776602
    expected_result_3d = np.array(
        [
            [
                [np.nan, np.nan, np.nan],
                [np.nan, 3.1622776602, np.nan],
                [np.nan, np.nan, np.nan],
            ],
            [
                [np.nan, np.nan, np.nan],
                [np.nan, 0.0, np.nan],
                [np.nan, np.nan, np.nan],
            ],
        ]
    )
    result_3d = velocity.compute_v_grad(v_map_3d)
    assert np.allclose(
        result_3d, expected_result_3d, equal_nan=True
    ), f"Test case 2 failed: {result_3d}"

    logger.info("All test cases passed!")
    

if __name__ == "__main__":
    test_get_wavelength_from_velocity()
    test_get_slices_in_velocity_range()
    test_compute_v_grad()

