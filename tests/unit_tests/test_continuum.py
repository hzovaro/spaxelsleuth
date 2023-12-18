import pandas as pd 
import numpy as np

from spaxelsleuth.config import configure_logger
configure_logger(level="DEBUG")
from spaxelsleuth.utils import continuum

def test_compute_d4000():
    assert True 

def test_compute_continuum_intensity():
    assert True 

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
    
    return

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

    return


if __name__ == "__main__":
    test_compute_continuum_luminosity()
    test_compute_EW()
    


