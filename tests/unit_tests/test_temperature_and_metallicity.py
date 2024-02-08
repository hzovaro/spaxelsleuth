import numpy as np
import pandas as pd

from spaxelsleuth.utils.linefns import compute_electron_temperature, get_T_e_Proxauf2014

def test_compute_T_e():

    #//////////////////////////////////////////////////////
    # Test set 1: Proxauf+2014 [SII]-based measurements 
    # Test case 1: valid input 
    r = 2.5
    R = 10**r
    T_e_truth = 8729.1004842615
    T_e, lolim_mask, uplim_mask = get_T_e_Proxauf2014(np.array([R, R, R]))
    assert np.all(np.isclose(T_e, T_e_truth))
    assert not any(lolim_mask)
    assert not any(uplim_mask)

    # Test case 2: upper/lower limit saturation
    T_e, lolim_mask, uplim_mask = get_T_e_Proxauf2014(np.array([10**1.1, 10**0.9, 10**-1]))
    assert np.all(T_e == 24000)
    assert not any(lolim_mask)
    assert all(uplim_mask)

    # Lower limit 
    T_e, lolim_mask, uplim_mask = get_T_e_Proxauf2014(np.array([10**3.9, 10**4.0, 10**6]))
    assert np.all(T_e == 5000)
    assert all(lolim_mask)
    assert not any(uplim_mask)

    # Test case 3: NaNs
    T_e, lolim_mask, uplim_mask = get_T_e_Proxauf2014(np.array([np.nan, np.nan, np.nan]))
    assert all(np.isnan(T_e))
    assert not any(lolim_mask)
    assert not any(uplim_mask)

    # Test case 3: infs
    T_e, lolim_mask, uplim_mask = get_T_e_Proxauf2014(np.array([np.inf, np.inf, np.inf]))
    assert all(np.isnan(T_e))
    assert not any(lolim_mask)
    assert not any(uplim_mask)

    # Test case 4: test inside DataFrame 
    df = pd.DataFrame({
        "OIII4959+OIII5007 (total)": [10**2.5, 10**2.5, 10**2.5, 2.5, np.nan,],
        "OIII4363 (total)": [1.0, 10**2.5, 0.0000001, 0.0, np.nan,],
    })
    df["R"] = df["OIII4959+OIII5007 (total)"] / df["OIII4363 (total)"]
    df["log(R)"] = np.log10(df["R"])
    df_updated = compute_electron_temperature(df=df, diagnostic="Proxauf2014", s=" (total)")
    assert df_updated.loc[0, f"T_e (Proxauf2014 ([OIII])) (total)"] == T_e_truth
    assert df_updated.loc[1, f"T_e (Proxauf2014 ([OIII])) (total)"] == 24e3
    assert df_updated.loc[2, f"T_e (Proxauf2014 ([OIII])) (total)"] == 5e3
    assert np.isnan(df_updated.loc[3, f"T_e (Proxauf2014 ([OIII])) (total)"])
    assert np.isnan(df_updated.loc[4, f"T_e (Proxauf2014 ([OIII])) (total)"])

    # Test case 5: test w/o suffix
    df = pd.DataFrame({
        "OIII4959+OIII5007": [10**2.5, 10**2.5, 10**2.5, 2.5, np.nan,],
        "OIII4363": [1.0, 10**2.5, 0.0000001, 0.0, np.nan,],
    })
    df["R"] = df["OIII4959+OIII5007"] / df["OIII4363"]
    df["log(R)"] = np.log10(df["R"])
    df_updated = compute_electron_temperature(df=df, diagnostic="Proxauf2014")
    assert df_updated.loc[0, f"T_e (Proxauf2014 ([OIII]))"] == T_e_truth
    assert df_updated.loc[1, f"T_e (Proxauf2014 ([OIII]))"] == 24e3
    assert df_updated.loc[2, f"T_e (Proxauf2014 ([OIII]))"] == 5e3
    assert np.isnan(df_updated.loc[3, f"T_e (Proxauf2014 ([OIII]))"])
    assert np.isnan(df_updated.loc[4, f"T_e (Proxauf2014 ([OIII]))"])


if __name__ == "__main__":

    test_compute_T_e()

    import matplotlib.pyplot as plt 
    plt.ion()
    plt.close("all")
   
    R_vals = 10**np.linspace(1.0, 4.0, 1000)
    T_e, lolim_mask, uplim_mask = get_T_e_Proxauf2014(R_vals)

    fig, ax = plt.subplots()
    ax.plot(T_e, R_vals, color="k", ls="-", label="[O II]")
    ax.set_xlim(0, 30e3)
    ax.set_ylim(10**1.0, 10**4.0)
    ax.set_ylabel(r"$\log(R)$")
    ax.set_xlabel(r"$T_e$ (K)")
    ax.set_yscale("log")
    ax.legend()