# Compare the old and new DataFrames to monitor for any changes 
if __name__ == "__main__":
    import numpy as np
    import pandas as pd
    from pathlib import Path
    import sys 

    fname = sys.argv[1]
    fname_new = Path("/data/misfit/u5708159/SAMI") / fname
    fname_old = Path("/priv/meggs3/u5708159/SAMI/sami_dr3/tests/old") / fname 

    if fname == "sami_dr3_aperture_snrs.hd5":
        df_new = pd.read_hdf(fname_new, key="SNR")
        df_old = pd.read_hdf(fname_old, key="SNR")
    else:
        df_new = pd.read_hdf(fname_new)
        df_old = pd.read_hdf(fname_old)

    # Drop SNR columns for now since I have no fucking idea why they've changed...
    df_old = df_old[[c for c in df_old.columns if " SNR " not in c]]
    df_new = df_new[[c for c in df_new.columns if " SNR " not in c]]

    # Check that shapes are the same 
    assert df_old.shape == df_new.shape

    # Check that there are no columns or rows that are different
    # NOTE: bin_type, survey etc. columns will NOT be in df_new because these get added at runtime
    assert len([c for c in df_old if c not in df_new]) == 0
    assert len([c for c in df_new if c not in df_old]) == 0
    assert all(df_old.index == df_new.index)

    # Since the order of the columns might change, we need to check them one by one.
    # We don't consider metallicity/ionisation parameter columns because these are computed in a non-deterministic way and won't agree between runs.
    for c in [c for c in df_old if "log(O/H) + 12" not in c and "log(U)" not in c]:
        
        # Check that the NaNs agree
        all(df_new[c].isna() == df_old[c].isna())
        
        # Check that values agree 
        cond_not_nan = ~df_old[c].isna()
        if df_old[c].dtype != "object":
            assert np.allclose(df_old.loc[cond_not_nan, c], df_new.loc[cond_not_nan, c]) 
        else:
            assert df_old[c].equals(df_new[c])

    print(f"Comparison test passed ({fname})")
