# Compare the old and new DataFrames to monitor for any changes 
if __name__ == "__main__":
    import numpy as np
    import pandas as pd
    from pathlib import Path
    import sys 

    fname = sys.argv[1]
    fname_new = Path("/data/misfit/u5708159/SAMI") / fname
    fname_old = Path("/data/misfit/u5708159/SAMI/reference") / fname

    if fname == "sami_dr3_aperture_snrs.hd5":
        df_new = pd.read_hdf(fname_new, key="SNR")
        df_old = pd.read_hdf(fname_old, key="SNR")
    else:
        df_new = pd.read_hdf(fname_new)
        df_old = pd.read_hdf(fname_old)

    # Check that shapes are the same 
    assert df_old.shape == df_new.shape

    # Check that there are no columns or rows that are different
    # NOTE: bin_type, survey etc. columns will NOT be in df_new because these get added at runtime
    assert len([c for c in df_old if c not in df_new]) == 0
    assert len([c for c in df_new if c not in df_old]) == 0
    assert all(df_old.index == df_new.index)

    # Since the order of the columns might change, we need to check them one by one.
    # We don't consider metallicity/ionisation parameter columns because these are computed in a non-deterministic way and won't agree between runs.
    cols_where_nans_are_inconsistent = []
    cols_where_values_are_different = []
    for c in [c for c in df_old if "log(O/H) + 12" not in c and "log(U)" not in c]:

        # Check that values agree 
        # cond_not_nan = ~df_old[c].isna()
        if df_old[c].dtype != "object":

            # Check that the NaNs agree
            if not all(df_new[c].isna() == df_old[c].isna()):
                cols_where_nans_are_inconsistent.append(c)

            cond_new_is_nan = (df_new[c].isna()) & (~df_old[c].isna())
            cond_old_is_nan = (~df_new[c].isna()) & (df_old[c].isna())
            cond_both_are_nan =  (df_new[c].isna()) & (df_old[c].isna())
            cond_neither_are_nan = (~df_new[c].isna()) & (~df_old[c].isna())
            cond_neither_are_nan_and_they_dont_agree = cond_neither_are_nan & (~np.isclose(df_new[c], df_old[c]))
            cond_neither_are_nan_and_they_agree = cond_neither_are_nan & (np.isclose(df_new[c], df_old[c]))

            if not np.allclose(df_new.loc[cond_neither_are_nan, c].values, df_old.loc[cond_neither_are_nan, c].values):
                cols_where_values_are_different.append(c)

        else:
            if not df_old[c].equals(df_new[c]):
                cols_where_values_are_different.append(c)

    if len(cols_where_values_are_different) == 0 and len(cols_where_nans_are_inconsistent) == 0:
        print(f"Comparison test passed ({fname})")
    else:
        raise ValueError(f"Comparison test FAILED ({fname})")
