import numpy as np
import pandas as pd


def compare_dataframes(df_new, df_old):
    """Compare df_new with df_reference and raise AssertionErrors if there are any differences."""
    
    # Check that there are no columns or rows that are different
    # NOTE: bin_type, survey etc. columns will NOT be in df_new because these get added at runtime
    removed_cols = [c for c in df_old if c not in df_new]
    added_cols = [c for c in df_new if c not in df_old]
    # assert len(removed_cols) == 0, f"The following columns in df_old are missing in df_new: {', '.join(removed_cols)}"
    # assert len(added_cols) == 0, f"The following columns in df_new do not exist in df_old: {', '.join(added_cols)}"
    assert all(df_old.index == df_new.index), f"The indices of df_old and df_new do not match!"

    if len(added_cols) > 0:
        df_new = df_new[[c for c in df_new if c in df_old]]
    if len(removed_cols) > 0:
        df_old = df_old[[c for c in df_new if c in df_old]]

    # Check that shapes are the same
    assert df_old.shape == df_new.shape, f"df_new has shape {df_new.shape} but df_old has shape {df_old.shape}!"
    
    # Since the order of the columns might change, we need to check them one by one.
    for c in df_old.columns:
        print(f"Checking column {c}...")
        if df_old[c].dtype != "object":

            # Check that the NaNs agree
            cond_new_is_nan = (df_new[c].isna()) & (~df_old[c].isna())
            assert not any(cond_new_is_nan), f"In column {c}: there are NaN entries in df_new that are non-NaN in df_old!"
            cond_old_is_nan = (~df_new[c].isna()) & (df_old[c].isna())
            assert not any(cond_old_is_nan), f"In column {c}: there are non-NaN entries in df_new that are NaN in df_old!"

            # Check equivalence for numerical values 
            cond_neither_are_nan = (~df_new[c].isna()) & (~df_old[c].isna())
            assert np.allclose(
                df_new.loc[cond_neither_are_nan, c].values,
                df_old.loc[cond_neither_are_nan, c].values,
            ), f"In column {c}: there are non-NaN entries in df_new and df_old that disagree!"

        else:
            assert df_old[c].equals(df_new[c]), f"In column {c}: there are entries that do not agree!"

    return


if __name__ == "__main__":
    # Using the same input params, make DataFrames using the methods in io.py and io_new.py
    # Load them side-by-side to check what has changed.

    from spaxelsleuth import load_user_config
    load_user_config("/home/u5708159/.spaxelsleuthconfig.json")
    from spaxelsleuth.io import io, io_new

    nthreads = 10
    DEBUG = True

    # Create the DataFrames using the old & new methods
    # io.make_df(survey="sami",
    #         bin_type="default", 
    #         ncomponents="recom", 
    #         eline_SNR_min=5, 
    #         eline_ANR_min=3, 
    #         correct_extinction=True,
    #         metallicity_diagnostics=["N2Ha_PP04", "N2Ha_K19"],
    #         nthreads=nthreads,
    #         debug=DEBUG)
    # io_new.make_df(survey="sami",
    #         bin_type="default", 
    #         ncomponents="recom", 
    #         eline_SNR_min=5, 
    #         eline_ANR_min=3, 
    #         correct_extinction=True,
    #         metallicity_diagnostics=["N2Ha_PP04", "N2Ha_K19"],
    #         nthreads=nthreads,
    #         debug=DEBUG)
    
    # Load 
    df_old = io.load_df(survey="sami",
            bin_type="default", 
            ncomponents="recom", 
            eline_SNR_min=5, 
            eline_ANR_min=3, 
            correct_extinction=True,
            debug=DEBUG)
    
    df_new = io_new.load_df(survey="sami",
            bin_type="default", 
            ncomponents="recom", 
            eline_SNR_min=5, 
            eline_ANR_min=3, 
            correct_extinction=True,
            debug=DEBUG)
    
    
    # Run a regression test here
    df_old = df_old.sort_values(by=["ID", "x (projected, arcsec)", "y (projected, arcsec)"]).reset_index(drop=True)
    # df_new = df_new.sort_values(by=["ID", "x (projected, arcsec)", "y (projected, arcsec)"]).reset_index(drop=True)
    compare_dataframes(df_new=df_new, df_old=df_old)


