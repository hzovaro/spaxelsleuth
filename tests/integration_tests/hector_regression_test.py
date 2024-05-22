import numpy as np
import os
import pandas as pd
from pathlib import Path
import sys

if (len(sys.argv) > 1) and ("pytest" not in sys.modules):  # Needed to prevent errors when running pytest
    config_fname = sys.argv[1]
else:
    config_fname = "test_config.json"

from spaxelsleuth import load_user_config, configure_logger
load_user_config(config_fname)
configure_logger(level="INFO")
from spaxelsleuth.config import settings
from spaxelsleuth.io.io import make_df, load_df, find_matching_files
from spaxelsleuth.config import settings

import logging
logger = logging.getLogger(__name__)


def delete_test_dataframes(**kwargs):
    """Delete the dataframes created in run_hector_assertion_tests."""
    output_fnames = find_matching_files(output_path=settings["hector"]["output_path"], **kwargs)
    for fname in output_fnames:
        logger.warning(f"deleting file {settings['hector']['output_path']}{fname}...")
        os.system(f"rm {settings['hector']['output_path']}/{fname}")


def test_regression_hector():
    """Compare current and reference DataFrames to check that they are the same."""
    for ncomponents in ["rec"]:
        for bin_type in ["default"]:
            logger.info(f"running regression tests for hector DataFrame with ncomponents={ncomponents}, bin_type={bin_type}, extinction_correction=True...")
            run_hector_regression_tests(ncomponents=ncomponents, bin_type=bin_type, correct_extinction=True)
            logger.info(f"running regression tests for hector DataFrame with ncomponents={ncomponents}, bin_type={bin_type}, extinction_correction=False...")
            run_hector_regression_tests(ncomponents=ncomponents, bin_type=bin_type, correct_extinction=False)
            logger.info(f"regression tests passed for hector DataFrame with ncomponents={ncomponents}, bin_type={bin_type}!")


def run_hector_regression_tests(
    ncomponents,
    bin_type,
    eline_SNR_min=5,
    eline_ANR_min=3,
    nthreads=10,
    correct_extinction=True,
    debug=False,
):
    """Run make_df for the given inputs and compare the output against a reference DataFrame."""

    kwargs = {
        "survey": "hector",
        "ncomponents": ncomponents,
        "bin_type": bin_type,
        "eline_SNR_min": eline_SNR_min,
        "eline_ANR_min": eline_ANR_min,
        "correct_extinction": correct_extinction,
        "debug": debug,
        "metallicity_diagnostics": ["N2Ha_PP04", "N2Ha_K19"],   
    }

    # First, delete any existing files 
    delete_test_dataframes(**kwargs)

    # Create the DataFrame
    make_df(**kwargs, nthreads=nthreads)

    # Load the newly-create and reference DataFrames
    df_reference, _ = load_df(**kwargs, output_path=settings["hector"]["reference_output_path"])
    df_new, _ = load_df(**kwargs)    

    # Compare
    compare_dataframes(df_new, df_reference)

    logger.info(f"Comparison test passed")

    return


def compare_dataframes(df_new, df_old):
    """Compare df_new with df_reference and raise AssertionErrors if there are any differences."""
    
    # Check that there are no columns or rows that are different
    # NOTE: bin_type, survey etc. columns will NOT be in df_new because these get added at runtime
    removed_cols = [c for c in df_old if c not in df_new]
    added_cols = [c for c in df_new if c not in df_old]
    assert len(removed_cols) == 0, f"The following columns in df_old are missing in df_new: {', '.join(removed_cols)}"
    assert len(added_cols) == 0, f"The following columns in df_new do not exist in df_old: {', '.join(added_cols)}"
    assert all(df_old.index == df_new.index), f"The indices of df_old and df_new do not match!"

    # Check that shapes are the same
    assert df_old.shape == df_new.shape, f"df_new has shape {df_new.shape} but df_old has shape {df_old.shape}!"

    # Since the order of the columns might change, we need to check them one by one.
    for c in [c for c in df_old.columns if c not in ["timestamp", "df_fname_tag"]]:
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
    test_regression_hector()