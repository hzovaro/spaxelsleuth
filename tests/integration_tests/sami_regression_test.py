import numpy as np
import pandas as pd
from pathlib import Path

from spaxelsleuth import load_user_config, configure_logger
load_user_config("test_config.json")
configure_logger(level="INFO")
from spaxelsleuth.loaddata.sami import make_sami_df
from spaxelsleuth.config import settings

import logging
logger = logging.getLogger(__name__)


def test_regression_sami():
    """Compare current and reference DataFrames to check that they are the same."""
    for ncomponents in ["recom", "1"]:
        for bin_type in ["default", "adaptive", "sectors"]:
            logger.info(f"running regression tests for SAMI DataFrame with ncomponens={ncomponents}, bin_type={bin_type}...")
            run_sami_regression_tests(ncomponents=ncomponents, bin_type=bin_type)
            logger.info(f"regression tests passed for SAMI DataFrame with ncomponens={ncomponents}, bin_type={bin_type}!")


def run_sami_regression_tests(
    ncomponents,
    bin_type,
    eline_SNR_min=5,
    eline_ANR_min=3,
    nthreads=10,
    correct_extinction=True,
    debug=False,
):
    """Run make_sami_df and load_sami_df for the given inputs and compare the output against a reference DataFrame."""

    kwargs = {
        "ncomponents": ncomponents,
        "bin_type": bin_type,
        "eline_SNR_min": eline_SNR_min,
        "eline_ANR_min": eline_ANR_min,
        "correct_extinction": correct_extinction,
        "debug": debug,
    }

    # Create the DataFrame
    make_sami_df(**kwargs, nthreads=nthreads)

    # Get the filename
    df_fname = f"sami_{bin_type}_{ncomponents}-comp"
    if correct_extinction:
        df_fname += "_extcorr"
    df_fname += f"_minSNR={eline_SNR_min}_minANR={eline_ANR_min}"
    if debug:
        df_fname += "_DEBUG"
    df_fname += ".hd5"

    # Load the DataFrame. Note that we do not use load_sami_df() here because we don't need to check the extra columns added at runtime. 
    df_new = pd.read_hdf(
        Path(settings["sami"]["output_path"]) / df_fname,
        key=f"{bin_type}{ncomponents}comp",
    )

    # Load the reference DataFrame
    df_reference = pd.read_hdf(
        Path(settings["sami"]["reference_output_path"]) / df_fname,
        key=f"{bin_type}{ncomponents}comp",
    )

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
    for c in df_old.columns:
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
    run_sami_regression_tests(
        ncomponents="recom",
        bin_type="default",
        eline_SNR_min=5,
        eline_ANR_min=3,
        correct_extinction=True,
    )
