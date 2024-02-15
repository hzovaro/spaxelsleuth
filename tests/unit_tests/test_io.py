import os
import pandas as pd
from pathlib import Path

from spaxelsleuth import load_user_config, configure_logger
load_user_config("test_config.json")
configure_logger(level="INFO")
from spaxelsleuth.config import settings
from spaxelsleuth.io.io_new import load_df, make_df, find_matching_files

import logging
logger = logging.getLogger(__name__)


def test_find_matching_files():
    """Unit test for find_matching_files()."""

    # TODO how to deal with situations where we end up making 100s of the same DataFrame & 
    # they all get saved as different copies thanks to the timestamp? 

    make_dataframes_again = False
    output_fnames = [f for f in os.listdir("../output/io/") if "metadata" not in f and "snrs" not in f]
    if make_dataframes_again:
    
        # First, delete all spaxelsleuth DataFrame files in the output directory.
        for fname in output_fnames:
            os.system(f"rm ../output/io/{fname}")

        # Make some DataFrames w/ dfferent settings 
            
        # 1: standard 
        # sami_default_recom-comp_minSNR=5_minANR=3_20240215120417.hd5
        make_df(survey="sami",
            bin_type="default", 
            ncomponents="recom", 
            eline_SNR_min=5, 
            eline_ANR_min=3, 
            correct_extinction=False,
            metallicity_diagnostics=["N2Ha_PP04", "N2Ha_K19"],
            line_flux_SNR_cut=True,
            line_amplitude_SNR_cut=True,
            sigma_gas_SNR_cut=True,
            stekin_cut=True,
            nthreads=10,
            )
        
        # 1: standard (but w/ different timestamp)
        # sami_default_recom-comp_minSNR=5_minANR=3_20240215120440.hd5
        make_df(survey="sami",
            bin_type="default", 
            ncomponents="recom", 
            eline_SNR_min=5, 
            eline_ANR_min=3, 
            correct_extinction=False,
            metallicity_diagnostics=["N2Ha_PP04", "N2Ha_K19"],
            line_flux_SNR_cut=True,
            line_amplitude_SNR_cut=True,
            sigma_gas_SNR_cut=True,
            stekin_cut=True,
            nthreads=10,
            )
        
        # 1: standard (cuts turned off)
        # sami_default_recom-comp_minSNR=5_minANR=3_20240215120505.hd5
        make_df(survey="sami",
            bin_type="default", 
            ncomponents="recom", 
            eline_SNR_min=5, 
            eline_ANR_min=3, 
            correct_extinction=False,
            metallicity_diagnostics=["N2Ha_PP04", "N2Ha_K19"],
            line_flux_SNR_cut=False,
            line_amplitude_SNR_cut=False,
            sigma_gas_SNR_cut=False,
            stekin_cut=False,
            nthreads=10,
            )
        
        # 2: same as standard but for specific galaxies 
        # sami_default_recom-comp_minSNR=5_minANR=3_20240215120523.hd5
        make_df(survey="sami",
            bin_type="default", 
            ncomponents="recom", 
            eline_SNR_min=5, 
            eline_ANR_min=3, 
            correct_extinction=False,
            metallicity_diagnostics=["N2Ha_PP04", "N2Ha_K19"],
            gals=[572402, 209807],
            line_flux_SNR_cut=True,
            line_amplitude_SNR_cut=True,
            sigma_gas_SNR_cut=True,
            stekin_cut=True,
            nthreads=10,
            )

        # 3: same as standard but w/ name tag 
        # sami_default_recom-comp_minSNR=5_minANR=3_special_20240215120544.hd5
        make_df(survey="sami",
            bin_type="default", 
            ncomponents="recom", 
            eline_SNR_min=5, 
            eline_ANR_min=3, 
            correct_extinction=False,
            metallicity_diagnostics=["N2Ha_PP04", "N2Ha_K19"],
            line_flux_SNR_cut=True,
            line_amplitude_SNR_cut=True,
            sigma_gas_SNR_cut=True,
            stekin_cut=True,
            df_fname_tag="special",
            nthreads=10,
            )
        
        # 4: same as standard but w/ different Z diagnostic
        # sami_default_recom-comp_minSNR=5_minANR=3_20240215120557.hd5
        make_df(survey="sami",
            bin_type="default", 
            ncomponents="recom", 
            eline_SNR_min=5, 
            eline_ANR_min=3, 
            correct_extinction=False,
            metallicity_diagnostics=["N2S2Ha_D16"],
            line_flux_SNR_cut=True,
            line_amplitude_SNR_cut=True,
            sigma_gas_SNR_cut=True,
            stekin_cut=True,
            nthreads=10,
            )
        
    output_path = Path(settings["sami"]["output_path"])
    
    #//////////////////////////////////////////////////////////////
    # Test: try to load DF #1 
    files = find_matching_files(output_path,
                                survey="sami",
                                bin_type="default", 
                                ncomponents="recom", 
                                eline_SNR_min=5, 
                                eline_ANR_min=3, 
                                correct_extinction=False,
                                metallicity_diagnostics=["N2Ha_PP04", "N2Ha_K19"],  # TODO how does it deal with lists?
                                line_flux_SNR_cut=True,
                                line_amplitude_SNR_cut=True,
                                sigma_gas_SNR_cut=True,
                                stekin_cut=True,)
    files.sort()
    files_truth = [
        "sami_default_recom-comp_minSNR=5_minANR=3_20240215120417.hd5",
        "sami_default_recom-comp_minSNR=5_minANR=3_20240215120440.hd5",
        # "sami_default_recom-comp_minSNR=5_minANR=3_20240215120505.hd5",
        "sami_default_recom-comp_minSNR=5_minANR=3_20240215120523.hd5",
        "sami_default_recom-comp_minSNR=5_minANR=3_special_20240215120544.hd5",
        # "sami_default_recom-comp_minSNR=5_minANR=3_20240215120557.hd5",
    ]
    files_truth.sort()
    assert files == files_truth
    
    # Test: try to load DF #1, Z diags in different order
    files = find_matching_files(output_path,
                                survey="sami",
                                bin_type="default", 
                                ncomponents="recom", 
                                eline_SNR_min=5, 
                                eline_ANR_min=3, 
                                correct_extinction=False,
                                metallicity_diagnostics=["N2Ha_K19", "N2Ha_PP04", ],  # TODO how does it deal with lists?
                                line_flux_SNR_cut=True,
                                line_amplitude_SNR_cut=True,
                                sigma_gas_SNR_cut=True,
                                stekin_cut=True,)
    files.sort()
    files_truth = [
        "sami_default_recom-comp_minSNR=5_minANR=3_20240215120417.hd5",
        "sami_default_recom-comp_minSNR=5_minANR=3_20240215120440.hd5",
        # "sami_default_recom-comp_minSNR=5_minANR=3_20240215120505.hd5",
        "sami_default_recom-comp_minSNR=5_minANR=3_20240215120523.hd5",
        "sami_default_recom-comp_minSNR=5_minANR=3_special_20240215120544.hd5",
        # "sami_default_recom-comp_minSNR=5_minANR=3_20240215120557.hd5",
    ]
    files_truth.sort()
    assert files == files_truth


    # Test: don't specify metallicity diagnostics 
    files = find_matching_files(output_path,
                                survey="sami",
                                bin_type="default", 
                                ncomponents="recom", 
                                eline_SNR_min=5, 
                                eline_ANR_min=3, 
                                correct_extinction=False,
                                line_flux_SNR_cut=True,
                                line_amplitude_SNR_cut=True,
                                sigma_gas_SNR_cut=True,
                                stekin_cut=True,)
    files.sort()
    files_truth = [
        "sami_default_recom-comp_minSNR=5_minANR=3_20240215120417.hd5",
        "sami_default_recom-comp_minSNR=5_minANR=3_20240215120440.hd5",
        # "sami_default_recom-comp_minSNR=5_minANR=3_20240215120505.hd5",
        "sami_default_recom-comp_minSNR=5_minANR=3_20240215120523.hd5",
        "sami_default_recom-comp_minSNR=5_minANR=3_special_20240215120544.hd5",
        "sami_default_recom-comp_minSNR=5_minANR=3_20240215120557.hd5",
    ]
    files_truth.sort()
    assert files == files_truth


    # Test: only specify one metallicity diagnostic
    files = find_matching_files(output_path,
                                survey="sami",
                                bin_type="default", 
                                ncomponents="recom", 
                                eline_SNR_min=5, 
                                eline_ANR_min=3, 
                                correct_extinction=False,
                                metallicity_diagnostics=["N2Ha_K19"],
                                line_flux_SNR_cut=True,
                                line_amplitude_SNR_cut=True,
                                sigma_gas_SNR_cut=True,
                                stekin_cut=True,)
    files.sort()
    files_truth = [
        "sami_default_recom-comp_minSNR=5_minANR=3_20240215120417.hd5",
        "sami_default_recom-comp_minSNR=5_minANR=3_20240215120440.hd5",
        # "sami_default_recom-comp_minSNR=5_minANR=3_20240215120505.hd5",
        "sami_default_recom-comp_minSNR=5_minANR=3_20240215120523.hd5",
        "sami_default_recom-comp_minSNR=5_minANR=3_special_20240215120544.hd5",
        # "sami_default_recom-comp_minSNR=5_minANR=3_20240215120557.hd5",
    ]
    files_truth.sort()
    assert files == files_truth


    # Test 2: find a file we know doesn't exist 
    files = find_matching_files(output_path,
                                survey="sami",
                                bin_type="default", 
                                ncomponents="recom", 
                                eline_SNR_min=5, 
                                eline_ANR_min=3, 
                                correct_extinction=True,
                                metallicity_diagnostics=["N2Ha_K19", "N2Ha_PP04", ],  # TODO how does it deal with lists?
                                line_flux_SNR_cut=True,
                                line_amplitude_SNR_cut=True,
                                sigma_gas_SNR_cut=True,
                                stekin_cut=True,)
    assert len(files) == 0

    # Test:  match on galaxies 
    files = find_matching_files(output_path,
                                gals=[572402],
                                )
    files.sort()
    files_truth = [
        "sami_default_recom-comp_minSNR=5_minANR=3_20240215120417.hd5",
        "sami_default_recom-comp_minSNR=5_minANR=3_20240215120440.hd5",
        "sami_default_recom-comp_minSNR=5_minANR=3_20240215120505.hd5",
        "sami_default_recom-comp_minSNR=5_minANR=3_20240215120523.hd5",
        "sami_default_recom-comp_minSNR=5_minANR=3_special_20240215120544.hd5",
        "sami_default_recom-comp_minSNR=5_minANR=3_20240215120557.hd5",
    ]
    files_truth.sort()
    assert files == files_truth


    # Test: load dataframe at specific name tag w/o identifying any other information
    files = find_matching_files(output_path,
                                df_fname_tag="special",
                                )
    files.sort()
    files_truth = [
        # "sami_default_recom-comp_minSNR=5_minANR=3_20240215120417.hd5",
        # "sami_default_recom-comp_minSNR=5_minANR=3_20240215120440.hd5",
        # "sami_default_recom-comp_minSNR=5_minANR=3_20240215120505.hd5",
        # "sami_default_recom-comp_minSNR=5_minANR=3_20240215120523.hd5",
        "sami_default_recom-comp_minSNR=5_minANR=3_special_20240215120544.hd5",
        # "sami_default_recom-comp_minSNR=5_minANR=3_20240215120557.hd5",
    ]
    files_truth.sort()
    assert files == files_truth

    # Test: load dataframe with specific timestamp w/o identifying any other information
    files = find_matching_files(output_path,
                                timestamp="20240215120417",
                                )
    files.sort()
    files_truth = [
        "sami_default_recom-comp_minSNR=5_minANR=3_20240215120417.hd5",
        # "sami_default_recom-comp_minSNR=5_minANR=3_20240215120440.hd5",
        # "sami_default_recom-comp_minSNR=5_minANR=3_20240215120505.hd5",
        # "sami_default_recom-comp_minSNR=5_minANR=3_20240215120523.hd5",
        # "sami_default_recom-comp_minSNR=5_minANR=3_special_20240215120544.hd5",
        # "sami_default_recom-comp_minSNR=5_minANR=3_20240215120557.hd5",
    ]
    files_truth.sort()
    assert files == files_truth

    logger.info("All tests passed!")


