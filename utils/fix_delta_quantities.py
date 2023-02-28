import datetime
import pandas as pd 
import os 
from itertools import product
from spaxelsleuth.utils.dqcut import compute_component_offsets
from IPython.core.debugger import Tracer

################################################################################
# Paths
sami_data_path = os.environ["SAMI_DIR"]
assert "SAMI_DIR" in os.environ, "Environment variable SAMI_DIR is not defined!"

# Path for LZIFU data products
__lzifu_products_path = "/priv/sami/sami_data/Final_SAMI_data/LZIFU/lzifu_default_products_old"

################################################################################
# Fix annoying mistake I made in compute_component_offsets() whereby "delta" quantities were the wrong way around...
bin_type_list = ["default", "adaptive", "sectors"]
ncomponents_list = ["recom", "1"]
correct_extinction_list = [True, False]
debug_list = [True, False]
__use_lzifu_fits_list = [False, True]
__lzifu_ncomponents_list = ["1", "2", "3", "recom"]
eline_SNR_min = 5

################################################################################
# Aperture fits 


################################################################################
__use_lzifu_fits = False
for bin_type, ncomponents, correct_extinction, debug in product(bin_type_list, ncomponents_list, correct_extinction_list, debug_list):

    assert (ncomponents == "recom") | (ncomponents == "1"), "ncomponents must be 'recom' or '1'!!"
    assert bin_type in ["default", "adaptive", "sectors"], "bin_type must be 'default' or 'adaptive' or 'sectors'!!"
    if __use_lzifu_fits:
        assert __lzifu_ncomponents in ["recom", "1", "2", "3"], "__lzifu_ncomponents must be 'recom', '1', '2' or '3'!!"
        assert os.path.exists(__lzifu_products_path), f"lzifu_products_path directory {__lzifu_products_path} not found!!"
        print(f"WARNING: using LZIFU {__lzifu_ncomponents}-component fits to obtain emission line fluxes & kinematics, NOT DR3 data products!!")

    # Input file name 
    df_fname = f"sami_{bin_type}_{ncomponents}-comp"
    if correct_extinction:
        df_fname += "_extcorr"
    df_fname += f"_minSNR={eline_SNR_min}"
    if __use_lzifu_fits:
        df_fname += f"_lzifu_{__lzifu_ncomponents}-comp"
    if debug:
        df_fname += "_DEBUG"
    df_fname += ".hd5"

    assert os.path.exists(os.path.join(sami_data_path, df_fname)),\
        f"File {os.path.join(sami_data_path, df_fname)} does does not exist!"

    # Load the data frame
    t = os.path.getmtime(os.path.join(sami_data_path, df_fname))
    print(f"In load_sami_df(): Loading DataFrame from file {os.path.join(sami_data_path, df_fname)} [last modified {datetime.datetime.fromtimestamp(t)}]...")
    df = pd.read_hdf(os.path.join(sami_data_path, df_fname))

    # Fix columns... 
    bad_cols =  [c for c in df if "delta sigma_gas" in c]
    bad_cols += [c for c in df if "delta v_gas" in c]
    bad_cols += [c for c in df if "HALPHA EW ratios" in c]
    bad_cols += [c for c in df if "Delta HALPHA EWs" in c]
    for col in ["log O3", "log N2", "log S2", "log O1"]:
        bad_cols += [c for c in df if f"delta {col}" in c]

    df = df.drop(bad_cols, axis=1)

    df = compute_component_offsets(df)

    # Save back to file 
    print(f"In load_sami_df(): Writing DataFrame to file {os.path.join(sami_data_path, df_fname)} [last modified {datetime.datetime.fromtimestamp(t)}]...")
    df.to_csv(os.path.join(sami_data_path, df_fname.split("hd5")[0] + "csv"))
    try:
        df.to_hdf(os.path.join(sami_data_path, df_fname), key=f"{bin_type}, {ncomponents}-comp")
    except:
        print(f"Unable to save to HDF file... sigh...")

################################################################################
__use_lzifu_fits = True
for bin_type, ncomponents, correct_extinction, debug, __lzifu_ncomponents in product(bin_type_list, ncomponents_list, correct_extinction_list, debug_list, __lzifu_ncomponents_list):
    print(f"bin_type={bin_type}, ncomponents={ncomponents}, correct_extinction={correct_extinction}, debug={debug}, __use_lzifu_fits={__use_lzifu_fits}, __lzifu_ncomponents={__lzifu_ncomponents}")

    assert (ncomponents == "recom") | (ncomponents == "1"), "ncomponents must be 'recom' or '1'!!"
    assert bin_type in ["default", "adaptive", "sectors"], "bin_type must be 'default' or 'adaptive' or 'sectors'!!"
    if __use_lzifu_fits:
        assert __lzifu_ncomponents in ["recom", "1", "2", "3"], "__lzifu_ncomponents must be 'recom', '1', '2' or '3'!!"
        assert os.path.exists(__lzifu_products_path), f"lzifu_products_path directory {__lzifu_products_path} not found!!"
        print(f"WARNING: using LZIFU {__lzifu_ncomponents}-component fits to obtain emission line fluxes & kinematics, NOT DR3 data products!!")

    # Input file name 
    df_fname = f"sami_{bin_type}_{ncomponents}-comp"
    if correct_extinction:
        df_fname += "_extcorr"
    df_fname += f"_minSNR={eline_SNR_min}"
    if __use_lzifu_fits:
        df_fname += f"_lzifu_{__lzifu_ncomponents}-comp"
    if debug:
        df_fname += "_DEBUG"
    df_fname += ".hd5"

    if not os.path.exists(os.path.join(sami_data_path, df_fname)):
        f"File {os.path.join(sami_data_path, df_fname)} does does not exist!"
    else:
        # Load the data frame
        t = os.path.getmtime(os.path.join(sami_data_path, df_fname))
        print(f"In load_sami_df(): Loading DataFrame from file {os.path.join(sami_data_path, df_fname)} [last modified {datetime.datetime.fromtimestamp(t)}]...")
        df = pd.read_hdf(os.path.join(sami_data_path, df_fname))

        # Fix columns... 
        bad_cols =  [c for c in df if "delta sigma_gas" in c]
        bad_cols += [c for c in df if "delta v_gas" in c]
        bad_cols += [c for c in df if "HALPHA EW ratios" in c]
        bad_cols += [c for c in df if "Delta HALPHA EWs" in c]
        for col in ["log O3", "log N2", "log S2", "log O1"]:
            bad_cols += [c for c in df if f"delta {col}" in c]

        df = df.drop(bad_cols, axis=1)

        df = compute_component_offsets(df)

        # Save back to file 
        print(f"In load_sami_df(): Writing DataFrame to file {os.path.join(sami_data_path, df_fname)} [last modified {datetime.datetime.fromtimestamp(t)}]...")
        df.to_csv(os.path.join(sami_data_path, df_fname.split("hd5")[0] + "csv"))
        try:
            df.to_hdf(os.path.join(sami_data_path, df_fname), key=f"{bin_type}, {ncomponents}-comp")
        except:
            print(f"Unable to save to HDF file... sigh...")

