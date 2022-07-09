"""
File:       sami.py
Author:     Henry Zovaro
Email:      henry.zovaro@anu.edu.au

DESCRIPTION
------------------------------------------------------------------------------
This script contains the function load_sami_galaxies() which returns the Pandas 
DataFrame containing spaxel-by-spaxel information for all SAMI galaxies that 
was created in make_df_sami.py.

USAGE
------------------------------------------------------------------------------
Run from the command line as follows:

    >>> from spaxelsleuth.loaddata.sami import load_sami_galaxies
    >>> df = load_sami_galaxies(ncomponents, bin_type, correct_extinction, 
                                eline_SNR_min, debug)

OUTPUTS
------------------------------------------------------------------------------
The Dataframe.

PREREQUISITES
------------------------------------------------------------------------------
SAMI_DIR and must be defined as an environment variable.

Both make_df_sami.py and make_sami_metadata_df.py must be run first.

------------------------------------------------------------------------------
Copyright (C) 2022 Henry Zovaro
"""
###############################################################################
import os
import pandas as pd

###############################################################################
# Paths
assert "SAMI_DIR" in os.environ, "Environment variable SAMI_DIR is not defined!"
sami_data_path = os.environ["SAMI_DIR"]

###############################################################################
def load_sami_galaxies(ncomponents, bin_type, correct_extinction, eline_SNR_min,
                       debug=False):

    #######################################################################
    # INPUT CHECKING
    #######################################################################
    assert (ncomponents == "recom") | (ncomponents == "1"), "ncomponents must be 'recom' or '1'!!"
    assert bin_type in ["default", "adaptive", "sectors"], "bin_type must be 'default' or 'adaptive' or 'sectors'!!"

    # Input file name 
    df_fname = f"sami_{bin_type}_{ncomponents}-comp"
    if correct_extinction:
        df_fname += "_extcorr"
    df_fname += f"_minSNR={eline_SNR_min}"
    if debug:
        df_fname += "_DEBUG"
    df_fname += ".hd5"

    assert os.path.exists(os.path.join(sami_data_path, df_fname)),\
        f"File {os.path.join(sami_data_path, df_fname)} does does not exist!"

    # Load the data frame
    df = pd.read_hdf(os.path.join(sami_data_path, df_fname))

    # Return
    return df.sort_index()
