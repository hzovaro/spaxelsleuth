
import multiprocessing
import os
import pandas as pd
from pathlib import Path

from spaxelsleuth.config import settings
from spaxelsleuth.utils.addcolumns import add_columns

import logging
logger = logging.getLogger(__name__)

# TODO implement enum for surveys

def load_metadata_df():
    # TODO implement this 
    return 

def make_df(survey, 
            bin_type, 
            ncomponents,
            eline_SNR_min,
            eline_ANR_min,
            correct_extinction,
            sigma_gas_SNR_min=3,
            line_flux_SNR_cut=True,
            missing_fluxes_cut=True,
            missing_kinematics_cut=True,
            line_amplitude_SNR_cut=True,
            flux_fraction_cut=False,
            sigma_gas_SNR_cut=True,
            vgrad_cut=False,
            stekin_cut=True,
            metallicity_diagnostics=[
                "N2Ha_PP04",
                "N2Ha_M13",
                "O3N2_PP04",
                "O3N2_M13",
                "N2S2Ha_D16",
                "N2O2_KD02",
                "Rcal_PG16",
                "Scal_PG16",
                "ON_P10",
                "ONS_P10",
                "N2Ha_K19",
                "O3N2_K19",
                "N2O2_K19",
                "R23_KK04",
            ],
            debug=False,
            nthreads=None,
            **kwargs):
    """
    TODO write docstring
    """
    #######################################################################
    # Input checking
    #######################################################################
    assert ncomponents in settings[survey]["ncomponents"],\
        f"bin_type must be {' or '.join(settings[survey]['ncomponents'])}!!"
    assert bin_type in settings[survey]["bin_types"],\
        f"bin_type must be {' or '.join(settings[survey]['bin_types'])}!!"

    if survey == "sami" and kwargs["__use_lzifu_fits"]:
        assert kwargs["__lzifu_ncomponents"] in [
            "recom", "1", "2", "3"
        ], "__lzifu_ncomponents must be 'recom', '1', '2' or '3'!!"
        assert os.path.exists(
            __lzifu_products_path,
        ), f"lzifu_products_path directory {__lzifu_products_path} not found!!"
        logger.warning(
            "using LZIFU %s-component fits to obtain emission line fluxes & kinematics, NOT DR3 data products!!" % (__lzifu_ncomponents),
            RuntimeWarning)

    logger.info(f"input parameters: survey={survey}, bin_type={bin_type}, ncomponents={ncomponents}, debug={debug}, eline_SNR_min={eline_SNR_min}, eline_ANR_min={eline_ANR_min}, correct_extinction={correct_extinction}")

    # List of emission lines (do we need this here? Not really...)
    eline_list = settings[survey]["eline_list"]

    # Determine number of threads
    if nthreads is None:
        nthreads = os.cpu_count()
        logger.warning(f"nthreads not specified: running make_sami_df() on {nthreads} threads...")
   
    #######################################################################
    # Paths
    #######################################################################
    # input_path = Path(settings[survey]["input_path"])
    output_path = Path(settings[survey]["output_path"])
    # __lzifu_products_path = Path(settings[survey]["lzifu_products_path"])

    #######################################################################
    # Filenames
    #######################################################################
    df_metadata_fname = "sami_dr3_metadata.hd5"

    # Output file names
    if correct_extinction:
        df_fname = f"sami_{bin_type}_{ncomponents}-comp_extcorr_minSNR={eline_SNR_min}_minANR={eline_ANR_min}"
    else:
        df_fname = f"sami_{bin_type}_{ncomponents}-comp_minSNR={eline_SNR_min}_minANR={eline_ANR_min}"
    if survey == "sami" and kwargs["__use_lzifu_fits"]:
        df_fname += f"_lzifu_{kwargs['__use_lzifu_fits']}-comp"
    if debug:
        df_fname += "_DEBUG"
    df_fname += ".hd5"

    logger.info(f"saving to file {df_fname}...")

    ###############################################################################
    # Read metadata
    ###############################################################################
    try:
        df_metadata = pd.read_hdf(output_path / df_metadata_fname,
                                  key="metadata")
    except:
        raise FileNotFoundError(
            f"metadata DataFrame file not found ({output_path / df_metadata_fname}). Please run make_sami_metadata_df.py first!"
        )

    # Only include galaxies flagged as "good" & for which we have data
    if survey == "sami":
        gals = df_metadata[df_metadata["Good?"] == True].index.values
        input_path = Path(settings["sami"]["input_path"]) # TODO this is so clunky!!
        gals = [
            g for g in gals
            if os.path.exists(input_path / f"ifs/{g}/")
        ]
        if len(gals) == 0:
            raise FileNotFoundError(f"I could not find any galaxy data in {input_path / 'ifs'}!")
    else:
        gals = df_metadata.index.values

    # If running in DEBUG mode, run on a subset to speed up execution time
    if debug:
        gals_debug = gals[:10]
        for gal in [
                572402, 209807
        ]:  # Add these two galaxies because they are very distinctive, making debugging a bit easier (for me at least...)
            if gal in gals:
                gals_debug += [gal]
        gals = gals_debug
        # Also only run on a subset of metallicity diagnostics to speed up execution time
        metallicity_diagnostics = ["N2Ha_PP04", "N2Ha_K19"]

    # Cast to flaot to avoid issues around Object data types
    df_metadata["Good?"] = df_metadata["Good?"].astype("float")

    ###############################################################################
    # Scrape measurements for each galaxy from FITS files
    ###############################################################################
    args_list = [[
        gg, gal, ncomponents, bin_type, df_metadata,
        kwargs["__use_lzifu_fits"], __lzifu_ncomponents
    ] for gg, gal in enumerate(gals)]

    if len(gals) == 1:
        res_list = [_process_gals(args_list[0])]
    else:
        if nthreads > 1:
            logger.info(f"beginning pool...")
            pool = multiprocessing.Pool(
                min([nthreads, len(gal_ids_dq_cut)]))
            res_list = np.array((pool.map(_process_gals, args_list)), dtype=object)
            pool.close()
            pool.join()
        else:
            logger.info(f"running sequentially...")
            res_list = []
            for args in args_list:
                res = _process_gals(args)
                res_list.append(res)

    ###############################################################################
    # Convert to a Pandas DataFrame
    ###############################################################################
    rows_list_all = [r[0] for r in res_list]
    colnames = res_list[0][1]
    df_spaxels = pd.DataFrame(np.vstack(tuple(rows_list_all)),
                              columns=colnames)

    # Merge with metadata
    df_spaxels = df_spaxels.merge(df_metadata, on="ID", how="left")

    ###############################################################################
    # Add extra columns
    ###############################################################################
    df_spaxels["r/R_e"] = df_spaxels[
        "r (relative to galaxy centre, deprojected, arcsec)"] / df_spaxels[
            "R_e (arcsec)"]

    ###############################################################################
    # Generic stuff: compute additional columns - extinction, metallicity, etc.
    ###############################################################################
    df_spaxels = add_columns(df_spaxels,
                             eline_SNR_min=eline_SNR_min,
                             eline_ANR_min=eline_ANR_min,
                             sigma_gas_SNR_min=sigma_gas_SNR_min,
                             eline_list=eline_list,  # TODO read this from settings inside 
                             line_flux_SNR_cut=line_flux_SNR_cut,
                             missing_fluxes_cut=missing_fluxes_cut,
                             missing_kinematics_cut=missing_kinematics_cut,
                             line_amplitude_SNR_cut=line_amplitude_SNR_cut,
                             flux_fraction_cut=flux_fraction_cut,
                             sigma_gas_SNR_cut=sigma_gas_SNR_cut,
                             vgrad_cut=vgrad_cut,
                             stekin_cut=stekin_cut,  # TODO make SAMI-specific
                             correct_extinction=correct_extinction,
                             metallicity_diagnostics=metallicity_diagnostics,
                             compute_sfr=False,  # Do this based on which survey is selected 
                             flux_units=settings["sami"]["flux_units"], # TODO read this from settings inside 
                             sigma_inst_kms=settings["sami"]["sigma_inst_kms"], # TODO read this from settings inside 
                             nthreads=nthreads,
                             **kwargs,
                            #  __use_lzifu_fits=kwargs["__use_lzifu_fits"],
                            #  __lzifu_ncomponents=__lzifu_ncomponents
                             )

    ###############################################################################
    # Save
    ###############################################################################
    logger.info(f"saving to file {output_path / df_fname}...")

    # Remove object-type columns
    bad_cols = [c for c in df_spaxels if df_spaxels[c].dtype == "object"]
    if len(bad_cols) > 0:
        logger.warning(f"The following object-type columns are present in the DataFrame: {','.join(bad_cols)}")

    # Save
    df_spaxels.to_hdf(output_path / df_fname, key=f"{bin_type}{ncomponents}comp")
    
    logger.info("finished!")
    return


def load_df():


