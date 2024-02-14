import datetime
from importlib import import_module
import multiprocessing
import numpy as np
import os
import pandas as pd
from pathlib import Path

from spaxelsleuth.config import settings
from spaxelsleuth.utils.addcolumns import add_columns
from spaxelsleuth.utils.misc import morph_num_to_str
from spaxelsleuth.utils.linefns import bpt_num_to_str

import logging
logger = logging.getLogger(__name__)


def make_metadata_df(survey, **kwargs):
    """Create a "metadata" DataFrame.
    """
    if survey in ["hector", "sami", "s7"]:
        import_module(f"spaxelsleuth.io.{survey}").make_metadata_df(**kwargs)
    else:
        logger.warning(f"make_metadata_df() has not been implemented for survey {survey}!")


def get_df_fname(survey,
                 bin_type,
                 ncomponents,
                 correct_extinction,
                 eline_SNR_min,
                 eline_ANR_min,
                 debug,
                 df_fname_tag=None,
                 **kwargs):
    """Returns the DataFrame filename corresponding to a set of input parameters."""
    df_fname = f"{survey}_{bin_type}_{ncomponents}-comp"
    if correct_extinction:
        df_fname += f"_extcorr"
    df_fname += f"_minSNR={eline_SNR_min}_minANR={eline_ANR_min}"
    if survey == "sami":
        if "__use_lzifu_fits" in kwargs and "__lzifu_ncomponents" in kwargs:
            if kwargs["__use_lzifu_fits"]:
                df_fname += f"_lzifu_{kwargs['__lzifu_ncomponents']}-comp"
    if debug:
        df_fname += "_DEBUG"
    if df_fname_tag is not None:
        df_fname += f"_{df_fname_tag}"
    df_fname += ".hd5"
    return df_fname


def make_df(survey, 
            bin_type, 
            ncomponents,
            eline_SNR_min,
            eline_ANR_min,
            correct_extinction,
            gals=None,
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
            df_fname_tag=None,
            **kwargs):
    """Make a spaxelsleuth DataFrame, where each row represents a single spaxel in a SAMI galaxy.

    DESCRIPTION
    ---------------------------------------------------------------------------
    This function is used to create a Pandas DataFrame containing emission line 
    fluxes & kinematics, stellar kinematics, extinction, star formation rates, 
    and other quantities for individual spaxels in SAMI galaxies as taken from 
    SAMI DR3.

    The output is stored in HDF format as a Pandas DataFrame in which each row 
    corresponds to a given spaxel (or Voronoi/sector bin) for every galaxy. 

    USAGE
    ---------------------------------------------------------------------------
    
        >>> from spaxelsleuth.io.io import make_df()
        >>> make_df(survey="sami",ncomponents="1", bin_type="default", 
                         eline_SNR_min=5, eline_ANR_min=3, correct_extinction=True)

    will create a DataFrame using the data products from 1-component Gaussian 
    fits to the unbinned datacubes, and will adopt minimum S/N and A/N 
    thresholds of 5 and 3 respectively to mask out unreliable emission line 
    fluxes and associated quantities. sigma_inst_kms refers to the Gaussian 
    sigma of the instrumental line function in km/s.    

    Other input arguments may be configured to control other aspects of the data 
    quality and S/N cuts made.

    Running this function on the full sample takes some time (~10-20 minutes 
    when threaded across 20 threads). Execution time can be sped up by tweaking 
    the nthreads parameter. 

    If you wish to run in debug mode, set the DEBUG flag to True: this will run 
    the script on a subset (by default 10) galaxies to speed up execution. 

    INPUTS
    ---------------------------------------------------------------------------
    survey:                     str
        Survey name, e.g. "sami", "hector". Must be an entry in settings.
    
    ncomponents:                str
        Controls which data products are used, depending on the number of 
        Gaussian components fitted to the emission lines. 
        Must be an entry in settings[survey]["ncomponents"].
        
        NOTE (SAMI only): if __use_lzifu_fits is True, then ncomponents is 
        ONLY used in  loading data products that are NOT contained in the 
        output LZIFU files - i.e., SFRs/SFR surface densities and HALPHA 
        extinction correction factors. Use parameter __lzifu_ncomponents to 
        control which data derived from the LZIFU fits is loaded. 

    bin_type:                   str
        Spatial binning strategy. Must be an entry in 
        settings[survey]["bin_types"].

    eline_SNR_min:              int 
        Minimum emission line flux S/N to adopt when making S/N and data 
        quality cuts.

    eline_ANR_min:          float
        Minimum A/N to adopt for emission lines in each kinematic component,
        defined as the Gaussian amplitude divided by the continuum standard
        deviation in a nearby wavelength range.

    correct_extinction:         bool
        If True, correct emission line fluxes for extinction. 

        Note that the Halpha equivalent widths are NOT corrected for extinction if 
        correct_extinction is True. This is because stellar continuum extinction 
        measurements are not available, and so applying the correction only to the 
        Halpha fluxes may over-estimate the true EW.

    sigma_gas_SNR_min:          float (optional)
        Minimum velocity dipersion S/N to accept. Defaults to 3.

    line_flux_SNR_cut:          bool (optional)
        Whether to NaN emission line components AND total fluxes 
        (corresponding to emission lines in eline_list) below a specified S/N 
        threshold, given by eline_SNR_min. The S/N is simply the flux dividied 
        by the formal 1sigma uncertainty on the flux. Default: True.

    missing_fluxes_cut:         bool (optional)
        Whether to NaN out "missing" fluxes - i.e., cells in which the flux
        of an emission line (total or per component) is NaN, but the error 
        is not for some reason. Default: True.

    missing_kinematics_cut: bool
        Whether to NaN out "missing" values for v_gas/sigma_gas/v_*/sigma_* - 
        i.e., cells in which the measurement itself is NaN, but the error 
        is not for some reason. Default: True.

    line_amplitude_SNR_cut:     bool (optional)
        If True, removes components with Gaussian amplitudes < 3 * RMS of the 
        continuum in the vicinity of Halpha. By default this is set to True
        because this does well at removing shallow components which are most 
        likely due to errors in the stellar continuum fit. Default: True.

    flux_fraction_cut:          bool (optional)
        If True, and if ncomponents > 1, remove intermediate and broad 
        components with line amplitudes < 0.05 that of the narrow componet.
        Set to False by default b/c it's unclear whether this is needed to 
        reject unreliable components. Default: False.

    sigma_gas_SNR_cut:          bool (optional)
        If True, mask component velocity dispersions where the S/N on the 
        velocity dispersion measurement is below sigma_gas_SNR_min. 
        By default this is set to True as it's a robust way to account for 
        emission line widths < instrumental.  Default: True.

    vgrad_cut:                  bool (optional)     
        If True, mask component kinematics (velocity and velocity dispersion)
        that are likely to be affected by beam smearing.
        By default this is set to False because it tends to remove nuclear spaxels 
        which may be of interest to your science case, & because it doesn't 
        reliably remove spaxels with quite large beam smearing components.
        Default: False.

    stekin_cut:                 bool (optional)
        If True, mask stellar kinematic quantities that do not meet the DQ and 
        S/N requirements specified in Croom et al. (2021). Default: True.

    metallicity_diagnostics:    list of str (optional)
        List of strong-line metallicity diagnostics to compute. 
        Options:
            N2Ha_K19    N2Ha diagnostic from Kewley (2019).
            S2Ha_K19    S2Ha diagnostic from Kewley (2019).
            N2S2_K19    N2S2 diagnostic from Kewley (2019).
            S23_K19     S23 diagnostic from Kewley (2019).
            O3N2_K19    O3N2 diagnostic from Kewley (2019).
            O2S2_K19    O2S2 diagnostic from Kewley (2019).
            O2Hb_K19    O2Hb diagnostic from Kewley (2019).
            N2O2_K19    N2O2 diagnostic from Kewley (2019).
            R23_K19     R23 diagnostic from Kewley (2019).
            N2Ha_PP04   N2Ha diagnostic from Pilyugin & Peimbert (2004).
            N2Ha_M13    N2Ha diagnostic from Marino et al. (2013).
            O3N2_PP04   O3N2 diagnostic from Pilyugin & Peimbert (2004).
            O3N2_M13    O3N2 diagnostic from Marino et al. (2013).
            R23_KK04    R23 diagnostic from Kobulnicky & Kewley (2004).
            N2S2Ha_D16  N2S2Ha diagnostic from Dopita et al. (2016).
            N2O2_KD02   N2O2 diagnostic from Kewley & Dopita (2002).
            Rcal_PG16   Rcal diagnostic from Pilyugin & Grebel (2016).
            Scal_PG16   Scal diagnostic from Pilyugin & Grebel (2016).
            ONS_P10     ONS diagnostic from Pilyugin et al. (2010).
            ON_P10      ON diagnostic from Pilyugin et al. (2010).

    nthreads:                   int (optional)           
        Maximum number of threads to use. Defaults to os.cpu_count().

    __use_lzifu_fits (SAMI only):       bool (optional)
        If True, load the DataFrame containing emission line quantities
        (including fluxes, kinematics, etc.) derived directly from the LZIFU
        output FITS files, rather than those included in DR3. Default: False.

    __lzifu_ncomponents (SAMI only):    str  (optional)
        Number of components corresponding to the LZIFU fit, if 
        __use_lzifu_fits is specified. May be '1', '2', '3' or 'recom'. Note 
        that this keyword ONLY affects emission line fluxes and gas kinematics;
        other quantities including SFR/SFR surface densities and HALPHA 
        extinction correction factors are loaded from DR3 data products as per
        the ncomponents keyword. Default: False.

    debug:                      bool (optional)
        If True, run on a subset of the entire sample (10 galaxies) and save
        the output with "_DEBUG" appended to the filename. This is useful for
        tweaking S/N and DQ cuts since running the function on the entire 
        sample is quite slow.  Default: False.

    df_fname_tag:               str (optional)
        If specified, add a tag to the end of the default filename. Can be 
        useful in cases where you want to make multiple DataFrames with differing
        options that are not reflected in the default filename, e.g. 
        line_flux_SNR_cut, or if you want to make separate DataFrames for different 
        sets of galaxies.

    OUTPUTS
    ---------------------------------------------------------------------------
    The resulting DataFrame will be stored as 

        settings[<survey>]["output_path"]/<survey>_<bin_type>_<ncomponents>-comp_extcorr_minSNR=<eline_SNR_min>_<df_fname_tag>_minANR=<eline_ANR_min>_<df_fname_tag>.hd5

    if correct_extinction is True, or else

        settings[<survey>]["output_path"]/<survey>_<bin_type>_<ncomponents>-comp_minSNR=<eline_SNR_min>_<df_fname_tag>_minANR=<eline_ANR_min>_<df_fname_tag>.hd5

    The DataFrame will be stored in CSV format in case saving in HDF format 
    fails for any reason.

    PREREQUISITES
    ---------------------------------------------------------------------------
    make_metadata_df() for the corresponding survey must be run first.

    All required data products must be stored in the folders specified in the
    config file.
    """
    # Save input params as a dict
    ss_params = locals()

    # Input checking
    try:
        survey_module = import_module(f"spaxelsleuth.io.{survey}")
    except AttributeError:
        raise ValueError(f"I could not find a survey module for {survey}!")
    if ncomponents not in settings[survey]["ncomponents"]:
        raise ValueError(f"bin_type must be {' or '.join(settings[survey]['ncomponents'])}!!")
    if bin_type not in settings[survey]["bin_types"]:
        raise ValueError(f"bin_type must be {' or '.join(settings[survey]['bin_types'])}!!")

    if survey == "sami":
        if "__use_lzifu_fits" not in kwargs:
            kwargs["__use_lzifu_fits"] = False 
            kwargs["__lzifu_ncomponents"] = 'recom'  # TODO check that this doesn't get used 
        else:
            if kwargs["__use_lzifu_fits"]:
                if kwargs["__lzifu_ncomponents"] not in ["recom", "1", "2", "3"]:
                    raise ValueError("__lzifu_ncomponents must be 'recom', '1', '2' or '3'!!")
                if not os.path.exists(settings["sami"]["__lzifu_products_path"]):
                    raise ValueError(f"lzifu_products_path directory {settings['sami']['__lzifu_products_path']} not found!!")
                logger.warning(
                    "using LZIFU %s-component fits to obtain emission line fluxes & kinematics, NOT DR3 data products!!" % (settings['sami']['__lzifu_ncomponents']),
                    RuntimeWarning)

    logger.info(f"input parameters: survey={survey}, bin_type={bin_type}, ncomponents={ncomponents}, debug={debug}, eline_SNR_min={eline_SNR_min}, eline_ANR_min={eline_ANR_min}, correct_extinction={correct_extinction}")

    # Determine number of threads
    if nthreads is None:
        nthreads = os.cpu_count()
        logger.warning(f"nthreads not specified: running make_df() on {nthreads} threads...")
   
    # Paths and filenames
    output_path = Path(settings[survey]["output_path"])
    df_fname = get_df_fname(survey,
                            bin_type,
                            ncomponents,
                            correct_extinction,
                            eline_SNR_min,
                            eline_ANR_min,
                            debug,
                            df_fname_tag,
                            **kwargs)
    logger.info(f"saving to file {df_fname}...")

    # Load metadata
    df_metadata = load_metadata_df(survey)

    # List of galaxies 
    if gals is None:

        # Get list of galaxies from the metadata DataFrame
        if survey == "sami":
            # Only include galaxies flagged as "good" & for which we have data
            gals = df_metadata[df_metadata["Good?"] == True].index.values
            input_path = Path(settings["sami"]["input_path"]) # TODO this is so clunky!!
            gals = [
                g for g in gals
                if os.path.exists(input_path / f"ifs/{g}/")
            ]
            if len(gals) == 0:
                raise FileNotFoundError(f"I could not find any galaxy data in {input_path / 'ifs'}!")
        else:
            if df_metadata is not None:
                gals = df_metadata.index.values
            else:
                raise ValueError(f"because there is no metadata DataFrame for {survey}, you must specify a list of galaxies!")
        
        # If running in DEBUG mode, run on a subset to speed up execution time
        if debug:
            if survey == "sami":
                gals = gals[:10] + [572402, 209807]
            else:
                gals = gals[:10]
        
    # Also only run on a subset of metallicity diagnostics to speed up execution time
    if debug:
        metallicity_diagnostics = ["N2Ha_PP04", "N2Ha_K19"]

    # Scrape measurements for each galaxy from FITS files
    args_list = [[
        gg, gal, ncomponents, bin_type, df_metadata,
        kwargs
    ] for gg, gal in enumerate(gals)]

    # Multithread galaxy processing
    if len(gals) == 1:
        res_list = [survey_module.process_galaxies(args_list[0])]
    else:
        if nthreads > 1:
            logger.info(f"beginning pool...")
            pool = multiprocessing.Pool(
                min([nthreads, len(gals)]))
            res_list = np.array((pool.map(survey_module.process_galaxies, args_list)), dtype=object)
            pool.close()
            pool.join()
        else:
            logger.info(f"running sequentially...")
            res_list = []
            for args in args_list:
                res = survey_module.process_galaxies(args)
                res_list.append(res)

    # Convert to a Pandas DataFrame
    rows_list_all = [r[0] for r in res_list]
    colnames = res_list[0][1]
    df_spaxels = pd.DataFrame(np.vstack(tuple(rows_list_all)),
                              columns=colnames)

    # Merge with metadata (numeric-type columns only)
    # The question is... are there any metadata columns that are required for the below?
    # ALSO: can just pass ss_params in here rather than spelling out every arg individually
    # if df_metadata is not None:
        # cols_to_merge = [c for c in df_metadata if df_metadata[c].dtypes != "object"]
        # df_spaxels = df_spaxels.merge(df_metadata[cols_to_merge], on="ID", how="left")

    # Generic stuff: compute additional columns - extinction, metallicity, etc.
    df_spaxels = add_columns(survey,
                             df_spaxels,
                             eline_SNR_min=eline_SNR_min,
                             eline_ANR_min=eline_ANR_min,
                             sigma_gas_SNR_min=sigma_gas_SNR_min,
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
                             nthreads=nthreads,
                             **kwargs,
                             )

    # Save
    logger.info(f"saving to file {output_path / df_fname}...")

    # Remove object-type columns
    bad_cols = [c for c in df_spaxels if df_spaxels[c].dtype == "object"]
    if len(bad_cols) > 0:
        logger.warning(f"The following object-type columns are present in the DataFrame: {','.join(bad_cols)}")

    # Save
    with pd.HDFStore(output_path / df_fname) as store:
        store["DataFrame"] = df_spaxels
        store["Metadata"] = df_metadata
        store["Spaxelsleuth parameters"] = ss_params
    
    logger.info("finished!")
    return


def load_metadata_df(survey):
    try:
        return import_module(f"spaxelsleuth.io.{survey}").load_metadata_df()
    except AttributeError:
        logger.warning(f"I could not find a load_metadata_df() function in spaxelsleuth.io.{survey}!")
        return None


def load_df(survey,
            ncomponents,
            bin_type,
            eline_SNR_min,
            eline_ANR_min,
            correct_extinction,
            debug=False,
            df_fname_tag=None,
            **kwargs):
    """Load a spaxelsleuth DataFrame created using make_df().

    DESCRIPTION
    ---------------------------------------------------------------------------
    Load and return the Pandas DataFrame containing spaxel-by-spaxel 
    information for all galaxies from a given survey which was created using 
    make_df().

    INPUTS
    ---------------------------------------------------------------------------
    survey:             str
        Survey name, e.g. "sami", "hector". Must be an entry in the 
        configuration file.
    
    ncomponents:        str
        Number of components; may either be "1" (corresponding to the 
        1-component Gaussian fits) or "recom" (corresponding to the multi-
        component Gaussian fits).

    bin_type:           str
        Binning scheme used. Must be one of 'default' or 'adaptive' or 

    eline_SNR_min:      int 
        Minimum flux S/N to accept. Fluxes below the threshold (plus associated
        data products) are set to NaN.
        'sectors'.

    eline_ANR_min:          float
        Minimum A/N to adopt for emission lines in each kinematic component,
        defined as the Gaussian amplitude divided by the continuum standard
        deviation in a nearby wavelength range.

    correct_extinction: bool
        If True, load the DataFrame in which the emission line fluxes (but not 
        EWs) have been corrected for intrinsic extinction.

    eline_list:                 list of str
        List of emission lines to which the flagging operations are applied
        for S/N cuts, etc. 

    __use_lzifu_fits:           bool (optional)
        If True, load the DataFrame containing emission line quantities
        (including fluxes, kinematics, etc.) derived directly from the LZIFU
        output FITS files, rather than those included in DR3. 

    __lzifu_ncomponents:        str  (optional)
        Number of components corresponding to the LZIFU fit, if 
        __use_lzifu_fits is specified. May be '1', '2', '3' or 'recom'. Note 
        that this keyword ONLY affects emission line fluxes and gas kinematics;
        other quantities including SFR/SFR surface densities and HALPHA 
        extinction correction factors are loaded from DR3 data products as per
        the ncomponents keyword. 

    debug:                      bool
        If True, load the "debug" version of the DataFrame created when 
        running make_df(survey="sami",) with debug=True.
    
    USAGE
    ---------------------------------------------------------------------------
    load_sami_df() is called as follows:

        >>> from spaxelsleuth.io.sami import load_sami_df
        >>> df = load_sami_df(ncomponents, bin_type, correct_extinction, 
                              eline_SNR_min, debug)

    OUTPUTS
    ---------------------------------------------------------------------------
    The Dataframe.

    """

    #######################################################################
    # INPUT CHECKING
    #######################################################################
    if ncomponents not in settings[survey]["ncomponents"]:
        raise ValueError(f"bin_type must be {' or '.join(settings[survey]['ncomponents'])}!!")
    if bin_type not in settings[survey]["bin_types"]:
        raise ValueError(f"bin_type must be {' or '.join(settings[survey]['bin_types'])}!!")

    if survey == "sami":
        if "__use_lzifu_fits" not in kwargs:
            kwargs["__use_lzifu_fits"] = False 
            kwargs["__lzifu_ncomponents"] = 'recom'  # TODO check that this doesn't get used 
        else:
            if kwargs["__use_lzifu_fits"]:
                if kwargs["__lzifu_ncomponents"] not in ["recom", "1", "2", "3"]:
                    raise ValueError("__lzifu_ncomponents must be 'recom', '1', '2' or '3'!!")
                if not os.path.exists(settings["sami"]["__lzifu_products_path"]):
                    raise ValueError(f"lzifu_products_path directory {settings['sami']['__lzifu_products_path']} not found!!")
                logger.warning(
                    "using LZIFU %s-component fits to obtain emission line fluxes & kinematics, NOT DR3 data products!!" % (settings['sami']['__lzifu_ncomponents']),
                    RuntimeWarning)

    logger.info(f"input parameters: survey={survey}, bin_type={bin_type}, ncomponents={ncomponents}, debug={debug}, eline_SNR_min={eline_SNR_min}, eline_ANR_min={eline_ANR_min}, correct_extinction={correct_extinction}")

    # Filename & path
    output_path = Path(settings[survey]["output_path"])
    df_fname = get_df_fname(survey,
                            bin_type,
                            ncomponents,
                            correct_extinction,
                            eline_SNR_min,
                            eline_ANR_min,
                            debug,
                            df_fname_tag,
                            **kwargs)
    if not os.path.exists(output_path / df_fname):
        raise FileNotFoundError(f"DataFrame file {output_path / df_fname} does not exist!")

    # Load the data frame
    t = os.path.getmtime(output_path / df_fname)
    logger.info(
        f"Loading DataFrame from file {output_path / df_fname} [last modified {datetime.datetime.fromtimestamp(t)}]..."
    )

    # df = pd.read_hdf(output_path / df_fname, key=f"{bin_type}{ncomponents}comp")
    with pd.HDFStore(output_path / df_fname) as store:
        df_spaxels = store["DataFrame"]
        df_metadata = store["Metadata"]
        ss_params = store["Spaxelsleuth parameters"]

    # Merge df_spaxels with df_metadata 
    cols_to_merge = [c for c in df_metadata if df_metadata[c].dtypes != "object"]
    df = df_spaxels.merge(df_metadata[cols_to_merge], on="ID", how="left")

    # Merge spaxelsleuth params 
    for param in ss_params:
        df[param] = ss_params[param]

    # # Add "metadata" columns to the DataFrame
    # df["survey"] = survey
    # df["ncomponents"] = ncomponents
    # df["bin_type"] = bin_type
    # df["debug"] = debug

    # Take other keywords from the config file if they do not exist in the DataFrame 
    # TODO look into using e.g. xarray instead for storing these as "tags"
    # Log this as info
    for kw in [
        "as_per_px",
        "N_x",
        "N_y",
        "x_0 (pixels)",
        "y_0 (pixels)",
    ]:
        if kw not in df:
            if kw in settings[survey]:
                df[kw] = settings["sami"][kw]
            else:
                logger.warning(f"I could not find keyword '{kw}' in settings[{survey}] so I am not adding them to the DataFrame!")
    
    # Add additional keys
    # for kw in kwargs.keys():
        # df[kw] = kwargs[kw]
    
    # Units 
    df["flux_units"] = f"E{str(settings[survey]['flux_units']).lstrip('1e')} erg/cm^2/s"  # Units of emission line flux
    df["continuum_units"] = f"E{str(settings[survey]['flux_units']).lstrip('1e')} erg/cm^2/Å/s"  # Units of continuum flux density

    # Add back in object-type columns
    if "Morphology (numeric)" in df:
        df["Morphology"] = morph_num_to_str(df["Morphology (numeric)"])
    bpt_cols = [c for c in df if c.startswith("BPT") and "numeric" in c]
    for bpt_col in bpt_cols:
        df[bpt_col.replace(" (numeric)", "")] = bpt_num_to_str(df[bpt_col])

    # Return
    logger.info("finished!")
    return df.sort_index()