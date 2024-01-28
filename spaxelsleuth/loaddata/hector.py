if __name__ == "__main__":

    """
    Place to test _process_hector() before placing it in the function body.
    """
    import os
    from pathlib import Path
    from astropy.io import fits
    from astropy.cosmology import FlatLambdaCDM
    import matplotlib.pyplot as plt 
    import numpy as np
    import pandas as pd
    
    from spaxelsleuth import load_user_config
    load_user_config("/Users/u5708159/Desktop/spaxelsleuth_test/.myconfig.json")
    from spaxelsleuth.config import settings

    from spaxelsleuth.utils.continuum import compute_d4000, compute_continuum_intensity
    from spaxelsleuth.utils.dqcut import compute_measured_HALPHA_amplitude_to_noise
    from spaxelsleuth.utils.addcolumns import add_columns
    from spaxelsleuth.utils.linefns import bpt_num_to_str

    import logging
    logger = logging.getLogger(__name__)

    plt.ion()
    plt.close("all")

    from IPython.core.debugger import set_trace

    # Paths
    input_path = Path(settings["hector"]["input_path"])
    output_path = Path(settings["hector"]["output_path"])
    data_cube_path = Path(settings["hector"]["data_cube_path"])
    eline_fit_path = input_path / "emission_cubes"
    stekin_path = input_path / "initial_stel_kin"
    continuum_fit_path = input_path / "cont_subtracted"

    gal = 901006821908960
    tile = 29
    id_str = f"{gal:d}_T{tile:03d}"
    ncomponents = "rec"
    if ncomponents == "rec":
        ncomponents_max = 3
    else:
        ncomponents_max = ncomponents



    #--------------------------------------------------------------------------
    # Filenames
    stekin_fname = stekin_path / f"{id_str}_initial_kinematics.fits"
    eline_fit_fname = eline_fit_path / id_str / f"{id_str}_{ncomponents}comp.fits"  
    cont_fit_B_fname = continuum_fit_path / f"{id_str}_blue_stel_subtract_final.fits"
    cont_fit_R_fname = continuum_fit_path / f"{id_str}_red_stel_subtract_final.fits"
    datacube_fnames_all = os.listdir(data_cube_path)  # Because (in the current format) the data cubes contain the field name (which we don't know in advance), to find the correct data cube file name we need to traverse the entire list. Not ideal but it's all we can do for now.
    datacube_B_fname = data_cube_path / [fname for fname in datacube_fnames_all if str(gal) in fname and "blue" in fname][0]
    datacube_R_fname = data_cube_path / [fname for fname in datacube_fnames_all if str(gal) in fname and "red" in fname][0]
    for file in [stekin_fname, eline_fit_fname, cont_fit_B_fname, cont_fit_R_fname, datacube_B_fname, datacube_R_fname]:
        if not os.path.exists(str(file)):
            raise FileNotFoundError(f"File {file} not found!")

    #--------------------------------------------------------------------------
    # Get redshift
    with fits.open(stekin_fname) as hdulist_stekin:
        z = hdulist_stekin[0].header["Z"]

    # Blue cube & wavelength array
    with fits.open(datacube_B_fname) as hdulist_B_cube:
        header_B = hdulist_B_cube[0].header
        data_cube_B = hdulist_B_cube[0].data
        var_cube_B = hdulist_B_cube[1].data
        # Wavelength information
        lambda_0_A = header_B["CRVAL3"] - header_B["CRPIX3"] * header_B["CDELT3"]
        dlambda_A = header_B["CDELT3"]
        N_lambda = header_B["NAXIS3"]
        lambda_vals_B_A = np.array(range(N_lambda)) * dlambda_A + lambda_0_A
        # Centre coordinates 
        x0_px, y0_px = np.floor(header_B["CRPIX1"]).astype(int), np.floor(header_B["CRPIX2"]).astype(int)

    # Red cube & wavelength array
    with fits.open(datacube_R_fname) as hdulist_R_cube:
        header_R = hdulist_R_cube[0].header
        data_cube_R = hdulist_R_cube[0].data
        var_cube_R = hdulist_R_cube[1].data
        # Wavelength information
        lambda_0_A = header_R["CRVAL3"] - header_R["CRPIX3"] * header_R["CDELT3"]
        dlambda_A = header_R["CDELT3"]
        N_lambda = header_R["NAXIS3"]
        lambda_vals_R_A = np.array(range(N_lambda)) * dlambda_A + lambda_0_A

    # Rest-frame wavelength arrays
    lambda_vals_R_rest_A = lambda_vals_R_A / (1 + z)
    lambda_vals_B_rest_A = lambda_vals_B_A / (1 + z)

    # Get coordinate lists corresponding to non-empty spaxels
    im_empty_B = np.all(np.isnan(data_cube_B), axis=0)
    im_empty_R = np.all(np.isnan(data_cube_R), axis=0)
    im_empty = np.logical_and(im_empty_B, im_empty_R)
    mask = ~im_empty
    ny, nx = im_empty.shape
    y_c_list, x_c_list = np.where(~im_empty)

    #--------------------------------------------------------------------------
    # Construct a dict where the keys are the FINAL column names, and the 
    # values are 2D arrays storing the corresponding quantity
    _2dmap_dict = {}

    #--------------------------------------------------------------------------
    # EMISSION LINES 
    with fits.open(eline_fit_fname) as hdulist_eline_fit:
        # Emission lines: fluxes
        for eline in settings["hector"]["eline_list"]:
            _2dmap_dict[f"{eline} (total)"] = hdulist_eline_fit[eline].data[0]
            _2dmap_dict[f"{eline} error (total"] = hdulist_eline_fit[eline + "_ERR"].data[0]
            if ncomponents_max > 1:
                for component in range(1, ncomponents_max):
                    _2dmap_dict[f"{eline} (component {component})"] = hdulist_eline_fit[eline].data[component]
                    _2dmap_dict[f"{eline} error (component {component})"] = hdulist_eline_fit[eline + "_ERR"].data[component]
            else:
                _2dmap_dict[f"{eline} (component 1)"] = hdulist_eline_fit[eline].data[0]
                _2dmap_dict[f"{eline} error (component 1)"] = hdulist_eline_fit[eline + "_ERR"].data[0]
        
        # Emission lines: gas kinematics 
        for component in range(1, ncomponents_max):
            _2dmap_dict[f"v_gas (component {component})"] = hdulist_eline_fit["V"].data[component]
            _2dmap_dict[f"v_gas error (component {component})"] = hdulist_eline_fit["V_ERR"].data[component]
            _2dmap_dict[f"sigma_gas (component {component})"] = hdulist_eline_fit["VDISP"].data[component]
            _2dmap_dict[f"sigma_gas error (component {component})"] = hdulist_eline_fit["VDISP_ERR"].data[component]

        # Emission lines: goodness-of-fit  
        _2dmap_dict["chi2 (emission lines)"] = hdulist_eline_fit["CHI2"].data

    #--------------------------------------------------------------------------
    # STELLAR CONTINUUM
    with fits.open(stekin_fname) as hdulist_stekin:
        # Stellar continuum fit: kinematics
        _2dmap_dict[f"v_*"] = hdulist_stekin["V_STAR"].data
        _2dmap_dict[f"v_* error"] = hdulist_stekin["V_STAR_ERR"].data
        _2dmap_dict[f"sigma_*"] = hdulist_stekin["SIG_STAR"].data
        _2dmap_dict[f"sigma_* error"] = hdulist_stekin["SIG_STAR_ERR"].data

        # Stellar continuum fit: goodness-of-fit/continuum S/N
        _2dmap_dict["chi2 (ppxf)"] = hdulist_stekin["CHI2"].data
        _2dmap_dict["Median continuum S/N (ppxf)"] = hdulist_stekin["PRIMARY"].data

    #--------------------------------------------------------------------------
    # CONTINUUM PROPERTIES 
    # Continuum properties: Compute the d4000 Angstrom break 
    # TODO: place these checks inside the function - return arrays of NaNs if they fail
    plt.figure(); plt.imshow(_2dmap_dict["v_*"])
    if lambda_vals_B_rest_A[0] <= 3850 and lambda_vals_B_rest_A[-1] >= 4100:
        d4000_map, d4000_map_err = compute_d4000(
            data_cube=data_cube_B,
            var_cube=var_cube_B,
            lambda_vals_rest_A=lambda_vals_B_rest_A,
            v_star_map=_2dmap_dict["v_*"])
        _2dmap_dict["D4000"] = d4000_map
        _2dmap_dict["D4000 error"] = d4000_map_err

    # Continuum properties: Compute the continuum intensity so that we can compute the Halpha equivalent width.
    # TODO: place these checks inside the function - return arrays of NaNs if they fail
    plt.figure(); plt.imshow(_2dmap_dict["v_*"])
    if lambda_vals_R_rest_A[0] <= 6500 and lambda_vals_R_rest_A[-1] >= 6540:
        cont_HALPHA_map, cont_HALPHA_map_std, cont_HALPHA_map_err = compute_continuum_intensity(
            data_cube=data_cube_R,
            var_cube=var_cube_R,
            lambda_vals_rest_A=lambda_vals_R_rest_A,
            start_A=6500,
            stop_A=6540,
            v_map=_2dmap_dict["v_*"])
        _2dmap_dict["HALPHA continuum"] = cont_HALPHA_map
        _2dmap_dict["HALPHA continuum std. dev."] = cont_HALPHA_map_std
        _2dmap_dict["HALPHA continuum error"] = cont_HALPHA_map_err

    # Continuum properties: Compute the approximate B-band continuum. (from raw cube or fit?)
    # TODO: place these checks inside the function - return arrays of NaNs if they fail
    plt.figure(); plt.imshow(_2dmap_dict["v_*"])
    if lambda_vals_B_rest_A[0] <= 4000 and lambda_vals_B_rest_A[-1] >= 5000:
        cont_B_map, cont_B_map_std, cont_B_map_err = compute_continuum_intensity(
            data_cube=data_cube_B,
            var_cube=var_cube_B,
            lambda_vals_rest_A=lambda_vals_B_rest_A,
            start_A=4000,
            stop_A=5000,
            v_map=_2dmap_dict["v_*"])
        _2dmap_dict[f"B-band continuum"] = cont_B_map
        _2dmap_dict[f"B-band continuum std. dev."] = cont_B_map_std
        _2dmap_dict[f"B-band continuum error"] = cont_B_map_err

    # Compute the continuum intensity so that we can compute the Halpha equivalent width.
    # Continuum wavelength range taken from here: https://ui.adsabs.harvard.edu/abs/2019MNRAS.485.4024V/abstract
    # TODO: place these checks inside the function - return arrays of NaNs if they fail
    plt.figure(); plt.imshow(_2dmap_dict["v_*"])
    if lambda_vals_R_rest_A[0] <= 6562.8 and lambda_vals_R_rest_A[-1] >= 6562.8:
        AN_HALPHA_map = compute_measured_HALPHA_amplitude_to_noise(
            data_cube=data_cube_R,
            var_cube=var_cube_R,
            lambda_vals_rest_A=lambda_vals_R_rest_A,
            v_star_map=_2dmap_dict["v_*"],
            v_map=_2dmap_dict["v_gas (component 1)"],
            dv=300)
        _2dmap_dict["HALPHA A/N (measured)"] = AN_HALPHA_map

    plt.figure(); plt.imshow(_2dmap_dict["v_*"])

    # Median value in blue/red clubes 
    for cont_fit_fname, side in zip([cont_fit_B_fname, cont_fit_R_fname], ["blue", "red"]):
        with fits.open(cont_fit_fname) as hdulist_cont_fit:
            _2dmap_dict[f"Median spectral value ({side})"] = hdulist_cont_fit["MED_SPEC"].data

    #--------------------------------------------------------------------------
    # Other quantitites 
    _2dmap_dict["Galaxy centre x0_px (projected, arcsec)"] = mask * x0_px
    _2dmap_dict["Galaxy centre y0_px (projected, arcsec)"] = mask * y0_px

    # TODO also need to store regular coordinates relative to galaxy centre 
    # TODO v_grad

    # rows_list.append(
    #     np.array([settings["sami"]["x0_px"]] * ngood_bins) *
    #     settings["sami"]["as_per_px"])
    # colnames.append("Galaxy centre x0_px (projected, arcsec)")
    # rows_list.append(
    #     np.array([settings["sami"]["y0_px"]] * ngood_bins) *
    #     settings["sami"]["as_per_px"])
    # # colnames.append("Galaxy centre y0_px (projected, arcsec)")
    # rows_list.append(
    #     np.array(x_c_list).flatten() * settings["sami"]["as_per_px"])
    # colnames.append("x (projected, arcsec)")
    # rows_list.append(
    #     np.array(y_c_list).flatten() * settings["sami"]["as_per_px"])
    # colnames.append("y (projected, arcsec)")
    # rows_list.append(
    #     np.array(x_prime_list).flatten() * settings["sami"]["as_per_px"])
    # colnames.append("x (relative to galaxy centre, deprojected, arcsec)")
    # rows_list.append(
    #     np.array(y_prime_list).flatten() * settings["sami"]["as_per_px"])
    # colnames.append("y (relative to galaxy centre, deprojected, arcsec)")
    # rows_list.append(
    #     np.array(r_prime_list).flatten() * settings["sami"]["as_per_px"])
    # colnames.append("r (relative to galaxy centre, deprojected, arcsec)")
    # rows_list.append(np.array(bin_number_list))
    # colnames.append("Bin number")
    # rows_list.append(np.array(bin_size_list_px))
    # colnames.append("Bin size (pixels)")
    # rows_list.append(
    #     np.array(bin_size_list_px) * settings["sami"]["as_per_px"]**2)
    # colnames.append("Bin size (square arcsec)")
    # rows_list.append(
    #     np.array(bin_size_list_px) * settings["sami"]["as_per_px"]**2 *
    #     df_metadata.loc[gal, "kpc per arcsec"]**2)
    # colnames.append("Bin size (square kpc)")

    #--------------------------------------------------------------------------
    # Convert 2D maps to 1D rows 

    # Function for extracting data from 2D maps
    def _2d_map_to_1d_list(colmap):
        """Returns a 1D array of values extracted from from spaxels in x_c_list and y_c_list in 2D array colmap."""
        if colmap.ndim != 2:
            raise ValueError(
                f"colmap must be a 2D array but has ndim = {colmap.ndim}!")
        row = np.full_like(x_c_list, np.nan, dtype="float")
        for jj, coords in enumerate(zip(x_c_list, y_c_list)):
            x_c, y_c = coords
            y, x = (int(np.round(y_c)), int(np.round(x_c)))
            if x > nx or y > ny:
                x = min([x, nx])
                y = min([y, ny])
            row[jj] = colmap[y, x]
        return row
    
    # DEBUGGING ONLY
    def _1d_map_to_2d_list(rows, x_c_list, y_c_list):
        """Reconstructs a 2D array of values from row with coordinates specified in x_c_list and y_c_list"""
        colmap = np.full((ny, nx), np.nan)
        for d, x, y in zip(rows, x_c_list, y_c_list):
            colmap[y, x] = d
        return colmap
    
    def check_row(rows, colname, data):
        fig, axs = plt.subplots(ncols=3, figsize=(10, 5))
        colmap = _1d_map_to_2d_list(rows, x_c_list, y_c_list)
        axs[0].imshow(data)
        axs[1].imshow(colmap)
        axs[2].imshow(data - colmap)
        fig.suptitle(colname)
        axs[0].set_title("From FITS file")
        axs[1].set_title("Reconstructed from row")
        axs[2].set_title("Difference")

    rows_list = []
    for colname in _2dmap_dict.keys():

        rows = _2d_map_to_1d_list(_2dmap_dict[colname])
        rows_list.append(rows)

        # For debugging 
        if colname == "v_*" or colname == "sigma_*":
            check_row(rows, colname, _2dmap_dict[colname])
            set_trace()
            plt.close("all")

def _process_hector():
    return


def make_hector_df(gals,
                  ncomponents,
                  eline_SNR_min,
                  eline_ANR_min,
                  correct_extinction,
                  sigma_inst_kms,
                  df_fname=None,
                  sigma_gas_SNR_min=3,
                  line_flux_SNR_cut=True,
                  missing_fluxes_cut=True,
                  line_amplitude_SNR_cut=True,
                  flux_fraction_cut=False,
                  sigma_gas_SNR_cut=True,
                  vgrad_cut=False,
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
                      "N2Ha_PP04",
                      "N2Ha_K19",
                      "R23_KK04",
                  ],
                  nthreads=None):
    """
    TODO: write docstring
    """
    ###############################################################################
    # input checking
    ###############################################################################
    if df_fname is not None:
        if not df_fname.endswith(".hd5"):
            df_fname += ".hd5"
    else:
        # Input file name
        df_fname = f"hector_{ncomponents}-comp"
        if correct_extinction:
            df_fname += "_extcorr"
        df_fname += f"_minSNR={eline_SNR_min}_minANR={eline_ANR_min}.hd5"

    if (type(ncomponents) not in [int, str]) or (type(ncomponents) == str
                                                 and ncomponents != "merge"):
        raise ValueError("ncomponents must be either an integer or 'merge'!")

    logger.info(f"input parameters: ncomponents={ncomponents}, eline_SNR_min={eline_SNR_min}, eline_ANR_min={eline_ANR_min}, correct_extinction={correct_extinction}")

    # Determine number of threads
    if nthreads is None:
        nthreads = os.cpu_count()
        logger.warning(f"nthreads not specified: running make_hector_df() on {nthreads} threads...")

    ###############################################################################
    # Scrape measurements for each galaxy from FITS files
    ###############################################################################
    args_list = [[gg, gal, ncomponents, data_cube_path]
                 for gg, gal in enumerate(gals)]

    if len(gals) == 1:
        res_list = [_process_hector(args_list[0])]
    else:
        if nthreads > 1:
            logger.info(f"beginning pool...")
            pool = multiprocessing.Pool(min([nthreads, len(gals)]))
            res_list = pool.map(_process_hector, args_list)
            pool.close()
            pool.join()
        else:
            logger.info(f"running sequentially...")
            res_list = []
            for args in args_list:
                res = _process_hector(args)
                res_list.append(res)

    ###############################################################################
    # Convert to a Pandas DataFrame
    ###############################################################################
    rows_list_all = [r[0] for r in res_list]
    colnames = res_list[0][1]
    eline_list = res_list[0][2]  #TODO
    df_spaxels = pd.DataFrame(np.vstack(tuple(rows_list_all)),
                              columns=colnames)

    # Cast to float data types
    for col in df_spaxels.columns:
        # Check if column values can be cast to float
        df_spaxels[col] = pd.to_numeric(df_spaxels[col], errors="coerce")

    ###############################################################################
    # Generic stuff: compute additional columns - extinction, metallicity, etc.
    ###############################################################################
    df_spaxels = add_columns(
        df_spaxels.copy(),
        eline_SNR_min=eline_SNR_min,
        eline_ANR_min=eline_ANR_min,
        sigma_gas_SNR_min=sigma_gas_SNR_min,
        eline_list=eline_list,
        line_flux_SNR_cut=line_flux_SNR_cut,
        missing_fluxes_cut=missing_fluxes_cut,
        line_amplitude_SNR_cut=line_amplitude_SNR_cut,
        flux_fraction_cut=flux_fraction_cut,
        sigma_gas_SNR_cut=sigma_gas_SNR_cut,
        vgrad_cut=vgrad_cut,
        stekin_cut=False,
        correct_extinction=correct_extinction,
        metallicity_diagnostics=metallicity_diagnostics,
        compute_sfr=True,
        sigma_inst_kms=sigma_inst_kms,
        nthreads=nthreads,
        base_missing_flux_components_on_HALPHA=False,  # NOTE: this is important!!
        )

    ###############################################################################
    # Save to file
    ###############################################################################
    logger.info(f"saving to file {output_path / df_fname}...")
    df_spaxels.to_hdf(output_path / df_fname, key="hector")
    logger.info(f"finished!")

    return


def load_hector_df(ncomponents,
                  eline_SNR_min,
                  eline_ANR_min,
                  correct_extinction,
                  df_fname=None,
                  key=None):
    """
    DESCRIPTION
    ---------------------------------------------------------------------------
    Load and return the Pandas DataFrame containing spaxel-by-spaxel 
    information for all hector galaxies which was created using make_hector_df().

    INPUTS
    ---------------------------------------------------------------------------
    ncomponents:        int or str
        Number of components; may either be 1, 2, 3... N where N is the number 
        of Gaussian components fitted to the emission lines or "merge" 
        (corresponding to the "recommendend"-component Gaussian fits where 
        the number of components in each spaxel is determined using the 
        Likelihood Ratio Test as detailed in Ho et al. 2016 
        (https://ui.adsabs.harvard.edu/abs/2016Ap%26SS.361..280H/abstract)).

    eline_SNR_min:      int 
        Minimum flux S/N to accept. Fluxes below the threshold (plus associated
        data products) are set to NaN.

    eline_ANR_min:      float
        Minimum A/N to adopt for emission lines in each kinematic component,
        defined as the Gaussian amplitude divided by the continuum standard
        deviation in a nearby wavelength range.
   
    correct_extinction: bool
        If True, load the DataFrame in which the emission line fluxes (but not 
        EWs) have been corrected for intrinsic extinction.

    df_fname:           str
        (Optional) If specified, load DataFerame from file <df_fname>.hd5. 
        Otherwise, the DataFrame filename is automatically determined using 
        the other input arguments.

    key:                str
        (Optional) key used to read the HDF file if df_fname is specified.
    
    USAGE
    ---------------------------------------------------------------------------
    load_hector_df() is called as follows:

        >>> from spaxelsleuth.loaddata.hector import load_hector_df
        >>> df = load_hector_df(ncomponents, eline_SNR_min, eline_ANR_min, 
                               correct_extinction)

    OUTPUTS
    ---------------------------------------------------------------------------
    The Dataframe.
    """

    #######################################################################
    # INPUT CHECKING
    #######################################################################
    # Input file name
    if df_fname is not None:
        logger.warning(
            f"loading DataFrame from user-provided filename {output_path / df_fname} which may not correspond to the provided ncomponents, etc. Proceed with caution!",
            RuntimeWarning)
        if not df_fname.endswith(".hd5"):
            df_fname += ".hd5"
    else:
        # Input file name
        df_fname = f"hector_{ncomponents}-comp"
        if correct_extinction:
            df_fname += "_extcorr"
        df_fname += f"_minSNR={eline_SNR_min}_minANR={eline_ANR_min}.hd5"

    if not os.path.exists(output_path / df_fname):
        raise FileNotFoundError(
            f"File {output_path / df_fname} does does not exist!")

    # Load the data frame
    t = os.path.getmtime(output_path / df_fname)
    logger.info(
        f"loading DataFrame from file {output_path / df_fname} [last modified {datetime.datetime.fromtimestamp(t)}]..."
    )
    if key is not None:
        df = pd.read_hdf(output_path / df_fname, key=key)
    else:
        df = pd.read_hdf(output_path / df_fname)

    # Add "metadata" columns to the DataFrame
    df["survey"] = "hector"
    df["ncomponents"] = ncomponents

    # Add back in object-type columns
    df["x, y (pixels)"] = list(
    zip(df["x (projected, arcsec)"] / 0.5,
        df["y (projected, arcsec)"] / 0.5))
    df["BPT (total)"] = bpt_num_to_str(df["BPT (numeric) (total)"])

    # Return
    logger.info("finished!")
    return df.sort_index()
