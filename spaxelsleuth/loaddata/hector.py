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

    # Filenames
    stekin_fname = stekin_path / f"{id_str}_initial_kinematics.fits"
    eline_fit_fname = eline_fit_path / id_str / f"{id_str}_{ncomponents}comp.fits"  
    datacube_fnames_all = os.listdir(data_cube_path)  # Because (in the current format) the data cubes contain the field name (which we don't know in advance), to find the correct data cube file name we need to traverse the entire list. Not ideal but it's all we can do for now.
    datacube_B_fname = data_cube_path / [fname for fname in datacube_fnames_all if str(gal) in fname and "blue" in fname][0]
    datacube_R_fname = data_cube_path / [fname for fname in datacube_fnames_all if str(gal) in fname and "red" in fname][0]
    if not os.path.exists(datacube_B_fname):
        raise FileNotFoundError(f"Blue data cube for galaxy {gal} ({datacube_B_fname}) not found!")
    if not os.path.exists(datacube_R_fname):
        raise FileNotFoundError(f"Red data cube for galaxy {gal} ({datacube_R_fname}) not found!")
    if not os.path.exists(stekin_fname):
        raise FileNotFoundError(f"Stellar kinematics data products for galaxy {gal} ({stekin_fname}) not found!")
    if not os.path.exists(eline_fit_fname):
        raise FileNotFoundError(f"Emission line data products for galaxy {gal} ({eline_fit_fname}) not found!")

    # Blue cube
    with fits.open(datacube_B_fname) as hdulist_B_cube:
        header_B = hdulist_B_cube[0].header
        data_cube_B = hdulist_B_cube[0].data
        var_cube_B = hdulist_B_cube[1].data

        # Wavelength values
        lambda_0_A = header_B["CRVAL3"] - header_B["CRPIX3"] * header_B["CDELT3"]
        dlambda_A = header_B["CDELT3"]
        N_lambda = header_B["NAXIS3"]
        lambda_vals_B_A = np.array(range(N_lambda)) * dlambda_A + lambda_0_A

    # Red cube
    with fits.open(datacube_R_fname) as hdulist_R_cube:
        header_R = hdulist_R_cube[0].header
        data_cube_R = hdulist_R_cube[0].data
        var_cube_R = hdulist_R_cube[1].data

        # Wavelength values
        lambda_0_A = header_R["CRVAL3"] - header_R["CRPIX3"] * header_R["CDELT3"]
        dlambda_A = header_R["CDELT3"]
        N_lambda = header_R["NAXIS3"]
        lambda_vals_R_A = np.array(range(N_lambda)) * dlambda_A + lambda_0_A

    # Stellar kinematics (want v_* map plus redshift)
    with fits.open(stekin_fname) as hdulist_stekin:
        z = hdulist_stekin[0].header["Z"]
        v_star = hdulist_stekin["V_STAR"].data

    # Stellar kinematics (want v_* map plus redshift)
    with fits.open(eline_fit_fname) as hdulist_eline_fit:
        v_gas = hdulist_eline_fit["V"].data

    # Rest-frame wavelength arrays
    lambda_vals_R_rest_A = lambda_vals_R_A / (1 + z)
    lambda_vals_B_rest_A = lambda_vals_B_A / (1 + z)


    # TODO: Compute stuff based on the cubes
    # TODO: Compute the d4000 Angstrom break 
    # TODO: Compute the continuum intensity so that we can compute the Halpha equivalent width.
    # TODO: Compute the approximate B-band continuum. (from raw cube or fit?)
    # TODO: Compute the measured HALPHA amplitude-to-noise

    # EMISSION LINE STUFF 

    # Create masks of empty pixels
    im_empty_B = np.all(np.isnan(data_cube_B), axis=0)
    im_empty_R = np.all(np.isnan(data_cube_R), axis=0)
    im_empty = np.logical_and(im_empty_B, im_empty_R)

    # Get coordinate lists corresponding to non-empty spaxels
    ny, nx = im_empty.shape
    y_c_list, x_c_list = np.where(~im_empty)

    # Empty lists to store values extracted from each spaxel
    rows_list = []
    colnames = []

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

    # Dict containing filenames of the "map"-like data products - i.e. 2D maps of various quantities - and which extensions we want to extract from each
    map_data_products = {
        stekin_fname: {
            "PRIMARY" : "Median continuum S/N",  # TODO is this is in the red cube or the blue cube?
            "V_STAR" : "v_*",
            "SIG_STAR" : "sigma_*",
            "V_STAR_ERR" : "v_* error",
            "SIG_STAR_ERR" : "sigma_* error",
            "CHI2" : "ppxf Chi-squared",  # TODO is this the reduced chi2?
        },
        eline_fit_fname: {
            "V": "v_gas",
            "V_ERR": "v_gas error",
            "VDISP": "sigma_gas",
            "VDISP_ERR": "sigma_gas error",
            "CHI2": "Emission line Chi-squared",
            "OII3726": "OII3726",
            "OII3726_ERR": "OII3726 error",
            "OII3729": "OII3729",
            "OII3729_ERR": "OII3729 error",
            "HDELTA": "HDELTA",
            "HDELTA_ERR": "HDELTA error",
            "HGAMMA": "HGAMMA",
            "HGAMMA_ERR": "HGAMMA error",
            "HBETA": "HBETA",
            "HBETA_ERR": "HBETA error",
            "OIII4959": "OIII4959",
            "OIII4959_ERR": "OIII4959 error",
            "OIII5007": "OIII5007",
            "OIII5007_ERR": "OIII5007 error",
            "OI6300": "OI6300",
            "OI6300_ERR": "OI6300 error",
            "OI6364": "OI6364",
            "OI6364_ERR": "OI6364 error",
            "NII6548": "NII6548",
            "NII6548_ERR": "NII6548 error",
            "NII6583": "NII6583",
            "NII6583_ERR": "NII6583 error",
            "HALPHA": "HALPHA",
            "HALPHA_ERR": "HALPHA error",
            "SII6716": "SII6716",
            "SII6716_ERR": "SII6716 error",
            "SII6731": "SII6731",
            "SII6731_ERR": "SII6731 error",
        }, 
    }
    
    def is_eline(extension):
        """Returns True if extension is of the form <eline> or <eline>_ERR where eline is an emission line in settings["hector"]["eline_list"]."""
        return extension.rstrip("_ERR") in settings["hector"]["eline_list"]

    # For each quantity, 
    fnames = map_data_products.keys()
    for fname in fnames:
        hdulist = fits.open(fname)
        rename_dict = map_data_products[fname]
        for extension in map_data_products[fname]:
            data = hdulist[extension].data
            
            # If it's an emission line, then data is a 2D array containing the total flux - se we need to append it twice (once for the total flux & once for component 1)
            if ncomponents == 1 and is_eline(extension):
                data = np.repeat(data[None, :, :], 2, axis=0)
            
            if data.ndim > 2:
                # The 0th extension contains "total" quantities, unless it's the gas velocity or velocity dispersion.
                if extension not in ["V", "V_ERR", "VDISP", "VDISP_ERR"]:
                    rows = _2d_map_to_1d_list(data[0])
                    colname = rename_dict[extension] + f" (total)"
                    
                    # data_reconstructed = _1d_map_to_2d_list(rows, x_c_list, y_c_list)
                    # data_original = np.copy(data[0])
                    # assert np.all(np.isclose(data_original[~np.isnan(data_original)], data_reconstructed[~np.isnan(data_reconstructed)]))
                    # check_row(rows, colname, data_original)
                
                for component in range(1, data.shape[0]):
                    rows = _2d_map_to_1d_list(data[component])
                    colname = rename_dict[extension] + f" (component {component})"
                    
                    # data_reconstructed = _1d_map_to_2d_list(rows, x_c_list, y_c_list)
                    # data_original = np.copy(data[component])
                    # assert np.all(np.isclose(data_original[~np.isnan(data_original)], data_reconstructed[~np.isnan(data_reconstructed)]))
                    # check_row(rows, colname, data_original)
            
            elif data.ndim == 2:
                rows = _2d_map_to_1d_list(data)
                colname = rename_dict[extension]
                
                # data_reconstructed = _1d_map_to_2d_list(rows, x_c_list, y_c_list)
                # data_original = np.copy(data)
                # assert np.all(np.isclose(data_original[~np.isnan(data_original)], data_reconstructed[~np.isnan(data_reconstructed)]))
                # check_row(rows, colname, data_original)
            
            rows_list.append(rows)
            colnames.append(colname)

            set_trace()


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
