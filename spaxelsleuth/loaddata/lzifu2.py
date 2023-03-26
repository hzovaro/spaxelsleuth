import os
from path import Path

from astropy.io import fits
import multiprocessing
import numpy as np
import pandas as pd
from scipy import constants

from spaxelsleuth.config import settings
from .generic import add_columns, compute_d4000, compute_continuum_intensity, compute_HALPHA_amplitude_to_noise, compute_v_grad

from IPython.core.debugger import Tracer

###############################################################################
# Paths
# TODO make this more flexible? perhaps an input arg?
input_path = Path(settings["lzifu"]["input_path"])
output_path = Path(settings["lzifu"]["output_path"])
data_cube_path = Path(settings["lzifu"]["data_cube_path"])

#/////////////////////////////////////////////////////////////////////////////////
def _process_lzifu(args):

    #######################################################################
    # Parse arguments
    _, gal, ncomponents, bin_type, data_cube_path, _ = args
    #TODO can we get ncomponents from SET?
    lzifu_ncomponents = ncomponents if type(ncomponents) == int else 3

    #######################################################################
    # Scrape outputs from LZIFU output
    hdulist_lzifu = fits.open(input_path + f"/{gal}_{ncomponents}_comp.fits")
    hdr = hdulist_lzifu[0].header
    t = hdulist_lzifu["SET"].data  # Table storing fit parameters
    
    # Path to data cubes used in the fit 
    if data_cube_path is None:
        data_cube_path = t["DATA_PATH"][0]
    
    # Was the fit one-sided or two-sided (i.e., red and blue cubes?)
    # In either case, use SET to find the original datacubes used in the fit.
    onesided = True if t["ONLY_1SIDE"][0] == 1 else False
    if onesided:
        datacube_fname = data_cube_path + f"/{gal}.fits"
        datacube_fname_alt = data_cube_path + f"/{gal}.fits.gz"
        if not os.path.exists(datacube_fname):
            if os.path.exists(datacube_fname_alt):
                datacube_fname = datacube_fname_alt 
            else:
                raise FileNotFoundError("Cannot find files {datacube_fname} or {datacube_fname_alt}!")
    else:
        datacube_fnames = []
        for side in ["B", "R"]:
            datacube_fname = data_cube_path + f"/{gal}_{side}.fits"
            datacube_fname_alt = data_cube_path + f"/{gal}_{side}.fits.gz"
            if not os.path.exists(datacube_fname):
                if os.path.exists(datacube_fname_alt):
                    datacube_fname = datacube_fname_alt 
                else:
                    raise FileNotFoundError("Cannot find files {datacube_fname} or {datacube_fname_alt}!")
            datacube_fnames.append(datacube_fname)
            
        datacube_B_fname, datacube_R_fname = datacube_fnames
    
    # Redshift
    z = t["Z"][0]
    sigma_inst_A = t["RESOL_SIGMA"] #TODO not sure what effect this will have in 2-sided fits since B and R are different 
    sigma_inst_kms = sigma_inst_A / 6562.8 * constants.c / 1e3 #TODO CHECK THIS FORMULA! Evaluate at Halpha I guess

    # Get size from one of the extensions
    nx = t["XSIZE"][0]
    ny = t["YSIZE"][0]
    if bin_type == "default":
        yy, xx = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
        x_c_list = xx.flatten()
        y_c_list = yy.flatten()
    else:
        # Load the bin map
        raise ValueError("Other binning methods have not been implemented yet!")

    #/////////////////////////////////////////////////////////////////////////
    def _2d_map_to_1d_list(colmap):
        """Returns a 1D array of values extracted from from spaxels in x_c_list and y_c_list in 2D array colmap."""
        if colmap.ndim != 2:
            raise ValueError(f"colmap must be a 2D array but has ndim = {colmap.ndim}!")
        row = np.full_like(x_c_list, np.nan, dtype="float")
        for jj, coords in enumerate(zip(x_c_list, y_c_list)):
            x_c, y_c = coords
            y, x = (int(np.round(y_c)), int(np.round(x_c)))
            #TODO: replace magic numbers with ENUM or Survey class definitions
            if x > nx or y > ny:
                x = min([x, nx])
                y = min([y, ny])
            row[jj] = colmap[y, x]
        return row

    #######################################################################
    # SCRAPE LZIFU MEASUREMENTS
    #######################################################################
    # Get extension names
    extnames = [hdr[e] for e in hdr if e.startswith("EXT") and type(hdr[e]) == str]
    quantities = [e.rstrip("_ERR") for e in extnames if e.endswith("_ERR")] + ["CHI2", "DOF"]
    rows_list = []
    colnames = []
    eline_list = [q for q in quantities if q not in ["V", "VDISP", "CHI2", "DOF"]]

    # Scrape the FITS file: emission line flues, velocity/velocity dispersion, fit quality
    for quantity in quantities:
        data = hdulist_lzifu[quantity].data
        if f"{quantity}_ERR" in hdulist_lzifu:
            err = hdulist_lzifu[f"{quantity}_ERR"].data
        # If 'data' has 3 dimensions, then it has been measured for all quantities
        if data.ndim == 3:
            # Total fluxes (only for emission lines)
            if quantity not in ["V", "VDISP"]:
                rows_list.append(_2d_map_to_1d_list(data[0])); colnames.append(f"{quantity} (total)")
                if f"{quantity}_ERR" in hdulist_lzifu:
                    rows_list.append(_2d_map_to_1d_list(err[0])); colnames.append(f"{quantity} error (total)")
            # Fluxes/values for individual components
            for nn in range(lzifu_ncomponents):
                rows_list.append(_2d_map_to_1d_list(data[nn + 1])); colnames.append(f"{quantity} (component {nn + 1})")
                if f"{quantity}_ERR" in hdulist_lzifu:
                    rows_list.append(_2d_map_to_1d_list(err[nn + 1])); colnames.append(f"{quantity} error (component {nn + 1})")
        # Otherwise it's a 2D map
        elif data.ndim == 2:
            rows_list.append(_2d_map_to_1d_list(data)); colnames.append(f"{quantity}")
            if f"{quantity}_ERR" in hdulist_lzifu:
                rows_list.append(_2d_map_to_1d_list(err)); colnames.append(f"{quantity} error")

    #######################################################################
    # Scrape data from original data cube
    #######################################################################
    if onesided:
        with fits.open(datacube_fname) as hdulist_cube:
            header = hdulist_cube[0].header
            data_cube = hdulist_cube[0].data
            var_cube = hdulist_cube[1].data

            # Plate scale
            if header_B["CUNIT1"].lower().startswith("deg"):
                as_per_px = np.abs(header_B["CDELT1"]) * 3600.
            elif header_B["CUNIT1"].lower().startswith("arc"):
                as_per_px = np.abs(header_B["CDELT1"])

            # Wavelength values
            lambda_0_A = header["CRVAL3"] - header["CRPIX3"] * header["CDELT3"]
            dlambda_A = header["CDELT3"]
            N_lambda = header["NAXIS3"]
            lambda_vals_A = np.array(range(N_lambda)) * dlambda_A + lambda_0_A

            data_cube_B = data_cube.copy()
            var_cube_B = var_cube.copy()
            data_cube_R = data_cube.copy()
            var_cube_R = var_cube.copy()
            lambda_vals_B_A = lambda_vals_A.copy()
            lambda_vals_R_A = lambda_vals_A.copy()
    else:
        # Blue cube
        with fits.open(datacube_B_fname) as hdulist_B_cube:
            header_B = hdulist_B_cube[0].header
            data_cube_B = hdulist_B_cube[0].data
            var_cube_B = hdulist_B_cube[1].data

            # Plate scale
            if header_B["CUNIT1"].lower().startswith("deg"):
                as_per_px = np.abs(header_B["CDELT1"]) * 3600.
            elif header_B["CUNIT1"].lower().startswith("arc"):
                as_per_px = np.abs(header_B["CDELT1"])

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

    # Rest-frame wavelength arrays
    lambda_vals_R_rest_A = lambda_vals_R_A / (1 + z)
    lambda_vals_B_rest_A = lambda_vals_B_A / (1 + z)

    # Compute the D4000Å break
    if lambda_vals_B_rest_A[0] >= 3850 and lambda_vals_B_rest_A[-1] <= 4100:
        d4000_map, d4000_map_err = compute_d4000(data_cube=data_cube_B, var_cube=var_cube_B, lambda_vals_rest_A=lambda_vals_B_rest_A)
        rows_list.append(_2d_map_to_1d_list(d4000_map));     colnames.append(f"D4000")
        rows_list.append(_2d_map_to_1d_list(d4000_map_err)); colnames.append(f"D4000 error")

    # Compute the continuum intensity so that we can compute the Halpha equivalent width.
    # Continuum wavelength range taken from here: https://ui.adsabs.harvard.edu/abs/2019MNRAS.485.4024V/abstract
    if lambda_vals_R_rest_A[0] >= 6500 and lambda_vals_R_rest_A[-1] <= 6540:
        cont_HALPHA_map, cont_HALPHA_map_std, cont_HALPHA_map_err = compute_continuum_intensity(data_cube=data_cube_R, var_cube=var_cube_R, lambda_vals_rest_A=lambda_vals_R_rest_A, start_A=6500, stop_A=6540)
        rows_list.append(_2d_map_to_1d_list(cont_HALPHA_map));     colnames.append(f"HALPHA continuum")
        rows_list.append(_2d_map_to_1d_list(cont_HALPHA_map_std)); colnames.append(f"HALPHA continuum std. dev.")
        rows_list.append(_2d_map_to_1d_list(cont_HALPHA_map_err)); colnames.append(f"HALPHA continuum error")

    # Compute the approximate B-band continuum
    if lambda_vals_B_rest_A[0] >= 4000 and lambda_vals_B_rest_A[-1] <= 5000:
        cont_B_map, cont_B_map_std, cont_B_map_err = compute_continuum_intensity(data_cube=data_cube_B, var_cube=var_cube_B, lambda_vals_rest_A=lambda_vals_B_rest_A, start_A=4000, stop_A=5000)
        rows_list.append(_2d_map_to_1d_list(cont_B_map));     colnames.append(f"B-band continuum")
        rows_list.append(_2d_map_to_1d_list(cont_B_map_std)); colnames.append(f"B-band continuum std. dev.")
        rows_list.append(_2d_map_to_1d_list(cont_B_map_err)); colnames.append(f"B-band continuum error")

    # Compute the HALPHA amplitude-to-noise
    if lambda_vals_R_rest_A[0] >= 6562.8 and lambda_vals_R_rest_A[-1] <= 6562.8:
        v_map = hdulist_lzifu["V"].data  # Get velocity field from LZIFU fit
        AN_HALPHA_map = compute_HALPHA_amplitude_to_noise(data_cube=data_cube_R, var_cube=var_cube_R, lambda_vals_rest_A=lambda_vals_R_rest_A, v_map=v_map[0], dv=300)
        rows_list.append(_2d_map_to_1d_list(AN_HALPHA_map)); colnames.append(f"HALPHA A/N (measured)")

    ##########################################################
    # Add other stuff
    rows_list.append([gal] * len(x_c_list)); colnames.append("ID")
    rows_list.append(np.array(x_c_list).flatten()); colnames.append("x (pixels)")
    rows_list.append(np.array(y_c_list).flatten()); colnames.append("y (pixels)")  
    rows_list.append(np.array(x_c_list).flatten() * as_per_px); colnames.append("x (projected, arcsec)")
    rows_list.append(np.array(y_c_list).flatten() * as_per_px); colnames.append("y (projected, arcsec)")

    ##########################################################
    # Transpose so that each row represents a single pixel & each column a measured quantity.
    rows_arr = np.array(rows_list).T

    # Get rid of rows that are all NaNs
    bad_rows = np.all(np.isnan(rows_arr), axis=1)
    rows_good = rows_arr[~bad_rows]

    return rows_good, colnames, eline_list

#/////////////////////////////////////////////////////////////////////////////////
def make_lzifu_df(gals,
                  ncomponents,
                  df_fname,
                  sigma_inst_kms,
                  data_cube_path=None,
                  bin_type="default",
                  eline_SNR_min=5,
                  sigma_gas_SNR_min=3,
                  line_flux_SNR_cut=True,
                  missing_fluxes_cut=True,
                  line_amplitude_SNR_cut=True,
                  flux_fraction_cut=False,
                  sigma_gas_SNR_cut=True,
                  vgrad_cut=False,
                  stekin_cut=False,
                  correct_extinction=True,
                  nthreads_max=20):
    """TODO: WRITE DOCSTRING"""

    # TODO: input checking
    if not df_fname.endswith(".hd5"):
        df_fname += ".hd5"
    # TODO store valid values in settings?
    if (type(ncomponents) not in [int, str]) or (type(ncomponents) == str and ncomponents != "merge"):
        raise ValueError("ncomponents must be either an integer or 'merge'!")
    lzifu_ncomponents = ncomponents if type(ncomponents) == int else 3
    if bin_type not in ["default", "voronoi"]:
        raise ValueError("bin_type must be 'default' or 'voronoi'!")

    # NOTE: don't worry about metadata for now

    # TODO: scan for galaxies -OR- use inputs
    status_str = f"In lzifu2.make_lzifu_df() [ncomponents={ncomponents}, bin_type={bin_type}, eline_SNR_min={eline_SNR_min}]"

    ###############################################################################
    # Scrape measurements for each galaxy from FITS files
    ###############################################################################
    args_list = [[gg, gal, ncomponents, bin_type, data_cube_path, status_str] for gg, gal in enumerate(gals)]

    if len(gals) == 1:
        res_list = [_process_lzifu(args_list[0])]
    else:
        if nthreads_max > 1:
            print(f"{status_str}: Beginning pool...")
            pool = multiprocessing.Pool(min([nthreads_max, len(gals)]))
            res_list = np.array((pool.map(_process_lzifu, args_list)))
            pool.close()
            pool.join()
        else:
            print(f"{status_str}: Running sequentially...")
            res_list = []
            for args in args_list:
                res = _process_lzifu(args)
                res_list.append(res)

    ###############################################################################
    # Convert to a Pandas DataFrame
    ###############################################################################
    rows_list_all = [r[0] for r in res_list]
    colnames = res_list[0][1]
    eline_list = res_list[0][2] #TODO 
    df_spaxels = pd.DataFrame(np.vstack(tuple(rows_list_all)), columns=colnames)

    # Rename columns
    oldcol_v = [f"V (component {nn + 1})" for nn in range(lzifu_ncomponents)]
    oldcol_v_err = [f"V error (component {nn + 1})" for nn in range(lzifu_ncomponents)]
    oldcol_sigma = [f"VDISP (component {nn + 1})" for nn in range(lzifu_ncomponents)]
    oldcol_sigma_err = [f"VDISP error (component {nn + 1})" for nn in range(lzifu_ncomponents)]
    newcol_v = [f"v_gas (component {nn + 1})" for nn in range(lzifu_ncomponents)]
    newcol_v_err = [f"v_gas error (component {nn + 1})" for nn in range(lzifu_ncomponents)]
    newcol_sigma = [f"sigma_gas (component {nn + 1})" for nn in range(lzifu_ncomponents)]
    newcol_sigma_err = [f"sigma_gas error (component {nn + 1})" for nn in range(lzifu_ncomponents)]
    v_dict = dict(zip(oldcol_v, newcol_v))
    v_err_dict = dict(zip(oldcol_v_err, newcol_v_err))
    sigma_dict = dict(zip(oldcol_sigma, newcol_sigma))
    sigma_err_dict = dict(zip(oldcol_sigma_err, newcol_sigma_err))
    rename_dict = {**v_dict, **v_err_dict, **sigma_dict, **sigma_err_dict}
    df_spaxels = df_spaxels.rename(columns=rename_dict)

    Tracer()()

    ###############################################################################
    # Generic stuff: compute additional columns - extinction, metallicity, etc.
    ###############################################################################
    df_spaxels = add_columns(
        df_spaxels,
        eline_SNR_min=eline_SNR_min,
        sigma_gas_SNR_min=sigma_gas_SNR_min,
        eline_list=eline_list,
        line_flux_SNR_cut=line_flux_SNR_cut,
        missing_fluxes_cut=missing_fluxes_cut,
        line_amplitude_SNR_cut=line_amplitude_SNR_cut,
        flux_fraction_cut=flux_fraction_cut,
        sigma_gas_SNR_cut=sigma_gas_SNR_cut,
        vgrad_cut=vgrad_cut,
        stekin_cut=stekin_cut,
        correct_extinction=correct_extinction,
        sigma_inst_kms=sigma_inst_kms,
        nthreads_max=nthreads_max,
        debug=True) #TODO fix debug 


    ###############################################################################
    # TODO: Save
    ###############################################################################
    print(f"{status_str}: Saving to file...")
    try:
        df_spaxels.to_hdf(os.path.join(output_path, df_fname), key=f"{bin_type}, {ncomponents}-comp")
    except:
        print(f"{status_str}: ERROR: Unable to save to HDF file! Saving to .csv instead")
        df_spaxels.to_csv(os.path.join(output_path, df_fname.split("hd5")[0] + "csv"))
    return