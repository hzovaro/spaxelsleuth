import os
from path import Path
import warnings

import datetime
from astropy.io import fits
from astropy.cosmology import FlatLambdaCDM
import multiprocessing
import numpy as np
import pandas as pd

from spaxelsleuth.config import settings
from .generic import add_columns, compute_d4000, compute_continuum_intensity, compute_HALPHA_amplitude_to_noise, compute_v_grad

from IPython.core.debugger import Tracer

###############################################################################
# Paths
input_path = Path(settings["lzifu"]["input_path"])
output_path = Path(settings["lzifu"]["output_path"])
data_cube_path = Path(settings["lzifu"]["data_cube_path"])

#/////////////////////////////////////////////////////////////////////////////////
def add_metadata(df, df_metadata):
    """Merge an input DataFrame with that was created using make_lzifu_df()."""
    if "ID" not in df_metadata:
        raise ValueError("df_metadata must contain an 'ID' column!")
    
    df = df.merge(df_metadata, on="ID", how="left")

    return df

#/////////////////////////////////////////////////////////////////////////////////
def _process_lzifu(args):

    #######################################################################
    # Parse arguments
    #######################################################################
    _, gal, ncomponents, bin_type, data_cube_path, _ = args
    #TODO can we get ncomponents from SET?
    lzifu_ncomponents = ncomponents if type(ncomponents) == int else 3

    #######################################################################
    # Scrape outputs from LZIFU output
    #######################################################################
    hdulist_lzifu = fits.open(input_path + f"/{gal}_{ncomponents}_comp.fits")
    hdr = hdulist_lzifu[0].header

    # NOTE: for some reason, the SET extension is missing from the merge_comp FITS file so we need to get it from one of the others :/
    if ncomponents == "merge":
        hdulist_lzifu_1comp = fits.open(input_path + f"/{gal}_1_comp.fits")
        t = hdulist_lzifu_1comp["SET"].data  # Table storing fit parameters
    else:
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

    # Calculate cosmological distances from the redshift
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    D_A_Mpc = cosmo.angular_diameter_distance(z).value
    D_L_Mpc = cosmo.luminosity_distance(z).value
    kpc_per_arcsec = D_A_Mpc * 1e3 * np.pi / 180.0 / 3600.0

    #######################################################################
    # LOAD THE DATACUBES
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

    # Create masks of empty pixels
    im_empty_B = np.all(np.isnan(data_cube_B), axis=0)
    im_empty_R = np.all(np.isnan(data_cube_R), axis=0)
    im_empty = np.logical_and(im_empty_B, im_empty_R)

    # Get coordinate lists corresponding to non-empty spaxels
    ny, nx = im_empty.shape
    if bin_type == "default":
        y_c_list, x_c_list = np.where(~im_empty)
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
            if x > nx or y > ny:
                x = min([x, nx])
                y = min([y, ny])
            row[jj] = colmap[y, x]
        return row

    #######################################################################
    # SCRAPE LZIFU MEASUREMENTS
    #######################################################################
    # Get extension names
    extnames = [hdr[e] for e in hdr if e.startswith("EXT") and type(hdr[e]) == str]  #TODO check whether the [SII]6716,31 lines are here
    quantities = [e.rstrip("_ERR") for e in extnames if e.endswith("_ERR")] + ["CHI2", "DOF"]
    # NOTE: for some stupid reason, the SII6731_ERR extension is MISSING from the FITS file when ncomponents = "merge", so we need to add this in manually.
    if "SII6731" in extnames and "SII6731_ERR" not in extnames:
        quantities += ["SII6731"]
    rows_list = []
    colnames = []
    eline_list = [q for q in quantities if q not in ["V", "VDISP", "CHI2", "DOF"]]
    lzifu_ncomponents = hdulist_lzifu["V"].shape[0] - 1

    # Scrape the FITS file: emission line flues, velocity/velocity dispersion, fit quality
    for quantity in quantities:
        data = hdulist_lzifu[quantity].data
        if f"{quantity}_ERR" in hdulist_lzifu:
            err = hdulist_lzifu[f"{quantity}_ERR"].data
        elif quantity == "SII6731" and "SII6731_ERR" not in extnames:
            # For now, let's just copy the SII6716 error.
            err = hdulist_lzifu[f"SII6716_ERR"].data
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

    ##########################################################
    # COMPUTE QUANTITIES DIRECTLY FROM THE DATACUBES
    ##########################################################
    # Compute the D4000Å break
    if lambda_vals_B_rest_A[0] <= 3850 and lambda_vals_B_rest_A[-1] >= 4100:
        d4000_map, d4000_map_err = compute_d4000(data_cube=data_cube_B, var_cube=var_cube_B, lambda_vals_rest_A=lambda_vals_B_rest_A)
        rows_list.append(_2d_map_to_1d_list(d4000_map));     colnames.append(f"D4000")
        rows_list.append(_2d_map_to_1d_list(d4000_map_err)); colnames.append(f"D4000 error")

    # Compute the continuum intensity so that we can compute the Halpha equivalent width.
    # Continuum wavelength range taken from here: https://ui.adsabs.harvard.edu/abs/2019MNRAS.485.4024V/abstract
    if lambda_vals_R_rest_A[0] <= 6500 and lambda_vals_R_rest_A[-1] >= 6540:
        cont_HALPHA_map, cont_HALPHA_map_std, cont_HALPHA_map_err = compute_continuum_intensity(data_cube=data_cube_R, var_cube=var_cube_R, lambda_vals_rest_A=lambda_vals_R_rest_A, start_A=6500, stop_A=6540)
        rows_list.append(_2d_map_to_1d_list(cont_HALPHA_map));     colnames.append(f"HALPHA continuum")
        rows_list.append(_2d_map_to_1d_list(cont_HALPHA_map_std)); colnames.append(f"HALPHA continuum std. dev.")
        rows_list.append(_2d_map_to_1d_list(cont_HALPHA_map_err)); colnames.append(f"HALPHA continuum error")

    # Compute the approximate B-band continuum
    if lambda_vals_B_rest_A[0] <= 4000 and lambda_vals_B_rest_A[-1] >= 5000:
        cont_B_map, cont_B_map_std, cont_B_map_err = compute_continuum_intensity(data_cube=data_cube_B, var_cube=var_cube_B, lambda_vals_rest_A=lambda_vals_B_rest_A, start_A=4000, stop_A=5000)
        rows_list.append(_2d_map_to_1d_list(cont_B_map));     colnames.append(f"B-band continuum")
        rows_list.append(_2d_map_to_1d_list(cont_B_map_std)); colnames.append(f"B-band continuum std. dev.")
        rows_list.append(_2d_map_to_1d_list(cont_B_map_err)); colnames.append(f"B-band continuum error")

    # Compute the HALPHA amplitude-to-noise
    if lambda_vals_R_rest_A[0] <= 6562.8 and lambda_vals_R_rest_A[-1] >= 6562.8:
        v_map = hdulist_lzifu["V"].data  # Get velocity field from LZIFU fit
        AN_HALPHA_map = compute_HALPHA_amplitude_to_noise(data_cube=data_cube_R, var_cube=var_cube_R, lambda_vals_rest_A=lambda_vals_R_rest_A, v_map=v_map[0], dv=300)
        rows_list.append(_2d_map_to_1d_list(AN_HALPHA_map)); colnames.append(f"HALPHA A/N (measured)")

    ##########################################################
    # Add other stuff
    rows_list.append([gal] * len(x_c_list)); colnames.append("ID")
    rows_list.append([D_A_Mpc] * len(x_c_list)); colnames.append("D_A (Mpc)")
    rows_list.append([D_L_Mpc] * len(x_c_list)); colnames.append("D_L (Mpc)")
    rows_list.append([as_per_px] * len(x_c_list)); colnames.append("as_per_px")
    rows_list.append([kpc_per_arcsec] * len(x_c_list)); colnames.append("kpc per arcsec")
    rows_list.append([as_per_px**2] * len(x_c_list)); colnames.append("Bin size (square arcsec)")
    rows_list.append([(as_per_px * kpc_per_arcsec)**2] * len(x_c_list)); colnames.append("Bin size (square kpc)")
    rows_list.append([nx] * len(x_c_list)); colnames.append("N_x")
    rows_list.append([ny] * len(x_c_list)); colnames.append("N_y")
    rows_list.append(np.array(x_c_list).flatten()); colnames.append("x (pixels)")
    rows_list.append(np.array(y_c_list).flatten()); colnames.append("y (pixels)")  
    rows_list.append(np.array(x_c_list).flatten() * as_per_px); colnames.append("x (projected, arcsec)")
    rows_list.append(np.array(y_c_list).flatten() * as_per_px); colnames.append("y (projected, arcsec)")

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
    for cc in range(len(colnames)):
        if colnames[cc] in rename_dict:
            colnames[cc] = rename_dict[colnames[cc]]

    ##########################################################
    # Transpose so that each row represents a single pixel & each column a measured quantity.
    rows_arr = np.array(rows_list).T
    return rows_arr, colnames, eline_list

#/////////////////////////////////////////////////////////////////////////////////
def make_lzifu_df(gals,
                  ncomponents,
                  sigma_inst_kms,
                  df_fname=None,
                  bin_type=None,
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
                  nthreads_max=20,
                  debug=False):
    """TODO: WRITE DOCSTRING"""

    # TODO: input checking
    if df_fname is not None:
        if not df_fname.endswith(".hd5"):
            df_fname += ".hd5"
    else:
        # Input file name
        df_fname = f"lzifu_{bin_type}_{ncomponents}-comp"
        if correct_extinction:
            df_fname += "_extcorr"
        df_fname += f"_minSNR={eline_SNR_min}"
        if debug:
            df_fname += "_DEBUG"
        df_fname += ".hd5"

    if (type(ncomponents) not in [int, str]) or (type(ncomponents) == str and ncomponents != "merge"):
        raise ValueError("ncomponents must be either an integer or 'merge'!")
    if bin_type != "default":
        raise ValueError("bin_types other than 'default' have not yet been implemented!")

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

    # Cast to float data types 
    for col in df_spaxels.columns:
        # Check if column values can be cast to float 
        df_spaxels[col] = pd.to_numeric(df_spaxels[col], errors="coerce")

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
        compute_sfr=True,
        sigma_inst_kms=sigma_inst_kms,
        nthreads_max=nthreads_max,
        base_missing_flux_components_on_HALPHA=False,  # NOTE: this is important!!
        debug=debug)

    ###############################################################################
    # Add extra columns
    ###############################################################################
    df_spaxels["x, y (pixels)"] = list(zip(df_spaxels["x (pixels)"], df_spaxels["y (pixels)"]))

    ###############################################################################
    # Save to file
    ###############################################################################
    print(f"{status_str}: Saving to file {df_fname}...")
    df_spaxels.to_hdf(os.path.join(output_path, df_fname), key=f"{bin_type}, {ncomponents}-comp")
    print(f"{status_str}: Finished!")
    
    return

#/////////////////////////////////////////////////////////////////////////////////
def load_lzifu_df(ncomponents=None,
                  bin_type=None,
                  correct_extinction=None,  
                  eline_SNR_min=None,
                  debug=False,
                  df_fname=None, key=None):

    #######################################################################
    # INPUT CHECKING
    #######################################################################
    # Input file name
    if df_fname is not None:
        warnings.warn(f"Loading DataFrame from user-provided filename {os.path.join(output_path, df_fname)} which may not correspond to the provided ncomponents, bin_type, etc. Proceed with caution!", RuntimeWarning)
        if not df_fname.endswith(".hd5"):
            df_fname += ".hd5"
    else:
        if bin_type not in settings["lzifu"]["bin_types"]:
            raise ValueError(f"bin_type {bin_type} is invalid for survey lzifu!")
        # Input file name
        df_fname = f"lzifu_{bin_type}_{ncomponents}-comp"
        if correct_extinction:
            df_fname += "_extcorr"
        df_fname += f"_minSNR={eline_SNR_min}"
        if debug:
            df_fname += "_DEBUG"
        df_fname += ".hd5"

    if not os.path.exists(os.path.join(output_path, df_fname)):
        raise FileNotFoundError(f"File {os.path.join(output_path, df_fname)} does does not exist!")

    # Load the data frame
    t = os.path.getmtime(os.path.join(output_path, df_fname))
    print(f"In load_lzifu_df(): Loading DataFrame from file {os.path.join(output_path, df_fname)} [last modified {datetime.datetime.fromtimestamp(t)}]...")
    if key is not None:
        df = pd.read_hdf(os.path.join(output_path, df_fname), key=key)
    else:
        df = pd.read_hdf(os.path.join(output_path, df_fname))

    # Add "metadata" columns to the DataFrame
    df["survey"] = "lzifu"
    df["ncomponents"] = ncomponents
    df["bin_type"] = bin_type
    df["debug"] = debug

    # Return
    print("In load_lzifu_df(): Finished!")
    return df.sort_index()