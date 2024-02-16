from astropy.io import fits
from astropy.cosmology import FlatLambdaCDM
import numpy as np
import os
from pathlib import Path

from spaxelsleuth.config import settings
from spaxelsleuth.utils.continuum import compute_d4000, compute_continuum_intensity
from spaxelsleuth.utils.dqcut import compute_measured_HALPHA_amplitude_to_noise
from spaxelsleuth.utils.misc import _2d_map_to_1d_list

import logging
logger = logging.getLogger(__name__)

# Paths
input_path = Path(settings["lzifu"]["input_path"])
output_path = Path(settings["lzifu"]["output_path"])
data_cube_path = Path(settings["lzifu"]["data_cube_path"])


def add_metadata(df, df_metadata):
    """Merge an input DataFrame with that was created using make_lzifu_df()."""
    if "ID" not in df_metadata:
        raise ValueError("df_metadata must contain an 'ID' column!")
    df = df.merge(df_metadata, on="ID", how="left")
    return df


def process_galaxies(args):

    # Parse arguments
    gg, gal, ncomponents, bin_type, df_metadata, kwargs = args 
    lzifu_ncomponents = ncomponents if type(ncomponents) == int else 3

    # Scrape outputs from LZIFU output
    hdulist_lzifu = fits.open(input_path / f"{gal}_{ncomponents}_comp.fits")
    hdr = hdulist_lzifu[0].header

    # NOTE: for some reason, the SET extension is missing from the merge_comp FITS file so we need to get it from one of the others :/
    if ncomponents == "merge":
        hdulist_lzifu_1comp = fits.open(input_path / f"{gal}_1_comp.fits")
        t = hdulist_lzifu_1comp["SET"].data  # Table storing fit parameters
    else:
        t = hdulist_lzifu["SET"].data  # Table storing fit parameters

    # Was the fit one-sided or two-sided (i.e., red and blue cubes?)
    # In either case, use SET to find the original datacubes used in the fit.
    onesided = True if t["ONLY_1SIDE"][0] == 1 else False
    if onesided:
        datacube_fname = data_cube_path / f"{gal}.fits"
        datacube_fname_alt = data_cube_path / f"{gal}.fits.gz"
        if not os.path.exists(datacube_fname):
            if os.path.exists(datacube_fname_alt):
                datacube_fname = datacube_fname_alt
            else:
                raise FileNotFoundError(
                    "Cannot find files {datacube_fname} or {datacube_fname_alt}!"
                )
    else:
        datacube_fnames = []
        for side in ["B", "R"]:
            datacube_fname = data_cube_path / f"{gal}_{side}.fits"
            datacube_fname_alt = data_cube_path / f"{gal}_{side}.fits.gz"
            if not os.path.exists(datacube_fname):
                if os.path.exists(datacube_fname_alt):
                    datacube_fname = datacube_fname_alt
                else:
                    raise FileNotFoundError(
                        "Cannot find files {datacube_fname} or {datacube_fname_alt}!"
                    )
            datacube_fnames.append(datacube_fname)

        datacube_B_fname, datacube_R_fname = datacube_fnames

    # Redshift
    z = t["Z"][0]

    # Calculate cosmological distances from the redshift
    cosmo = FlatLambdaCDM(H0=settings["H_0"], Om0=settings["Omega_0"])
    D_A_Mpc = cosmo.angular_diameter_distance(z).value
    D_L_Mpc = cosmo.luminosity_distance(z).value
    kpc_per_arcsec = D_A_Mpc * 1e3 * np.pi / 180.0 / 3600.0

    # LOAD THE DATACUBES
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
            lambda_0_A = header_B[
                "CRVAL3"] - header_B["CRPIX3"] * header_B["CDELT3"]
            dlambda_A = header_B["CDELT3"]
            N_lambda = header_B["NAXIS3"]
            lambda_vals_B_A = np.array(
                range(N_lambda)) * dlambda_A + lambda_0_A

        # Red cube
        with fits.open(datacube_R_fname) as hdulist_R_cube:
            header_R = hdulist_R_cube[0].header
            data_cube_R = hdulist_R_cube[0].data
            var_cube_R = hdulist_R_cube[1].data

            # Wavelength values
            lambda_0_A = header_R[
                "CRVAL3"] - header_R["CRPIX3"] * header_R["CDELT3"]
            dlambda_A = header_R["CDELT3"]
            N_lambda = header_R["NAXIS3"]
            lambda_vals_R_A = np.array(
                range(N_lambda)) * dlambda_A + lambda_0_A

    # Rest-frame wavelength arrays
    lambda_vals_R_rest_A = lambda_vals_R_A / (1 + z)
    lambda_vals_B_rest_A = lambda_vals_B_A / (1 + z)

    # Create masks of empty pixels
    im_empty_B = np.all(np.isnan(data_cube_B), axis=0)
    im_empty_R = np.all(np.isnan(data_cube_R), axis=0)
    im_empty = np.logical_and(im_empty_B, im_empty_R)

    # Get coordinate lists corresponding to non-empty spaxels
    ny, nx = im_empty.shape
    y_c_list, x_c_list = np.where(~im_empty)

    # SCRAPE LZIFU MEASUREMENTS
    # Get extension names
    extnames = [
        hdr[e] for e in hdr if e.startswith("EXT") and type(hdr[e]) == str
    ]  #TODO check whether the [SII]6716,31 lines are here
    quantities = [e.rstrip("_ERR")
                  for e in extnames if e.endswith("_ERR")] + ["CHI2", "DOF"]
    # NOTE: for some stupid reason, the SII6731_ERR extension is MISSING from the FITS file when ncomponents = "merge", so we need to add this in manually.
    if "SII6731" in extnames and "SII6731_ERR" not in extnames:
        quantities += ["SII6731"]
    rows_list = []
    colnames = []
    eline_list = [
        q for q in quantities if q not in ["V", "VDISP", "CHI2", "DOF"]
    ]
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
                rows_list.append(_2d_map_to_1d_list(data[0], x_c_list, y_c_list, nx, ny))
                colnames.append(f"{quantity} (total)")
                if f"{quantity}_ERR" in hdulist_lzifu:
                    rows_list.append(_2d_map_to_1d_list(err[0], x_c_list, y_c_list, nx, ny))
                    colnames.append(f"{quantity} error (total)")
            # Fluxes/values for individual components
            for nn in range(lzifu_ncomponents):
                rows_list.append(_2d_map_to_1d_list(data[nn + 1], x_c_list, y_c_list, nx, ny))
                colnames.append(f"{quantity} (component {nn + 1})")
                if f"{quantity}_ERR" in hdulist_lzifu:
                    rows_list.append(_2d_map_to_1d_list(err[nn + 1], x_c_list, y_c_list, nx, ny))
                    colnames.append(f"{quantity} error (component {nn + 1})")
        # Otherwise it's a 2D map
        elif data.ndim == 2:
            rows_list.append(_2d_map_to_1d_list(data, x_c_list, y_c_list, nx, ny))
            colnames.append(f"{quantity}")
            if f"{quantity}_ERR" in hdulist_lzifu:
                rows_list.append(_2d_map_to_1d_list(err, x_c_list, y_c_list, nx, ny))
                colnames.append(f"{quantity} error")

    # COMPUTE QUANTITIES DIRECTLY FROM THE DATACUBES
    # NOTE: because we do not have access to stellar velocities, we assume no peculiar velocities within the object when calculating continuum quantities
    v_star_map = np.zeros(data_cube_B.shape[1:])
    # Compute the D4000Å break
    d4000_map, d4000_map_err = compute_d4000(
        data_cube=data_cube_B,
        var_cube=var_cube_B,
        lambda_vals_rest_A=lambda_vals_B_rest_A,
        v_star_map=v_star_map)
    rows_list.append(_2d_map_to_1d_list(d4000_map, x_c_list, y_c_list, nx, ny))
    colnames.append(f"D4000")
    rows_list.append(_2d_map_to_1d_list(d4000_map_err, x_c_list, y_c_list, nx, ny))
    colnames.append(f"D4000 error")

    # Compute the continuum intensity so that we can compute the Halpha equivalent width.
    # Continuum wavelength range taken from here: https://ui.adsabs.harvard.edu/abs/2019MNRAS.485.4024V/abstract
    cont_HALPHA_map, cont_HALPHA_map_std, cont_HALPHA_map_err = compute_continuum_intensity(
        data_cube=data_cube_R,
        var_cube=var_cube_R,
        lambda_vals_rest_A=lambda_vals_R_rest_A,
        start_A=6500,
        stop_A=6540,
        v_map=v_star_map)
    rows_list.append(_2d_map_to_1d_list(cont_HALPHA_map, x_c_list, y_c_list, nx, ny))
    colnames.append(f"HALPHA continuum")
    rows_list.append(_2d_map_to_1d_list(cont_HALPHA_map_std, x_c_list, y_c_list, nx, ny))
    colnames.append(f"HALPHA continuum std. dev.")
    rows_list.append(_2d_map_to_1d_list(cont_HALPHA_map_err, x_c_list, y_c_list, nx, ny))
    colnames.append(f"HALPHA continuum error")

    # Compute the approximate B-band continuum
    cont_B_map, cont_B_map_std, cont_B_map_err = compute_continuum_intensity(
        data_cube=data_cube_B,
        var_cube=var_cube_B,
        lambda_vals_rest_A=lambda_vals_B_rest_A,
        start_A=4000,
        stop_A=5000,
        v_map=v_star_map)
    rows_list.append(_2d_map_to_1d_list(cont_B_map, x_c_list, y_c_list, nx, ny))
    colnames.append(f"B-band continuum")
    rows_list.append(_2d_map_to_1d_list(cont_B_map_std, x_c_list, y_c_list, nx, ny))
    colnames.append(f"B-band continuum std. dev.")
    rows_list.append(_2d_map_to_1d_list(cont_B_map_err, x_c_list, y_c_list, nx, ny))
    colnames.append(f"B-band continuum error")

    # Compute the HALPHA amplitude-to-noise
    v_map = hdulist_lzifu["V"].data  # Get velocity field from LZIFU fit
    AN_HALPHA_map = compute_measured_HALPHA_amplitude_to_noise(
        data_cube=data_cube_R,
        var_cube=var_cube_R,
        lambda_vals_rest_A=lambda_vals_R_rest_A,
        v_star_map=v_star_map,
        v_map=v_map[0],
        dv=300)
    rows_list.append(_2d_map_to_1d_list(AN_HALPHA_map, x_c_list, y_c_list, nx, ny))
    colnames.append(f"HALPHA A/N (measured)")

    # Add other stuff
    rows_list.append([gg] * len(x_c_list))
    colnames.append("ID (numeric)")
    rows_list.append([gal] * len(x_c_list))
    colnames.append("ID")
    rows_list.append([D_A_Mpc] * len(x_c_list))
    colnames.append("D_A (Mpc)")
    rows_list.append([D_L_Mpc] * len(x_c_list))
    colnames.append("D_L (Mpc)")
    rows_list.append([as_per_px] * len(x_c_list))
    colnames.append("as_per_px")
    rows_list.append([kpc_per_arcsec] * len(x_c_list))
    colnames.append("kpc per arcsec")
    rows_list.append([as_per_px**2] * len(x_c_list))
    colnames.append("Bin size (square arcsec)")
    rows_list.append([(as_per_px * kpc_per_arcsec)**2] * len(x_c_list))
    colnames.append("Bin size (square kpc)")
    rows_list.append([nx] * len(x_c_list))
    colnames.append("N_x")
    rows_list.append([ny] * len(x_c_list))
    colnames.append("N_y")
    rows_list.append(np.array(x_c_list).flatten())
    colnames.append("x (pixels)")
    rows_list.append(np.array(y_c_list).flatten())
    colnames.append("y (pixels)")
    rows_list.append(np.array(x_c_list).flatten() * as_per_px)
    colnames.append("x (projected, arcsec)")
    rows_list.append(np.array(y_c_list).flatten() * as_per_px)
    colnames.append("y (projected, arcsec)")

    # Rename columns
    oldcol_v = [f"V (component {nn + 1})" for nn in range(lzifu_ncomponents)]
    oldcol_v_err = [
        f"V error (component {nn + 1})" for nn in range(lzifu_ncomponents)
    ]
    oldcol_sigma = [
        f"VDISP (component {nn + 1})" for nn in range(lzifu_ncomponents)
    ]
    oldcol_sigma_err = [
        f"VDISP error (component {nn + 1})" for nn in range(lzifu_ncomponents)
    ]
    newcol_v = [
        f"v_gas (component {nn + 1})" for nn in range(lzifu_ncomponents)
    ]
    newcol_v_err = [
        f"v_gas error (component {nn + 1})" for nn in range(lzifu_ncomponents)
    ]
    newcol_sigma = [
        f"sigma_gas (component {nn + 1})" for nn in range(lzifu_ncomponents)
    ]
    newcol_sigma_err = [
        f"sigma_gas error (component {nn + 1})"
        for nn in range(lzifu_ncomponents)
    ]
    v_dict = dict(zip(oldcol_v, newcol_v))
    v_err_dict = dict(zip(oldcol_v_err, newcol_v_err))
    sigma_dict = dict(zip(oldcol_sigma, newcol_sigma))
    sigma_err_dict = dict(zip(oldcol_sigma_err, newcol_sigma_err))
    rename_dict = {**v_dict, **v_err_dict, **sigma_dict, **sigma_err_dict}
    for cc in range(len(colnames)):
        if colnames[cc] in rename_dict:
            colnames[cc] = rename_dict[colnames[cc]]

    # Transpose so that each row represents a single pixel & each column a measured quantity.
    rows_arr = np.array(rows_list).T

    logger.info(f"Finished processing galaxy {gal} ({gg})")

    return rows_arr, colnames