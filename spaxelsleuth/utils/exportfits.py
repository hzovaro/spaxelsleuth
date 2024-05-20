from astropy.io import fits
import datetime
from itertools import product
import numpy as np
from pathlib import Path
from time import time
import warnings

from spaxelsleuth import __version__
from spaxelsleuth.config import settings
from spaxelsleuth.utils.linefns import bpt_dict
from spaxelsleuth.io.io import load_metadata_df, load_df

import logging
logger = logging.getLogger(__name__)

from IPython.core.debugger import set_trace

"""
TODO: units for each quantity 
"""

# spaxelsleuth parameters to store in the PrimaryHDU
spaxelsleuth_metadata_cols = [
    "eline_SNR_min",
    "eline_ANR_min",
    "sigma_gas_SNR_min",
    "line_flux_SNR_cut",
    "missing_fluxes_cut",
    "missing_kinematics_cut",
    "line_amplitude_SNR_cut",
    "flux_fraction_cut",
    "vgrad_cut",
    "sigma_gas_SNR_cut",
    "stekin_cut",
    "survey",
    "ncomponents",
    "as_per_px",
    "bin_type",
    "flux_units",
    "continuum_units",
    "correct_extinction",
]

# Dict for converting DataFrame column names into <7-character strings for use as FITS header keys
header_strs = {
    "RA (J2000)": "RA_DEG",
    "Dec (J2000)": "DEC_DEG",
    "z": "Z",
    "D_A (Mpc)": "DA_MPC",
    "D_L (Mpc)": "DL_MPC",
    "N_x": "NX_PX",
    "N_y": "NY_PX",
    "x_0 (pixels)": "X0_PX",
    "y_0 (pixels)": "Y0_PX",
    "x_0 (arcsec)": "X0_AS",
    "y_0 (arcsec)": "Y0_AS",
    "kpc per arcsec": "KPCPRAS",
    "Field": "FIELD",
    "Tile": "TILE",
    "Plate ID": "PLATEID",
    "Spectrograph": "INSTRUME",
    "Bundle": "IFUPROBE",
    "Blue data cube FITS file": "BCUBE",
    "Red data cube FITS file": "RCUBE",
    "Stellar kinematics FITS file": "STEKIN",
    "Blue continuum fit FITS file": "BCONT",
    "Red continuum fit FITS file": "RCONT",
    "1-component fit emission line FITS file": "1COMP",
    "2-component fit emission line FITS file": "2COMP",
    "3-component fit emission line FITS file": "3COMP",
    "rec-component fit emission line FITS file": "RECCOMP",
}


# TODO these need to be added back in, in a new section
bad_keys = [
    "Blue data cube FITS file",
    "Red data cube FITS file",
    "Stellar kinematics FITS file",
    "Blue continuum fit FITS file",
    "Red continuum fit FITS file",
    "1-component fit emission line FITS file",
    "2-component fit emission line FITS file",
    "3-component fit emission line FITS file",
    "rec-component fit emission line FITS file",
]


def replace_unicode_chars(s):
    s = s.replace("å", "Ang").replace("Å", "Ang")
    for c in s:
        if ord(c) >= 128:
            s = s.replace(c, "")
    return s


def export_fits(
    survey,
    gals=None, 
    cols_to_store_no_suffixes=None, 
    include_data_cubes=False,
    fname_suffix="",
    **kwargs,
):
    """Export a multi-extension FITS file from columns in df."""

    # Load DataFrame
    df_metadata = load_metadata_df(survey=survey)
    df = load_df(survey=survey, **kwargs)

    # Get number of components
    if df["ncomponents"].unique()[0] == "rec":
        ncomponents_max = 3
    else:
        ncomponents_max = df["ncomponents"].unique()[0]

    # Get numeric-type columns that can be stored as extensions in the FITS file
    all_numeric_cols = [c for c in df.columns if df[c].dtype != "object"]

    # User to optionally specify numeric-type columns. These must have their suffixes removed (i.e., "HALPHA" is OK, but "HALPHA (component 1)" is not)
    if cols_to_store_no_suffixes is None:
        cols_to_store_no_suffixes = list(
            set(
                [
                    c.split(" (component ")[0].split(" (total)")[0]
                    for c in all_numeric_cols
                ]
            )
        )
    assert not any(
        ["(component" in c for c in cols_to_store_no_suffixes]
    ), "columns must not contain any suffixes, e.g. '(component N)' or '(total)'!"
    assert not any(
        ["(total" in c for c in cols_to_store_no_suffixes]
    ), "columns must not contain any suffixes, e.g. '(component N)' or '(total)'!"

    # Figure out which columns have associated components
    cols_2d_no_suffixes = []
    cols_3d_no_suffixes = (
        []
    )  # Contains all columns with associated "total" or "per-components" quantities
    for col in cols_to_store_no_suffixes:
        if (
            any(
                [
                    f"{col} (component {component})" in df.columns
                    for component in range(1, ncomponents_max + 1)
                ]
            )
            or f"{col} (total)" in df.columns
        ):
            cols_3d_no_suffixes.append(col)
        else:
            assert col in df.columns, f"column {col} not found in the DataFrame!"
            cols_2d_no_suffixes.append(col)

    # Determine list of galaxies for which to create FITS files
    if gals is None:
        gals = df["ID"].unique()

    for gal in gals:
        # Get subset of rows belonging to this galaxy
        df_gal = df.loc[df["ID"] == gal]

        # Get FITS file dimensions & coordinates
        nx = df_gal["N_x"].unique()[0].astype(int)
        ny = df_gal["N_y"].unique()[0].astype(int)
        as_per_px = df_gal["as_per_px"].unique()[0]
        ra = df_gal["RA (J2000)"].unique()[0]
        dec = df_gal["Dec (J2000)"].unique()[0]

        # Create FITS file
        hdulist = fits.HDUList()

        # Add galaxy metadata to Primary HDU
        phdu = fits.PrimaryHDU()
        phdu.header["SURVEY"] = survey
        lastkey = "SURVEY"
        for col in [c for c in df_metadata.columns if c not in bad_keys]:
            value = df_metadata.loc[gal, col]
            if isinstance(value, float):
                if np.isnan(value):
                    value = "NaN"
            if col in header_strs:
                key = header_strs[col]
            else:
                key = col
            phdu.header[key] = (
                replace_unicode_chars(value) if type(value) == str else value,
                col,
            )
        # Append section header
        # I can't believe that this is the only way to get the comment to go where I want it to go... fml
        phdu.header.insert(lastkey, ("", ""), after=True)
        phdu.header.insert(lastkey, ("", "Galaxy metadata"), after=True)
        phdu.header.insert(lastkey, ("", ""), after=True)

        # Add spaxelsleuth metadata to Primary HDU
        lastkey = key
        for col in spaxelsleuth_metadata_cols:
            key = "hierarch " + col
            value = df_gal[col].unique()[0]
            phdu.header[key] = (
                replace_unicode_chars(value) if type(value) == str else value
            )
        # Append section header
        phdu.header["NOTES"] = (
            "See https://github.com/hzovaro/spaxelsleuth/wiki/Column-descriptions/ for parameter descriptions"
        )
        phdu.header.insert(lastkey, ("", ""), after=True)
        phdu.header.insert(lastkey, ("", "spaxelsleuth parameters"), after=True)
        phdu.header.insert(lastkey, ("", ""), after=True)

        # Add filenames to Primary HDU
        lastkey = "NOTES"
        for col in bad_keys:
            value = df_metadata.loc[gal, col]
            if col in header_strs:
                key = header_strs[col]
            else:
                key = col
            phdu.header[key] = (
                replace_unicode_chars(value) if type(value) == str else value,
                col,
            )
        # Append section header
        # I can't believe that this is the only way to get the comment to go where I want it to go... fml
        phdu.header.insert(lastkey, ("", ""), after=True)
        phdu.header.insert(lastkey, ("", "Input FITS files"), after=True)
        phdu.header.insert(lastkey, ("", ""), after=True)

        # Add other info
        lastkey = key
        phdu.header["DATE"] = (
            f"{datetime.datetime.fromtimestamp(time())}",
            "Date/time modified",
        )
        phdu.header["FNAME"] = (str(df["fname"].unique()[0]), "Input Spaxelsleuth DataFrame filename")
        phdu.header["TSTAMP"] = (str(df["timestamp"].unique()[0]), "Input Spaxelsleuth DataFrame timestamp")
        phdu.header["VERSION"] = (__version__, "Spaxelsleuth version")
        phdu.header["AUTHOR"] = "Henry Zovaro"
        # Append section header
        phdu.header.insert(lastkey, ("", ""), after=True)
        phdu.header.insert(lastkey, ("", "Other info"), after=True)
        phdu.header.insert(lastkey, ("", ""), after=True)
        hdulist.append(phdu)

        # Add the data and variance cubes
        if include_data_cubes:
            for side in ["blue", "red"]:
                # Open the DataCube
                datacube_fname = df_metadata.loc[gal, "Blue data cube FITS file"]
                with fits.open(datacube_fname) as hdulist_cube:
                    header = hdulist_cube[0].header
                    data_cube = hdulist_cube[0].data
                    var_cube = hdulist_cube[1].data

                    # Create the HDU
                    hdu = fits.ImageHDU(data=data_cube)
                    for axis, key in product(
                        ["1", "2", "3"], ["CRVAL", "CRPIX", "CDELT", "CUNIT"]
                    ):
                        hdu.header[key + axis] = header[key + axis]
                    hdu.header["BUNIT"] = header["BUNIT"]
                    hdu.header["EXTNAME"] = f"Data cube - {side}"
                    hdulist.append(hdu)
                    hdu = fits.ImageHDU(data=var_cube)
                    for axis, key in product(
                        ["1", "2", "3"], ["CRVAL", "CRPIX", "CDELT", "CUNIT"]
                    ):
                        hdu.header[key + axis] = header[key + axis]
                    hdu.header["BUNIT"] = header["BUNIT"]
                    hdu.header["EXTNAME"] = f"Variance cube - {side}"
                    hdulist.append(hdu)

        # Extract 2D maps corresponding to each column in df_gal
        _2d_maps = {}
        df_arr = df_gal.loc[:, all_numeric_cols].to_numpy(dtype="float")
        xs = [int(x) for x in df_gal["x (pixels)"].values]
        ys = [int(y) for y in df_gal["y (pixels)"].values]
        for cc, col in enumerate(all_numeric_cols):
            colmap = np.full((ny, nx), np.nan)
            for rr in range(df_arr.shape[0]):
                colmap[ys[rr], xs[rr]] = df_arr[rr, cc]
            _2d_maps[col] = colmap

        # Construct 3D arrays in cases where the quantity is associated with multiple components
        for col in cols_to_store_no_suffixes:
            if col in cols_2d_no_suffixes:  # then store the 2D array
                colmap = _2d_maps[col]
            else:
                colmap = np.full((ncomponents_max + 1, ny, nx), np.nan)
                if f"{col} (total)" in all_numeric_cols:
                    colmap[0] = _2d_maps[f"{col} (total)"]
                for component in range(1, ncomponents_max + 1):
                    if f"{col} (component {component})" in all_numeric_cols:
                        colmap[component] = _2d_maps[f"{col} (component {component})"]

            # Create the HDU
            hdu = fits.ImageHDU(data=colmap)
            hdu.header["CRVAL1"] = ra
            hdu.header["CRVAL2"] = dec
            hdu.header["CDELT1"] = -as_per_px / 3600.0
            hdu.header["CDELT2"] = as_per_px / 3600.0
            hdu.header["CUNIT1"] = "deg"
            hdu.header["CUNIT2"] = "deg"
            hdu.header["CTYPE1"] = "RA---TAN"
            hdu.header["CTYPE2"] = "DEC--TAN"
            hdu.header["EXTNAME"] = replace_unicode_chars(col)

            # Add a comment in the header to say what data is stored in each slice
            if col not in cols_2d_no_suffixes:
                if f"{col} (total)" in all_numeric_cols:
                    hdu.header["SLICE0"] = "Total"
                else:
                    hdu.header["SLICE0"] = "N/A"
                for component in range(1, ncomponents_max + 1):
                    if f"{col} (component {component})" in all_numeric_cols:
                        hdu.header[f"SLICE{component}"] = f"Component {component}"
                    else:
                        hdu.header[f"SLICE{component}"] = "N/A"

            # Add explainers for "numeric" versions of quantitative columns e.g. BPT
            if col.startswith("BPT (numeric)"):
                for cat in bpt_dict.keys():
                    hdu.header[f"CAT_{int(float(cat))}"] = (
                        bpt_dict[cat],
                        f"BPT category corresponding to pixels with value {int(float(cat))}",
                    )

            # Store in the FITS file
            hdulist.append(hdu)

        # Save to file
        with warnings.catch_warnings():
            warnings.filterwarnings(
                action="ignore",
                category=fits.verify.VerifyWarning,
                message="Card is too long, comment will be truncated.",
            )
            fits_path = Path(settings[survey]["fits_output_path"])
            if len(fname_suffix) > 0:
                fits_fname = f"{gal}_data_products_{fname_suffix}.fits"
            else:
                fits_fname = f"{gal}_data_products.fits"
            logger.info(
                f"Creating FITS file {fits_path / fits_fname} for galaxy {gal}..."
            )
            hdulist.writeto(
                fits_path / fits_fname, overwrite=True, output_verify="ignore"
            )
