# Imports
from astropy.io import fits
from astropy.cosmology import FlatLambdaCDM
from glob import glob
import numpy as np
import os
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from spaxelsleuth.config import settings
from spaxelsleuth.utils.continuum import compute_d4000, compute_continuum_intensity
from spaxelsleuth.utils.dqcut import compute_measured_HALPHA_amplitude_to_noise
from spaxelsleuth.utils.velocity import compute_v_grad

import logging
logger = logging.getLogger(__name__)

# Paths
input_path = Path(settings["hector"]["input_path"])
output_path = Path(settings["hector"]["output_path"])
data_cube_path = Path(settings["hector"]["data_cube_path"])
eline_fit_path = input_path / "emission_cubes"
stekin_path = input_path / "initial_stel_kin"
continuum_fit_path = input_path / "cont_subtracted"


def _2d_map_to_1d_list(colmap, x_c_list, y_c_list, nx, ny):
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


def get_filenames():
    """
    Return a DataFrame containing galaxies with at least one data product + associated file names.
    Putting this in its own function because the file structure is complicated and subject to change...
    """
    # Get ALL filenames
    all_data_cube_files = glob(str(data_cube_path) + "/**/*.fits", recursive=True)
    data_cube_files_B = [Path(f) for f in all_data_cube_files if "blue" in f]
    data_cube_files_R = [Path(f) for f in all_data_cube_files if "red" in f]
    
    all_continuum_fit_files = glob(str(continuum_fit_path) + "/**/*.fits", recursive=True)
    continuum_fit_files_B = [Path(f) for f in all_continuum_fit_files if "blue" in f]
    continuum_fit_files_R = [Path(f) for f in all_continuum_fit_files if "red" in f]
    
    stekin_files = [Path(f) for f in glob(str(stekin_path) + "/**/*.fits", recursive=True)]

    all_eline_fit_files = glob(str(eline_fit_path) + "/**/*.fits", recursive=True)
    eline_fit_files_reccomp = [Path(f) for f in all_eline_fit_files if f.endswith(".fits") and "reccomp" in f]
    eline_fit_files_1comp = [Path(f) for f in all_eline_fit_files if f.endswith(".fits") and "1comp" in f]
    eline_fit_files_2comp = [Path(f) for f in all_eline_fit_files if f.endswith(".fits") and "2comp" in f]
    eline_fit_files_3comp = [Path(f) for f in all_eline_fit_files if f.endswith(".fits") and "3comp" in f]

    # Now, check that each of these has each of Gabby's data products 
    ids_with_initial_stel_kin = [f.stem.split("_initial_kinematics")[0] for f in stekin_files]
    # ids_with_initial_stel_kin = [g for g in set([f.split("_initial_kinematics")[0] for f in os.listdir(stekin_path) if f.endswith(".fits")])    ]
    logger.info(f"len(ids_with_initial_stel_kin) = {len(ids_with_initial_stel_kin)}")

    ids_with_cont_subtracted = [f.stem.split("_blue_stel_subtract_final")[0] for f in continuum_fit_files_B]
    # ids_with_cont_subtracted = [g for g in set([f.split("_blue_stel_subtract_final")[0] for f in os.listdir(continuum_fit_path) if f.endswith(".fits") and "blue" in f])    ]
    logger.info(f"len(ids_with_cont_subtracted) = {len(ids_with_cont_subtracted)}")

    ids_with_emission_cubes = [f.stem.split("_reccomp")[0] for f in eline_fit_files_reccomp]
    # ids_with_emission_cubes = [g for g in set([f.split("_reccomp")[0] for f in os.listdir(eline_fit_path) if f.endswith(".fits") and "rec" in f])]
    logger.info(f"len(ids_with_emission_cubes) = {len(ids_with_emission_cubes)}")

    ids_with_all_data_products = list(set(ids_with_initial_stel_kin) & set(ids_with_cont_subtracted) & set(ids_with_emission_cubes))
    logger.info(f"len(ids_with_all_data_products) = {len(ids_with_all_data_products)}")

    # Susbet of IDs containing at least one data product
    ids_all = list(set(ids_with_initial_stel_kin) | set(ids_with_cont_subtracted) | set(ids_with_emission_cubes))
    df_filenames = pd.DataFrame(index=ids_all)

    file_types = [
        "Blue data cube FITS file",
        "Red data cube FITS file",
        "Blue continuum fit FITS file",
        "Red continuum fit FITS file",
        "Stellar kinematics FITS file",
        f"rec-component fit emission line FITS file",
        f"1-component fit emission line FITS file",
        f"2-component fit emission line FITS file",
        f"3-component fit emission line FITS file",
    ]
    file_lists = [
        data_cube_files_B,
        data_cube_files_R,
        continuum_fit_files_B,
        continuum_fit_files_R,
        stekin_files,
        eline_fit_files_reccomp,
        eline_fit_files_1comp,
        eline_fit_files_2comp,
        eline_fit_files_3comp,
    ]

    # TODO check for duplicate galaxies - discard whichever comes second in the list

    # Now, hunt down the data cubes with the same gal AND tile number 
    for id_str in tqdm(ids_all):

        # Split into galaxy + tile
        gal, tile = id_str.split("_")
        tile_number = tile[1:]
        df_filenames.loc[id_str, "ID"] = gal 
        df_filenames.loc[id_str, "Tile"] = tile
        df_filenames.loc[id_str, "Tile number"] = tile_number

        # Find files 
        for file_type, file_list in zip(file_types, file_lists):
            
            # Count how many files contain the galaxy and the tile number 
            gal_file_list = []
            for fname in file_list:
                if gal in str(fname) and ((tile in str(fname)) or (f"tile_{tile_number}" in str(fname))):
                    gal_file_list.append(fname)
            
            # Determine how to 
            if len(gal_file_list) == 0:
                logger.info(f"{id_str} is missing {file_type}!")
                df_filenames.loc[id_str, f"Has {file_type}"] = False
                df_filenames.loc[id_str, f"Duplicate {file_type}"] = False
                df_filenames.loc[id_str, file_type] = ""
            elif len(gal_file_list) > 1:
                logger.info(f"{id_str} has {len(gal_file_list)} {file_type} files:")
                for fname in gal_file_list:
                    logger.info("\t" + fname)
                    df_filenames.loc[id_str, f"Has {file_type}"] = True
                df_filenames.loc[id_str, f"Duplicate {file_type}"] = True
                df_filenames.loc[id_str, file_type] = gal_file_list
            else:    
                df_filenames.loc[id_str, f"Has {file_type}"] = True
                df_filenames.loc[id_str, f"Duplicate {file_type}"] = False
                df_filenames.loc[id_str, file_type] = gal_file_list[0]

    # Determine how many IDs have all required data 
    cond_good = np.ones(df_filenames.shape[0], dtype="bool")
    for file_type in file_types:
        cond_good &= df_filenames[f"Has {file_type}"]
        cond_good &= ~df_filenames[f"Duplicate {file_type}"]
    logger.info(f"{df_filenames[cond_good].shape[0]} / {df_filenames.shape[0]} have all required files")
    df_filenames["Has all required files"] = cond_good

    # Check that all files exist 
    logger.info("Checking that all files actually exist...")
    for id_str in df_filenames[cond_good].index.values:
        for file_type in file_types:
            assert os.path.exists(df_filenames.loc[id_str, file_type])

    return df_filenames


def make_metadata_df():
    """Create the Hector "metadata" DataFrame.

    DESCRIPTION
    ---------------------------------------------------------------------------
    This function is used to create a DataFrame containing "metadata", including
    coordinates, redshifts, distances and angular scales for each Hector galaxy. 
    Also stored are paths to the FITS files containing the raw data cubes, 
    emission line and stellar continuum fitting products.

    This script must be run before make_hector_df() as the resulting DataFrame
    is used there.

    Details:
        - Distances are computed from the redshifts assuming a flat ΛCDM cosmology 
    with cosmological parameters (H0 and ΩM) specified in the config file.

    USAGE
    ---------------------------------------------------------------------------
            
            >>> from spaxelsleuth.io.hector import make_metadata_df
            >>> make_metadata_df()

    INPUTS
    ---------------------------------------------------------------------------
    None.

    OUTPUTS
    ---------------------------------------------------------------------------
    The DataFrame is saved to 

        settings["hector"]["output_path"]/hector_metadata.hd5

    PREREQUISITES
    ---------------------------------------------------------------------------
    TODO: complete this.
    """
    logger.info("creating metadata DataFrame...")

    # Output DataFrame filename
    df_fname = f"hector_metadata.hd5"

    # Get list of unique galaxy IDs 
    df_filenames_duplicates = get_filenames()

    # Remove rows with missing files 
    cond_good = df_filenames_duplicates["Has all required files"]
    df_filenames_duplicates_has_all_files = df_filenames_duplicates.loc[cond_good]

    # Check how many repeat gals there are 
    # NOTE: this will automatically drop the 2nd galaxy if there are duplicates!!! 
    df_filenames = df_filenames_duplicates_has_all_files.loc[~df_filenames_duplicates_has_all_files["ID"].duplicated()]
    df_filenames.loc[:, "ID string"] = df_filenames.index.values
    df_filenames = df_filenames.set_index("ID")
    df_filenames.index = df_filenames.index.astype(int)
    gals = df_filenames.index.values

    df_metadata = pd.DataFrame(data = {"ID": gals,}).set_index("ID")
    # Initialise object-type columns to avoid pandas FutureWarning
    for col in [
        "Field",
        "Tile",
        "Spectrograph",
        "Bundle",
        "Plate ID",
        "Blue data cube FITS file",
        "Red data cube FITS file",
        "Stellar kinematics FITS file",
        "Blue continuum fit FITS file",
        "Red continuum fit FITS file",
        f"1-component fit emission line FITS file",
        f"2-component fit emission line FITS file",
        f"3-component fit emission line FITS file",
        f"rec-component fit emission line FITS file",
    ]:
        df_metadata[col] = ""

    # Iterate through all galaxies and scrape data from FITS files 
    for gal in tqdm(gals):
        # Get the blue & red data cube names
        datacube_B_fname = df_filenames.loc[gal, "Blue data cube FITS file"]
        datacube_R_fname = df_filenames.loc[gal, "Red data cube FITS file"]
        assert os.path.exists(datacube_B_fname)
        assert os.path.exists(datacube_R_fname)

        # Get RA, Dec, spectrograph, bundle, field & tile numbers
        with fits.open(datacube_B_fname) as hdulist_B_cube:
            header_B = hdulist_B_cube[0].header
            spectrograph_B = header_B["INSTRUME"]
            bundle_B = header_B["IFUPROBE"]
            ra_B = header_B["CRVAL1"]
            dec_B = header_B["CRVAL2"]
            plateid_B = header_B["PLATEID"]
            x0_px_B, y0_px_B = np.floor(header_B["CRPIX1"]).astype(int), np.floor(header_B["CRPIX2"]).astype(int)
            nx_B, ny_B = np.floor(header_B["NAXIS1"]).astype(int), np.floor(header_B["NAXIS2"]).astype(int)
            
        with fits.open(datacube_R_fname) as hdulist_R_cube:
            header_R = hdulist_R_cube[0].header
            spectrograph_R = header_R["INSTRUME"]
            bundle_R = header_R["IFUPROBE"]
            ra_R = header_R["CRVAL1"]
            dec_R = header_R["CRVAL2"]
            plateid_R = header_R["PLATEID"]
            x0_px_R, y0_px_R = np.floor(header_R["CRPIX1"]).astype(int), np.floor(header_R["CRPIX2"]).astype(int)
            nx_R, ny_R = np.floor(header_R["NAXIS1"]).astype(int), np.floor(header_R["NAXIS2"]).astype(int)
            
        # Check for consistency between blue & red cubes 
        assert spectrograph_B == spectrograph_R
        assert bundle_B == bundle_R
        assert ra_B == ra_R
        assert dec_B == dec_R
        assert plateid_B == plateid_R
        assert nx_B == nx_R
        assert ny_B == ny_R
        assert x0_px_B == x0_px_R
        assert y0_px_B == y0_px_R
        field, tile = plateid_B.split("_") 

        # FITS filenames for stellar kinematics & continuum fit data products
        stekin_fname = df_filenames.loc[gal, "Stellar kinematics FITS file"]
        cont_fit_B_fname = df_filenames.loc[gal, "Blue continuum fit FITS file"]
        cont_fit_R_fname = df_filenames.loc[gal, "Red continuum fit FITS file"]
        assert os.path.exists(stekin_fname)
        assert os.path.exists(cont_fit_B_fname)
        assert os.path.exists(cont_fit_R_fname)

        # Get FITS filenames for emission line fit data products
        eline_fit_fnames = []
        for ncomponents in [1, 2, 3, "rec"]:
            eline_fit_fname = df_filenames.loc[gal, f"{ncomponents}-component fit emission line FITS file"]
            eline_fit_fnames.append(eline_fit_fname)
            assert os.path.exists(eline_fit_fname)

        # Get redshift & calculate distances 
        with fits.open(stekin_fname) as hdulist_stekin:
            z = hdulist_stekin[0].header["Z"]
        cosmo = FlatLambdaCDM(H0=settings["H_0"], Om0=settings["Omega_0"])
        D_A_Mpc = cosmo.angular_diameter_distance(z).value
        D_L_Mpc = cosmo.luminosity_distance(z).value
        kpc_per_arcsec = D_A_Mpc * 1e3 * np.pi / 180.0 / 3600.0

        # Store in DataFrame
        df_metadata.loc[gal, "RA (J2000)"] = ra_B
        df_metadata.loc[gal, "Dec (J2000)"] = dec_B
        df_metadata.loc[gal, "z"] = z
        df_metadata.loc[gal, "D_A (Mpc)"] = D_A_Mpc
        df_metadata.loc[gal, "D_L (Mpc)"] = D_L_Mpc
        df_metadata.loc[gal, "N_x"] = int(nx_B)
        df_metadata.loc[gal, "N_y"] = int(ny_B)
        df_metadata.loc[gal, "x_0 (pixels)"] = int(x0_px_B)
        df_metadata.loc[gal, "y_0 (pixels)"] = int(y0_px_B)
        df_metadata.loc[gal, "x_0 (arcsec)"] = x0_px_B * settings["hector"]["as_per_px"]
        df_metadata.loc[gal, "y_0 (arcsec)"] = y0_px_B * settings["hector"]["as_per_px"]
        df_metadata.loc[gal, "kpc per arcsec"] = kpc_per_arcsec
        df_metadata.loc[gal, "Field"] = field
        df_metadata.loc[gal, "Tile"] = tile
        df_metadata.loc[gal, "Spectrograph"] = spectrograph_B
        df_metadata.loc[gal, "Bundle"] = bundle_B
        df_metadata.loc[gal, "Plate ID"] = plateid_B
        df_metadata.loc[gal, "Blue data cube FITS file"] = str(datacube_B_fname)
        df_metadata.loc[gal, "Red data cube FITS file"] = str(datacube_R_fname)
        df_metadata.loc[gal, "Stellar kinematics FITS file"] = str(stekin_fname)
        df_metadata.loc[gal, "Blue continuum fit FITS file"] = str(cont_fit_B_fname)
        df_metadata.loc[gal, "Red continuum fit FITS file"] = str(cont_fit_R_fname)
        for ncomponents, eline_fit_fname in zip([1, 2, 3, "rec"], eline_fit_fnames): 
            df_metadata.loc[gal, f"{ncomponents}-component fit emission line FITS file"] = str(eline_fit_fname)

    # Save to file
    logger.info(f"saving metadata DataFrame to file {output_path / df_fname}...")
    df_metadata = df_metadata.sort_index()
    df_metadata.to_hdf(output_path / df_fname, key="metadata")
    logger.info(f"finished!")
    return


def load_metadata_df():
    """Load the Hector metadata DataFrame, containing "metadata" for each galaxy."""
    if not os.path.exists(output_path / "hector_metadata.hd5"):
        raise FileNotFoundError(
            f"File {output_path / 'hector_metadata.hd5'} not found. Did you remember to run make_metadata_df() first?"
        )
    df_metadata = pd.read_hdf(output_path / "hector_metadata.hd5")
    return df_metadata


def process_galaxies(args):
    # Extract input arguments
    gg, gal, ncomponents, bin_type, df_metadata, kwargs = args 
    ncomponents = "rec"
    if ncomponents == "rec":
        ncomponents_max = 3
    else:
        ncomponents_max = ncomponents

    z = df_metadata.loc[gal, "z"]
    kpc_per_arcsec = df_metadata.loc[gal, "kpc per arcsec"]
    nx = df_metadata.loc[gal, "N_x"].astype(int)
    ny = df_metadata.loc[gal, "N_y"].astype(int)
    x0_px = df_metadata.loc[gal, "x_0 (pixels)"].astype(int)
    y0_px = df_metadata.loc[gal, "y_0 (pixels)"].astype(int)

    # Filenames
    datacube_B_fname = df_metadata.loc[gal, "Blue data cube FITS file"]
    datacube_R_fname = df_metadata.loc[gal, "Red data cube FITS file"]
    stekin_fname = df_metadata.loc[gal, "Stellar kinematics FITS file"]
    eline_fit_fname = df_metadata.loc[gal, f"{ncomponents}-component fit emission line FITS file"]
    cont_fit_B_fname = df_metadata.loc[gal, "Blue continuum fit FITS file"]
    cont_fit_R_fname = df_metadata.loc[gal, "Red continuum fit FITS file"]
    for file in [stekin_fname, eline_fit_fname, cont_fit_B_fname, cont_fit_R_fname, datacube_B_fname, datacube_R_fname]:
        if not os.path.exists(str(file)):
            raise FileNotFoundError(f"File {file} not found!")

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
    y_c_list, x_c_list = np.where(mask)

    # Construct a dict where the keys are the FINAL column names, and the 
    # values are 2D arrays storing the corresponding quantity
    _2dmap_dict = {}

    # EMISSION LINES 
    with fits.open(eline_fit_fname) as hdulist_eline_fit:
        # Emission lines: fluxes
        for eline in settings["hector"]["eline_list"]:
            _2dmap_dict[f"{eline} (total)"] = hdulist_eline_fit[eline].data[0]
            _2dmap_dict[f"{eline} error (total)"] = hdulist_eline_fit[eline + "_ERR"].data[0]
            if ncomponents_max > 1:
                for component in range(1, ncomponents_max + 1):
                    _2dmap_dict[f"{eline} (component {component})"] = hdulist_eline_fit[eline].data[component]
                    _2dmap_dict[f"{eline} error (component {component})"] = hdulist_eline_fit[eline + "_ERR"].data[component]
            else:
                _2dmap_dict[f"{eline} (component 1)"] = hdulist_eline_fit[eline].data[0]
                _2dmap_dict[f"{eline} error (component 1)"] = hdulist_eline_fit[eline + "_ERR"].data[0]
        
        # Emission lines: gas kinematics 
        for component in range(1, ncomponents_max + 1):
            _2dmap_dict[f"v_gas (component {component})"] = hdulist_eline_fit["V"].data[component]
            _2dmap_dict[f"v_gas error (component {component})"] = hdulist_eline_fit["V_ERR"].data[component]
            _2dmap_dict[f"sigma_gas (component {component})"] = hdulist_eline_fit["VDISP"].data[component]
            _2dmap_dict[f"sigma_gas error (component {component})"] = hdulist_eline_fit["VDISP_ERR"].data[component]

        # Emission lines: goodness-of-fit  
        _2dmap_dict["chi2 (emission lines)"] = hdulist_eline_fit["CHI2"].data

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

    # CONTINUUM PROPERTIES 
    # Continuum properties: Compute the d4000 Angstrom break 
    d4000_map, d4000_map_err = compute_d4000(
        data_cube=data_cube_B,
        var_cube=var_cube_B,
        lambda_vals_rest_A=lambda_vals_B_rest_A,
        v_star_map=_2dmap_dict["v_*"])
    _2dmap_dict["D4000"] = d4000_map
    _2dmap_dict["D4000 error"] = d4000_map_err

    # Continuum properties: Compute the continuum intensity so that we can compute the Halpha equivalent width.
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
    AN_HALPHA_map = compute_measured_HALPHA_amplitude_to_noise(
        data_cube=data_cube_R,
        var_cube=var_cube_R,
        lambda_vals_rest_A=lambda_vals_R_rest_A,
        v_star_map=_2dmap_dict["v_*"],
        v_map=_2dmap_dict["v_gas (component 1)"],
        dv=300)
    _2dmap_dict["HALPHA A/N (measured)"] = AN_HALPHA_map

    # Median value in blue/red clubes 
    for cont_fit_fname, side in zip([cont_fit_B_fname, cont_fit_R_fname], ["blue", "red"]):
        with fits.open(cont_fit_fname) as hdulist_cont_fit:
            _2dmap_dict[f"Median spectral value ({side})"] = hdulist_cont_fit["MED_SPEC"].data

    # Compute v_grad using eqn. 1 of Zhou+2017
    for component in range(1, ncomponents_max + 1):
        _2dmap_dict[f"v_grad (component {component})"] = compute_v_grad(_2dmap_dict[f"v_gas (component {component})"])

    # OTHER QUANTITIES 
    # NOTE: to create the final list of rows, only values at coordinates in x/y_c_list are extracted from the maps in _2dmap_dict.
    ys, xs = np.meshgrid(range(nx), range(ny), indexing="ij")
    _2dmap_dict["x (pixels)"] = xs
    _2dmap_dict["y (pixels)"] = ys
    _2dmap_dict["x (projected, arcsec)"] = xs * settings["hector"]["as_per_px"]
    _2dmap_dict["y (projected, arcsec)"] = ys * settings["hector"]["as_per_px"]
    _2dmap_dict["x (relative to galaxy centre, projected, arcsec)"] = (xs - x0_px) * settings["hector"]["as_per_px"]
    _2dmap_dict["y (relative to galaxy centre, projected, arcsec)"] = (ys - y0_px) * settings["hector"]["as_per_px"]
    _2dmap_dict["r (relative to galaxy centre, projected, arcsec)"] = np.sqrt((ys - y0_px)**2 + (xs - x0_px)**2) * settings["hector"]["as_per_px"]
    _2dmap_dict["x (relative to galaxy centre, projected, kpc)"] = (xs - x0_px) * settings["hector"]["as_per_px"] * kpc_per_arcsec
    _2dmap_dict["y (relative to galaxy centre, projected, kpc)"] = (ys - y0_px) * settings["hector"]["as_per_px"] * kpc_per_arcsec
    _2dmap_dict["r (relative to galaxy centre, projected, kpc)"] = np.sqrt((ys - y0_px)**2 + (xs - x0_px)**2) * settings["hector"]["as_per_px"] * kpc_per_arcsec

    # Convert 2D maps to 1D rows 
    rows_list = []
    colnames = list(_2dmap_dict.keys())
    for colname in colnames:
        rows = _2d_map_to_1d_list(_2dmap_dict[colname], x_c_list, y_c_list, nx, ny)
        rows_list.append(rows)

    # Add galaxy ID 
    rows_list.append([gal] * len(x_c_list))
    colnames.append("ID")

    # Add pixel sizes in arcsec and kpc 
    rows_list.append([settings["sami"]["as_per_px"]**2] * len(x_c_list))
    colnames.append("Bin size (square arcsec)")
    rows_list.append([settings["sami"]["as_per_px"]**2 * df_metadata.loc[gal, "kpc per arcsec"]**2] * len(x_c_list))
    colnames.append("Bin size (square kpc)")

    # Transpose so that each row represents a single pixel & each column a measured quantity.
    rows_arr = np.array(rows_list).T

    logger.info(f"Finished processing galaxy {gal} ({gg})")

    return rows_arr, colnames