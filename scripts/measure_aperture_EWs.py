import os
import numpy as np
from astropy.io import fits
from astroquery.sdss import SDSS
from astropy import coordinates as coords
import pandas as pd
from tqdm import tqdm
import multiprocessing
from scipy import constants
import sys

# import matplotlib
# matplotlib.use("Agg")

import matplotlib.pyplot as plt
plt.ion()
plt.close("all")

from IPython.core.debugger import Tracer

import warnings
warnings.filterwarnings(action="ignore", message="Mean of empty slice")
warnings.filterwarnings(action="ignore", message="invalid value encountered in sqrt")

###############################################################################
# User inputs
ap = sys.argv[1]
apertures = ["1_4_arcsecond", "2_arcsecond", "3_arcsecond", "4_arcsecond", "3kpc_round", "re", "re_mge"]
assert ap in apertures, f"{ap} is not a valid aperture!"

###############################################################################
sami_data_path = "/data/misfit/u5708159/SAMI/"
sami_datacube_path = "/priv/myrtle1/sami/sami_data/Final_SAMI_data/cube/sami/dr3/"

input_data_path = "/home/u5708159/python/Modules/spaxelsleuth/spaxelsleuth/data/"

###############################################################################
# Filenames
df_metadata_fname = "sami_dr3_metadata.hd5"
fits_ap_ssp_fname = "sami_SSPAperturesDR3.csv"
fits_ap_eline_fname = "sami_EmissionLine1compDR3.csv"
df_ap_stekin_fname = "sami_samiDR3gaskinPA.csv"
df_mge_data_fname = "sami_MGEPhotomUnregDR3.csv"
df_eline_ews_fname = f"sami_{ap}_aperture_1-comp_EWs_20230725.hd5"

###############################################################################
# READ IN THE METADATA
###############################################################################
df_metadata = pd.read_hdf(os.path.join(sami_data_path, df_metadata_fname), key="metadata")
gal_ids_dq_cut = df_metadata[df_metadata["Good?"] == True].index.values
df_metadata["Good?"] = df_metadata["Good?"].astype("float")

###############################################################################
# READ IN THE MGE INFORMATION
###############################################################################
df_mge_data = pd.read_csv(os.path.join(input_data_path, df_mge_data_fname))
df_mge_data = df_mge_data.set_index("catid").drop(columns=["Unnamed: 0"])

###############################################################################
# READ SSP APERTURE DATA
###############################################################################
df_ssp = pd.read_csv(os.path.join(input_data_path, fits_ap_ssp_fname))
df_ssp = df_ssp.drop(columns=["Unnamed: 0"])

###############################################################################
# READ STELLAR KINEMATICS
###############################################################################
df_stekin = pd.read_csv(os.path.join(input_data_path, df_ap_stekin_fname))
df_stekin = df_stekin.set_index("catid")

# Get the subset of stellar kinematics info for this aperture
df_stekin = df_stekin[[c for c in df_stekin.columns if (c.endswith(ap) or c.endswith(f"{ap}_err"))]]

# Only interested in stellar velocity dispersion
df_stekin = df_stekin[[f"sigma_{ap}", f"sigma_{ap}_err"]]
df_stekin = df_stekin.rename(columns={f"sigma_{ap}": f"sigma_*",
                                      f"sigma_{ap}_err": f"sigma_* error"})

###############################################################################
# READ EMISSION LINE APERTURE DATA
###############################################################################
df_elines = pd.read_csv(os.path.join(input_data_path, fits_ap_eline_fname))

hdulist = fits.open(os.path.join(sami_data_path, fits_ap_eline_fname))
t = hdulist[1].data

# Convert table into a dataframe
rows_list = []
for rr in range(t.shape[0]):
    row_dict = {t.columns[ii].name: t[rr][ii] for ii in range(len(t.columns))}
    rows_list.append(row_dict)
df_eline_aps = pd.DataFrame(rows_list)

for col in t.columns:
    if col.name in df_ssp.columns:
        df_eline_aps = df_eline_aps.astype({col.name: str(df_ssp[col.name].dtype)})
    elif col.name in df_metadata.columns:
        df_eline_aps = df_eline_aps.astype({col.name: str(df_metadata[col.name].dtype)})

# Convert numeric values to float
for aperture in apertures:
    df_eline_aps = df_eline_aps.astype({c: float for c in df_eline_aps.columns if c.endswith(aperture)})
    df_eline_aps = df_eline_aps.astype({c: float for c in df_eline_aps.columns if c.endswith(f"{aperture}_err")})

# Drop duplicate rows: keep the "A" data cube 
df_eline_aps = df_eline_aps.drop_duplicates(subset="catid", keep="first")

# Get the subset of emission line info for this aperture
df_eline_ap = df_eline_aps[["catid"] + [c for c in df_eline_aps.columns if (c.endswith(ap) or c.endswith(f"{ap}_err"))]]

# Re-name emission line columns
df_eline_ap = df_eline_ap.rename(columns=dict(
    [(c, c.split(f"_{ap}")[0].upper()) for c in df_eline_ap.columns if c.endswith(ap)] +\
    [(c, c.split(f"_{ap}_err")[0].upper() + " error") for c in df_eline_ap.columns if c.endswith(f"{ap}_err")])
)
df_eline_ap = df_eline_ap.rename(columns={
    "VDISP_GAS": "sigma_gas",
    "V_GAS": "v_gas",
    "VDISP_GAS error": "sigma_gas error",
    "V_GAS error": "v_gas error",
    }
)
df_eline_ap["OII3726+OII3729"] = df_eline_ap["OII3726"] + df_eline_ap["OII3729"]
df_eline_ap["OII3726+OII3729 error"] = \
    np.sqrt((df_eline_ap["OII3726 error"] / df_eline_ap["OII3726"])**2 +\
            (df_eline_ap["OII3729 error"] / df_eline_ap["OII3729"])**2)


# Merge into a single dataframe
df_eline_ap = df_eline_ap.set_index("catid")
df_eline_ap = df_eline_ap.join(df_metadata)

###############################################################################
# COMPUTE D4000Å BREAK STRENGTHS AND HALPHA CONTINUUM STRENGTHS FOR EACH
###############################################################################
# X, Y pixel coordinates
ys, xs = np.meshgrid(np.arange(50), np.arange(50), indexing="ij")
as_per_px = 0.5

# Centre galaxy coordinates (see p16 of Croom+2021)
x0_px = 25.5
y0_px = 25.5

################################################################################
def process_gals(args, plotit=False):
    ap, gal_idx, gal = args

    if ap == "re_mge" and gal not in df_mge_data.index:
        print(f"MGE measurements not available for {gal}! Skipping...")
        return [int(gal), np.nan, np.nan, np.nan, np.nan]

    try:
        ######################################################################
        # Open the red & blue cubes.
        hdulist_B_cube = fits.open(os.path.join(sami_datacube_path, f"ifs/{gal}/{gal}_A_cube_blue.fits.gz"))
        hdulist_R_cube = fits.open(os.path.join(sami_datacube_path, f"ifs/{gal}/{gal}_A_cube_red.fits.gz"))

        header_B = hdulist_B_cube[0].header
        data_cube_B = hdulist_B_cube[0].data
        var_cube_B = hdulist_B_cube[1].data
        hdulist_B_cube.close()

        header_R = hdulist_R_cube[0].header
        data_cube_R = hdulist_R_cube[0].data 
        var_cube_R = hdulist_R_cube[1].data  
        hdulist_R_cube.close() 

        #######################################################################
        if ap.endswith("_arcsecond"):
            r_as = float(ap.strip("_arcsecond")[0].replace("_", ".")) / 2 
            r_px = r_as / as_per_px
            ap_mask = (xs - x0_px)**2 + (ys - y0_px)**2 <= r_px**2

        elif ap == "3kpc_round": # Diameter
            r_kpc = 3 / 2
            r_as = r_kpc / df_metadata.loc[gal, "kpc per arcsec"]
            r_px = r_as / as_per_px
            ap_mask = (xs - x0_px)**2 + (ys - y0_px)**2 <= r_px**2

        elif ap.startswith("re"):

            # Need to treat MGE and non-MGE R_e differently
            if ap == "re":
                r_e_as = df_metadata.loc[gal, "r_e"]  # in arcseoncds
                PA_deg = df_metadata.loc[gal, "pa"]
                ellip = df_metadata.loc[gal, "ellip"]

            elif ap == "re_mge":
                # Create a mask so that we can compute the EW within 1R_e.
                if df_mge_data.loc[gal].ndim > 1:
                    # Some targets have 2 entries in the table (VST and SDSS photometry)
                    # For now, take the VST measurement.
                    cond = df_mge_data.loc[gal, "photometry"] == "VST"
                    r_e_as = df_mge_data.loc[gal][cond]["remge"].values[0]
                    PA_deg = df_mge_data.loc[gal][cond]["pamge"].values[0]
                    ellip = df_mge_data.loc[gal][cond]["epsmge_re"].values[0]
                else:
                    r_e_as = df_mge_data.loc[gal, "remge"]  # in arcseoncds
                    PA_deg = df_mge_data.loc[gal, "pamge"]  # +ve E from N
                    ellip = df_mge_data.loc[gal, "epsmge_re"]
            
            r_e_px = r_e_as / as_per_px
            b_over_a = 1 - ellip
            a = r_e_px
            b = a * b_over_a
            beta_rad = np.deg2rad(PA_deg)
            xs_sfhited = xs - x0_px  # de-shift coordinates
            ys_shifted = ys - y0_px
            xs_prime = xs_sfhited * np.cos(beta_rad) + ys_shifted * np.sin(beta_rad)
            ys_prime = (- xs_sfhited * np.sin(beta_rad) + ys_shifted * np.cos(beta_rad))
            ap_mask = (xs_prime / b)**2 + (ys_prime / a)**2 <= 1
            # ap_mask_orig = ((xs - x0_px) / b)**2 + ((ys - y0_px) / a)**2 <= 1  # Un-rotated mask for reference

        # CHECK: plot an image of the galaxy; plot everything else on top
        if plotit:
            fig, axs = plt.subplots(nrows=1, ncols=2)
            axs[0].imshow(np.nansum(data_cube_R, axis=0))
            axs[0].scatter(x=[x0_px], y=[y0_px], s=50, c="red")
            axs[1].imshow(ap_mask)
            fig.suptitle(f"Gal ID {gal}; PA = {PA_deg:.2f} degrees, R_e = {r_e_px:.2f} px, b/a = {b_over_a:.2f}")
            fig.canvas.draw()

        #######################################################################
        # Compute the d4000 Angstrom break.
        spec_B = np.nansum(data_cube_B[:, ap_mask], axis=1)
        spec_var_B = np.nansum(var_cube_B[:, ap_mask], axis=1)

        # Wavelength values
        lambda_0_A = header_B["CRVAL3"] - header_B["CRPIX3"] * header_B["CDELT3"]
        dlambda_A = header_B["CDELT3"]
        N_lambda = header_B["NAXIS3"]
        lambda_vals_A = np.array(range(N_lambda)) * dlambda_A + lambda_0_A 

        # Compute the D4000Å break
        # Definition from Balogh+1999 (see here: https://arxiv.org/pdf/1611.07050.pdf, page 3)
        start_b_idx = np.nanargmin(np.abs(lambda_vals_A / (1 + df_metadata.loc[gal, "z_spec"]) - 3850))
        stop_b_idx = np.nanargmin(np.abs(lambda_vals_A / (1 + df_metadata.loc[gal, "z_spec"]) - 3950))
        start_r_idx = np.nanargmin(np.abs(lambda_vals_A / (1 + df_metadata.loc[gal, "z_spec"]) - 4000))
        stop_r_idx = np.nanargmin(np.abs(lambda_vals_A / (1 + df_metadata.loc[gal, "z_spec"]) - 4100))
        N_b = stop_b_idx - start_b_idx
        N_r = stop_r_idx - start_r_idx

        # Convert datacube & variance cubes to units of F_nu
        spec_B_Hz = spec_B * lambda_vals_A**2 / (constants.c * 1e10)
        spec_var_B_Hz2 = spec_var_B * (lambda_vals_A**2 / (constants.c * 1e10))**2

        num = np.nanmean(spec_B_Hz[start_r_idx:stop_r_idx], axis=0)
        denom = np.nanmean(spec_B_Hz[start_b_idx:stop_b_idx], axis=0)
        err_num = 1 / N_r * np.sqrt(np.nansum(spec_var_B_Hz2[start_r_idx:stop_r_idx], axis=0))
        err_denom = 1 / N_b * np.sqrt(np.nansum(spec_var_B_Hz2[start_b_idx:stop_b_idx], axis=0))

        d4000 = num / denom
        d4000_err = d4000 * np.sqrt((err_num / num)**2 + (err_denom / denom)**2)

        #######################################################################
        # Use the red cube to calculate the continuum intensity so 
        # that we can compute the HALPHA equivalent width.
        # Units of 10**(-16) erg /s /cm**2 /angstrom /pixel
        # Continuum wavelength range taken from here: https://ui.adsabs.harvard.edu/abs/2019MNRAS.485.4024V/abstract
        spec_R = np.nansum(data_cube_R[:, ap_mask], axis=1)
        spec_var_R = np.nansum(var_cube_R[:, ap_mask], axis=1)

        # Wavelength values
        lambda_0_A = header_R["CRVAL3"] - header_R["CRPIX3"] * header_R["CDELT3"]
        dlambda_A = header_R["CDELT3"]
        N_lambda = header_R["NAXIS3"]
        lambda_vals_A = np.array(range(N_lambda)) * dlambda_A + lambda_0_A 

        # Compute continuum intensity
        start_idx = np.nanargmin(np.abs(lambda_vals_A / (1 + df_metadata.loc[gal, "z_spec"]) - 6500))
        stop_idx = np.nanargmin(np.abs(lambda_vals_A / (1 + df_metadata.loc[gal, "z_spec"]) - 6540))
        halpha_cont = np.nanmean(spec_R[start_idx:stop_idx], axis=0)
        halpha_cont_err = 1 / (stop_idx - start_idx) * np.sqrt(np.nansum(spec_var_R[start_idx:stop_idx], axis=0))

        #######################################################################
        print(f"Finished processing {gal} ({gal_idx}/{len(gal_ids_dq_cut)})")

        return [int(gal), d4000, d4000_err, halpha_cont, halpha_cont_err]
    except:
        print(f"Processing of galaxy {gal} failed for some reason :(")
        return [int(gal), np.nan, np.nan, np.nan, np.nan]

###############################################################################
# Run in parallel
###############################################################################
# for gal in gal_ids_dq_cut[:20]:
#     process_gals(["re_mge", 0, gal], plotit=False)
#     Tracer()()

print("Beginning pool...")
args_list = [[ap, ii, g] for ii, g in enumerate(gal_ids_dq_cut)]
pool = multiprocessing.Pool(20)
res_list = np.array((pool.map(process_gals, args_list)))
pool.close()
pool.join()

###############################################################################
# SAVE TO FILE
###############################################################################
# Convert to a Pandas DataFrame
df_ews = pd.DataFrame(res_list, columns=["catid", "D4000", "D4000 error", "HALPHA continuum", "HALPHA continuum error"])
df_ews = df_ews.set_index("catid")

# Merge EW and D4000 info with existing data frame
df_eline_ews = df_eline_ap.join(df_ews)

# Compute EWs
df_eline_ews["HALPHA EW"] = df_eline_ews["HALPHA"] / df_eline_ews["HALPHA continuum"]
df_eline_ews["HALPHA EW error"] = df_eline_ews["HALPHA EW"] * np.sqrt(\
        (df_eline_ews["HALPHA error"] / df_eline_ews["HALPHA"])**2 +\
        (df_eline_ews["HALPHA continuum error"] / df_eline_ews["HALPHA continuum"])**2)

# Save
df_eline_ews.to_hdf(os.path.join(sami_data_path, df_eline_ews_fname), key=f"{ap} aperture, 1-comp")
df_eline_ews.to_csv(os.path.join(sami_data_path, df_eline_ews_fname.split(".hd5")[0] + ".csv"))

# Plot
fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
axs[0].hist(np.log10(df_eline_ews["HALPHA EW"]), range=(-1, 3), bins=30)
axs[0].set_xlabel(r"log EW(H$\alpha$)")
axs[0].set_ylabel(r"$N$")
axs[1].hist(df_eline_ews["D4000"], range=(0.5, 2.5), bins=30)
axs[1].set_xlabel(r"D$_n$4000\AA break strength")
axs[1].set_ylabel(r"$N$")
fig.suptitle(f"{ap} aperture")
plt.show()

"""
# Re-making old files to update stuff
for ap in ["re_mge", "1_4_arcsecond", "2_arcsecond", "3_arcsecond"]:
    df_eline_ews_fname = f"sami_{ap}_aperture_1-comp.hd5"
    df_eline_ews = pd.read_hdf(os.path.join(sami_data_path, df_eline_ews_fname), key=f"Aperture EWs")

    # Read in stellar data
    df_stekin = pd.read_csv(os.path.join(sami_data_path, df_ap_stekin_fname))
    df_stekin = df_stekin.set_index("catid")

    # Get the subset of stellar kinematics info for this aperture
    df_stekin = df_stekin[[c for c in df_stekin.columns if (c.endswith(ap) or c.endswith(f"{ap}_err"))]]

    # Only interested in stellar velocity dispersion
    df_stekin = df_stekin[[f"sigma_{ap}", f"sigma_{ap}_err"]]
    df_stekin = df_stekin.rename(columns={f"sigma_{ap}": f"sigma_*",
                                          f"sigma_{ap}_err": f"sigma_* error"})

    df_eline_ews = df_eline_ews.join(df_stekin)
    df_eline_ews["v_*"] = np.nan
    df_eline_ews["v_* error"] = np.nan

    # Save back to file 
    df_eline_ews.to_hdf(os.path.join(sami_data_path, df_eline_ews_fname), key=f"{ap} aperture, 1-comp")
"""


