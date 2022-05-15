import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import copy

from astroquery.ipac.ned import Ned
from astroquery.exceptions import TableParseError, RemoteServiceError
from astropy.coordinates import SkyCoord
import astropy.units as u

from spaxelsleuth.loaddata.sami import load_sami_galaxies
from spaxelsleuth.loaddata.linefns import ratio_fn, bpt_fn
from spaxelsleuth.plotting.plottools import plot_BPT_lines

import matplotlib.pyplot as plt
plt.ion()
plt.close()

from IPython.core.debugger import Tracer
"""
In this script:

* compute line fluxes for a nuclear spectrum
* compute a nuclear BPT classifications for each galaxy 
* compute a nuclear WHAN classification for each galaxy 
* extract WISE fluxes to compute an IR AGN classification for each galaxy
* extract FIRST 1.4 GHz power to compute a radio AGN classification for each galaxy

Store the result in df_info

TODO:
- check if any source have LOWER limits for IRAS fluxes & 1.4 GHz flux densities
- count how many sources have both upper limits for S_1.4 AND IRAS fluxes

"""
sami_data_path = "/priv/meggs3/u5708159/SAMI/sami_dr3/"

###############################################################################
# LOAD SAMI DATA
###############################################################################
df_metadata = pd.read_hdf(os.path.join(sami_data_path, f"sami_dr3_metadata_extended.hd5"), key="Extended metadata")

# Load MGE R_e emission line measurements
ap = "3_arcsecond"
df_ap_elines = pd.read_hdf(os.path.join(sami_data_path, f"sami_{ap}_aperture_1-comp.hd5"), 
                           key=f"{ap} aperture, 1-comp")

# Compute S/N in the optical emission lines
elines = ["OII3726+OII3729", "HBETA", "OIII5007", "OI6300", "NII6583", "HALPHA", "SII6716", "SII6731"]
for eline in elines:
    df_ap_elines[f"{eline} S/N"] = df_ap_elines[eline] / df_ap_elines[f"{eline} error"]

###############################################################################
# AGN CLASSIFICATION: OPTICAL (BPT)
###############################################################################
df_ap_elines = ratio_fn(df_ap_elines, s=" (total)")
df_ap_elines = bpt_fn(df_ap_elines, s=" (total)")

df_ap_elines.loc[df_ap_elines["BPT (total)"] == "Seyfert", "is AGN (BPT)?"] = "Yes (Seyfert)"
df_ap_elines.loc[df_ap_elines["BPT (total)"] == "LINER", "is AGN (BPT)?"] = "Maybe (LINER)"
df_ap_elines.loc[df_ap_elines["BPT (total)"] == "SF", "is AGN (BPT)?"] = "No"
df_ap_elines.loc[df_ap_elines["BPT (total)"] == "Composite", "is AGN (BPT)?"] = "No"
df_ap_elines.loc[df_ap_elines["BPT (total)"] == "Ambiguous", "is AGN (BPT)?"] = "Maybe (ambiguous)"
df_ap_elines.loc[df_ap_elines["BPT (total)"] == "Not classified", "is AGN (BPT)?"] = "n/a"

# Now, make S/N cut 
cond_poor_SN = df_ap_elines["HBETA S/N"] < 5
cond_poor_SN |= df_ap_elines["OIII5007 S/N"] < 5
cond_poor_SN |= df_ap_elines["HALPHA S/N"] < 5
cond_poor_SN |= df_ap_elines["NII6583 S/N"] < 5
cond_poor_SN |= df_ap_elines["SII6716 S/N"] < 5
cond_poor_SN |= df_ap_elines["SII6731 S/N"] < 5
df_ap_elines.loc[cond_poor_SN, "is AGN (BPT)?"] = "n/a"

###############################################################################
# AGN CLASSIFICATION: OPTICAL (WHAN)
###############################################################################
cond_Sy_AGN = (df_ap_elines["HALPHA EW"] > 6) & (df_ap_elines["log N2 (total)"] > -0.4)
cond_LINER_AGN = (df_ap_elines["HALPHA EW"] <= 6) & (df_ap_elines["HALPHA EW"] > 3) & (df_ap_elines["log N2 (total)"] > -0.4)
cond_not_AGN = (df_ap_elines["log N2 (total)"] <= -0.4) | (df_ap_elines["HALPHA EW"] <= 3)
cond_na = np.isnan(df_ap_elines["HALPHA EW"]) | np.isnan(df_ap_elines["log N2 (total)"])

df_ap_elines.loc[cond_Sy_AGN, "is AGN (WHAN)?"] = "Yes (Seyfert)"
df_ap_elines.loc[cond_LINER_AGN, "is AGN (WHAN)?"] = "Maybe (LINER)"
df_ap_elines.loc[cond_not_AGN, "is AGN (WHAN)?"] = "No"
df_ap_elines.loc[cond_na, "is AGN (WHAN)?"] = "n/a"

# Now, make S/N cut 
cond_poor_SN = df_ap_elines["HALPHA S/N"] < 5
cond_poor_SN |= df_ap_elines["NII6583 S/N"] < 5
df_ap_elines.loc[cond_poor_SN, "is AGN (WHAN)?"] = "n/a"

###############################################################################
# Make a new dataframe only containing the columns we need
###############################################################################
# List of "good" galaxies
gals = df_metadata[df_metadata["Good?"] == 1.0].index.unique()
gals = [g for g in gals if g in df_ap_elines.index]  # there is 1 galaxy missing from df_ap_elines for some reason - remove it 

# Make a NEW DataFrame containing classifications
df_classifications = df_ap_elines.loc[gals]

df_classifications["log HALPHA EW"] = np.log10(df_classifications["HALPHA EW"])

df_classifications["HBETA S/N"] = df_classifications["HBETA"] / df_classifications["HBETA error"]

###############################################################################
# GET NED DATA
###############################################################################
# for gal in tqdm(gals):

# gals = [572402, 396833, 47460, 106549, 9008501058, 9008500074, 492883, 543895, 548946]

for gal in tqdm(gals):
    # print("----------------------------------------------------------------")
    try:
        table_phot = Ned.get_table(f"GAMA {gal}", table="photometry")
        # print(f"GAMA {gal} found in NED")
    
    except (RemoteServiceError, TableParseError) as e:
        # print(f"GAMA {gal} not found in NED, using cone search...")
        
        # Look up coordinates instead
        ra = df_ap_elines.loc[gal, "ra_obj"]
        dec = df_ap_elines.loc[gal, "dec_obj"]
        z = df_ap_elines.loc[gal, "z_spec"]
        coords = SkyCoord(ra=ra, dec=dec, unit=(u.deg, u.deg), frame="icrs")
        table_region = Ned.query_region(coordinates=coords, radius=30 * u.arcsec)
        
        # Get entries within a narrow redshift range
        cond_z = (table_region["Redshift"] < z + 0.001) & (table_region["Redshift"] > z - 0.001)
        table_region = table_region[cond_z]

        # if len(table_region) == 1:
            # print(f"Entry found for {gal} using cone search!")

        if len(table_region) > 1:
            # print(f"WARNING: multiple entries found in NED for {gal} - taking entry with nearest RA, dec... ")
            table_region.sort("Separation")
            table_region = table_region[0]

        elif len(table_region) == 0:
            # print(f"ERROR: {gal} not found in NED! Skipping...")
            df_classifications.loc[gal, "W1 - W2 undetermined?"] = True
            df_classifications.loc[gal, "is AGN (MIR)?"] = "n/a"
            df_classifications.loc[gal, "FIR undetermined?"] = True
            df_classifications.loc[gal, "is AGN (FIR)?"] = "n/a"
            continue

        # Get the object name & get the photometry table
        obj_name = table_region["Object Name"] if type(table_region["Object Name"]) == np.str_ else table_region["Object Name"].value[0]
        try:
            table_phot = Ned.get_table(obj_name, table="photometry")
        # If there is no photometry available for this object, then skip it.
        except RemoteServiceError:
            # print(f"No photometric data can be found for galaxy {gal} ({obj_name})!")
            df_classifications.loc[gal, "FIR undetermined?"] = True
            df_classifications.loc[gal, "FIR q"] = np.nan
            df_classifications.loc[gal, "is AGN (FIR)?"] = "n/a"
            df_classifications.loc[gal, "W1 - W2 undetermined?"] = True
            df_classifications.loc[gal, "W1 - W2"] = np.nan
            df_classifications.loc[gal, "is AGN (MIR)?"] = "n/a"
            continue

    ###############################################################################
    # AGN CLASSIFICATION: FIR-RADIO CORRELATION
    ###############################################################################
    # FIR fluxes (and upper limits) from IRAS
    # print(f"Computing FIR, MIR AGN classifications for {gal}...")

    if len(table_phot[table_phot["Observed Passband"] == "60 microns (IRAS)"]) > 0 and\
        len(table_phot[table_phot["Observed Passband"] == "100 microns (IRAS)"]) > 0 and\
        len(table_phot[table_phot["Observed Passband"] == "1.4 GHz (FIRST)"]) > 0:

        # Some galaxies have multiple IRAS fluxes from different publications - make sure we only get one.
        cond_60 = (table_phot["Observed Passband"] == "60 microns (IRAS)")
        cond_100 = (table_phot["Observed Passband"] == "100 microns (IRAS)")

        # Note: for some objects there are multiple measurements with the same refcode & comments - in these cases take the 1st entry 
        S_60_Jy = float(table_phot[cond_60]["Photometry Measurement"].value) if len(table_phot[cond_60]) == 1 else float(table_phot[cond_60]["Photometry Measurement"].value[0]) # mag
        S_100_Jy = float(table_phot[cond_100]["Photometry Measurement"].value) if len(table_phot[cond_100]) == 1 else float(table_phot[cond_100]["Photometry Measurement"].value[0]) # mag
        S_60_ulim_Jy = float(table_phot[cond_60]["Upper limit of Flux Density"].value) if len(table_phot[cond_60]) == 1 else float(table_phot[cond_60]["Upper limit of Flux Density"].value[0]) # mag
        S_100_ulim_Jy = float(table_phot[cond_100]["Upper limit of Flux Density"].value) if len(table_phot[cond_100]) == 1 else float(table_phot[cond_100]["Upper limit of Flux Density"].value[0]) # mag

        # Check: one of S_60_Jy or S_60_ulim_Jy must be NaN
        assert not (np.isnan(S_60_Jy) and np.isnan(S_60_ulim_Jy)),\
            "ERROR: both S_60_Jy and S_60_ulim_Jy are undefined!"
        assert not (~np.isnan(S_60_Jy) and ~np.isnan(S_60_ulim_Jy)),\
            "ERROR: both S_60_Jy and S_60_ulim_Jy are defined!"
        assert not (np.isnan(S_100_Jy) and np.isnan(S_100_ulim_Jy)),\
            "ERROR: both S_100_Jy and S_100_ulim_Jy are undefined!"
        assert not (~np.isnan(S_100_Jy) and ~np.isnan(S_100_ulim_Jy)),\
            "ERROR: both S_100_Jy and S_100_ulim_Jy are defined!"

        # 1.4 GHz flux densities in W m^-2 Hz^-1
        S_1_4GHz_mJy = float(table_phot[table_phot["Observed Passband"] == "1.4 GHz (FIRST)"]["Photometry Measurement"].value[0])
        S_1_4GHz_ulim_mJy = float(table_phot[table_phot["Observed Passband"] == "1.4 GHz (FIRST)"]["Upper limit of Flux Density"].value[0])
        assert not (np.isnan(S_1_4GHz_mJy) and np.isnan(S_1_4GHz_ulim_mJy)),\
            "ERROR: both S_1_4GHz_mJy or S_1_4GHz_ulim_mJy are undefined!"
        assert not (~np.isnan(S_1_4GHz_mJy) and ~np.isnan(S_1_4GHz_ulim_mJy)),\
            "ERROR: both S_1_4GHz_mJy or S_1_4GHz_ulim_mJy are defined!"

        # Convert to W m^-2 Hz^-1
        S_1_4GHz_Wm2Hz = S_1_4GHz_mJy * 1e-3 * 1e-26
        S_1_4GHz_ulim_Wm2Hz = S_1_4GHz_ulim_mJy * 1e-3 * 1e-26

        # Determine whether any measurements are upper limits
        if ~np.isnan(S_60_ulim_Jy) or ~np.isnan(S_100_ulim_Jy):
            FIR_is_upper_limit = True
        else:
            FIR_is_upper_limit = False

        if ~np.isnan(S_1_4GHz_ulim_mJy):
            S_14GHz_is_upper_limit = True
        else:
            S_14GHz_is_upper_limit = False

        # Determine whether q is constrained, and whether it is an upper or lower limit
        if FIR_is_upper_limit and not S_14GHz_is_upper_limit:
            q_is_lower_limit = False
            q_is_upper_limit = True 
            q_not_constrained = False
        elif not FIR_is_upper_limit and S_14GHz_is_upper_limit:
            q_is_lower_limit = True
            q_is_upper_limit = False 
            q_not_constrained = False
        elif FIR_is_upper_limit and S_14GHz_is_upper_limit:
            q_is_lower_limit = False
            q_is_upper_limit = False 
            q_not_constrained = True
        else:
            q_is_lower_limit = False
            q_is_upper_limit = False 
            q_not_constrained = False

        # Compute FIR using eqn. 1 of Mauch & Sadler (2002)
        if not FIR_is_upper_limit:
            FIR = 1.26e-14 * (2.58 * S_60_Jy + S_100_Jy)
            FIR_ulim = np.nan
        else:
            # Compue an upper limit for FIR, if necessary
            FIR = np.nan
            if np.isnan(S_60_Jy):
                FIR_ulim = 1.26e-14 * (2.58 * S_60_ulim_Jy + S_100_Jy)
            elif np.isnan(S_100_Jy):
                FIR_ulim = 1.26e-14 * (2.58 * S_60_Jy + S_100_ulim_Jy)

        # Compute "q" using eqn. 2 of Mauch & Sadler (2002) 
        q = np.log10((FIR / (3.75e12)) / S_1_4GHz_Wm2Hz)
        q_ulim = np.log10((FIR_ulim / (3.75e12)) / S_1_4GHz_Wm2Hz)
        q_lolim = np.log10((FIR / (3.75e12)) / S_1_4GHz_ulim_Wm2Hz)

        # q > 1.8: star-forming
        # q < 1.8: AGN
        # Add to DataFrame
        df_classifications.loc[gal, "FIR (W m^-2)"] = FIR
        df_classifications.loc[gal, "FIR upper limit (W m^-2)"] = FIR
        df_classifications.loc[gal, "FIR upper limit?"] = FIR_is_upper_limit

        df_classifications.loc[gal, "Flux density at 1.4 GHz (mJy)"] = S_1_4GHz_mJy
        df_classifications.loc[gal, "Flux density at 1.4 GHz upper limit (mJy)"] = S_1_4GHz_ulim_mJy
        df_classifications.loc[gal, "Flux density at 1.4 GHz upper limit?"] = S_14GHz_is_upper_limit

        df_classifications.loc[gal, "FIR q"] = q
        df_classifications.loc[gal, "FIR q upper limit"] = q_ulim
        df_classifications.loc[gal, "FIR q lower limit"] = q_lolim
        df_classifications.loc[gal, "FIR q upper limit?"] = q_is_upper_limit
        df_classifications.loc[gal, "FIR q lower limit?"] = q_is_lower_limit
        df_classifications.loc[gal, "FIR undetermined?"] = q_not_constrained

        if ~np.isnan(q):
            df_classifications.loc[gal, "is AGN (FIR)?"] = "Yes" if q < 1.8 else "No" 
        elif q_is_upper_limit:
            df_classifications.loc[gal, "is AGN (FIR)?"] = "Yes" if q_ulim < 1.8 else "No"
        elif q_is_lower_limit or q_not_constrained:
            df_classifications.loc[gal, "is AGN (FIR)?"] = "n/a"

    else:
        df_classifications.loc[gal, "FIR undetermined?"] = True
        df_classifications.loc[gal, "FIR q"] = np.nan
        df_classifications.loc[gal, "is AGN (FIR)?"] = "n/a"

    ###############################################################################
    # AGN CLASSIFICATION: IR (WISE)
    ###############################################################################
    # IR fluxes from WISE 
    if len(table_phot[table_phot["Observed Passband"] == "W1 (WISE)"]) > 0 and len(table_phot[table_phot["Observed Passband"] == "W2 (WISE)"]) > 0:
        cond_W1 = (table_phot["Observed Passband"] == "W1 (WISE)") & (table_phot["Qualifiers"] == "Profile-fit;extended")
        cond_W2 = (table_phot["Observed Passband"] == "W2 (WISE)") & (table_phot["Qualifiers"] == "Profile-fit;extended")
        cond_W3 = (table_phot["Observed Passband"] == "W3 (WISE)") & (table_phot["Qualifiers"] == "Profile-fit;extended")
        if len(table_phot[cond_W1]) == 0:
            cond_W1 = (table_phot["Observed Passband"] == "W1 (WISE)") & (table_phot["Qualifiers"] == "Profile-fit")
            cond_W2 = (table_phot["Observed Passband"] == "W2 (WISE)") & (table_phot["Qualifiers"] == "Profile-fit")  
            cond_W3 = (table_phot["Observed Passband"] == "W3 (WISE)") & (table_phot["Qualifiers"] == "Profile-fit") 

        # Note: for some objects there are multiple measurements with the same refcode & comments - in these cases take the 1st entry 
        try:
            W1 = float(table_phot[cond_W1]["Photometry Measurement"].value) if len(table_phot[cond_W1]) == 1 else float(table_phot[cond_W1]["Photometry Measurement"].value[0]) # mag
            W2 = float(table_phot[cond_W2]["Photometry Measurement"].value) if len(table_phot[cond_W2]) == 1 else float(table_phot[cond_W2]["Photometry Measurement"].value[0]) # mag
            W3 = float(table_phot[cond_W3]["Photometry Measurement"].value) if len(table_phot[cond_W3]) == 1 else float(table_phot[cond_W3]["Photometry Measurement"].value[0]) # mag
        except IndexError as e:
            # print("Error extracting WISE fluxes. Skipping...")
            df_classifications.loc[gal, "W1 - W2 undetermined?"] = True
            df_classifications.loc[gal, "W1 - W2"] = np.nan
            df_classifications.loc[gal, "is AGN (MIR)?"] = "n/a"
            continue

        W1_lolim = float(table_phot[cond_W1]["Uncertainty"].value[0].split(">")[1]) if table_phot[cond_W1]["Uncertainty"].value[0].startswith(">") else np.nan
        W2_lolim = float(table_phot[cond_W2]["Uncertainty"].value[0].split(">")[1]) if table_phot[cond_W2]["Uncertainty"].value[0].startswith(">") else np.nan

        # Check: one of S_60_Jy or S_60_ulim_Jy must be NaN
        assert not (np.isnan(W1) and np.isnan(W1_lolim)),\
            "ERROR: both W1 and W1_ulim are undefined!"
        assert not (~np.isnan(W1) and ~np.isnan(W1_lolim)),\
            "ERROR: both W1 and W1_ulim are defined!"
        assert not (np.isnan(W2) and np.isnan(W2_lolim)),\
            "ERROR: both W2 and W2_ulim are undefined!"
        assert not (~np.isnan(W2) and ~np.isnan(W2_lolim)),\
            "ERROR: both W2 and W2_ulim are defined!"


        # Determine whether any measurements are upper limits
        W1_is_lower_limit = True if np.isnan(W1) and ~np.isnan(W2_lolim) else False
        W2_is_lower_limit = True if np.isnan(W2) and ~np.isnan(W2_lolim) else False
        W12_not_constrained = True if W1_is_lower_limit and W2_is_lower_limit else False
        W12_is_lower_lim = True if W1_is_lower_limit and not W2_is_upper_limit else False
        W12_is_upper_lim = True if W2_is_lower_limit and not W1_is_lower_limit else False

        # Apply the AGN criterion of Stern+2012 (https://ui.adsabs.harvard.edu/abs/2012ApJ...753...30S/abstract)
        # W1 - W2 >= 0.8: AGN
        # W1 - W2 < 0.8: not an AGN!
        W12 = W1 - W2
        W12_ulim = W1 - W2_lolim
        W12_lolim = W1_lolim - W2

        df_classifications.loc[gal, "W1 - W2"] = W12
        df_classifications.loc[gal, "W2 - W3"] = W2 - W3  # For plotting only
        df_classifications.loc[gal, "W1 - W2 upper limit"] = W12_ulim
        df_classifications.loc[gal, "W1 - W2 lower limit"] = W12_lolim
        df_classifications.loc[gal, "W1 - W2 is upper limit?"] = W12_is_upper_lim
        df_classifications.loc[gal, "W1 - W2 is lower limit?"] = W12_is_lower_lim
        df_classifications.loc[gal, "W1 - W2 undetermined?"] = W12_not_constrained
        if ~np.isnan(W12):
            df_classifications.loc[gal, "is AGN (MIR)?"] = "Yes" if W12 >= 0.8 else "No"
        elif W12_is_lower_lim:
            df_classifications.loc[gal, "is AGN (MIR)?"] = "Yes" if W12_lolim >= 0.8 else "No"
        elif W12_is_upper_lim:
            df_classifications.loc[gal, "is AGN (MIR)?"] = "No" if W12_ulim < 0.8 else "n/a"
        elif W12_not_constrained:
            df_classifications.loc[gal, "is AGN (MIR)?"] = "n/a"
    else:
        df_classifications.loc[gal, "W1 - W2 undetermined?"] = True
        df_classifications.loc[gal, "W1 - W2"] = np.nan
        df_classifications.loc[gal, "is AGN (MIR)?"] = "n/a"

    # print("----------------------------------------------------------------")

###############################################################################
# SAVE
###############################################################################
df_classifications.to_hdf(os.path.join(sami_data_path, "sami_dr3_agn_classifications.hd5"), key="AGN")


df_classifications = pd.read_hdf(os.path.join(sami_data_path, "sami_dr3_agn_classifications.hd5"), key="AGN")
gals = [int(g) for g in df_classifications.index.unique()]

###############################################################################
# PLOT TO CHECK
###############################################################################
df = df_classifications.loc[gals, :]

# Make numerical labels to make plotting easier
df.loc[df["is AGN (BPT)?"] == "Yes (Seyfert)", "is AGN (BPT)? (numeric)"] = 3
df.loc[df["is AGN (BPT)?"] == "Maybe (LINER)", "is AGN (BPT)? (numeric)"] = 2
df.loc[df["is AGN (BPT)?"] == "No", "is AGN (BPT)? (numeric)"] = 1
df.loc[df["is AGN (BPT)?"] == "n/a", "is AGN (BPT)? (numeric)"] = 0

df.loc[df["is AGN (WHAN)?"] == "Yes (Seyfert)", "is AGN (WHAN)? (numeric)"] = 3
df.loc[df["is AGN (WHAN)?"] == "Maybe (LINER)", "is AGN (WHAN)? (numeric)"] = 2
df.loc[df["is AGN (WHAN)?"] == "No", "is AGN (WHAN)? (numeric)"] = 1
df.loc[df["is AGN (WHAN)?"] == "n/a", "is AGN (WHAN)? (numeric)"] = 0

df.loc[df["is AGN (MIR)?"] == "Yes", "is AGN (MIR)? (numeric)"] = 2
df.loc[df["is AGN (MIR)?"] == "No", "is AGN (MIR)? (numeric)"] = 1
df.loc[df["is AGN (MIR)?"] == "n/a", "is AGN (MIR)? (numeric)"] = 0

df.loc[df["is AGN (FIR)?"] == "Yes", "is AGN (FIR)? (numeric)"] = 2
df.loc[df["is AGN (FIR)?"] == "No", "is AGN (FIR)? (numeric)"] = 1
df.loc[df["is AGN (FIR)?"] == "n/a", "is AGN (FIR)? (numeric)"] = 0

colours = ["grey", "blue", "orange", "yellow"]

###############################################################################
# Compare AGN classifications
###############################################################################
fig, axs = plt.subplots(nrows=1, ncols=2)

ii = 0
for cat in ["No", "Maybe (LINER)", "Yes (Seyfert)"]:
    axs[ii].hist(df_classifications.loc[df_classifications["is AGN (BPT)?"] == cat, "W1 - W2"],
                range=(-1, 6), bins=100, label=cat, density=True)
axs[ii].legend()

###############################################################################
# Check 1: BPT
###############################################################################
fig, axs = plt.subplots(nrows=1, ncols=4, figsize=(20, 4))
caxs = []
for ax in axs:
    plot_BPT_lines(ax=ax, col_x="log N2")

for cc, col in enumerate(["BPT", "WHAN", "MIR", "FIR"]):
    for ii, nn in enumerate(df[f"is AGN ({col})? (numeric)"].unique()):
        cond = df[f"is AGN ({col})? (numeric)"] == nn
        assert len(df.loc[cond, f"is AGN ({col})?"].unique()) <= 1
        if len(df.loc[cond, f"is AGN ({col})?"].unique()) > 0:
            label = df.loc[cond, f"is AGN ({col})?"].unique()[0]

            axs[cc].scatter(x=df.loc[cond, "log N2 (total)"],
                           y=df.loc[cond, "log O3 (total)"],
                           color=colours[int(nn)], s=5,
                           label=label)
    # Decorations
    axs[cc].legend()
    axs[cc].set_title(col)

###############################################################################
# Check 2: WHAN
###############################################################################
fig, axs = plt.subplots(nrows=1, ncols=4, figsize=(20, 5))
for cc, col in enumerate(["BPT", "WHAN", "MIR", "FIR"]):
    for ii, nn in enumerate(df[f"is AGN ({col})? (numeric)"].unique()):
        cond = df[f"is AGN ({col})? (numeric)"] == nn
        assert len(df.loc[cond, f"is AGN ({col})?"].unique()) <= 1
        if len(df.loc[cond, f"is AGN ({col})?"].unique()) > 0:
            label = df.loc[cond, f"is AGN ({col})?"].unique()[0]

            axs[cc].scatter(x=df.loc[cond, "log N2 (total)"],
                           y=df.loc[cond, "HALPHA EW"],
                           color=colours[int(nn)], s=5,
                           label=label)
    # Decorations
    axs[cc].legend()
    axs[cc].set_title(col)
    axs[cc].axhline(3, color="black")
    axs[cc].plot([-0.4, -0.4], [3, 1e3], color="black")
    axs[cc].plot([-0.4, 100], [6, 6], color="black")
    axs[cc].set_xlim([-1.0, +0.6])
    axs[cc].set_yscale("log")
    axs[cc].set_ylim([0.1, 210])

###############################################################################
# Check 3: WISE colour-colour plot
###############################################################################
fig, axs = plt.subplots(nrows=1, ncols=4, figsize=(20, 4))
caxs = []
for ax in axs:
    bbox = ax.get_position()
    caxs.append(fig.add_axes([bbox.x0 + bbox.width, bbox.y0, 0.01, bbox.height]))

fig, axs = plt.subplots(nrows=1, ncols=4, figsize=(20, 5))
for cc, col in enumerate(["BPT", "WHAN", "MIR", "FIR"]):
    for ii, nn in enumerate(df[f"is AGN ({col})? (numeric)"].unique()):
        cond = df[f"is AGN ({col})? (numeric)"] == nn
        assert len(df.loc[cond, f"is AGN ({col})?"].unique()) <= 1
        if len(df.loc[cond, f"is AGN ({col})?"].unique()) > 0:
            label = df.loc[cond, f"is AGN ({col})?"].unique()[0]

            axs[cc].scatter(x=df.loc[cond, "W2 - W3"],
                           y=df.loc[cond, "W1 - W2"],
                           color=colours[int(nn)], s=5,
                           label=label)
    # Decorations
    axs[cc].legend()
    axs[cc].set_title(col)
    axs[cc].axhline(0.8, color="black")
    axs[cc].set_xlim([-1.0, +6.0])
    axs[cc].set_ylim([-1.0, +3.0])

###############################################################################
# Check 2: for the MIR and FIR classifications, colour each point by W12 or q
###############################################################################
fig, axs = plt.subplots(nrows=1, ncols=4, figsize=(20, 4))
caxs = []
for ax in axs:
    bbox = ax.get_position()
    caxs.append(fig.add_axes([bbox.x0 + bbox.width, bbox.y0, 0.01, bbox.height]))

cmap = copy.copy(plt.cm.get_cmap("Spectral_r"))
cmap.set_bad("gray")
for ax in axs:
    plot_BPT_lines(ax=ax, col_x="log N2")

m = axs[0].scatter(x=df["log N2 (total)"], y=df["log O3 (total)"], c=df["FIR q"],
                   cmap=cmap, vmin=1.4, vmax=2.2, s=3)
plt.colorbar(mappable=m, cax=caxs[0], orientation="vertical")
caxs[0].set_ylabel("FIR q")

m = axs[1].scatter(x=df["log N2 (total)"], y=df["log O3 (total)"], c=df["W1 - W2"],
                   cmap=cmap, vmin=0, vmax=1.6, s=3)
plt.colorbar(mappable=m, cax=caxs[1], orientation="vertical")
caxs[1].set_ylabel(r"$W1 - W2$")

m = axs[2].scatter(x=df["log N2 (total)"], y=df["log O3 (total)"], c=df["log HALPHA EW"],
                   cmap="Spectral", vmin=-1, vmax=2, s=3)
plt.colorbar(mappable=m, cax=caxs[2], orientation="vertical")
caxs[2].set_ylabel(r"$\log_{10} \rm H\alpha$ EW")

m = axs[3].scatter(x=df["log N2 (total)"], y=df["log O3 (total)"], c=df["HBETA"] / df["HBETA error"] ,
                   cmap="Spectral", vmin=0, vmax=6, s=3)
plt.colorbar(mappable=m, cax=caxs[3], orientation="vertical")
caxs[3].set_ylabel(r"$\rm H\beta$ S/N")




