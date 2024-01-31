"""
Rename the columns below in the full-sized versions of the DataFrames (instead of running make_sami_df again).
"""
import pandas as pd
from pathlib import Path

from spaxelsleuth import load_user_config
load_user_config("/home/u5708159/.spaxelsleuthconfig.json")
from spaxelsleuth.config import settings

rename_dict = {
    "Galaxy centre x0_px (projected, arcsec)" : "x_0 (arcsec)",
    "Galaxy centre y0_px (projected, arcsec)" : "y_0 (arcsec)",
    "HALPHA lambda_obs (component 1) (Å)" : "HALPHA lambda_obs (Å) (component 1)",
    "HALPHA sigma_gas (component 1) (Å)" : "HALPHA sigma_gas (Å) (component 1)",
    "HALPHA lambda_obs (component 2) (Å)" : "HALPHA lambda_obs (Å) (component 2)",
    "HALPHA sigma_gas (component 2) (Å)" : "HALPHA sigma_gas (Å) (component 2)",
    "HALPHA lambda_obs (component 3) (Å)" : "HALPHA lambda_obs (Å) (component 3)",
    "HALPHA sigma_gas (component 3) (Å)" : "HALPHA sigma_gas (Å) (component 3)",
    "S2 ratio (total)" : "[SII] ratio (total)",
    "S2 ratio error (total)" : "[SII] ratio error (total)",
    "log S2 ratio (total)" : "log [SII] ratio (total)",
    "log S2 ratio error (lower) (total)" : "log [SII] ratio error (lower) (total)",
    "log S2 ratio error (upper) (total)" : "log [SII] ratio error (upper) (total)",
}

eline_SNR_min = 5
eline_ANR_min = 3
for ncomponents in ["recom", "1"]:
    for bin_type in ["default", "adaptive", "sectors"]:
        for correct_extinction in [True, False]:
            for debug in [False,]:

                # Get the filename
                df_fname = f"sami_{bin_type}_{ncomponents}-comp"
                if correct_extinction:
                    df_fname += "_extcorr"
                df_fname += f"_minSNR={eline_SNR_min}_minANR={eline_ANR_min}"
                if debug:
                    df_fname += "_DEBUG"
                df_fname += ".hd5"

                # Load the DataFrame. Note that we do not use load_sami_df() here because we don't need to check the extra columns added at runtime. 
                try:
                    print(f'Renaming columns in DataFrame {Path(settings["sami"]["output_path"]) / df_fname}...')
                    df = pd.read_hdf(
                        Path(settings["sami"]["output_path"]) / df_fname,
                        key=f"{bin_type}{ncomponents}comp",
                    )

                    # rename
                    df_renamed = df.rename(columns=rename_dict)
                    removed_cols = [c for c in df if c not in df_renamed]
                    added_cols = [c for c in df_renamed if c not in df]
                    print("The following columns have been ADDED:")
                    for col in added_cols:
                        print(f"\t{col}")
                    print("The following columns have been REMOVED:")
                    for col in removed_cols:
                        print(f"\t{col}")

                    # Check that nothing else has changedf
                    assert df_renamed.shape == df.shape

                    # Overwrite 
                    df_renamed.to_hdf(
                        Path(settings["sami"]["output_path"]) / df_fname,
                        key=f"{bin_type}{ncomponents}comp",
                    )

                except FileNotFoundError:
                    print(f'Could not locate DataFrame {Path(settings["sami"]["output_path"]) / df_fname}!')
                    pass