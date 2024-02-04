"""
Add missing columns to full-sized versions of the DataFrames (instead of running make_sami_df again).
"""
import pandas as pd
from pathlib import Path

from spaxelsleuth import load_user_config
load_user_config("/home/u5708159/.spaxelsleuthconfig.json")
from spaxelsleuth.config import settings

eline_SNR_min = 5
eline_ANR_min = 3
for ncomponents in ["recom", "1"]:
    for bin_type in ["default", "adaptive", "sectors"]:
        for correct_extinction in [True, False]:
            for debug in [True, False]:

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

                    # Add                    
                    old_cols = df.columns
                    if "missing_kinematics_cut" not in df.columns:
                        df["missing_kinematics_cut"] = True
                    new_cols = df.columns  
                    added_cols = [c for c in new_cols if c not in old_cols]                      
                    removed_cols = [c for c in old_cols if c not in new_cols]                      
                    print("The following columns have been ADDED:")
                    for col in added_cols:
                        print(f"\t{col}")
                    print("The following columns have been REMOVED:")
                    for col in removed_cols:
                        print(f"\t{col}")

                    # Overwrite 
                    df.to_hdf(
                        Path(settings["sami"]["output_path"]) / df_fname,
                        key=f"{bin_type}{ncomponents}comp",
                    )

                except FileNotFoundError:
                    print(f'Could not locate DataFrame {Path(settings["sami"]["output_path"]) / df_fname}!')
                    pass