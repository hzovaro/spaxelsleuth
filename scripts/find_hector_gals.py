from glob import glob
import numpy as np
import os
import pandas as pd    
from pathlib import Path
from tqdm import tqdm

from spaxelsleuth import load_user_config
load_user_config(sys.argv[1])
from spaxelsleuth.config import settings

input_path = Path(settings["hector"]["input_path"])
output_path = Path(settings["hector"]["output_path"])
data_cube_path = Path(settings["hector"]["data_cube_path"])
eline_fit_path = input_path / "emission_cubes"
stekin_path = input_path / "initial_stel_kin"
continuum_fit_path = input_path / "cont_subtracted"


"""

NEW APPROACH
treat unique gal, tile pairs as an individual ENTITIY
i.e. replace gal -- > gal __ tile 

"""


################################

# # Locate all data cubes 
# subdirs = [f for f in os.listdir(data_cube_path) if os.path.isdir(data_cube_path / f)]
# gal_subdirs = {}
# gals_with_data_cubes = []
# for subdir in subdirs:
#     gals_in_this_subdir = [int(g) for g in os.listdir(data_cube_path / subdir) if os.path.isdir(data_cube_path / subdir / g)]
#     gals_with_data_cubes += gals_in_this_subdir
#     for gal in gals_in_this_subdir:
#         gal_subdirs[gal] = subdir

# # Record which galaxies have duplicates 
# gals_with_duplicates = []
# seen = []
# for gal in gals_with_data_cubes:
#     if gal in seen:
#         gals_with_duplicates.append(gal)
#     else:
#         seen.append(gal)
# print(f"len(gals_with_duplicates) = {len(gals_with_duplicates)}")

# # Store galaxies WITHOUT duplicates in the final list 
# gals_with_data_cubes = [g for g in gals_with_data_cubes if g not in gals_with_duplicates]
# print(f"len(gals_with_data_cubes) = {len(gals_with_data_cubes)}")

all_data_cube_files = glob(str(data_cube_path) + "/**/*.fits", recursive=True)
data_cube_files_B = [Path(f) for f in all_data_cube_files if "blue" in f]
data_cube_files_R = [Path(f) for f in all_data_cube_files if "red" in f]


# continuum_fit_files_B = [Path(f) for f in glob(str(continuum_fit_path) + "/**/*.fits", recursive=True) if "_blue_" in f]
# continuum_fit_files_R = [Path(f) for f in glob(str(continuum_fit_path) + "/**/*.fits", recursive=True) if "_red_" in f]
# stekin_files = [Path(f) for f in glob(str(stekin_path) + "/**/*.fits", recursive=True)]
# eline_fit_files_1comp = [Path(f) for f in glob(str(eline_fit_path) + "/**/*.fits", recursive=True) if "1comp" in f]
# eline_fit_files_2comp = [Path(f) for f in glob(str(eline_fit_path) + "/**/*.fits", recursive=True) if "2comp" in f]
# eline_fit_files_3comp = [Path(f) for f in glob(str(eline_fit_path) + "/**/*.fits", recursive=True) if "3comp" in f]
# eline_fit_files_reccomp = [Path(f) for f in glob(str(eline_fit_path) + "/**/*.fits", recursive=True) if "reccomp" in f]

# os.listdir is much faster 
continuum_fit_files_B = [continuum_fit_path / f for f in os.listdir(continuum_fit_path) if "blue" in f]
continuum_fit_files_R = [continuum_fit_path / f for f in os.listdir(continuum_fit_path) if "red" in f]
stekin_files = [stekin_path / f for f in os.listdir(stekin_path)]
eline_fit_files_reccomp = [eline_fit_path / f for f in os.listdir(eline_fit_path) if f.endswith(".fits") and "reccomp" in f]
eline_fit_files_1comp = [eline_fit_path / f for f in os.listdir(eline_fit_path) if f.endswith(".fits") and "1comp" in f]
eline_fit_files_2comp = [eline_fit_path / f for f in os.listdir(eline_fit_path) if f.endswith(".fits") and "2comp" in f]
eline_fit_files_3comp = [eline_fit_path / f for f in os.listdir(eline_fit_path) if f.endswith(".fits") and "3comp" in f]

# Now, check that each of these has each of Gabby's data products 
ids_with_initial_stel_kin = [
    g for g in set([f.split("_initial_kinematics")[0] for f in os.listdir(stekin_path) if f.endswith(".fits")])
]
print(f"len(ids_with_initial_stel_kin) = {len(ids_with_initial_stel_kin)}")

ids_with_cont_subtracted = [
    g for g in set([f.split("_blue_stel_subtract_final")[0] for f in os.listdir(continuum_fit_path) if f.endswith(".fits") and "blue" in f])
]
print(f"len(ids_with_cont_subtracted) = {len(ids_with_cont_subtracted)}")

ids_with_emission_cubes = [
    g for g in set([f.split("_reccomp")[0] for f in os.listdir(eline_fit_path) if f.endswith(".fits") and "rec" in f])
]
print(f"len(ids_with_emission_cubes) = {len(ids_with_emission_cubes)}")

ids_with_all_data_products = list(set(ids_with_initial_stel_kin) & set(ids_with_cont_subtracted) & set(ids_with_emission_cubes))
print(f"len(ids_with_all_data_products) = {len(ids_with_all_data_products)}")

# gals_final = list(set(gals_with_data_cubes) & set(gals_with_all_data_products))
# print(f"len(gals_final) = {len(gals_final)}")

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
            print(f"{id_str} has 0 {file_type} files!")
            df_filenames.loc[id_str, f"Has {file_type}"] = False
            df_filenames.loc[id_str, f"Duplicate {file_type}"] = False
            df_filenames.loc[id_str, file_type] = ""
        elif len(gal_file_list) > 1:
            print(f"{id_str} has {len(gal_file_list)} {file_type} files:")
            for fname in gal_file_list:
                print("\t" + fname)
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
print(f"{df_filenames[cond_good].shape[0]} / {df_filenames.shape[0]} have all required files")

# Check that all files exist 
# TODO logger call
for id_str in df_filenames[cond_good].index.values:
    for file_type in file_types:
        assert os.path.exists(df_filenames.loc[id_str, file_type])



# Check how many repeat gals there are 
df_no_duplicates = df_filenames.loc[~df_filenames["ID"].duplicated()]
df_no_duplicates.loc[:, "ID string"] = df_no_duplicates.index.values
df_no_duplicates = df_no_duplicates.set_index("ID")

    # # Find the data cube files 
    # subdirs = [f for f in os.listdir(data_cube_path) if os.path.isdir(data_cube_path / f)]
    # for subdir in subdirs:
    #     list_of_folders = os.listdir(data_cube_path / subdir)
    #     if str(gal) in list_of_folders:
    #         list_of_files = [f for f in os.listdir(data_cube_path / subdir / str(gal)) if "blue" in f]
    #         for fname_B in list_of_files:
    #             if gal in fname_B and ((tile in fname_B) or (f"T_{tile_number}" in fname_B)):
    #                 print(f"{id_str}: {fname_B}")
    #                 df_filenames.loc[id_str, "Blue data cube FITS file"] = data_cube_path / subdir / str(gal) / fname_B
    #                 fname_R = fname_B.replace("blue", "red") 
    #                 df_filenames.loc[id_str, "Red data cube FITS file"] = data_cube_path / subdir / str(gal) / fname_B

    # Find the stellar kinematics files 


    # Check for cases where both 6/7 dither frames are available...


# # NOW try to open each of them to check that the files are accessible... 
# for gal in gals_final:
#     # Get the blue & read data cube names 
#     #TODO check this 
#     data_cube_subdirs = [f for f in os.listdir(data_cube_path / gal_subdirs[gal]) if f.startswith(str(gal))]
#     assert len(data_cube_subdirs) == 1
#     data_cube_subdir = data_cube_subdirs[0]
#     datacube_B_fnames = [data_cube_path / gal_subdirs[gal] / data_cube_subdir / f for f in os.listdir(data_cube_path / gal_subdirs[gal] / data_cube_subdir) if f.startswith(str(gal)) and "blue" in f]
#     datacube_R_fnames = [data_cube_path / gal_subdirs[gal] / data_cube_subdir / f for f in os.listdir(data_cube_path / gal_subdirs[gal] / data_cube_subdir) if f.startswith(str(gal)) and "red" in f]
    
#     # TODO figure out WHICH data cube is the right one!!! For now just assume it's the first one... 
#     datacube_B_fnames = datacube_B_fnames[:1]
#     datacube_R_fnames = datacube_R_fnames[:1]
    
#     assert len(datacube_B_fnames) == 1
#     assert len(datacube_R_fnames) == 1
#     datacube_B_fname = datacube_B_fnames[0]
#     datacube_R_fname = datacube_R_fnames[0]
#     assert os.path.exists(datacube_B_fname)
#     assert os.path.exists(datacube_R_fname)

#     # FITS filenames for stellar kinematics & continuum fit data products
#     # TODO why the fuck doesn't this work in data_products
#     stekin_fnames = [stekin_path / f for f in os.listdir(stekin_path) if f.startswith(str(gal))]
#     cont_fit_B_fnames = [continuum_fit_path / f for f in os.listdir(continuum_fit_path) if f.startswith(str(gal)) and "blue" in f]
#     cont_fit_R_fnames = [continuum_fit_path / f for f in os.listdir(continuum_fit_path) if f.startswith(str(gal)) and "red" in f]
#     assert len(stekin_fnames) == 1
#     assert len(cont_fit_B_fnames) == 1
#     assert len(cont_fit_R_fnames) == 1
#     stekin_fname = stekin_fnames[0]
#     cont_fit_B_fname = cont_fit_B_fnames[0]
#     cont_fit_R_fname = cont_fit_R_fnames[0]
#     assert os.path.exists(stekin_fname)
#     assert os.path.exists(cont_fit_B_fname)
#     assert os.path.exists(cont_fit_R_fname)

#     # Get FITS filenames for emission line fit data products
#     # TODO check this
#     # eline_fit_subdirs = [f for f in os.listdir(eline_fit_path) if f.startswith(str(gal))]
#     # assert len(eline_fit_subdirs) == 1
#     # eline_fit_subdir = eline_fit_subdirs[0]
#     eline_fit_fnames = []
#     for ncomponents in [1, 2, 3, "rec"]:
#         eline_component_fit_fnames = [eline_fit_path / f for f in os.listdir(eline_fit_path) if f.startswith(str(gal)) and f"{ncomponents}comp" in f]
#         # eline_component_fit_fnames = [eline_fit_path / eline_fit_subdir / f for f in os.listdir(eline_fit_path / eline_fit_subdir) if f.startswith(str(gal)) and f"{ncomponents}comp" in f]
#         assert len(eline_component_fit_fnames) == 1
#         eline_fit_fname = eline_component_fit_fnames[0]
#         assert os.path.exists(eline_fit_fname)
#         eline_fit_fnames.append(eline_fit_fname)

