This folder contains scripts that Matt Alger wrote/edited during the code review session on Sunday August 27, 2023.

Their suggestions:
* don't use pandas to write the DataFrame to file, because it's not very flexible when it comes to metadata, mixing data types, etc. Better to use a dedicated HDF IO library e.g. h5py (?). Can save metadata e.g. 'survey', 'bin_type' separately to the table to save disk space. Could also save all numeric data in 1 table, and string data in another table, but in the same file, and merge back together within the loader functions.
* could also save metadata and the full data table in a .zip or .tar file instead of using hd5. 
* consider using custom dataclasses (see dcs.py) w/ numpy arrays when manipulating data, calculating stuff etc. and only use DataFrames when reading/writing to file. This will speed up processing considerably.
* consider storing column names somewhere and referring to them as variables rather than typing them out manually each time - easy to make mistakes, typos etc. this way. 