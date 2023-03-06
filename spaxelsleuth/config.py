# Example of how this is done in matplotlib:
# import matplotlib.pyplot as plt
# plt.style.use('./images/presentation.mplstyle')

# And then in __init__.py: from loadconfig import load_user_config.py()
# Maybe load the default one here too? 
# And then 
# merge into a big dict called settings? 

import json
from pathlib import Path

# Load the default config file
def load_default_config():
    with open(Path(__file__).parent / "config.json", "r") as f:
        global settings
        settings = json.load(f)
    # 


# def set_config(s):
#     global setting
#     setting = s

# # If we have run foo(), then bar() can "see" the value of setting
# def bar():
#     # Need to check that foo() has been run first
#     print(setting)

# Usage: 
# from spaxelsleuth import set_config (TODO: rename)
# set_config(path_to_custom_config_file)


# In __init__.py want to load the default config file
# then the user can run set_config() which overwrites the default values 

# To import into other modules:
# import spaxelsleuth.config

