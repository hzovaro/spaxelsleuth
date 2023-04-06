import json
from pathlib import Path

# Load the default config file
def load_default_config():
    """Load the default config file."""
    with open(Path(__file__).parent / "config.json", "r") as f:
        global settings
        settings = json.load(f)

# Allow user to upload custom settings - e.g. colourmaps, vmin/vmax limits, paths
def load_user_config(p):
    """Load a custom config file. Overwrites default configuration files."""
    with open(Path(p)) as f:
        user_settings = json.load(f)
    # Merge with existing settings
    print(f"Updating settings from {p}:")
    for key in user_settings:
        if key in settings:
            print(f"{key}:")
            for subkey in user_settings[key]:
                print(f"\t{subkey}:")
                new_setting = user_settings[key][subkey]
                if subkey in settings[key]:
                    old_setting = settings[key][subkey]
                    print(f"\t\t{old_setting} --> {new_setting}")
                else:
                    print(f"\t\tAdding new setting {new_setting}")
                settings[key][subkey] = new_setting
        else:
            print(f"Adding new key {key}: {user_settings[key]}")
            settings[key] = user_settings[key]
