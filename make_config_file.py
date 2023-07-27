# Write config file 
import json
from spaxelsleuth.plotting.plottools import vmin_dict, vmax_dict, cmap_dict, fname_dict, label_dict

settings_dict = {
    "plotting": {}
}
for key in vmin_dict:
    settings_dict["plotting"][key] = {
        "vmin": vmin_dict[key],
        "vmax": vmax_dict[key],
        "label": label_dict[key],
        "cmap": cmap_dict[key],
        "fname": fname_dict[key],
    }

with open("/home/u5708159/.spaxelsleuthconfig.json", "w") as f:
    json.dump(settings_dict, f, indent=4)
