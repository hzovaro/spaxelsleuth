from spaxelsleuth.plotting import plottools

cols = [
    "HALPHA",
    "HALPHA (total)",
    "HALPHA (component 1)", 
    "sigma_gas (component 3)",
    "sigma_gas (R_e)",
    "D4000 (R_e (MGE))",
    "log(O/H) + 12 (PP04/K19)",
    "BPT (numeric)",
    "Morphology (numeric)",
]

# label function
for col in cols:
    label = plottools.get_cmap(col)
    print(f"{col} --> {label}")

# label function
for col in cols:
    label = plottools.get_label(col)
    print(f"{col} --> {label}")

# vmin/vmax functions
for col in cols:
    vmin = plottools.get_vmin(col)
    vmax = plottools.get_vmax(col)
    print(f"{col} --> {vmin}, {vmax}")

# Colname helper function
for col in cols:
    c, s = plottools.trim_suffix(col)
    print(f"{col} --> {c} + {s}")
