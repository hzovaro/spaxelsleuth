from spaxelsleuth.plotting import plottools, plottools_new

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
    # Old 
    label = plottools.cmap_fn(col)
    print(f"OLD: {col} --> {label}")

    # New   
    label = plottools_new.get_cmap(col)
    print(f"NEW: {col} --> {label}")

# label function
for col in cols:
    # Old 
    label = plottools.label_fn(col)
    print(f"OLD: {col} --> {label}")

    # New   
    label = plottools_new.get_label(col)
    print(f"NEW: {col} --> {label}")

# vmin/vmax functions
for col in cols:
    # Old 
    vmin = plottools.vmin_fn(col)
    vmax = plottools.vmax_fn(col)
    print(f"OLD: {col} --> {vmin}, {vmax}")

    # New   
    vmin = plottools_new.get_vmin(col)
    vmax = plottools_new.get_vmax(col)
    print(f"NEW: {col} --> {vmin}, {vmax}")


# Colname helper function
for col in cols:
    # Old 
    c, s = plottools._colname_helper_fn(col)
    print(f"OLD: {col} --> {c} + {s}")

    # New   
    c, s = plottools_new.trim_suffix(col)
    print(f"NEW: {col} --> {c} + {s}")
