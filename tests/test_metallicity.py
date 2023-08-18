# Imports
import numpy as np
from time import time

from spaxelsleuth import load_user_config
load_user_config("/home/u5708159/.spaxelsleuthconfig.json")
from spaxelsleuth.loaddata.sami import load_sami_df

from spaxelsleuth.utils.metallicity import calculate_metallicity, line_list_dict

from spaxelsleuth.plotting.plotgalaxies import plot2dscatter
import matplotlib.pyplot as plt
plt.ion()
plt.close("all")

from IPython.core.debugger import Tracer

##############################################################################
# CHECK: only SF-like spaxels have nonzero metallicities.
def assertion_checks(df):
    print("Running assertion checks...")
    cond_not_SF = df["BPT (total)"] != "SF"
    for c in [c for c in df.columns if "log(O/H) + 12" in c]:
        assert all(df.loc[cond_not_SF, c].isna())

    # CHECK: rows with NaN in any required emission lines have NaN metallicities and ionisation parameters. 
    for met_diagnostic in line_list_dict.keys():
        for line in [l for l in line_list_dict[met_diagnostic] if f"{l} (total)" in df.columns]:
            cond_isnan = np.isnan(df[f"{line} (total)"])
            cols = [c for c in df.columns if met_diagnostic in c]
            for c in cols:
                assert all(df.loc[cond_isnan, c].isna())
            
    # CHECK: all rows with NaN metallicities also have NaN log(U).
    for c in [c for c in df.columns if "log(O/H) + 12" in c and "error" not in c]:
        diagnostic_str = c.split("log(O/H) + 12 (")[1].split(")")[0]
        cond_nan_logOH12 = df[c].isna()
        if f"log(U) ({diagnostic_str}) (total)" in df.columns:
            assert all(df.loc[cond_nan_logOH12, f"log(U) ({diagnostic_str}) (total)"].isna())
            # Also check the converse 
            cond_finite_logU = ~df[f"log(U) ({diagnostic_str}) (total)"].isna()
            assert all(~df.loc[cond_finite_logU, f"log(O/H) + 12 ({diagnostic_str}) (total)"].isna())

    print("Passed assertion checks!")

##############################################################################
# Load DataFrame
df = load_sami_df(ncomponents="recom", bin_type="default", correct_extinction=True, eline_SNR_min=5, debug=True)
assertion_checks(df)

# Remove prior metallicity calculation results 
cols_to_remove = [c for c in df.columns if "log(O/H) + 12" in c]
cols_to_remove += [c for c in df.columns if "log(U)" in c]
df = df.drop(columns=cols_to_remove)

##############################################################################
# TIMING TEST: how long does it take to compute all metallicity diagnostics with errors?
print("Testing log(O/H) + 12 computation w/o log(U) calculation...")
t = time()
df = calculate_metallicity(met_diagnostic="N2Ha_K19", compute_logU=True, ion_diagnostic="O3O2_K19", compute_errors=True, niters=1000, df=df, s=" (total)")
df = calculate_metallicity(met_diagnostic="O3N2_K19", compute_logU=True, ion_diagnostic="O3O2_K19", compute_errors=True, niters=1000, df=df, s=" (total)")
df = calculate_metallicity(met_diagnostic="N2O2_K19", compute_logU=True, ion_diagnostic="O3O2_K19", compute_errors=True, niters=1000, df=df, s=" (total)")
df = calculate_metallicity(met_diagnostic="R23_KK04", compute_logU=True, ion_diagnostic="O3O2_KK04", compute_errors=True, niters=1000, df=df, s=" (total)")

df = calculate_metallicity(met_diagnostic="N2Ha_PP04", compute_errors=True, niters=1000, df=df, s=" (total)")
df = calculate_metallicity(met_diagnostic="N2Ha_M13", compute_errors=True, niters=1000, df=df, s=" (total)")
df = calculate_metallicity(met_diagnostic="O3N2_PP04", compute_errors=True, niters=1000, df=df, s=" (total)")
df = calculate_metallicity(met_diagnostic="O3N2_M13", compute_errors=True, niters=1000, df=df, s=" (total)")
df = calculate_metallicity(met_diagnostic="N2S2Ha_D16", compute_errors=True, niters=1000, df=df, s=" (total)")
df = calculate_metallicity(met_diagnostic="N2O2_KD02", compute_errors=True, niters=1000, df=df, s=" (total)")
df = calculate_metallicity(met_diagnostic="Rcal_PG16", compute_errors=True, niters=1000, df=df, s=" (total)")
df = calculate_metallicity(met_diagnostic="Scal_PG16", compute_errors=True, niters=1000, df=df, s=" (total)")
df = calculate_metallicity(met_diagnostic="ON_P10", compute_errors=True, niters=1000, df=df, s=" (total)")
df = calculate_metallicity(met_diagnostic="ONS_P10", compute_errors=True, niters=1000, df=df, s=" (total)")

dt = time() - t
print(f"Total time elapsed: {dt / 60} minutes")

assertion_checks(df)

##############################################################################
# TEST: metallicity calculation w/o errors, w/o log(U) calculation
print("Testing log(O/H) + 12 computation w/o log(U) calculation...")
df = calculate_metallicity(met_diagnostic="N2Ha_PP04", compute_errors=False, df=df, s=" (total)")
assertion_checks(df)
df = calculate_metallicity(met_diagnostic="N2Ha_M13", compute_errors=False, df=df, s=" (total)")
assertion_checks(df)
df = calculate_metallicity(met_diagnostic="N2Ha_K19", compute_errors=False, logU=-3.0, df=df, s=" (total)")
assertion_checks(df)
df = calculate_metallicity(met_diagnostic="O3N2_PP04", compute_errors=False, df=df, s=" (total)")
assertion_checks(df)
df = calculate_metallicity(met_diagnostic="O3N2_M13", compute_errors=False, df=df, s=" (total)")
assertion_checks(df)
df = calculate_metallicity(met_diagnostic="O3N2_K19", compute_errors=False, logU=-3.0, df=df, s=" (total)")
assertion_checks(df)
df = calculate_metallicity(met_diagnostic="N2S2Ha_D16", compute_errors=False, df=df, s=" (total)")
assertion_checks(df)
df = calculate_metallicity(met_diagnostic="N2O2_KD02", compute_errors=False, df=df, s=" (total)")
assertion_checks(df)
df = calculate_metallicity(met_diagnostic="N2O2_K19", compute_errors=False, logU=-3.0, df=df, s=" (total)")
assertion_checks(df)
df = calculate_metallicity(met_diagnostic="R23_KK04", ion_diagnostic="O3O2_KK04", compute_errors=False, compute_logU=True, df=df, s=" (total)")
assertion_checks(df)
df = calculate_metallicity(met_diagnostic="Rcal_PG16", compute_errors=False, df=df, s=" (total)")
assertion_checks(df)
df = calculate_metallicity(met_diagnostic="Scal_PG16", compute_errors=False, df=df, s=" (total)")
assertion_checks(df)
df = calculate_metallicity(met_diagnostic="ON_P10", compute_errors=False, df=df, s=" (total)")
assertion_checks(df)
df = calculate_metallicity(met_diagnostic="ONS_P10", compute_errors=False, df=df, s=" (total)")
assertion_checks(df)

# Plot to check: O3N2 calibrations 
fig, ax = plt.subplots(nrows=1, ncols=1)
ax.scatter(df["log(O/H) + 12 (N2Ha_PP04) (total)"], df["log N2 (total)"], label="N2Ha_PP04")
ax.scatter(df["log(O/H) + 12 (N2Ha_M13) (total)"], df["log N2 (total)"], label="N2Ha_M13")
ax.scatter(df["log(O/H) + 12 (N2Ha_K19) (total)"], df["log N2 (total)"], label="N2Ha_K19")
ax.set_xlabel("log(O/H) + 12")
ax.set_ylabel(r"$\log(R)$ (N2Ha)")
ax.legend()
ax.grid()

# Plot to check: O3N2 calibrations 
fig, ax = plt.subplots(nrows=1, ncols=1)
ax.scatter(df["log(O/H) + 12 (O3N2_PP04) (total)"], df["O3N2 (total)"], label="O3N2_PP04")
ax.scatter(df["log(O/H) + 12 (O3N2_M13) (total)"], df["O3N2 (total)"], label="O3N2_M13")
ax.scatter(df["log(O/H) + 12 (O3N2_K19) (total)"], df["O3N2 (total)"], label="O3N2_K19")
ax.set_xlabel("log(O/H) + 12")
ax.set_ylabel(r"$\log(R)$ (O3N2)")
ax.legend()
ax.grid()

# Plot to check: N2O2 calibrations 
fig, ax = plt.subplots(nrows=1, ncols=1)
ax.scatter(df["log(O/H) + 12 (N2O2_K19) (total)"], df["N2O2 (total)"], label="N2O2_K19")
ax.scatter(df["log(O/H) + 12 (N2O2_KD02) (total)"], df["N2O2 (total)"], label="N2O2_KD02")
ax.set_xlabel("log(O/H) + 12")
ax.set_ylabel(r"$\log(R)$ (N2O2)")
ax.legend()
ax.grid()

# Plot to check: ON vs. ONS 
fig, ax = plt.subplots(nrows=1, ncols=1)
plot2dscatter(df=df, col_y="log(O/H) + 12 (ON_P10) (total)", col_x="log(O/H) + 12 (ONS_P10) (total)", marker="^", markerfacecolor="green", markeredgecolor="none", ax=ax)
ax.plot([7.5, 9.5], [7.5, 9.5], color="k")
ax.grid()

#//////////////////////////////////////////////////////////////////////////
# TEST: metallicity calculation w/o errors, WITH log(U) calculation
print("Testing log(O/H) + 12 computation with log(U) calculation: R23_KK04...")
# R23 - KK04
df = calculate_metallicity(met_diagnostic="R23_KK04", logU=-3.0, compute_errors=False, df=df, s=" (total)")
assertion_checks(df)
df = calculate_metallicity(met_diagnostic="R23_KK04", compute_logU=True, ion_diagnostic="O3O2_KK04", compute_errors=False, df=df, s=" (total)")
assertion_checks(df)

# Plot to check 
fig, ax = plt.subplots(nrows=1, ncols=1)
plot2dscatter(df=df, col_x="log(O/H) + 12 (R23_KK04) (total)", col_y="R23 (total)", marker="o", markerfacecolor="white", markeredgecolor="black", ax=ax)
plot2dscatter(df=df, col_x="log(O/H) + 12 (R23_KK04/O3O2_KK04) (total)", col_y="R23 (total)", col_z="log(U) (R23_KK04/O3O2_KK04) (total)", marker="^", markeredgecolor="none", plot_colorbar=True, ax=ax)
ax.set_xlabel("log(O/H) + 12")
ax.set_ylabel(r"$\log(R)$ (R23)")
ax.grid()

# Comparison of log(O/H) + 12 w/ iterative log(U) computation vs. that without
fig, ax = plt.subplots(nrows=1, ncols=1)
plot2dscatter(df=df, col_x="log(O/H) + 12 (R23_KK04) (total)", col_y="log(O/H) + 12 (R23_KK04/O3O2_KK04) (total)", marker="o", markerfacecolor="white", markeredgecolor="black", ax=ax)
ax.grid()
ax.plot([7.5, 9.3], [7.5, 9.3], color="grey")

# Repeat for O3N2
print("Testing log(O/H) + 12 computation with log(U) calculation: O3N2_K19...")
df = calculate_metallicity(met_diagnostic="O3N2_K19", logU=-3.0, compute_errors=False, df=df, s=" (total)")
assertion_checks(df)
df = calculate_metallicity(met_diagnostic="O3N2_K19", compute_logU=True, ion_diagnostic="O3O2_K19", compute_errors=False, df=df, s=" (total)")
assertion_checks(df)

# Plot to check 
fig, ax = plt.subplots(nrows=1, ncols=1)
plot2dscatter(df=df, col_x="log(O/H) + 12 (O3N2_K19) (total)", col_y="R23 (total)", marker="o", markerfacecolor="white", markeredgecolor="black", ax=ax)
plot2dscatter(df=df, col_x="log(O/H) + 12 (O3N2_K19/O3O2_K19) (total)", col_y="R23 (total)", col_z="log(U) (O3N2_K19/O3O2_K19) (total)", marker="^", markeredgecolor="none", plot_colorbar=True, ax=ax)
ax.set_xlabel("log(O/H) + 12")
ax.set_ylabel(r"$\log(R)$ (R23)")
ax.grid()

# Comparison of log(O/H) + 12 w/ iterative log(U) computation vs. that without
fig, ax = plt.subplots(nrows=1, ncols=1)
plot2dscatter(df=df, col_x="log(O/H) + 12 (O3N2_K19) (total)", col_y="log(O/H) + 12 (O3N2_K19/O3O2_K19) (total)", col_z="log(U) (O3N2_K19/O3O2_K19) (total)", marker="o", markeredgecolor="black", ax=ax)
ax.grid()
ax.plot([7.5, 9.3], [7.5, 9.3], color="grey")

##############################################################################
# TEST: metallicity calculation WITH errors, WITH log(U) calculation
print("Testing log(O/H) + 12 computation with log(U) calculation AND errors: R23_KK04...")
# df = calculate_metallicity(met_diagnostic="R23_KK04", logU=-3.0, compute_errors=True, niters=100, df=df, s=" (total)")
# assertion_checks(df)
df = calculate_metallicity(met_diagnostic="R23_KK04", compute_logU=True, ion_diagnostic="O3O2_KK04", compute_errors=True, niters=100, df=df, s=" (total)")
assertion_checks(df)

# Plot to check 
fig, ax = plt.subplots(nrows=1, ncols=1)
plot2dscatter(df=df, col_x="log(O/H) + 12 (R23_KK04) (total)", col_y="R23 (total)", marker="^", markerfacecolor="green", markeredgecolor="none", ax=ax)
plot2dscatter(df=df, col_x="log(O/H) + 12 (R23_KK04/O3O2_KK04) (total)", col_y="R23 (total)", col_z="log(U) (R23_KK04/O3O2_KK04) (total)", marker="^", markeredgecolor="none", plot_colorbar=True, ax=ax)
ax.set_xlabel("log(O/H) + 12")
ax.set_ylabel(r"$\log(R)$ (R23)")

##############################################################################
# TEST: metallicity calculation with errors, w/o log(U) calculation
print("Testing log(O/H) + 12 computation with errors: N2Ha...")
df = calculate_metallicity(met_diagnostic="N2Ha_PP04", compute_errors=True, niters=100, df=df, s=" (total)")
assertion_checks(df)
df = calculate_metallicity(met_diagnostic="N2Ha_M13", compute_errors=True, niters=100, df=df, s=" (total)")
assertion_checks(df)
df = calculate_metallicity(met_diagnostic="N2Ha_K19", compute_errors=True, niters=100, logU=-3.0, df=df, s=" (total)")
assertion_checks(df)

# Plot to check 
fig, ax = plt.subplots(nrows=1, ncols=1)
plot2dscatter(df=df, col_x="log(O/H) + 12 (N2Ha_PP04) (total)", col_y="log N2 (total)", marker="^", markerfacecolor="yellow", markeredgecolor="none", ax=ax)
plot2dscatter(df=df, col_x="log(O/H) + 12 (N2Ha_M13) (total)", col_y="log N2 (total)", marker="o", markerfacecolor="green", markeredgecolor="none", ax=ax)
plot2dscatter(df=df, col_x="log(O/H) + 12 (N2Ha_K19) (total)", col_y="log N2 (total)", marker="v", markerfacecolor="red", markeredgecolor="none", ax=ax)
ax.set_xlabel("log(O/H) + 12")
ax.set_ylabel(r"$\log(R)$ (N2Ha)")
ax.legend()
ax.grid()

##############################################################################
# Make a big corner plot 
from itertools import product
from spaxelsleuth.plotting.plotgalaxies import plot2dhistcontours

met_diagnostics = ["N2Ha_PP04", "N2Ha_M13", "O3N2_PP04", "O3N2_M13",
                   "N2S2Ha_D16", "N2O2_KD02", "N2Ha_K19/O3O2_K19"]
# met_diagnostics = ["R23_KK04/O3O2_KK04", "N2O2_KD02", "O3N2_PP04", "N2Ha_PP04", "Scal_PG16"]
fig, axs = plt.subplots(nrows=len(met_diagnostics) - 1, ncols=len(met_diagnostics) - 1)
fig.subplots_adjust(hspace=0, wspace=0)
cnt = len(met_diagnostics) - 1
for rr, cc in product(range(cnt), range(cnt)):
    if rr >= cc:
        # print(f"In ax[{rr}][{cc}]: x = {met_diagnostics[cc]}, y = {met_diagnostics[rr + 1]}")
        plot2dhistcontours(df=df, 
                           col_x=f"log(O/H) + 12 ({met_diagnostics[cc]}) (total)",
                           col_y=f"log(O/H) + 12 ({met_diagnostics[rr + 1]}) (total)",
                           col_z="count", log_z=True,
                           nbins=50, cmap="cubehelix",
                           vmin=10, vmax=1e4, plot_colorbar=False,
                           ax=axs[rr][cc],)
        axs[rr][cc].plot([7.5, 9.3], [7.5, 9.3], color="grey")
        axs[rr][cc].grid()
        axs[rr][cc].autoscale(enable=True, tight=True)
        if cc == 0:
            axs[rr][cc].set_ylabel(f"{met_diagnostics[rr + 1]}")
        if rr == len(met_diagnostics) - 2:
            axs[rr][cc].set_xlabel(f"{met_diagnostics[cc]}")
        if cc > 0:
            axs[rr][cc].set_yticklabels([""])
            axs[rr][cc].set_ylabel("")
        if rr < len(met_diagnostics) - 2:
            axs[rr][cc].set_xticklabels([""])
            axs[rr][cc].set_xlabel("")
    else:
        axs[rr][cc].set_visible(False)

