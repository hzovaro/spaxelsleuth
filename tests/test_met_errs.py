

import numpy as np
import pandas as pd
from time import time
import sys

from spaxelsleuth.loaddata.sami import load_sami_df
from spaxelsleuth.utils.metallicity import calculate_metallicity, line_list_dict
from spaxelsleuth.plotting.plotgalaxies import plot2dscatter

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
df = load_sami_df(ncomponents="recom", bin_type="default", eline_SNR_min=5, correct_extinction=True, debug=True)

# Time operation
t = time()
df = calculate_metallicity(met_diagnostic="R23_KK04", compute_logU=True, ion_diagnostic="O3O2_KK04", compute_errors=True, df=df, s=" (total)")
assertion_checks(df)
df = calculate_metallicity(met_diagnostic="R23_KK04", compute_logU=True, ion_diagnostic="O3O2_KK04", compute_errors=False, df=df, s=" (total)")
assertion_checks(df)
dt = time() - t
print(f"With errors: N = {df.shape[0]}: Total time = {int(np.floor((dt) / 3600)):d}:{int(np.floor((dt) / 60)):d}:{np.mod(dt, 60):02.5f}" +\
      f" (Average {dt / df.shape[0]:.5f}s per iteration)")


sys.exit()

"""
Trying to find a more efficient way to compute errors for metallicities
"""
def foo(df):
    A = df["A"].values 
    # A_err = df["A_err"].values
    B = df["B"].values
    # B_err = df["B_err"].values

    A_tmp = A 
    B_tmp = B
    # A_tmp += np.random.normal(loc=0, scale=A_err)
    # B_tmp += np.random.normal(loc=0, scale=B_err)
    return A_tmp + B_tmp

# Figuring out how to use df.apply()
def myfun(df):
    # A = df["A"].values 
    # A_err = df["A_err"].values
    # B = df["B"].values
    # B_err = df["B_err"].values

    niters = 1000
    c_list = []
    for nn in range(niters):
        df_tmp = df.copy()
        df_tmp["A"] += np.random.normal(loc=0, scale=df_tmp["A_err"])
        df_tmp["B"] += np.random.normal(loc=0, scale=df_tmp["B_err"])
        # A_tmp = A 
        # B_tmp = B
        # A_tmp += np.random.normal(loc=0, scale=A_err)
        # B_tmp += np.random.normal(loc=0, scale=B_err)
        # c_list.append(A_tmp + B_tmp)
        c_list.append(foo(df))
    c = np.nanmean(c_list)
    c_err = np.nanstd(c_list)
    return c, c_err

def dt_test(N):
    N = int(N)
    df = pd.DataFrame({"A": np.random.normal(loc=10, scale=1.0, size=N),
                       "A_err": np.abs(np.random.normal(loc=0, scale=2.0, size=N)),
                       "B": np.random.normal(loc=10, scale=1.0, size=N),
                       "B_err": np.abs(np.random.normal(loc=0, scale=2.0, size=N))})

    # # Time operation: df.apply()
    # t = time()
    # df["C"] = df.apply(lambda x: myfun(x.A, x.A_err, x.B, x.B_err), axis=1)
    # dt = time() - t
    # print(f"N = {N}: Total time = {int(np.floor((dt) / 3600)):d}:{int(np.floor((dt) / 60)):d}:{np.mod(dt, 60):02.5f}" +\
    #       f" (Average {dt / N:.5f}s per iteration)")

    # Time operation: using arrays
    t = time()
    C, C_err = myfun(df)
    df["C"] = C
    df["C_err"] = C_err
    dt = time() - t
    print(f"N = {N}: Total time = {int(np.floor((dt) / 3600)):d}:{int(np.floor((dt) / 60)):d}:{np.mod(dt, 60):02.5f}" +\
          f" (Average {dt / N:.5f}s per iteration)")

dt_test(1e1)
dt_test(1e2)
dt_test(1e3)
dt_test(1e4)
dt_test(1e5)
# dt_test(1e6)

# Try with arrays instead 
