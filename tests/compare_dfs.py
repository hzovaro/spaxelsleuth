# Compare the old and new DataFrames to monitor for any changes 
if __name__ == "__main__":
    import sys 
    from pathlib import Path

    import pandas as pd

    fname = sys.argv[1]
    fname_new = Path("/priv/meggs3/u5708159/SAMI/sami_dr3") / fname
    fname_old = Path("/priv/meggs3/u5708159/SAMI/sami_dr3/tests/old") / fname 

    df_new = pd.read_hdf(fname_new)
    df_old = pd.read_hdf(fname_old)

    # Compare DataFrames
    if df_old.equals(df_new):
        print("Old and new DataFrames match!")
        sys.exit(0)
    else:
        print("ERROR: old and new DataFrames do not match!")
        sys.exit(1)