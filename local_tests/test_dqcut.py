import pandas as pd
import numpy as np

from spaxelsleuth.utils.dqcut import compute_SN

# Make some test data 
df_in = pd.DataFrame()
df_in["HALPHA (total)"] = [10.0, 5.0, 1.0, np.nan]
df_in["HALPHA error (total)"] = [1.0, 5.0, 0.0, 3.0]
eline_list = ["HALPHA"]
ncomponents_max = 1

def test_compute_SN():
    """Test dqcut.compute_SN()."""
    df_out = compute_SN(df_in.copy(), ncomponents_max=ncomponents_max, eline_list=eline_list)
    assert df_out.iloc[0]["HALPHA S/N (total)"] == 10.0
    assert df_out.iloc[1]["HALPHA S/N (total)"] == 1.0
    assert df_out.iloc[2]["HALPHA S/N (total)"] == np.inf
    assert np.isnan(df_out.iloc[3]["HALPHA S/N (total)"])
    return 