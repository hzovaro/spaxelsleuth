# Make DataFrames 
python make_sami_test_dfs.py

# Compare with old verions
python compare_sami_dfs.py sami_dr3_metadata.hd5
python compare_sami_dfs.py sami_dr3_aperture_snrs.hd5
python compare_sami_dfs.py sami_default_recom-comp_extcorr_minSNR=5_minANR=3_DEBUG.hd5
python compare_sami_dfs.py sami_adaptive_recom-comp_extcorr_minSNR=5_minANR=3_DEBUG.hd5
python compare_sami_dfs.py sami_sectors_recom-comp_extcorr_minSNR=5_minANR=3_DEBUG.hd5
python compare_sami_dfs.py sami_default_1-comp_extcorr_minSNR=5_minANR=3_DEBUG.hd5
python compare_sami_dfs.py sami_adaptive_1-comp_extcorr_minSNR=5_minANR=3_DEBUG.hd5
python compare_sami_dfs.py sami_sectors_1-comp_extcorr_minSNR=5_minANR=3_DEBUG.hd5
python compare_sami_dfs.py sami_default_recom-comp_minSNR=5_minANR=3_DEBUG.hd5
python compare_sami_dfs.py sami_adaptive_recom-comp_minSNR=5_minANR=3_DEBUG.hd5
python compare_sami_dfs.py sami_sectors_recom-comp_minSNR=5_minANR=3_DEBUG.hd5
python compare_sami_dfs.py sami_default_1-comp_minSNR=5_minANR=3_DEBUG.hd5
python compare_sami_dfs.py sami_adaptive_1-comp_minSNR=5_minANR=3_DEBUG.hd5
python compare_sami_dfs.py sami_sectors_1-comp_minSNR=5_minANR=3_DEBUG.hd5

# Run assertion tests 
python test_assertions_sami.py recom default 5
python test_assertions_sami.py 1 default 5
python test_assertions_sami.py recom adaptive 5
python test_assertions_sami.py 1 adaptive 5
python test_assertions_sami.py recom sectors 5
python test_assertions_sami.py 1 sectors 5