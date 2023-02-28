# Make DataFrames 
python make_sami_test_dfs.py

# Run assertion tests 
python test_assertions.py recom default 5
python test_assertions.py 1 default 5
python test_assertions.py recom adaptive 5
python test_assertions.py 1 adaptive 5
python test_assertions.py recom sectors 5
python test_assertions.py 1 sectors 5