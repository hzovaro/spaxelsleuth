# Imports
import sys

from spaxelsleuth.loaddata.sami import make_sami_df

###########################################################################
# Options
ncomponents, bin_type, eline_SNR_min = [sys.argv[1], sys.argv[2], int(sys.argv[3])]

###########################################################################
# Create the DataFrame
###########################################################################
make_sami_df(ncomponents=ncomponents,
             bin_type=bin_type,
             eline_SNR_min=eline_SNR_min, 
             debug=True)