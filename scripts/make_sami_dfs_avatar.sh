#!/bin/bash
#PBS -N spaxelsleuth
#PBS -q smallmem
#PBS -M henry.zovaro@anu.edu.au
#PBS -m abe
#PBS -l ncpus=56
#PBS -k oe
source /home/u5708159/.bashrc
cd /home/u5708159/python/Modules/spaxelsleuth/scripts
python make_sami_dfs.py 56
