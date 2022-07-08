# `spaxelsleuth`
A `python` package for analysing data from large IFU surveys, such as SAMI and S7, on a spaxel-by-spaxel basis.

`spaxelsleuth` is currently under development - download at your own risk! Stay tuned for stable releases.

## Prerequisites 
----

**Packages**:
* numpy, scipy, pandas
* matplotlib
* astropy
* scipy
* tqdm 


## Prerequisites
----

### Environment variables 

The following environment variables must be defined:

* `SAMI_DIR` - points to the location of the SAMI data products. Output DataFrames are also stored here.
* `SAMI_DATACUBE_DIR` - points to the location of the SAMI data cubes. `SAMI_DATACUBE_DIR` can be the same as SAMI_DIR (I just have them differently in my setup due to storage space limitations).
* `SAMI_FIG_DIR` - where figures are saved.

### SAMI data

**SAMI data products** must be downloaded from the (DataCentral Bulk Download page)[https://datacentral.org.au/services/download/] and stored as follows: 

    `SAMI_DIR/ifs/<gal>/<gal>_<quantity>_<bin type>_<number of components>-comp.fits`

This is essentially the default file structure when data products are downloaded from DataCentral and unzipped:

    `sami/dr3/ifs/<gal>/<gal>_<quantity>_<bin type>_<number of components>-comp.fits`

The following data products for each galaxy are required:

* `Halpha_{bin_type}_{ncomponents}-comp.fits`
* `Hbeta_{bin_type}_{ncomponents}-comp.fits`
* `NII6583_{bin_type}_{ncomponents}-comp.fits`
* `OI6300_{bin_type}_{ncomponents}-comp.fits`
* `OII3728_{bin_type}_{ncomponents}-comp.fits`
* `OIII5007_{bin_type}_{ncomponents}-comp.fits`
* `SII6716_{bin_type}_{ncomponents}-comp.fits`
* `SII6731_{bin_type}_{ncomponents}-comp.fits`
* `gas-vdisp_{bin_type}_{ncomponents}-comp.fits`
* `gas-velocity_{bin_type}_{ncomponents}-comp.fits`
* `stellar-velocity-dispersion_{bin_type}_two-moment.fits`
* `stellar-velocity_{bin_type}_two-moment.fits`
* `extinct-corr_{bin_type}_{ncomponents}-comp.fits`
* `sfr-dens_{bin_type}_{ncomponents}-comp.fits`
* `sfr_{bin_type}_{ncomponents}-comp.fits`

SAMI data cubes must also be downloaded from DataCentral and stored as follows: 

    `SAMI_DATACUBE_DIR/ifs/<gal>/<gal>_A_cube_<blue/red>.fits.gz`

A number of **SAMI galaxy metadata** must also be downloaded. These tables be downloaded in CSV format from the (DataCentral Schema)[https://datacentral.org.au/services/schema/] where they can be found under the following tabs:

* SAMI 
    * Data Release 3
        * Catalogues 
            * SAMI 
                * CubeObs:
                    - `sami_CubeObs`
                * Other
                    - `InputCatGAMADR3`
                    - `InputCatClustersDR3`
                    - `InputCatFiller`
                    - `VisualMorphologyDR3`

 and stored at SAMI_DIR/ using the naming convention

    * `sami_InputCatGAMADR3.csv`
    * `sami_InputCatClustersDR3.csv`
    * `sami_InputCatFiller.csv`
    * `sami_VisualMorphologyDR3.csv`
    * `sami_CubeObs.csv`.

## Usage 
----
1. Create the metadata DataFrame by running `loaddata/make_sami_metadata_df.py`.
2. Create SAMI spaxel DataFrames by running `loaddata/make_df_sami.py`. See the docstrings within for details on how to process data with different emission line fitting and/or binning schemes, how to apply different S/N cuts, etc.
3. The test files in `tests/`.
4. Play around with the example scripts in `examples/`. Enjoy!

## Citing this work
----
Please contact me at `henry.zovaro@anu.edu.au` if you decide to use `spaxelsleuth` for your science! 


