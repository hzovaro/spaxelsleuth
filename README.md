# `spaxelsleuth`
A `python` package for analysing data from large IFU surveys, such as SAMI and S7, on a spaxel-by-spaxel basis.

`spaxelsleuth` is currently under development - download at your own risk! Stay tuned for stable releases.

## Prerequisites 

**Packages**:
* numpy, scipy, pandas
* matplotlib
* astropy
* scipy
* tqdm 


## Prerequisites

### Environment variables 

The following environment variables must be defined:

* `SAMI_DIR` - points to the location of the SAMI data products. Output DataFrames are also stored here.
* `SAMI_DATACUBE_DIR` - points to the location of the SAMI data cubes. `SAMI_DATACUBE_DIR` can be the same as SAMI_DIR (I just have them differently in my setup due to storage space limitations).

### SAMI data

In order to use `spaxelsleuth`, **SAMI data products** must be downloaded from the (DataCentral Bulk Download page)[https://datacentral.org.au/services/download/].

The following data products for each galaxy are required. Note that if you are only interested in analysing e.g. the sector-binned data, then you only need to download the corresponding data. However, the unbinned data cubes are required in all cases. You can also download only a subset of the full sample, if you are only interested in certain galaxies.

* The original, unbinned data cubes:
    * SAMI DR3 SAMI Flux cube: blue
    * SAMI DR3 SAMI Flux cube: red

* The unbinned ("default") data:
    * SAMI DR3 SAMI 1-component line emission map: [OII] (3726Å+3729Å)
    * SAMI DR3 SAMI 1-component line emission map: Hα
    * SAMI DR3 SAMI 1-component line emission map: Hβ
    * SAMI DR3 SAMI 1-component line emission map: [OIII] (5007Å)
    * SAMI DR3 SAMI 1-component line emission map: [OI] (6300Å)
    * SAMI DR3 SAMI 1-component line emission map: [NII] (6583Å)
    * SAMI DR3 SAMI 1-component line emission map: [SII] (6716Å)
    * SAMI DR3 SAMI 1-component line emission map: [SII] (6731Å)
    * SAMI DR3 SAMI Recommended-component line emission map: [OII] (3726Å+3729Å)
    * SAMI DR3 SAMI Recommended-component line emission map: Hα
    * SAMI DR3 SAMI Recommended-component line emission map: Hβ
    * SAMI DR3 SAMI Recommended-component line emission map: [OIII] (5007Å)
    * SAMI DR3 SAMI Recommended-component line emission map: [OI] (6300Å)
    * SAMI DR3 SAMI Recommended-component line emission map: [NII] (6583Å)
    * SAMI DR3 SAMI Recommended-component line emission map: [SII] (6716Å)
    * SAMI DR3 SAMI Recommended-component line emission map: [SII] (6731Å)
    * SAMI DR3 SAMI 1-component ionised gas velocity map
    * SAMI DR3 SAMI 1-component ionised gas velocity dispersion map
    * SAMI DR3 SAMI Recommended-component ionised gas velocity map
    * SAMI DR3 SAMI Recommended-component ionised gas velocity dispersion map
    * SAMI DR3 SAMI Extinction correction map from 1-component Hα/Hβ flux ratio
    * SAMI DR3 SAMI Star formation rate map from 1-component Hα flux
    * SAMI DR3 SAMI Star formation rate surface density from 1-component Hα flux
    * SAMI DR3 SAMI Extinction correction map from recommended-component Hα/Hβ flux ratio
    * SAMI DR3 SAMI Star formation rate map from recommended-component Hα flux
    * SAMI DR3 SAMI Star formation rate surface density from recommended-component Hα flux
    * SAMI DR3 SAMI Stellar velocity map (two moment) from default cube
    * SAMI DR3 SAMI Stellar velocity dispersion map (two moment) from default cube

* For the sector-binned data:
    * SAMI DR3 SAMI Sectors-binned flux cube: blue
    * SAMI DR3 SAMI Sectors-binned flux cube: red
    * SAMI DR3 SAMI 1-component line emission map from sectors-binned cube: [OII] (3726Å+3729Å)
    * SAMI DR3 SAMI 1-component line emission map from sectors-binned cube: Hα
    * SAMI DR3 SAMI 1-component line emission map from sectors-binned cube: Hβ
    * SAMI DR3 SAMI 1-component line emission map from sectors-binned cube: [OIII] (5007Å)
    * SAMI DR3 SAMI 1-component line emission map from sectors-binned cube: [OI] (6300Å)
    * SAMI DR3 SAMI 1-component line emission map from sectors-binned cube: [NII] (6583Å)
    * SAMI DR3 SAMI 1-component line emission map from sectors-binned cube: [SII] (6716Å)
    * SAMI DR3 SAMI 1-component line emission map from sectors-binned cube: [SII] (6731Å)
    * SAMI DR3 SAMI Recommended-component line emission map from sectors-binned cube: [OII] (3726Å+3729Å)
    * SAMI DR3 SAMI Recommended-component line emission map from sectors-binned cube: Hα
    * SAMI DR3 SAMI Recommended-component line emission map from sectors-binned cube: Hβ
    * SAMI DR3 SAMI Recommended-component line emission map from sectors-binned cube: [OIII] (5007Å)
    * SAMI DR3 SAMI Recommended-component line emission map from sectors-binned cube: [OI] (6300Å)
    * SAMI DR3 SAMI Recommended-component line emission map from sectors-binned cube: [NII] (6583Å)
    * SAMI DR3 SAMI Recommended-component line emission map from sectors-binned cube: [SII] (6716Å)
    * SAMI DR3 SAMI Recommended-component line emission map from sectors-binned cube: [SII] (6731Å)
    * SAMI DR3 SAMI 1-component ionised gas velocity map from sectors-binned cube
    * SAMI DR3 SAMI 1-component ionised gas velocity dispersion map from sectors-binned cube
    * SAMI DR3 SAMI Recommended-component ionised gas velocity map from sectors-binned cube
    * SAMI DR3 SAMI Recommended-component ionised gas velocity dispersion map from sectors-binned cube
    * SAMI DR3 SAMI Extinction correction map from 1-component sectors-binned Hα/Hβ flux ratio
    * SAMI DR3 SAMI Star formation rate map from 1-component sectors-binned Hα flux
    * SAMI DR3 SAMI Star formation rate surface density from 1-component sectors-binned Hα flux
    * SAMI DR3 SAMI Extinction correction map from recommended-component sectors-binned Hα/Hβ flux ratio
    * SAMI DR3 SAMI Star formation rate map from recommended-component sectors-binned Hα flux
    * SAMI DR3 SAMI Star formation rate surface density from recommended-component sectors-binned Hα flux
    * SAMI DR3 SAMI Stellar velocity map (two moment) from sectors-binned cube
    * SAMI DR3 SAMI Stellar velocity dispersion map (two moment) from sectors-binned cube

* For the adaptively-binned data:
    * SAMI DR3 SAMI Adaptively-binned flux cube: blue
    * SAMI DR3 SAMI Adaptively-binned flux cube: red
    * SAMI DR3 SAMI 1-component line emission map from adaptively-binned cube: [OII] (3726Å+3729Å)
    * SAMI DR3 SAMI 1-component line emission map from adaptively-binned cube: Hα
    * SAMI DR3 SAMI 1-component line emission map from adaptively-binned cube: Hβ
    * SAMI DR3 SAMI 1-component line emission map from adaptively-binned cube: [OIII] (5007Å)
    * SAMI DR3 SAMI 1-component line emission map from adaptively-binned cube: [OI] (6300Å)
    * SAMI DR3 SAMI 1-component line emission map from adaptively-binned cube: [NII] (6583Å)
    * SAMI DR3 SAMI 1-component line emission map from adaptively-binned cube: [SII] (6716Å)
    * SAMI DR3 SAMI 1-component line emission map from adaptively-binned cube: [SII] (6731Å)
    * SAMI DR3 SAMI Recommended-component line emission map from adaptively-binned cube: [OII] (3726Å+3729Å)
    * SAMI DR3 SAMI Recommended-component line emission map from adaptively-binned cube: Hα
    * SAMI DR3 SAMI Recommended-component line emission map from adaptively-binned cube: Hβ
    * SAMI DR3 SAMI Recommended-component line emission map from adaptively-binned cube: [OIII] (5007Å)
    * SAMI DR3 SAMI Recommended-component line emission map from adaptively-binned cube: [OI] (6300Å)
    * SAMI DR3 SAMI Recommended-component line emission map from adaptively-binned cube: [NII] (6583Å)
    * SAMI DR3 SAMI Recommended-component line emission map from adaptively-binned cube: [SII] (6716Å)
    * SAMI DR3 SAMI Recommended-component line emission map from adaptively-binned cube: [SII] (6731Å)
    * SAMI DR3 SAMI 1-component ionised gas velocity map from adaptively-binned cube
    * SAMI DR3 SAMI 1-component ionised gas velocity dispersion map from adaptively-binned cube
    * SAMI DR3 SAMI Recommended-component ionised gas velocity map from adaptively-binned cube
    * SAMI DR3 SAMI Recommended-component ionised gas velocity dispersion map from adaptively-binned cube
    * SAMI DR3 SAMI Extinction correction map from 1-component adaptive-binned Hα/Hβ flux ratio
    * SAMI DR3 SAMI Star formation rate map from 1-component adaptive-binned Hα flux
    * SAMI DR3 SAMI Star formation rate surface density from 1-component adaptive-binned Hα flux
    * SAMI DR3 SAMI Extinction correction map from recommended-component adaptive-binned Hα/Hβ flux ratio
    * SAMI DR3 SAMI Star formation rate map from recommended-component adaptive-binned Hα flux
    * SAMI DR3 SAMI Star formation rate surface density from recommended-component adaptive-binned Hα flux
    * SAMI DR3 SAMI Stellar velocity map (two moment) from adaptively-binned cube
    * SAMI DR3 SAMI Stellar velocity dispersion map (two moment) from adaptively-binned cube

Which, when downloaded, have the following naming convention:

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

These files must be stored as follows:

`SAMI_DIR/ifs/<gal>/<gal>_<quantity>_<bin type>_<number of components>-comp.fits`

and the data cubes must be stored at:

`SAMI_DATACUBE_DIR/ifs/<gal>/<gal>_A_cube_<blue/red>.fits.gz`


This is essentially the default file structure when data products are downloaded from DataCentral and unzipped:

`sami/dr3/ifs/<gal>/<gal>_<quantity>_<bin type>_<number of components>-comp.fits`

**SAMI galaxy metadata**, such as galaxy redshifts and stellar masses, is also required. This data is provided in data/, but may be downloaded in CSV format from the (DataCentral Schema)[https://datacentral.org.au/services/schema/] where they can be found under the following tabs:

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

 and stored in data/ using the naming convention

* `sami_InputCatGAMADR3.csv`
* `sami_InputCatClustersDR3.csv`
* `sami_InputCatFiller.csv`
* `sami_VisualMorphologyDR3.csv`
* `sami_CubeObs.csv`.

## Usage 
The most basic way to use `spaxelsleuth` is as follows:

1. Create the metadata DataFrame, which containts redshifts, stellar masses, and other "global" galaxy properties for each SAMI galaxy by running `loaddata.sami.make_sami_metadata_df()`.
2. Create SAMI spaxel DataFrames by running `loaddata.sami.make_sami_df()`. See the docstrings within for details on how to process data with different emission line fitting and/or binning schemes, how to apply different S/N cuts, etc.
3. Run the assertion tests in tests/test_assertions.py to check that nothing has gone awry. 

A Jupyter notebook showing you how to get up and running with `spaxelsleuth` is provided in examples/Examples.ipynb. This notebook shows you how to create the necessary DataFrames and how to create plots. *I highly recommend you start here*.


## Citing this work
Please contact me at `henry.zovaro@anu.edu.au` if you decide to use `spaxelsleuth` for your science or are interested in adding new features!


