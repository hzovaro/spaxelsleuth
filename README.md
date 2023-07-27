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

### Config file 

Various important settings and variables required by `spaxelsleuth` are specified in a configuration file in JSON format. Default configurations are stored in `config.json` in the root `spaxelsleuth` directory. 

There is one top-level entry in `config.json` for each data source (e.g., `sami`, `s7` and `lzifu`). Each of these stores information such as default data cube sizes (`N_x`, `N_y`), spaxel sizes (`as_per_px`) and centre coordinates (`x0_px`, `y0_px`), as well as paths to the necessary input data products (e.g., data products and and raw data cubes for SAMI) and an output path (`output_path`) which is where outputs are saved.

The default values in this file can be easily overridden by the user by creating a custom configuration file. The file can be stored anywhere, but must be in the same JSON format, where you only need to enter key-value pairs for settings you'd like to update. For example, to change where `spaxelsleuth` looks for the input data products, you can create a file named `/home/.my_custom_config.json` with the contents
```
{
    "sami": {
        "output_path": "/some/path/spaxelsleuth_outputs/",
        "input_path": "/some/path/sami_data_products/",
        "data_cube_path": "/some/path/sami_data_cubes/",
        "sdss_im_path": "/some/path/sdss_images/"
    }
}
```
To override the default spaxelsleuth configuration settings, simply use the following lines at the start of your script or notebook: 
```
from spaxelsleuth import load_user_config
load_user_config("/home/.my_custom_config.json")
```
The settings themselves can be accessed in the form of a `dict` using 
```
from spaxelsleuth.config import settings
# e.g.
input_path = settings["sami"]["input_path"]
```

### SAMI data

In order to use `spaxelsleuth`, both the SAMI DR3 **data cubes** and **data products** must be downloaded from the (DataCentral Bulk Download page)[https://datacentral.org.au/services/download/].

The following data products for each galaxy are required. Note that if you are only interested in analysing e.g. the sector-binned data, then you only need to download the corresponding data. However, the unbinned data cubes are required in all cases. You can also download only a subset of the full sample, if you are only interested in certain galaxies.

* The original, unbinned **data cubes**:
    * SAMI DR3 SAMI Flux cube: blue
    * SAMI DR3 SAMI Flux cube: red

* The unbinned ("default") **data products**:
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

* For the binned **data products**:
    * SAMI DR3 SAMI <Sectors-binned>/<Adaptively-binned> flux cube: blue
    * SAMI DR3 SAMI <Sectors-binned>/<Adaptively-binned> flux cube: red
    * SAMI DR3 SAMI 1-component line emission map from <sectors-binned>/<adaptively-binned> cube: [OII] (3726Å+3729Å)
    * SAMI DR3 SAMI 1-component line emission map from <sectors-binned>/<adaptively-binned> cube: Hα
    * SAMI DR3 SAMI 1-component line emission map from <sectors-binned>/<adaptively-binned> cube: Hβ
    * SAMI DR3 SAMI 1-component line emission map from <sectors-binned>/<adaptively-binned> cube: [OIII] (5007Å)
    * SAMI DR3 SAMI 1-component line emission map from <sectors-binned>/<adaptively-binned> cube: [OI] (6300Å)
    * SAMI DR3 SAMI 1-component line emission map from <sectors-binned>/<adaptively-binned> cube: [NII] (6583Å)
    * SAMI DR3 SAMI 1-component line emission map from <sectors-binned>/<adaptively-binned> cube: [SII] (6716Å)
    * SAMI DR3 SAMI 1-component line emission map from <sectors-binned>/<adaptively-binned> cube: [SII] (6731Å)
    * SAMI DR3 SAMI Recommended-component line emission map from <sectors-binned>/<adaptively-binned> cube: [OII] (3726Å+3729Å)
    * SAMI DR3 SAMI Recommended-component line emission map from <sectors-binned>/<adaptively-binned> cube: Hα
    * SAMI DR3 SAMI Recommended-component line emission map from <sectors-binned>/<adaptively-binned> cube: Hβ
    * SAMI DR3 SAMI Recommended-component line emission map from <sectors-binned>/<adaptively-binned> cube: [OIII] (5007Å)
    * SAMI DR3 SAMI Recommended-component line emission map from <sectors-binned>/<adaptively-binned> cube: [OI] (6300Å)
    * SAMI DR3 SAMI Recommended-component line emission map from <sectors-binned>/<adaptively-binned> cube: [NII] (6583Å)
    * SAMI DR3 SAMI Recommended-component line emission map from <sectors-binned>/<adaptively-binned> cube: [SII] (6716Å)
    * SAMI DR3 SAMI Recommended-component line emission map from <sectors-binned>/<adaptively-binned> cube: [SII] (6731Å)
    * SAMI DR3 SAMI 1-component ionised gas velocity map from <sectors-binned>/<adaptively-binned> cube
    * SAMI DR3 SAMI 1-component ionised gas velocity dispersion map from <sectors-binned>/<adaptively-binned> cube
    * SAMI DR3 SAMI Recommended-component ionised gas velocity map from <sectors-binned>/<adaptively-binned> cube
    * SAMI DR3 SAMI Recommended-component ionised gas velocity dispersion map from <sectors-binned>/<adaptively-binned> cube
    * SAMI DR3 SAMI Extinction correction map from 1-component <sectors-binned>/<adaptively-binned> Hα/Hβ flux ratio
    * SAMI DR3 SAMI Star formation rate map from 1-component <sectors-binned>/<adaptively-binned> Hα flux
    * SAMI DR3 SAMI Star formation rate surface density from 1-component <sectors-binned>/<adaptively-binned> Hα flux
    * SAMI DR3 SAMI Extinction correction map from recommended-component <sectors-binned>/<adaptively-binned> Hα/Hβ flux ratio
    * SAMI DR3 SAMI Star formation rate map from recommended-component <sectors-binned>/<adaptively-binned> Hα flux
    * SAMI DR3 SAMI Star formation rate surface density from recommended-component <sectors-binned>/<adaptively-binned> Hα flux
    * SAMI DR3 SAMI Stellar velocity map (two moment) from <sectors-binned>/<adaptively-binned> cube
    * SAMI DR3 SAMI Stellar velocity dispersion map (two moment) from <sectors-binned>/<adaptively-binned> cube

The **data products** have the following naming convention:

* `Halpha_<bin_type>_<ncomponents>-comp.fits`
* `Hbeta_<bin_type>_<ncomponents>-comp.fits`
* `NII6583_<bin_type>_<ncomponents>-comp.fits`
* `OI6300_<bin_type>_<ncomponents>-comp.fits`
* `OII3728_<bin_type>_<ncomponents>-comp.fits`
* `OIII5007_<bin_type>_<ncomponents>-comp.fits`
* `SII6716_<bin_type>_<ncomponents>-comp.fits`
* `SII6731_<bin_type>_<ncomponents>-comp.fits`
* `gas-vdisp_<bin_type>_<ncomponents>-comp.fits`
* `gas-velocity_<bin_type>_<ncomponents>-comp.fits`
* `stellar-velocity-dispersion_<bin_type>_two-moment.fits`
* `stellar-velocity_<bin_type>_two-moment.fits`
* `extinct-corr_<bin_type>_<ncomponents>-comp.fits`
* `sfr-dens_<bin_type>_<ncomponents>-comp.fits`
* `sfr_<bin_type>_<ncomponents>-comp.fits`

These files must be stored as follows:

`<settings["sami"]["input_path"]>/ifs/<gal>/<gal>_<quantity>_<bin type>_<number of components>-comp.fits`

and the data cubes must be stored as as follows (note that `settings["sami"]["input_path"]` and `settings["sami"]["data_cube_path"]` can be the same path; they are only different in my setup due to storage limitations):

`<settings["sami"]["data_cube_path"]>/ifs/<gal>/<gal>_A_cube_<blue/red>.fits.gz`

This is essentially the default file structure when data products are downloaded from DataCentral and unzipped:

`sami/dr3/ifs/<gal>/<gal>_<quantity>_<bin type>_<number of components>-comp.fits`

So if you simply set `settings["sami"]["input_path"]` to `sami/dr3/` `spaxelsleuth` should be able to find the data.

**SAMI galaxy metadata**, such as galaxy redshifts and stellar masses, is also required. For your convenience, this data is provided in data/, but may be downloaded in CSV format from the (DataCentral Schema)[https://datacentral.org.au/services/schema/] where they can be found under the following tabs:

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
2. Create SAMI spaxel DataFrames by running `loaddata.sami.make_sami_df()`. See the docstrings within for details on how to process data with different emission line fitting and/or binning schemes, how to apply different S/N cuts, etc. **Note that you will need approximately 8 GB to store the DataFrame containing all SAMI galaxies.**
3. Run the assertion tests in tests/test_assertions.py to check that nothing has gone awry. 

A Jupyter notebook showing you how to get up and running with `spaxelsleuth` is provided in examples/Examples.ipynb. This notebook shows you how to create the necessary DataFrames and how to create plots. *I highly recommend you start here*.

## Citing this work
Please contact me at `henry.zovaro@anu.edu.au` if you decide to use `spaxelsleuth` for your science or are interested in adding new features!


