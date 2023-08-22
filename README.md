# `spaxelsleuth`
A `python` package for analysing data from large IFU surveys, such as SAMI, on a spaxel-by-spaxel basis.

`spaxelsleuth` was originally developed to work with data from the [Sydney-AAO Multi-object Integral field spectrograph (SAMI) survey](http://sami-survey.org/) but contains extensions to work with fitting output from LZIFU and from other surveys. 

# Using `spaxelsleuth` with SAMI

## Quick start guide 

A Jupyter notebook showing you how to get up and running with `spaxelsleuth` using the SAMI data set is provided in examples/Examples - SAMI.ipynb. This notebook shows you how to create the necessary DataFrames and how to create plots. *I highly recommend you start here*.

The most basic way to use `spaxelsleuth` with SAMI data is as follows:

1. Download the SAMI data following the steps below.

2. Create a config file and save it as `/path/to/config/file/.myconfig`:

```
{
    "sami": {
        "output_path": "/some/path/spaxelsleuth_outputs/",
        "input_path": "/some/path/sami_data_products/",
        "data_cube_path": "/some/path/sami_data_cubes/",
    }
}
```

3. Load the config file:

```
from spaxelsleuth import load_user_config
load_user_config("/path/to/config/file/.myconfig.json")
```

3. Create the metadata DataFrame, which containts redshifts, stellar masses, and other "global" galaxy properties for each SAMI galaxy:

```
from spaxelsleuth.loaddata.sami import make_sami_metadata_df
import os
make_sami_metadata_df(nthreads=os.cpu_count())
```

4. Create the SAMI spaxel DataFrame:
```
from spaxelsleuth.loaddata.sami import make_sami_df
make_sami_df(bin_type="default", 
             ncomponents="recom", 
             eline_SNR_min=5, 
             nthreads_max=N, 
             metallicity_diagnostics=["R23_KK04"])
```

See the docstrings within for details on how to process data with different emission line fitting and/or binning schemes, how to apply different S/N cuts, etc. **Note that you will need approximately 8 GB to store the DataFrame containing all SAMI galaxies.**


5. After running `make_sami_df()`, load the DataFrames:

```
from spaxelsleuth.loaddata.sami import load_sami_df, load_sami_metadata_df
df_metadata = load_sami_metadata_df()
df = load_sami_df(ncomponents="recom",
                  bin_type="default",
                  correct_extinction=True,
                  eline_SNR_min=5)
```

6. Do your analysis - e.g., make some plots:

```
# Histograms showing the distribution in velocity dispersion
import matplotlib.pyplot as plt
from astropy.visualization import hist
fig, ax = plt.subplots(nrows=1, ncols=1)
for nn in range(1, 4):
    hist(df[f"sigma_gas (component {nn})"].values, bins="scott", ax=ax, range=(0, 500), density=True, histtype="step", label=f"Component {nn}")
ax.legend()
ax.set_xlabel(r"$\sigma_{\rm gas}$")
ax.set_ylabel(r"$N$ (normalised)")


# Plot a 2D histogram showing the distribution of SAMI spaxels in the WHAN diagram
from spaxelsleuth.plotting.plottools import plot_empty_BPT_diagram, plot_BPT_lines
from spaxelsleuth.plotting.plotgalaxies import plot2dhistcontours, plot2dscatter
plot2dhistcontours(df=df,
              col_x=f"log N2 (total)",
              col_y=f"log HALPHA EW (total)",
              col_z="count", log_z=True,
              plot_colorbar=True)
```

## Prerequisites 

Required packages:

* astropy
* extinction
* matplotlib
* pandas
* pytables
* scipy
* tqdm

### Config file 

Various important settings and variables required by `spaxelsleuth` are specified in a configuration file in JSON format. Default configurations are stored in `config.json` in the root `spaxelsleuth` directory. 

There is one top-level entry in `config.json` for each data source (e.g., `sami`, `s7` and `lzifu`). Each of these stores paths to the necessary input data products (e.g., data products (`input_path`) and and data cubes (`data_cube_path`)) and an output path (`output_path`) which is where output DataFrames are saved. For surveys such as SAMI where the data format is the same for each object, information such as default data cube sizes (`N_x`, `N_y`), spaxel sizes (`as_per_px`) and centre coordinates (`x0_px`, `y0_px`) are also specified in `settings`.

The default values in this file can be easily overridden by the user by creating a custom configuration file. The file can be stored anywhere, but must be in the same JSON format, where you only need to enter key-value pairs for settings you'd like to update. For example, to change where `spaxelsleuth` looks for the input data products, you can create a file named `/home/.my_custom_config.json` with the contents
```
{
    "sdss_im_path": "/some/path/sdss_images/",
    "sami": {
        "output_path": "/some/path/spaxelsleuth_outputs/",
        "input_path": "/some/path/sami_data_products/",
        "data_cube_path": "/some/path/sami_data_cubes/"
    },
    "lzifu": {
        "output_path": "/some/path/spaxelsleuth_outputs/",
        "input_path": "/some/path/lzifu_data_products/"
        "data_cube_path": "/some/path/lzifu_data_cubes/"
    },
    ...
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
input_path = settings["sami"]["input_path"]
```
etc.

## Downloading SAMI data

### SAMI Data Products 
In order to use `spaxelsleuth` with SAMI data, the SAMI [Data Release 3](https://ui.adsabs.harvard.edu/abs/2021MNRAS.505..991C/abstract) **data cubes** and **data products** must be downloaded from the [DataCentral Bulk Download page](https://datacentral.org.au/services/download/).

Here is the simplest way to download the data:
1. In **Source List**: leave blank to download the full SAMI sample, or enter the IDs of the subset of galaxies you would like to analyse.
2. In **Data Release(s)**: select SAMI Data Release 3.
3. Leave **Loose matching** unchecked.
4. In **Data products: Integral Field Spectra (IFS)**, select all by using `ctrl+a` or `cmd+a`.
5. Do not select anything in **Data products: Spectra**.
6. If you are not logged in to DataCentral, make sure to enter your email address in **Email**.

This will download *all* SAMI DR3 data products, including some that are not used (yet) by `spaxelsleuth`; however, this is much easier than manually selecting only the necessary data products. **Note that the full list of files for each galaxy can exceed 100 MB, so you will need a lot of disk space to download the full data set for all 3068 SAMI galaxies!**

When downloaded and unzipped, the data products for each galaxy will be organised into sub-folders with name 

`/sami/dr3/ifs/<gal>/`

and will have the following naming convention:
* Unbinnned blue/red data cubes: `<gal>_<A/B>_cube_<blue/red>.fits`
* Binned blue/red data cubes: `<gal>_<A/B>_<bin_type>_<blue/red>.fits`
* Emission line flux maps: `<gal>_<A/B>_<emission line>_<bin_type>_<ncomponents>-comp.fits`
* Gas velociy map: `<gal>_<A/B>_gas-velocity_<bin_type>_<ncomponents>-comp.fits`
* Gas velocity dispersion map: `<gal>_<A/B>_gas-vdisp_<bin_type>_<ncomponents>-comp.fits`
* Stellar velocity dispersion map: `<gal>_<A/B>_stellar-velocity-dispersion_<bin_type>_two-moment.fits`
* Stellar velocity map: `<gal>_<A/B>_stellar-velocity_<bin_type>_two-moment.fits`
* Halpha extinction correction factor map: `<gal>_<A/B>_extinct-corr_<bin_type>_<ncomponents>-comp.fits`
* SFR surface density map: `<gal>_<A/B>_sfr-dens_<bin_type>_<ncomponents>-comp.fits`
* SFR map: `<gal>_<A/B>_sfr_<bin_type>_<ncomponents>-comp.fits`

For simplicity, `spaxelsleuth` assumes this default file structure when it searches for the files. To point `spaxelsleuth` to the right location, simply set `settings["sami"]["input_path"]` in your `.config.json` file to the folder containing `ifs/`, i.e. `/path/to/datacentral/data/sami/dr3/`. Note that `settings["sami"]["input_path"]` and `settings["sami"]["data_cube_path"]` can be the same path. 

### SAMI metadata
SAMI galaxy "metadata", such as galaxy redshifts and stellar masses, is also required. For your convenience, this data is provided in data/, but may be downloaded in CSV format from the (DataCentral Schema)[https://datacentral.org.au/services/schema/] where they can be found under the following tabs:

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

# Using `spaxelsleuth` with LZIFU

`spaxelsleuth` works directly with data output by [LZIFU](https://github.com/hoiting/LZIFU). Simply modify your configuration file to point to your LZIFU data products as follows:
```
{
    "lzifu": {
        "output_path": "/some/path/spaxelsleuth_outputs/",
        "input_path": "/some/path/lzifu_data_products/",
    },
    ...
}
```

## Usage 

A Jupyter notebook showing you how to get up and running with `spaxelsleuth` using LZIFU data is provided in examples/Examples - LZIFU.ipynb. This notebook shows you how to create the necessary DataFrames and how to create plots. *I highly recommend you start here*. The most basic way to use `spaxelsleuth` with LZIFU data is as follows:

1. Create LZIFU spaxel DataFrames by running `loaddaata.lzifu.make_lzifu_df()`. See the docstrings within for details on how to process data with different emission line fitting and/or binning schemes, how to apply different S/N cuts, etc.
2. After running `make_lzifu_df()`, the DataFrame can be loaded using `loaddata.lzifu.load_lzifu_df()`. 



# Using `spaxelsleuth` with S7 data

`spaxelsleuth` also works with data from the [Siding Spring Southern Seyfert Spectroscopic Snapshot Survey (S7)](https://miocene.anu.edu.au/S7/). Simply modify your configuration file to point to your S7 data products as follows:
```
{
    "s7": {
        "output_path": "/some/path/spaxelsleuth_outputs/",
        "input_path": "/some/path/s7_data_products/",
        "data_cube_path": "/some/path/s7_data_cubes/",
    },
    ...
}
```
## Usage

A Jupyter notebook showing you how to get up and running with `spaxelsleuth` using S7 data is provided in examples/Examples - S7.ipynb. This notebook shows you how to create the necessary DataFrames and how to create plots. *I highly recommend you start here*. The most basic way to use `spaxelsleuth` with S7 data is as follows:

1. Create the S7 metadata DataFrame by running `loaddata.s7.make_s7_metadata_df()`. 
1. Create S7 spaxel DataFrames by running `loaddaata.s7.make_s7_df()`. See the docstrings within for details on how to process data with different emission line fitting and/or binning schemes, how to apply different S/N cuts, etc.
2. After running `make_s7_df()`, the DataFrame can be loaded using `loaddata.s7.load_s7_df()`. 


# Using `spaxelsleuth` with other data 

`spaxelsleuth` contains a number of useful tools and functions you can use to manipulate data from any IFU survey or observations, as long as the data is presented in the correct format. 

`spaxelsleuth` is based on `pandas` DataFrames, where each row represents a single bin or spaxel. Column names use the following conventions:

* Emission lines are represented in upper case, e.g. `HALPHA`, `HBETA`, `OIII5007`, etc. 
* Quantities corresponding to individual emission line components are denoted by 'component N' where N is counted from 1 in order of increasing velocity dispersion, and 'total' represents quantities based on emission line fluxes summed across all kinematic components. 
For example, the DataFrame produced by `spaxelsleuth` on the recommended-component fits to SAMI data contains  the following columns:
    * `HALPHA (component 1)` - Ha emission line flux in the narrowest component
    * `HALPHA (component 2)` - Ha emission line flux in the intermediate component
    * `HALPHA (component 3)` - Ha emission line flux in the broadest component
    * `HALPHA error (component 1)` - uncertainty in the Ha emission line flux in the narrowest component
    * `HALPHA EW (component 1)` - Ha equivalent width (EW) in the narrowest component 
    * `sigma_gas (component 1)` - ionised gas velocity dispersion in the narrowest component
    * `HALPHA (total)` - Ha emission line flux summed across all components 

For example, the `utils` submodule contains the following functions to calculate various quantities and perform actions on a supplied DataFrame:
* `dqcut.compute_SN()` - compute emission line S/N
* `dqcut.set_flags()`, `dqcut.apply_flags()` - set and apply data quality and S/N cuts 
* `extcorr.compute_A_V()`, `extcorr.apply_extinction_correction()` - use the Balmer decrement to compute the interstellar extinction and apply extinction correction to emission line fluxes
* `continuum.compute_EW()` - compute Ha equivalent widths 
* `continuum.compute_continuum_luminosity()`, `linefns.compute_eline_luminosity()` - compute continuum and emission line luminosities
* `linefns.ratio_fn()` - evaluate commonly used emission line ratios (e.g., [OIII]5007/Hbeta)
* `linefns.bpt_fn()` - determine the BPT classification in each spaxel
* `linefns.compute_SFR()` - compute the star formation rate from the Ha flux
* `linefns.compute_FWHM()` - compute emission line FWHMs from Gaussian velocity dispersions
* `metallicity.calculate_metallicity()` - compute gas-phase metallicities using a selection of strong-line diagnostics
* `misc.compute_gas_stellar_offsets()` - compute offsets between gas and stellar kinematics 
* `misc.compute_log_columns()` - compute log quantities 
* `misc.compute_component_offsets()` - compute offsets between kinematic components, e.g. difference in LOS velocity between the narrow and broad emission line components

To apply all of these functions to a DataFrame, you can simply use `add_columns()` from `spaxelsleuth.utils.addcolumns`. 

The only requirement is that the data is presented in the correct format. 

# Citing this work
Please contact me at `henry.zovaro@anu.edu.au` if you decide to use `spaxelsleuth` for your science or are interested in adding new features!
