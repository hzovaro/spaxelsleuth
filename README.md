# `spaxelsleuth`

`spaxelsleuth` is a `python` package for analysing data from large IFU surveys, such as SAMI, on a spaxel-by-spaxel basis. 

`spaxelsleuth` takes as input data cubes and other data products, e.g. emission line fits and stellar kinematics measurements, and outputs a `pandas` DataFrame where each row represents a spatial pixel (*spaxel*) in a galaxy, and the columns represent various measurements. In addition to storing data products from large surveys in an easy-to-use format, `spaxelsleuth` computes extra quantities, such as extinctions, emission line ratios, and metallicities, so you don't have to! `spaxelsleuth` also comes with a set of useful plotting and data visualisation tools to help you analyse your data. 

`spaxelsleuth` was originally developed to work with data from the [Sydney-AAO Multi-object Integral field spectrograph (SAMI) survey](http://sami-survey.org/) but contains extensions to work with fitting output from LZIFU and S7, and can be adapted to ingest data from other surveys.

# Prerequisites 

## Required packages 

* astropy
* extinction
* ipympl (*only required for example notebooks*)
* matplotlib
* pandas
* pytables
* scipy
* tqdm

## Config file 

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

## Logging

By default, information messages and warnings are printed to the terminal. To save the output to a file instead, simply do the following:
```
from spaxelsleuth import configure_logger
configure_logger(logfile_name="output.log")
```
The minimum level of messages logged can be controlled using the `level` parameter:
```
configure_logger(level="DEBUG")    # print ALL messages
configure_logger(level="INFO")     # print information and warning messages (default - recommended)
configure_logger(level="WARNING")  # print only warnings 
```

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
ax.set_xlabel(r"\sigma_{\rm gas}")
ax.set_ylabel(r"N (normalised)")


# Plot a 2D histogram showing the distribution of SAMI spaxels in the WHAN diagram
from spaxelsleuth.plotting.plottools import plot_empty_BPT_diagram, plot_BPT_lines
from spaxelsleuth.plotting.plotgalaxies import plot2dhistcontours, plot2dscatter
plot2dhistcontours(df=df,
              col_x=f"log N2 (total)",
              col_y=f"log HALPHA EW (total)",
              col_z="count", log_z=True,
              plot_colorbar=True)
```



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

#### Sample cuts

Using the DR3 `CubeObs` table as described in Croom et al. (2021), galaxies with bad sky subtraction residuals (as indicated by `WARNSKYR` and `WARNSKYB`), those with flux calibration issues (`WARNFCAL` and `WARNFCBR`), and those containing multiple objects in the SAMI field-of-view (`WARNMULT`) are removed, leaving 2997 unique galaxies. 

#### Additional information

* **Distances** are computed from the redshifts assuming a flat ΛCDM cosmology with H0 = 70 km/s/Mpc, ΩM = 0.3 and ΩΛ = 0.7. For the SAMI sample, the flow-corrected redshifts are used to compute distances when available. 
* **Morphologies** are taken from the `VisualMorphologyDR3` catalogue. For simplicity, the `?`, `No agreement` and `Unknown` categories are all merged into a single category labelled `Unknown`.
* **MGE effective radius measurements** are taken from the `MGEPhotomUnregDR3` catalogue. For galaxies for which measurements from both VST and SDSS photometry are available, only the VST measurements are kept.

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
3. Use `loaddata.lzifu.add_metadata()` to merge the DataFrame with another containing metadata (e.g. stellar masses, position angles, etc.).


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

`spaxelsleuth` contains a number of useful tools and functions you can use to manipulate data from any IFU survey or observations, as long as the data is presented in the correct format. See the "column descriptions" section below for details on the naming conventions for various quantities. 

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

To apply all of these functions to a DataFrame in the correct order, you can simply use `utils.addcolumns.add_columns()`. 

# Citing this work
Please contact me at `henry.zovaro@anu.edu.au` if you decide to use `spaxelsleuth` for your science or are interested in adding new features!

# Details

This section provides some details as to how various quantities are computed in the DataFrames output by `spaxelsleuth`. Please see the docstrings for the relevant functions for further details.

## S/N and data quality cuts 
*Functions*: `utils.dqcut.set_flags()`, `utils.dqcut.apply_flags()`

To ensure high-quality kinematics and fluxes of individual components within individual spaxels, the following data quality and S/N cuts are made in each row and are denoted by a series of flags which can be accessed as columns in the DataFrame. In the following, `<line>` represents an emission line, and `<component>` corresponds to the kinematic component: "component 1/2/3/etc." denotes individual kinematic emission line components, and "total" corresponds to total fluxes etc. derived by summing together the values across all kinematic components in each spaxel. 

* **Flux S/N cuts (`Low flux S/N flag - <line> (<component>)`)**: flags emission line fluxes where the S/N < 5 (defined as the flux divided by the formal uncertainty). This applies both to total emission line fluxes and those within individual components. Discarded fluxes are replaced with NaNs.
    * *Values that get masked out*: all quantities based on the emission line flux in this component - e.g. metallicities, SFRs, etc.. If the emission line is is Hɑ, then *all* quantities based on this component - including kinematics - are also masked out, which is based on the assumption that Hɑ is generally the strongest line. 

* **Missing flux cuts (`Missing flux flag - <line> (<component>)`)**: flags emission lines with "missing" (i.e. NaN) fluxes in which the error on the flux is finite.
    * *Values that get masked out*: all quantities based on the emission line flux in this component - e.g. metallicities, SFRs, etc.. If the emission line is is Hɑ, then *all* quantities based on this component - including kinematics - are also masked out.

* **Amplitude-to-noise (A/N) cuts (`Low amplitude flag - <line> (<component>)`)**: flags emission line fluxes where the A/N < 3 (which is defined as the Gaussian amplitude of the component divided by the RMS noise in the rest-frame wavelength range 6500 Å - 6540 Å).  In spaxels where *all* components have low amplitudes, the flag `Low amplitude flag - <line> (total)` will be set to `True`. Discarded fluxes are replaced with NaNs.
    * *Values that get masked out*: all quantities based on the emission line flux in this component - e.g. metallicities, SFRs, etc.. If the emission line is is Hɑ, then *all* quantities based on this component - including kinematics - are also masked out.
    * *Note:* the column `HALPHA A/N (measured)` is unrelated to this cut: this column measures the 

* **Flux fraction cut (`Low flux fraction flag - {eline} (<component>)`)**: flags spaxels where the flux ratio of the broad (component 2) to narrow (component 1) fluxes is less than 0.05. *Note: not used by default with SAMI data.*
    * *Values that get masked out*: all quantities based on the emission line flux in this component - e.g. metallicities, SFRs, etc.. If the emission line is is Hɑ, then *all* quantities based on this component - including kinematics - are also masked out.

* **Beam smearing cut (`Beam smearing flag (<component>)`)**: flags emission line components likely to be affected by beam smearing, i.e. with `sigma_gas` (component n) < `v_grad` (component n) where `v_grad` is computed using eqn. 1 of Zhou et al. (2017). *Note: not used by default with SAMI data.*
    * *Values that get masked out*: gas kinematics (i.e., `sigma_gas` and `v_gas`) in the corresponding components and all quantities derived using these values.

* **Velocity dispersion S/N cut (`Low sigma_gas S/N flag (<component>)`)**: flags components where the gas velocity dispersion S/N (as defined by Federrath et al. 2017) is less than 3, which generally applies to individual emission line components that are very close to the instrumental resolution. 
    * *Values that get masked out*: gas velocity dispersions in the corresponding components and all quantities derived using these values.

* **Stellar kinematics cuts**: flags spaxels with unreliable stellar kinematics, defined as having `sigma_*` < 35 km/s, `v_*` error > 30 km/s, `sigma_*` error" >= `sigma_*` * 0.1 + 25, or where the stellar kinematics contain NaNs. *Note: this should generally not be used with data sets other than SAMI as the threshold values have been taken from Croom et al. (2018)*. 
    * *Values that get masked out*: stellar kinematics (i.e., `sigma_*` and `v_*`) in the corresponding component and all quantities derived using these values.

### "Missing components"

In datasets where emission lines have been fitted with multiple Gaussian components, it is important to consider how to handle spaxels where one or more individual components fail to meet S/N or data quality criteria. We refer to these as "**missing components**".

For example, a 2-component spaxel in which the 2nd component is a low-S/N component may still have high S/N *total* fluxes, in which case things like e.g. line ratios for the *total* flux are most likely still reliable. In this case, you would probably want to mask out the fluxes and kinematics of the low-S/N component, but keep the total fluxes.

In `spaxelsleuth`, flagging of spaxels with missing components is handled by the function `utils.dqcut.apply_flags()`. By default, (i.e. when `base_missing_flux_components_on_HALPHA = True`), we define a "missing component" as one in which both the HALPHA flux and velocity dispersion have been masked out for any reason. If `base_missing_flux_components_on_HALPHA = False`, it is based only on the velocity dispersion. 

Note that while `spaxelsleuth` will flag affected spaxels (denoted by the `Missing components flag` column), it will not automatically mask out anything out based on this criterion, allowing the user to control how spaxels with missing components are handled based on their specific use case. 

`utils.dqcut.apply_flags()` adds an additional `Number of components` column to the DataFrame (not to be confused with the `Number of components (original)` column, which records the number of kinematic components in each spaxel that were originally fitted to the data). `Number of components` records the number of *reliable* components ONLY IF they are in the right order. For example, consider a spaxel that originally has 2 components. Say that component 1 has a low S/N in the gas velocity dispersion, so sigma_gas (component 1) is masked out, but HALPHA (component 1) is not, and that component 2 passes all DQ and S/N cuts. In this case, the spaxel will NOT be recorded as having `Number of components = 1 or 2` because component 1 fails to meet the DQ and S/N criteria. It will therefore have an undefined `Number of components` and will be set to NaN. 

Spaxels that still retain their original number of components after making all DQ and S/N cuts can be selected as follows:

    df_good_quality_components = df[~df["Missing components flag"]]

## Extinction
*Functions*: `utils.extcorr.compute_A_V()`, `utils.extcorr.apply_extinction_correction()`

Total extinction in the V-band, A_V, is computed using the Balmer decrement, i.e. the Hɑ/Hβ ratio, assuming an intrinsic value of 2.86 (corresponding to a standard nebular temperature and density of 10,000 K and 100 cm^-3). The default reddening curve used is that of Fitzpatrick & Massa (2007). Extinctions are only computed where the S/N in both Hɑ and Hβ exceeds 5, and in rows where the measured Balmer decrement is *less* than the assumed intrinsic value, A_V is assumed to be 0. 

The A_V is only computed based on the total emission line fluxes, i.e. summed across all 3 kinematic components, because the fluxes in individual components are only supplied for Hɑ. However, the extinction correction is applied to all 3 Hɑ components.

The extinction correction corresponding to the computed A_V is applied to all emission line fluxes. Note that equivalent widths are *not* corrected for extinction, because the stellar continuum extinction is assumed to be unknown. 

## Metallicities 
*Functions*: `utils.metallicity.calculate_metallicity()`

Gas-phase metallicities (recorded as log(O/H) + 12) and corresopnding errors are computed using a variety of different strong-line metallicity diagnostics. For some metallicity diagnostics, namely those from Kewley (2019) and the R23 diagnostic from Kobulnicky & Kewley (2004), a self-consistent values for log(O/H) + 12 and log(U) are computed using the method of Kewley & Dopita (2002). For SAMI data, metallicities are computed based on the total emission line fluxes, i.e. summed across all 3 kinematic components.

Errors are computed using a Monte-Carlo approach, in which random noise is added to emission line fluxes (sampled from Gaussian distributions with a standard deviation equal to the flux uncertainty) before computing the metallicity and/or ionisation parameter. This process is repeated 1000 times and the final metallicities/ionisation parameters and errors are taken as the mean and standard deviation of the resulting distribution of metallicity/ionisation parameter values.

## Emission line ratios and BPT classifications
*Functions*: `utils.linefns.bpt_fn()`

Spaxels are spectrally classified using the standard optical diagnostic diagrams of Baldwin, Philips and Terlevich (1981) and Veilleux & Osterbrock (1987), which plot the [O III]5007/Hβ ratio (O3) against the [N II]6583/Hɑ (N2), [S II]6717,31/Hɑ (S2), and [O I]6300/Hɑ (O1) ratios. Classifications are based on the total emission line fluxes in each spaxel.

Only the O3 vs. N2 and O3 vs. S2 diagrams are used in spectral classification due to the relative weakness of the [O I] emission line. To ensure reliable classification, only spaxels with a flux S/N of at least 5 in all of Hβ, Hɑ, [O III]5007, [N II]6583, and [S II]6716,31 lines are classified.
Note that, in cases where multiple emission line components are present that arise from different excitation mechanisms (e.g., a star-forming component plus a shocked component), the classification is naturally weighted towards the component with the highest luminosity.

Each spaxel is assigned one of the following spectral classification, based on the total emission line fluxes in each spaxel:
* *Star-forming (SF)*: lies below the Kauffmann et al. (2003) line in the N2 diagram, and below the Kewley et al. (2001) "extreme starburst" line in the S2 diagram.
* *Composite*: lies above the Kauffmann et al. (2003) line in the N2 diagram but below the the Kewley et al. (2001) "extreme starburst" line in both the N2 and S2 diagrams.
* *LINER*: lies above the Kewley et al. (2001) "extreme starburst" line in both the N2 and S2 diagrams but below the Kewley et al. (2006) LINER/Seyfert line in the S2 diagram.
* *Seyfert*: lies above the Kewley et al. (2001) "extreme starburst" line in both the N2 and S2 diagrams but above the Kewley et al. (2006) LINER/Seyfert line in the S2 diagram.
* *Ambiguous*: inconsistent classifications between the O3 vs. N2 and O3 vs. S2 diagrams; e.g., composite-like in the N2 diagram, but LINER-like in the S2 diagram.
* *Not classified*: low S/N in or missing at least one of Hɑ, Hβ, [N II], [S II] or [O III] fluxes.


## Star formation rates (SFRs)
*Functions*: `utils.linefns.compute_SFR()`

SFRs are computed from the Hɑ luminosity using the calibration of Calzetti (2013), which assumes a stellar mass range 0.1 – 100 M_sun, star formation timescape >= 6 Myr, a temperature 10000 K, and an electron density 100 cm^-3. *Note: since SFR maps are provided with SAMI DR3, `spaxelsleuth` uses these values rather than computing new ones for the SAMI data set.*


# Column descriptions

`spaxelsleuth` is based on `pandas` DataFrames, where each row represents a single bin or spaxel. Detailed descriptions of the contents of each column are given below. 

In the table, \<component\> represents which Gaussian emission line component the quantity corresponds to, e.g. "component 1" or "component 2", or it may be "total", in which case the measurement corresponds to the quantity summed over all Gaussian components. \<line\> represents an emission line name.


| Column | Description |
| ------ | ----------- |
| ID                                                                               | Galaxy ID |
| **Flags** |  |
| Good?                                                                            | Whether galaxy is in final sample  |
| Bad class                                                                        | Flag for bad or problem objects - 0, 5 and 8 are "good" |
| **Coordinates** |  |
| RA (IFU) (J2000)                                                                 | IFU right ascension in degrees (J2000) |
| RA (J2000)                                                                       | Object right ascension in degrees (J2000) |
| Dec (IFU) (J2000)                                                                | IFU declination in degrees (J2000) |
| Dec (J2000)                                                                      | Object declination in degrees (J2000) |
| z (spectroscopic)                                                                | Spectroscopic redshift  |
| z (flow-corrected)                                                               | Flow-corrected redshift |
| z                                                                                | Redshift (flow-corrected, if available, otherwise spectroscopic) |
| D_A (Mpc)                                                                        | Angular diameter distance  |
| D_L (Mpc)                                                                        | Luminosity distance  |
| kpc per arcsec                                                                   | Angular scale  |
| **Morphologies** |  |
| Morphology (numeric)                                                             | Galaxy morphology (numerical representation) |
| Morphology                                                                       | Galaxy morphology (string representation) |
| **Masses, magnitudes, surface brightnesses, colours** |  |
| log M_*                                                                          | Logarithm of stellar mass |
| A_g                                                                              | g-band extinction |
| M_r                                                                              | Absolute r-band magnitude |
| g - i colour                                                                     | g - i colour  |
| mu_r at 1R_e                                                                     | r-band surface brightness at 1 effective radius (Kelvin et al. 2012) |
| mu_r at 2R_e                                                                     | r-band surface brightness at 2 effective radii (Kelvin et al. 2012) |
| mu_r within 1R_e                                                                 | Mean r-band surface brightness within 1 effective radius (Kelvin et al. 2012) |
| **Sizes, inclinations, etc. from Kelvin et al. (2012) fits** |  |
| e                                                                                | r-band ellipticity (Kelvin et al. 2012) |
| PA (degrees)                                                                     | r-band position angle (Kelvin et al. 2012) |
| R_e (arcsec)                                                                     | r-band major axis effective radius (Kelvin et al. 2012) |
| R_e (kpc)                                                                        | Effective radius in kpc (Kelvin et al. 2012) |
| i (degrees)                                                                      | Inclination (Kelvin et al. 2012) |
| **Multi-Gaussian Expansion (MGE) size measurements** |  |
| R_e (MGE) (arcsec)                                                               | Circularised effective radius (MGE fit)  |
| m_AB (MGE)                                                                       | AB magnitude within 1R_e (MGE fit) |
| PA (MGE) (degrees)                                                               | Position angle (MGE fit)  |
| e at 1R_e (MGE)                                                                  | Model isophotal ellipticity at one Re (MGE fit) |
| e, LW (MGE)                                                                      | Light-weighted ellipticity of the model (MGE fit) |
| R_e (MGE) (kpc)                                                                  | Effective radius in kpc (MGE fit) |
| **Environmental metrics** |  |
| Cluster member                                                                   | Flag indicating cluster membership (1=member, 0=non-member) |
| r/R_200                                                                          | Projected distance from cluster centre normalised by R200 |
| v/sigma_cluster                                                                  | Line-of-sight velocity relative to cluster redshift normalised by cluster velocity dispersion measured within R200 |
| **Proxies for stellar mass surface density/gravitational potential** |  |
| log(M/R_e) (MGE)                                                                 | Stellar mass divided by R_e (MGE fit) |
| log(M/R_e^2) (MGE)                                                               | Stellar mass divided by R_e^2 (MGE fit) |
| log(M/R_e)                                                                       | Stellar mass divided by R_e (Kelvin et al. 2012) |
| log(M/R_e^2)                                                                     | Stellar mass divided by R_e^2 (Kelvin et al. 2012) |
| **Spatially resolved quantities** |  |
| **Stellar kinematics** |  |
| sigma_*                                                                          | Stellar velocity dispersion (km/s) |
| sigma_* error                                                                    | 1-sigma uncertainty on stellar velocity dispersion (km/s) |
| v_*                                                                              | Stellar velocity (km/s) |
| v_* error                                                                        | 1-sigma uncertainty on stellar velocity (km/s) |
| **Continuum-derived properties** |  |
| HALPHA continuum                                                                 | Mean continuum level in the rest-frame wavelength range 6500 Å - 6540 Å (flux units) |
| HALPHA continuum std. dev.                                                       | Standard deviation in the continuum level in the rest-frame wavelength range 6500 Å - 6540 Å (flux units) |
| HALPHA continuum error                                                           | 1-sigma uncertainty on the mean continuum level in the rest-frame wavelength range 6500 Å - 6540 Å (flux units) |
| HALPHA continuum luminosity                                                      | Mean continuum luminosity measured in the rest-frame wavelength range 6500 Å - 6540 Å |
| HALPHA continuum luminosity error                                                | 1-sigma uncertainty on the mean continuum luminosity measured in the rest-frame wavelength range 6500 Å - 6540 Å |
| B-band continuum                                                                 | Mean continuum level in the rest-frame wavelength range 4000 Å - 5000 Å (flux units)         |
| B-band continuum std. dev.                                                       | Standard deviation in the continuum level in the rest-frame wavelength range 4000 Å - 5000 Å (flux units) |
| B-band continuum error                                                           | 1-sigmag uncertainty on the mean continuum level in the rest-frame wavelength range 4000 Å - 5000 Å (flux units) |
| D4000                                                                            | Dn4000 break strength using the definition of Balogh et al. (1999) |
| D4000 error                                                                      | 1-sigma uncertainty on the Dn4000 break strength using the definition of Balogh et al. (1999) |
| Median SNR (B, full field)                                                       | Median signal-to-noise ratio in the B data cube measured across all spaxels |
| Median SNR (R, full field)                                                       | Median signal-to-noise ratio in the R data cube measured across all spaxels |
| Median SNR (B, 1R_e)                                                             | Median signal-to-noise ratio in the B data cube measured within 1R_e (circularised MGE R_e) |
| Median SNR (R, 1R_e)                                                             | Median signal-to-noise ratio in the R data cube measured within 1R_e (circularised MGE R_e) |
| Median SNR (B, 1.5R_e)                                                           | Median signal-to-noise ratio in the R data cube measured within 1.5R_e (circularised MGE R_e) |
| Median SNR (R, 1.5R_e)                                                           | Median signal-to-noise ratio in the R data cube measured within 1.5R_e (circularised MGE R_e) |
| Median SNR (B, 2R_e)                                                             | Median signal-to-noise ratio in the B data cube measured within 2R_e (circularised MGE R_e) |
| Median SNR (R, 2R_e)                                                             | Median signal-to-noise ratio in the R data cube measured within 3R_e (circularised MGE R_e) |
| **Coordinates and geometry** |  |
| Galaxy centre x0_px (projected, arcsec)                                          | x-coordinate of galaxy centre in arcsec relative to bottom-left corner of image |
| Galaxy centre y0_px (projected, arcsec)                                          | y-coordinate of galaxy centre in arcsec relative to bottom-left corner of image |
| x (projected, arcsec)                                                            | Projected x-coordinate of spaxel (or bin centre) in arcsec relative to bottom-left corner of image                 |
| y (projected, arcsec)                                                            | Projected y-coordinate of spaxel (or bin centre) in arcsec relative to bottom-left corner of image                 |
| x (relative to galaxy centre, deprojected, arcsec)                               | De-projected x-coordinate of spaxel (or bin centre) in arcsec relative to galaxy centre |
| y (relative to galaxy centre, deprojected, arcsec)                               | De-projected y-coordinate of spaxel (or bin centre) in arcsec relative to galaxy centre |
| r (relative to galaxy centre, deprojected, arcsec)                               | De-projected radius of spaxel (or bin centre) in arcsec relative to galaxy centre |
| Bin number                                                                       | Spaxel or bin number |
| Bin size (pixels)                                                                | Spaxel or bin size (in pixels) |
| Bin size (square arcsec)                                                         | Spaxel or bin size (in square arcseconds) |
| Bin size (square kpc)                                                            | Spaxel or bin size (in square kpc) |
| r/R_e                                                                            | De-projected radius of spaxel (or bin centre) in units of R_e (Kelvin et al. 2012) relative to galaxy centre |
| **Emission line quantities** |  |
| Number of components (original)                                                  | Number of Gaussian emission line components identified by LZComp in this spaxel or bin |
| Number of components                                                             | "Final" number of components after accounting for data quality and S/N cuts   |
| \<line\> (\<component\>)                                                             | \<line\> flux (flux units) |
| \<line\> error (\<component\>)                                                       | 1-sigma uncertainty on \<line\> flux (flux units) |
| \<line\> S/N (\<component\>)                                                         | \<line\> S/N, defined as the flux divided by the 1-sigma uncertainty on the flux |
| \<line\> luminosity (\<component\>)                                                  | \<line\> luminosity (flux units) |
| \<line\> luminosity error (\<component\>)                                            | 1-sigma uncertainty on the \<line\> luminosity (flux units) |
| HALPHA EW (\<component\>)                                                          | Halpha equivalent width, defined as the Halpha flux divided by the mean continuum leven in the rest-frame wavelength range 6500 Å - 6540 Å |
| HALPHA EW error (\<component\>)                                                    | 1-sigma uncertainty on the Halpha equivalent width, defined as the Halpha flux divided by the mean continuum leven in the rest-frame wavelength range 6500 Å - 6540 Å |
| log HALPHA luminosity (\<component\>)                                              | Logarithm of the Halpha luminosity (log erg/s) |
| log HALPHA luminosity error (lower) (\<component\>)                                | Lower 1-sigma uncertainty on the logarithm of the Halpha luminosity (log erg/s  |
| log HALPHA luminosity error (upper) (\<component\>)                                | Upper 1-sigma uncertainty on the logarithm of the Halpha luminosity (log erg/s |
| log HALPHA EW (\<component\>)                                                      | Logarithm of the Halpha EW (log Å) |
| log HALPHA EW error (lower) (\<component\>)                                        | Lower 1-sigma uncertainty on logarithm of the Halpha EW (log Å) |
| log HALPHA EW error (upper) (\<component\>)                                        | Upper 1-sigma uncertainty on logarithm of the Halpha EW (log Å) |
| **Emission line kinematics** |  |
| v_gas (\<component\>)                                                              | LOS gas velocity (km/s) |
| v_gas error (\<component\>)                                                        | 1-sigma uncertainty on LOS gas velocity (km/s) |
| v_grad (\<component\>)                                                             | Velocity gradient (eqn. 1 of Zhou et al. 2017) |
| sigma_gas (\<component\>                                                           | LOS gas velocity dispersion corrected for instrumental resolution (km/s) |
| sigma_gas error (\<component\>                                                     | 1-sigma uncertainty on LOS gas velocity dispersion corrected for instrumental resolution (km/s) |
| log sigma_gas (\<component\>)                                                      | Logarithm of the gas velocity dispersion corrected for instrumental resolution (log km/s) |
| log sigma_gas error (lower) (\<component\>)                                        | Lower 1-sigma uncertainty on logarithm of the gas velocity dispersion corrected for instrumental resolution (log km/s) |
| log sigma_gas error (upper) (\<component\>)                                        | Upper 1-sigma uncertainty on logarithm of the gas velocity dispersion corrected for instrumental resolution (log km/s) |
| sigma_obs (\<component\>)                                                          | Observed velocity dispersion, un-corrected for instrumental resolution (km/s) |
| sigma_obs S/N (\<component\>)                                                      | Velocity dispersion S/N as defined by Federrath et al. (2017) |
| sigma_obs target S/N (\<component\>)                                               | Target velocity dispersion S/N as defined by Federrath et al. (2017) assuming sigma_gas_SNR_min = 3 |
| FWHM_gas (\<component\>)                                                           | Emission line component FWHM (corrected for instrumental resolution) |
| FWHM_gas error (\<component\>)                                                     | 1-sigma uncertainty on the emission line component FWHM (corrected for instrumental resolution) |
| **Star formation rates** |  |
| SFR (\<component\>)                                                                | SFR (Msun/yr/kpc^-2) |
| SFR error (\<component\>)                                                          | 1-sigma uncertainty on the SFR (Msun/yr/kpc^-2)     |
| log SFR (\<component\>)                                                            | Log SFR (\<component\>) |
| log SFR error (lower) (\<component\>)                                              | Lower 1-sigma uncertainty on log SFR (\<component\>) |
| log SFR error (upper) (\<component\>)                                              | Upper 1-sigma uncertainty on log SFR (\<component\>) |
| SFR surface density (\<component\>)                                                | SFR surface density (Msun/yr/kpc^-2) |
| SFR surface density error (\<component\>)                                          | 1-sigma uncertainty on the SFR surface density (Msun/yr/kpc^-2) |
| log SFR surface density (\<component\>)                                            | Log SFR surface density (\<component\>) |
| log SFR surface density error (lower) (\<component\>)                              | Lower 1-sigma uncertainty on log SFR surface density (\<component\>) |
| log SFR surface density error (upper) (\<component\>)                              | Upper 1-sigma uncertainty on log SFR surface density (\<component\>) |
|**Other emission line quantities** |  |
| HALPHA lambda_obs (\<component\>) (Å)                                              | Observed central wavelength of HALPHA in Angstroms |
| HALPHA sigma_gas (\<component\>) (Å)                                               | Gaussian sigma of HALPHA in Angstroms                                                   |
| HALPHA A (\<component\>)                                                           | Gaussian Amplitude of HALPHA in 'flux units' |
| HALPHA A/N (measured)                                                            | Halpha amplitude-to-noise measured directly from the spectrum. Measured as as ...  |
|**Extinction parameters** |  |
| HALPHA extinction correction (total)                                             | Halpha extinction correction factor (from SAMI DR3) |
| HALPHA extinction correction error (total)                                       | 1-sigma uncertainty on Halpha extinction correction factor (from SAMI DR3) |
| Balmer decrement (\<component\>)                                                   | Measured HALPHA/HBETA ratio |
| Balmer decrement error (\<component\>)                                             | 1-sigma uncertainty on the measured HALPHA/ HBETAratio |
| A_V (\<component\>)                                                                | Total extinction in the V-band computed using the Balmer decrement (mag) |
| A_V error (\<component\>)                                                          | 1-sigma uncertainty on the total extinction in the V-band computed using the Balmer decrement |
|**Emission line ratios** |  |
| N2O2 (\<component\>)                                                               | log NII6583 / OII3726+OII3729 ratio |
| N2S2 (\<component\>)                                                               | log NII6583 / SII6716+SII6731 ratio |
| O3N2 (\<component\>)                                                               | log (OIII5007/HBETA) / (NII6583/HALPHA) ratio |
| R23 (\<component\>)                                                                | log (OIII4959+OIII5007 + OII3726+OII3729) / HBETA ratio |
| O3O2 (\<component\>)                                                               | log OIII5007 / OII3726+OII3729 ratio |
| O1O3 (\<component\>)                                                               | log OI6300 / OIII5007 ratio |
| Dopita+2016 (\<component\>)                                                        | log (NII6583 / SII6716+SII6731) + 0.264 * np.log10(NII6583 / HALPHA) ratio |
| N2 (\<component\>)                                                                 | NII6583 / HALPHA ratio    |
| N2 error (\<component\>)                                                           | 1-sigma uncertainty on the NII6583 / HALPHA ratio |
| log N2 (\<component\>)                                                             | log NII6583 / HALPHA ratio |
| log N2 error (lower) (\<component\>)                                               | Lower 1-sigma uncertainty on log NII6583 / HALPHA ratio |
| log N2 error (upper) (\<component\>)                                               | Lower 1-sigma uncertainty on log NII6583 / HALPHA ratio |
| O1 (\<component\>)                                                                 | OI6300 / HALPHA ratio |
| O1 error (\<component\>)                                                           | 1-sigma uncertainty on the OI6300 / HALPHA ratio |
| log O1 (\<component\>)                                                             | log OI6300 / HALPHA ratio |
| log O1 error (lower) (\<component\>)                                               | Lower 1-sigma uncertainty on log OI6300 / HALPHA ratio |
| log O1 error (upper) (\<component\>)                                               | Lower 1-sigma uncertainty on log OI6300 / HALPHA ratio |
| S2 (\<component\>)                                                                 | SII6716+SII6731 / HALPHA ratio |
| S2 error (\<component\>)                                                           | 1-sigma uncertainty on the SII6716+SII6731 / HALPHA ratio |
| log S2 (\<component\>)                                                             | log SII6716+SII6731 / HALPHA ratio |
| log S2 error (lower) (\<component\>)                                               | Lower 1-sigma uncertainty on log SII6716+SII6731 / HALPHA ratio |
| log S2 error (upper) (\<component\>)                                               | Upper 1-sigma uncertainty on log SII6716+SII6731 / HALPHA ratio |
| O3 (\<component\>)                                                                 | OIII5007 / HBETA ratio |
| O3 error (\<component\>)                                                           | 1-sigma uncertainty on the OIII5007 / HBETA ratio |
| log O3 (\<component\>)                                                             | log OIII5007 / HBETA ratio |
| log O3 error (lower) (\<component\>)                                               | Lower 1-sigma uncertainty on log OIII5007 / HBETA ratio |
| log O3 error (upper) (\<component\>)                                               | Upper 1-sigma uncertainty on log OIII5007 / HBETA ratio |
| S2 ratio (\<component\>)                                                           | SII6716/SII6731 ratio |
| S2 ratio error (\<component\>)                                                     | 1-sigma uncertainty on the SII6716/SII6731 ratio |
| log S2 ratio (\<component\>)                                                       | log SII6716/SII6731 ratio |
| log S2 ratio error (lower) (\<component\>)                                         | Lower 1-sigma uncertainty on log SII6716/SII6731 ratio |
| log S2 ratio error (upper) (\<component\>)                                         | Upper 1-sigma uncertainty on log SII6716/SII6731 ratio |
| **Spectral categories** |  |
| BPT (numeric) (\<component\>)                                                      | BPT category (numerical representation)  |
| BPT (\<component\>)                                                                | BPT category (string representation)  |
| **Stellar/gas kinematic offsets** |  |
| v_gas - v_* (\<component\>)                                                        | Difference between stellar and gas LOS velocities |
| v_gas - v_* error (\<component\>)                                                  | 1-sigma uncertainty on the difference between stellar and gas LOS velocities |
| sigma_gas - sigma_* (\<component\>)                                                | Difference between stellar and gas LOS velocity dispersions |
| sigma_gas^2 - sigma_*^2 (\<component\>)                                            | Difference between the squares of the stellar and gas LOS velocity dispersions |
| sigma_gas/sigma_* (\<component\>)                                                  | Ratio of the stellar and gas LOS velocity dispersions |
| sigma_gas - sigma_* error (\<component\>)                                          | 1-sigma uncertainty on the difference between stellar and gas LOS velocity dispersions |
| sigma_gas^2 - sigma_*^2 error (\<component\>)                                      | 1-sigma uncertainty on the difference between the squares of the stellar and gas LOS velocity dispersions |
| sigma_gas/sigma_* error (\<component\>)                                            | 1-sigma uncertainty on the ratio of the stellar and gas LOS velocity dispersions |
| **Component offsets** |  |
| delta sigma_gas (A/B)                                                            | sigma_gas (component A) - sigma_gas (component B) |
| delta sigma_gas error (A/B)                                                      | 1-sigma uncertainty on sigma_gas (component A) - sigma_gas (component B) |
| delta v_gas (A/B)                                                                | v_gas (component A) - v_gas (component B) |
| delta v_gas error (A/B)                                                          | 1-sigma uncertainty on v_gas (component A) - v_gas (component B) |
| HALPHA EW ratio (A/B)                                                            | Ratio of EWs measured in components A and B |
| HALPHA EW ratio error (A/B)                                                      | 1-sigma uncertainty on ratio of EWs measured in components A and B |
| Delta HALPHA EW (A/B)                                                            | log(HALPHA EW (component A)) - log(HALPHA EW (component B)) |
| HALPHA EW/HALPHA EW (total) (\<component\>)                                        | HALPHA EW in \<component\> divided by the total EW in the spaxel |
| **Metallicities** |  |
| log(O/H) + 12 (\<diagnostic\>) (\<component\>)                                       | Gas-phase metallicity measured using \<diagnostic\> (\<component\>) |
| log(O/H) + 12 (\<diagnostic\>) error (lower) (\<component\>)                         | Lower 1-sigma uncertainty on the gas-phase metallicity measured using \<diagnostic\> (\<component\>), from MC simulations |
| log(O/H) + 12 (\<diagnostic\>) error (upper) (\<component\>)                         | Upper 1-sigma uncertainty on the gas-phase metallicity measured using \<diagnostic\> (\<component\>), from MC simulations |
| log(U) (\<diagnostic\>) (\<component\>)                                              | Ionisation parameter measured using \<diagnostic\> (\<component\>) |
| log(U) (\<diagnostic\>) error (lower) (\<component\>)                                | Lower 1-sigma uncertainty on the ionisation parameter measured using \<diagnostic\> (\<component\>) |
| log(U) (\<diagnostic\>) error (upper) (\<component\>)                                | Upper 1-sigma uncertainty on the ionisation parameter measured using \<diagnostic\> (\<component\>) |
| **Data quality and S/N settings** |  |
| eline_SNR_min                                                                    | Minimum emission line S/N adopted  |
| sigma_gas_SNR_min                                                                | Minimum gas velocity dispersion S/N adopted |
| line_flux_SNR_cut                                                                | True if S/N cuts have been applied to emission line fluxes  |
| missing_fluxes_cut                                                               | True if missing flux cuts have been applied to emission line fluxes  |
| line_amplitude_SNR_cut                                                           | True if line amplitude cuts have been applied to emission line fluxes  |
| flux_fraction_cut                                                                | True if flux fraction cuts have been applied to emission line fluxes  |
| vgrad_cut                                                                        | True if beam smearing cuts have been applied to gas velocity dispersion measurements  |
| sigma_gas_SNR_cut                                                                | True if S/N cuts have been applied to gas velocity dispersion measurements  |
| stekin_cut                                                                       | True if stellar kinematics data quality and S/N cuts have been applied |
| Extinction correction applied                                                    | True if extinction correction has been applied to emission line fluxes  |
| **Data quality and S/N flags** |  |
| Beam smearing flag (\<component\>)                                                 | True if sigma_gas < 2 * v_grad |
| Low sigma_gas S/N flag (\<component\>)                                             | True if sigma_obs S/N < sigma_obs target S/N |
| Bad stellar kinematics                                                           | True if stellar kinematics do not meet the minimum data quality and S/N requirements given in Croom et al. (2021) (p18) |
| Low flux S/N flag - \<line\> (\<component\>)                                         | True if \<line\> S/N < eline_SNR_min |
| Missing flux flag - \<line\> (\<component\>)                                         | True if \<line\> flux is NaN but the corresponding error is finite |
| Low flux fraction flag - \<line\> (\<component\>)                                    | True if \<line\> amplitude (\<component\>) < 0.05 * \<line\> amplitude (component 1) |
| Low amplitude flag - \<line\> (\<component\>)                                        | True if \<line\> (\<component\>) < 3 * RMS noise in the rest-frame wavelength range  |
| Missing components flag (\<component\>)                                            | True if any kinematic components in this spaxel fail to meet minimum data quality and S/N requirements |