# `spaxelsleuth`

spaxelsleuth is a `python` package for analysing data from large IFU surveys, such as SAMI, on a spaxel-by-spaxel basis. 

spaxelsleuth takes as input data cubes and other data products, e.g. emission line fits and stellar kinematics measurements, and outputs a `pandas` DataFrame where each row represents a spatial pixel (*spaxel*) in a galaxy, and the columns represent various measurements. In addition to storing data products from large surveys in an easy-to-use format, spaxelsleuth computes extra quantities, such as extinctions, emission line ratios, and metallicities, so you don't have to! spaxelsleuth also comes with a set of useful plotting and data visualisation tools to help you analyse your data. 

spaxelsleuth was originally developed to work with data from the [Sydney-AAO Multi-object Integral field spectrograph (SAMI) survey](http://sami-survey.org/) but contains extensions to work with fitting output from LZIFU and S7, and can be adapted to ingest data from other surveys.

# Installation

After cloning into the repository, cd into it and install spaxelsleuth using 
```sh
pip install .
```

# Help 
The [wiki pages](https://github.com/hzovaro/spaxelsleuth/wiki) provide detailed information about what spaxelsleuth does, what inputs it requires, and what it produces. 
For detailed instructions on how to use spaxelsleuth, please see the [example Jupyter notebooks](https://github.com/hzovaro/spaxelsleuth/tree/main/examples).

Please raise a Github issue (preferred) or send me an email at `henry.zovaro@anu.edu.au` if you encounter any problems or have questions that aren't covered in the wiki. 

# Citing this work
Please cite Zovaro et al. (2023) in any works making use of `spaxelsleuth`:
```bibtex
@ARTICLE{10.1093/mnras/stad3747,
       author = {{Zovaro}, Henry R.~M. and {Mendel}, J. Trevor and {Groves}, Brent and {Kewley}, Lisa J. and {Colless}, Matthew and {Ristea}, Andrei and {Cortese}, Luca and {Oh}, Sree and {D'Eugenio}, Francesco and {Croom}, Scott M. and {L{\'o}pez-S{\'a}nchez}, {\'A}ngel R. and {van de Sande}, Jesse and {Brough}, Sarah and {Medling}, Anne M. and {Bland-Hawthorn}, Joss and {Bryant}, Julia J.},
        title = "{The SAMI Galaxy Survey: {\ensuremath{\Sigma}}$_{SFR}$ drives the presence of complex emission line profiles in star-forming galaxies}",
      journal = {\mnras},
     keywords = {galaxies: ISM, ISM: jets and outflows, ISM: kinematics and dynamics, Astrophysics - Astrophysics of Galaxies},
         year = 2023,
        month = dec,
          doi = {10.1093/mnras/stad3747},
archivePrefix = {arXiv},
       eprint = {2312.03659},
 primaryClass = {astro-ph.GA},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2023MNRAS.tmp.3596Z},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```
Feel free to contact me at `henry.zovaro@anu.edu.au` if you decide to use spaxelsleuth for your science or are interested in adding new features!
