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
Please contact me at `henry.zovaro@anu.edu.au` if you decide to use spaxelsleuth for your science or are interested in adding new features!
