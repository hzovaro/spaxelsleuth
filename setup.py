#! /usr/bin/env python
"""
Set up for spaxelsleuth.
"""
from setuptools import setup
import os

setup(
    name="spaxelsleuth",
    packages=["spaxelsleuth"],
    version="0.9.0",
    # James: all I need is for people to make some kind of env w/ python > 3.10, and then install via pip install .
    # Don't need to faff around w/ conda environments.
    # Add anything directly imported here
    install_requires=[
        "astropy",
        "matplotlib",
        "scipy",
        "extinction",
        "matplotlib",
        "numpy",
        "pandas",
    ],
    python_requires=">=3.10",
)