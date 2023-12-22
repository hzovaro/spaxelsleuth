#! /usr/bin/env python
"""
Set up for spaxelsleuth.
"""
from setuptools import setup, find_packages
import os

setup(
    name="spaxelsleuth",
    packages=find_packages(),
    version="1.0.0",
    # Add anything directly imported here
    install_requires=[
        "astropy",
        "matplotlib",
        "scipy",
        "extinction",
        "matplotlib",
        "numpy",
        "pandas",
        "tables",
    ],
    python_requires=">=3.10",
    package_data={
        "spaxelsleuth": ["**/*.csv", "**/*.json"]
    }
)