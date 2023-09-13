#! /usr/bin/env python
"""
Set up for spaxelsleuth.
"""
from setuptools import setup
import os

def get_requirements():
    """
    Read the requirements from a file
    """
    requirements = []
    if os.path.exists('requirements.txt'):
        with open('requirements.txt') as req:
            for line in req:
                # skip commented lines
                if not line.startswith('#'):
                    requirements.append(line.strip())
    return requirements

setup(
    name="spaxelsleuth",
    packages=["spaxelsleuth"],
    version="0.9.0",
    # install_requires=get_requirements(),
    # James: all I need is for people to make some kind of env w/ python > 3.10, and then install via pip install .
    # Don't need to faff around w/ conda environments.
    # Add anything directly imported here
    install_requires=[
        "astropy",
        "matplotlib",
        # "pathlib",
        "scipy",
        # "setuptools",
        # "time",
        # "typing",
        # "urllib",
        # "copy",
        # "datetime",
        "extinction",
        # "json",
        # "logging",
        "matplotlib",
        # "multiprocessing",
        "numpy",
        "pandas",
        # "warnings"
    ],
    python_requires=">=3.10",
)