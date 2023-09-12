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
    install_requires=get_requirements(),
    python_requires=">=3.10",
)