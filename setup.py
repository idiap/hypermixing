# Copyright © <2023> Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Florian MAI <florian.ren.mai@googlemail.com>
# SPDX-FileContributor: Arnaud Pannatier <arnaud.pannatier@idiap.ch>
# SPDX-FileContributor: Fabio Fehr <fabio.fehr@idiap.ch>
# SPDX-FileContributor: Juan Pablo Zuluaga <juan-pablo.zuluaga@idiap.ch>
#
# SPDX-License-Identifier: MIT
"""Setup script for the Hypermixing package."""
from setuptools import setup

setup(
    name="hypermixing",
    version="0.1.1",
    description="A PyTorch implementation of the HyperMixer ",
    author="Florian Mai",
    author_email="florian.ren.mai@googlemail.com",
    packages=["hypermixing"],
    install_requires=[
        # List of your package dependencies
        # For example: 'numpy>=1.18.1'
        "torch>=1.8.0",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
    ],
)
