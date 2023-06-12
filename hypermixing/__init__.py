# Copyright © <2023> Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Florian MAI <florian.ren.mai@googlemail.com>
# SPDX-FileContributor: Arnaud Pannatier <arnaud.pannatier@idiap.ch>
# SPDX-FileContributor: Fabio Fehr <fabio.fehr@idiap.ch>
# SPDX-FileContributor: Juan Pablo Zuluaga <juan-pablo.zuluaga@idiap.ch>
#
# SPDX-License-Identifier: MIT
"""Init file of the hypermixing module.

This module mixes information from different tokens via HyperMixing.
It can be viewed as a linear-time drop-in replacement for (self-)attention.
source: https://arxiv.org/abs/2203.03691

Authors
 * Florian Mai 2023
 * Arnaud Pannatier 2023
 * Juan Pablo Zuluaga 2023
 * Fabio Fehr 2023
"""
from .hypermixing import HyperMixing
