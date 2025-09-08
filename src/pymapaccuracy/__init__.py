"""
PyMapAccuracy: Thematic Map Accuracy Assessment Package

A Python package for thematic map accuracy assessment and area estimation 
under stratified random sampling.
"""

from .estimators import olofsson, stehman2014

__version__ = "0.1.0"
__author__ = "Andrew Copenhaver"
__email__ = "acopenhaver@verra.org"

__all__ = ["olofsson", "stehman2014"]