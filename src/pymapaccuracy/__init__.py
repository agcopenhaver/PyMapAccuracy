"""
PyMapAccuracy: Thematic map accuracy assessment with stratified sampling.

This package provides statistically rigorous estimators for thematic map accuracy
assessment under stratified random sampling designs, implementing methods from:

- Stehman (2014): For sampling strata different from map classes
- Olofsson et al. (2014): For map classes as sampling strata

Both estimators provide unbiased accuracy metrics, area estimates, standard errors,
and confidence intervals appropriate for stratified sampling designs.
"""

__version__ = "0.1.0"
__author__ = "Andrew Copenhaver"

from .estimators import stehman2014, olofsson

__all__ = ["stehman2014", "olofsson"]