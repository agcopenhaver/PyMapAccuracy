PyMapAccuracy Documentation
===========================

.. image:: https://img.shields.io/badge/python-3.8+-blue.svg
   :target: https://www.python.org/downloads/
   :alt: Python Version

.. image:: https://img.shields.io/badge/License-MIT-yellow.svg
   :target: https://opensource.org/licenses/MIT
   :alt: License: MIT

PyMapAccuracy is a Python package for rigorous thematic map accuracy assessment and area estimation under stratified random sampling. It provides statistically sound implementations of accuracy estimators for remote sensing and land cover mapping applications.

Key Features
------------

* **Comprehensive Accuracy Metrics**: Overall accuracy, user's accuracy, producer's accuracy, and area proportions
* **Statistical Rigor**: Standard error estimates and confidence intervals for all metrics
* **Flexible Sampling Designs**: Support for stratified random sampling where strata differ from map classes
* **Two Estimator Methods**:

  * **Stehman (2014)**: When sampling strata differ from map classes (e.g., geographic regions)
  * **Olofsson et al. (2014)**: When map classes serve as sampling strata

* **Robust Input Validation**: Comprehensive error checking with informative messages
* **Production Ready**: Extensive test suite ensuring reliability and accuracy

Quick Start
-----------

Installation
~~~~~~~~~~~~

.. code-block:: bash

   pip install pymapaccuracy

Basic Usage
~~~~~~~~~~~

Stehman Estimator (Strata ≠ Map Classes)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from pymapaccuracy import stehman2014

   # Example: Administrative regions as strata, land cover as classes
   strata = ['region_A', 'region_A', 'region_B', 'region_B', 'region_C']
   reference = ['forest', 'grassland', 'water', 'forest', 'urban']
   map_pred = ['forest', 'forest', 'water', 'grassland', 'urban']

   # Area of each administrative region
   stratum_areas = {
       'region_A': 10000,
       'region_B': 8000,
       'region_C': 12000
   }

   results = stehman2014(strata, reference, map_pred, stratum_areas)
   print(f"Overall Accuracy: {results['OA']:.3f} ± {results['SEoa']:.3f}")

Olofsson Estimator (Map Classes = Strata)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from pymapaccuracy import olofsson

   # Example: Map classes serve as sampling strata
   reference = ['forest', 'forest', 'water', 'grassland', 'urban']
   map_pred = ['forest', 'grassland', 'water', 'grassland', 'urban']

   # Area of each map class stratum
   map_areas = {
       'forest': 15000,
       'grassland': 8000,
       'water': 3000,
       'urban': 4000
   }

   results = olofsson(reference, map_pred, map_areas)
   print(f"Overall Accuracy: {results['OA']:.3f} ± {results['SEoa']:.3f}")

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   quickstart
   api
   examples
   theory
   r_comparison

API Reference
-------------

.. autosummary::
   :toctree: _autosummary
   :recursive:

   pymapaccuracy

Statistical Methods
-------------------

This package implements the estimators described in:

* **Stehman, S.V. (2014)**: Estimating area and map accuracy for stratified random sampling when the strata are different from the map classes. *International Journal of Remote Sensing*, 35(13), 4923-4939. DOI: 10.1080/01431161.2014.930207

* **Olofsson, P., Foody, G.M., Stehman, S.V., & Woodcock, C.E. (2013)**: Making better use of accuracy data in land change studies: Estimating accuracy and area and quantifying uncertainty using stratified estimation. *Remote Sensing of Environment*, 129, 122-131. https://doi.org/10.1016/j.rse.2012.10.031

* **Olofsson, P., Foody, G.M., Herold, M., Stehman, S.V., Woodcock, C.E., & Wulder, M.A. (2014)**: Good practices for estimating area and assessing accuracy of land change. *Remote Sensing of Environment*, 148, 42-57. https://doi.org/10.1016/j.rse.2014.02.015

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
