# PyMapAccuracy

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

PyMapAccuracy is a Python package for thematic map accuracy assessment and area estimation under stratified random sampling. It provides statistically sound implementations of accuracy estimators for remote sensing and land cover mapping applications. PyMapAccuracy is inspired by and based on the R package [**mapaccuracy**](https://cran.r-project.org/web/packages/mapaccuracy/index.html) developed by Hugo Costa.

##  Key Features

- **Comprehensive Accuracy Metrics**: Overall accuracy, user's accuracy, producer's accuracy, and area proportions
- **Statistical Rigor**: Standard error estimates and confidence intervals for all metrics
- **Flexible Sampling Designs**: Support for stratified random sampling where strata differ from map classes
- **Two Estimator Methods**:
  - **Stehman (2014)**: When sampling strata differ from map classes
  - **Olofsson et al. (2014)**: When map classes serve as sampling strata
- **Robust Input Validation**: Comprehensive error checking with informative messages

## Statistical Methods

This package implements the estimators described in the following references:

- Olofsson, P.; Foody, G. M.; Herold, M.; Stehman, S. V.; Woodcock, C. E.; Wulder, M. A. (2014). *Good practices for estimating area and assessing accuracy of land change*. Remote Sensing of Environment, **148**, 42–57. [doi:10.1016/j.rse.2014.02.015](https://doi.org/10.1016/j.rse.2014.02.015)

- Olofsson, P.; Foody, G. M.; Stehman, S. V.; Woodcock, C. E. (2013). *Making better use of accuracy data in land change studies: Estimating accuracy and area and quantifying uncertainty using stratified estimation*. Remote Sensing of Environment, **129**, 122–131. [doi:10.1016/j.rse.2012.10.031](https://doi.org/10.1016/j.rse.2012.10.031)

- Stehman, S. V. (2014). *Estimating area and map accuracy for stratified random sampling when the strata are different from the map classes*. International Journal of Remote Sensing, **35**, 4923–4939. [doi:10.1080/01431161.2014.930207](https://doi.org/10.1080/01431161.2014.930207)


##  Quick Start

### Installation

```bash
pip install pymapaccuracy
```

For development installation:
```bash
git clone https://github.com/agcopenhaver/PyMapAccuracy.git
cd PyMapAccuracy
pip install -e .
```

### Basic Usage

#### Stehman Estimator (Strata ≠ Map Classes)

```python
from pymapaccuracy import stehman2014

# Example: Administrative regions as strata, land cover as classes
strata = ['region_A', 'region_A', 'region_B', 'region_B', 'region_C', 'region_C']
reference = ['forest', 'grassland', 'water', 'forest', 'urban', 'forest']
map_pred = ['forest', 'forest', 'water', 'grassland', 'urban', 'forest']

# Area of each administrative region (pixels or hectares)
stratum_areas = {
    'region_A': 10000,
    'region_B': 8000,
    'region_C': 12000
}

# Calculate accuracy metrics
results = stehman2014(strata, reference, map_pred, stratum_areas)

print(f"Overall Accuracy: {results['OA']:.3f} ± {results['SEoa']:.3f}")
print(f"Area Estimates:\n{results['area']}")
print(f"Confusion Matrix:\n{results['matrix']}")
```

#### Olofsson Estimator (Map Classes = Strata)

```python
from pymapaccuracy import olofsson

# Example: Map classes serve as sampling strata
reference = ['forest', 'forest', 'water', 'grassland', 'urban', 'forest']
map_pred = ['forest', 'grassland', 'water', 'grassland', 'urban', 'forest']

# Area of each map class stratum
map_areas = {
    'forest': 15000,
    'grassland': 8000,
    'water': 3000,
    'urban': 4000
}

results = olofsson(reference, map_pred, map_areas)
print(f"Overall Accuracy: {results['OA']:.3f} ± {results['SEoa']:.3f}")
```

## 📈 Output Structure

Both estimators return a comprehensive dictionary containing:

```python
{
    'OA': float,              # Overall accuracy
    'UA': pd.Series,          # User's accuracy by map class
    'PA': pd.Series,          # Producer's accuracy by reference class
    'area': pd.Series,        # Area proportion by reference class
    'SEoa': float,            # Standard error of overall accuracy
    'SEua': pd.Series,        # Standard error of user's accuracy
    'SEpa': pd.Series,        # Standard error of producer's accuracy
    'SEa': pd.Series,         # Standard error of area estimates
    'matrix': pd.DataFrame,   # Area-weighted confusion matrix
    'CI_oa': tuple,          # 95% confidence interval for OA
    'CI_ua': pd.Series,      # 95% confidence intervals for UA
    'CI_pa': pd.Series,      # 95% confidence intervals for PA
    'CI_area': pd.Series     # 95% confidence intervals for area
}
```

## 📋 Requirements

- Python 3.8+
- pandas >= 1.3.0
- numpy >= 1.21.0
- scipy >= 1.7.0

##  Testing

Run the comprehensive test suite:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=pymapaccuracy

# Run specific test file
pytest tests/test_stehman2014.py -v
```

##  Documentation

- Coming soon.

##  Contributing

## Contributing

Contributions to **PyMapAccuracy** are welcome. To contribute, please follow these steps:

1. Fork the repository.
2. Create a new feature branch:
   git checkout -b feature/your-feature-name
3. Commit your changes with a clear and descriptive message:
   git commit -m "Add detailed description of feature or fix"
4. Push the branch to your fork:
   git push origin feature/your-feature-name
5. Open a Pull Request to the main repository, providing a concise description of the changes and their purpose.
##  License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📖 Citation

If you use PyMapAccuracy in your research, please cite:

```bibtex
@software{pymapaccuracy,
  title = {PyMapAccuracy: Thematic Map Accuracy Assessment in Python},
  author = {Andrew Copenhaver},
  year = {2025},
  url = {https://github.com/agcopenhaver/PyMapAccuracy}
}
```

##  References

**Stehman Estimator:**
- Stehman, S. V. (2014). *Estimating area and map accuracy for stratified random sampling when the strata are different from the map classes*. International Journal of Remote Sensing, **35**, 4923–4939. [doi:10.1080/01431161.2014.930207](https://doi.org/10.1080/01431161.2014.930207)


**Olofsson Estimator:**
- Olofsson, P.; Foody, G. M.; Herold, M.; Stehman, S. V.; Woodcock, C. E.; Wulder, M. A. (2014). *Good practices for estimating area and assessing accuracy of land change*. Remote Sensing of Environment, **148**, 42–57. [doi:10.1016/j.rse.2014.02.015](https://doi.org/10.1016/j.rse.2014.02.015)

- Olofsson, P.; Foody, G. M.; Stehman, S. V.; Woodcock, C. E. (2013). *Making better use of accuracy data in land change studies: Estimating accuracy and area and quantifying uncertainty using stratified estimation*. Remote Sensing of Environment, **129**, 122–131. [doi:10.1016/j.rse.2012.10.031](https://doi.org/10.1016/j.rse.2012.10.031)

##  Support

- 🐛 Issues: [GitHub Issues](https://github.com/agcopenhaver/PyMapAccuracy/issues)
