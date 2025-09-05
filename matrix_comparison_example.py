"""
Quick example comparing Olofsson and Stehman estimators using IDENTICAL data
and IDENTICAL sampling design (map classes = strata) to verify they produce
the same results when the sampling designs are equivalent.
"""

import pandas as pd
import numpy as np
from src.pymapaccuracy.estimators import olofsson, stehman2014

# Set random seed for reproducibility
np.random.seed(42)

# Create identical reference and map data
n_samples = 100
classes = ['forest', 'grassland', 'water', 'urban']

# Generate reference classes (ground truth)
reference = np.random.choice(classes, size=n_samples, p=[0.4, 0.3, 0.2, 0.1]).tolist()

# Generate map predictions with some confusion
map_pred = reference.copy()
# Add some misclassification noise
noise_indices = np.random.choice(n_samples, size=20, replace=False)
for idx in noise_indices:
    # Change to a different random class
    other_classes = [c for c in classes if c != map_pred[idx]]
    map_pred[idx] = np.random.choice(other_classes)

print("=== DATA SUMMARY ===")
print(f"Total samples: {n_samples}")
print(f"Classes: {classes}")
print()

# Create confusion matrix for inspection
confusion = pd.crosstab(map_pred, reference, margins=True)
print("Raw confusion matrix (map vs reference):")
print(confusion)
print()

# ===================================================================
# IDENTICAL SAMPLING DESIGN: Map classes = Strata for BOTH estimators
# ===================================================================

# Map class areas (these will be used as both map areas AND strata areas)
class_areas = {
    'forest': 15000.0,
    'grassland': 8000.0, 
    'water': 3000.0,
    'urban': 4000.0
}

# The "strata" are the same as map classes - this should make results identical
strata = map_pred.copy()  # Strata = map classes

print("Sampling design:")
print("- Strata = Map classes (identical for both estimators)")
print("- Same reference data")
print("- Same map predictions")
print("- Same area weights")
print()

print("Stratum (= Map class) sample distribution:")
print(pd.Series(strata).value_counts().sort_index())
print()

# ===================================================================
# OLOFSSON ESTIMATOR
# ===================================================================

print("=" * 60)
print("OLOFSSON ESTIMATOR")
print("=" * 60)

olofsson_results = olofsson(reference, map_pred, class_areas)

print("Olofsson Area-weighted Matrix:")
print(olofsson_results['matrix'])
print()
print(f"Overall Accuracy: {olofsson_results['OA']:.6f}")
print()

# ===================================================================
# STEHMAN ESTIMATOR (with identical sampling design)
# ===================================================================

print("=" * 60)
print("STEHMAN ESTIMATOR (Same design as Olofsson)")
print("=" * 60)

stehman_results = stehman2014(strata, reference, map_pred, class_areas)

print("Stehman Area-weighted Matrix:")
print(stehman_results['matrix'])
print()
print(f"Overall Accuracy: {stehman_results['OA']:.6f}")
print()

# ===================================================================
# DETAILED COMPARISON
# ===================================================================

print("=" * 60)
print("DETAILED COMPARISON")
print("=" * 60)

print("Overall Accuracy:")
print(f"Olofsson: {olofsson_results['OA']:.8f}")
print(f"Stehman:  {stehman_results['OA']:.8f}")
print(f"Difference: {abs(olofsson_results['OA'] - stehman_results['OA']):.2e}")
print()

print("User's Accuracy Comparison:")
olof_ua = olofsson_results['UA'].sort_index()
steh_ua = stehman_results['UA'].sort_index()
for class_name in olof_ua.index:
    if class_name in steh_ua.index:
        diff = abs(olof_ua[class_name] - steh_ua[class_name])
        print(f"{class_name:>10}: Olofsson={olof_ua[class_name]:.6f}, Stehman={steh_ua[class_name]:.6f}, Diff={diff:.2e}")
print()

print("Producer's Accuracy Comparison:")
olof_pa = olofsson_results['PA'].sort_index()
steh_pa = stehman_results['PA'].sort_index()
for class_name in olof_pa.index:
    if class_name in steh_pa.index:
        diff = abs(olof_pa[class_name] - steh_pa[class_name])
        print(f"{class_name:>10}: Olofsson={olof_pa[class_name]:.6f}, Stehman={steh_pa[class_name]:.6f}, Diff={diff:.2e}")
print()

print("Area Estimates Comparison:")
olof_areas = olofsson_results['area'].sort_index()
steh_areas = stehman_results['area'].sort_index()
for class_name in olof_areas.index:
    if class_name in steh_areas.index:
        diff = abs(olof_areas[class_name] - steh_areas[class_name])
        print(f"{class_name:>10}: Olofsson={olof_areas[class_name]:.6f}, Stehman={steh_areas[class_name]:.6f}, Diff={diff:.2e}")
print()

print("Matrix Element-wise Comparison:")
olof_matrix = olofsson_results['matrix']
steh_matrix = stehman_results['matrix']

# Remove margins if present for clean comparison
if 'sum' in olof_matrix.index:
    olof_matrix = olof_matrix.drop('sum', axis=0)
if 'sum' in olof_matrix.columns:
    olof_matrix = olof_matrix.drop('sum', axis=1)
if 'sum' in steh_matrix.index:
    steh_matrix = steh_matrix.drop('sum', axis=0)
if 'sum' in steh_matrix.columns:
    steh_matrix = steh_matrix.drop('sum', axis=1)

max_diff = 0
for i in olof_matrix.index:
    for j in olof_matrix.columns:
        if i in steh_matrix.index and j in steh_matrix.columns:
            diff = abs(olof_matrix.loc[i, j] - steh_matrix.loc[i, j])
            max_diff = max(max_diff, diff)
            if diff > 1e-10:  # Only show non-trivial differences
                print(f"Matrix[{i}, {j}]: Olofsson={olof_matrix.loc[i, j]:.8f}, Stehman={steh_matrix.loc[i, j]:.8f}, Diff={diff:.2e}")

if max_diff < 1e-10:
    print("All matrix elements are identical (within numerical precision)")
else:
    print(f"Maximum matrix difference: {max_diff:.2e}")

print()
print("=" * 60)
print("CONCLUSION")
print("=" * 60)
print("When sampling design is identical (map classes = strata),")
print("both estimators should produce nearly identical results.")
print("Any differences are due to numerical precision or implementation details.")
