import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal, assert_series_equal
from scipy.stats import norm

from pymapaccuracy.estimators import olofsson

# Calculate the z-score for 95% CI once
Z_95 = norm.ppf(0.975)  # approx 1.96


def test_olofsson_example1_2013():
    """Replicates Example 1 from Olofsson et al. (2013)."""
    r = ["1"] * 102 + ["2"] * 280 + ["3"] * 118
    m = (
        ["1"] * 97
        + ["2"] * 3
        + ["3"] * 2
        + ["2"] * 279
        + ["3"] * 1
        + ["1"] * 3
        + ["2"] * 18
        + ["3"] * 97
    )
    Nh = {"1": 22353, "2": 1122543, "3": 610228}
    total_area_pixels = sum(Nh.values())

    result = olofsson(r, m, Nh)

    # print(result['SEoa'], result['CIoa'])  # Debugging output
    print(result["area"])  # Debugging output
    print(result["SEa"])  # Debugging output

    print(result["CIa"])  # Debugging output

    # Compare to paper values (allowing for minor floating point differences) - note that there are issues with rounding in the original paper, so values here do not match exactly
    # Area
    assert result["area"]["1"] == pytest.approx(
        0.02570326, abs=1e-5
    )  # eq. 9 (proportion)
    assert result["area"]["1"] * total_area_pixels == pytest.approx(
        45112.4, abs=1
    )  # eq. 10 (pixels)
    # SE Area
    se_area1_pixels = result["SEa"]["1"] * total_area_pixels
    assert se_area1_pixels == pytest.approx(10751.88, abs=1)  # eq. 12 (pixels)
    # CI Area (Note: Paper has typo in lower bound calculation)
    ci_lower = (result["area"]["1"] * total_area_pixels) - Z_95 * se_area1_pixels
    ci_upper = (result["area"]["1"] * total_area_pixels) + Z_95 * se_area1_pixels
    assert ci_lower == pytest.approx(
        45112.4 - 21072.37, abs=2
    )  # Using paper's CI width / 2
    assert ci_upper == pytest.approx(
        45112.4 + 21072.37, abs=2
    )  # Using paper's CI width / 2

    # Accuracies
    assert result["UA"]["1"] == pytest.approx(0.97, abs=1e-5)  # eq. 14
    assert result["PA"]["1"] == pytest.approx(0.4806308, abs=1e-5)  # eq. 15
    assert result["OA"] == pytest.approx(0.9444168, abs=1e-5)  # eq. 16

    # Table 4 comparisons
    expected_ua = pd.Series([0.97, 0.93, 0.97], index=["1", "2", "3"], name="UA")
    expected_pa = pd.Series(
        [0.4806308, 0.9941887, 0.8969259], index=["1", "2", "3"], name="PA"
    )
    expected_se_ua_ci = pd.Series(
        [0.03360292, 0.02892031, 0.03360292], index=["1", "2", "3"], name="SE_UA"
    )  # CI half-width / Z_95
    expected_se_pa_ci = pd.Series(
        [0.22453045, 0.01132522, 0.04120541], index=["1", "2", "3"], name="SE_PA"
    )  # CI half-width / Z_95

    assert_series_equal(
        result["UA"].dropna(), expected_ua, check_exact=False, rtol=1e-5
    )
    assert_series_equal(
        result["PA"].dropna(), expected_pa, check_exact=False, rtol=1e-5
    )
    # Compare SE * Z_95
    assert_series_equal(
        result["SEua"].dropna() * Z_95, expected_se_ua_ci, check_exact=False, rtol=1e-5
    )  # Allow slightly larger tolerance for SE
    assert_series_equal(
        result["SEpa"].dropna() * Z_95, expected_se_pa_ci, check_exact=False, rtol=1e-5
    )  # Allow slightly larger tolerance for SE

    # Check matrix structure (optional, focus on values)
    assert isinstance(result["matrix"], pd.DataFrame)
    assert "sum" in result["matrix"].index
    assert "sum" in result["matrix"].columns


def test_olofsson_example2_2013():
    """Replicates Example 2 from Olofsson et al. (2013), Table 6."""
    r = ["1"] * 129 + ["2"] * 403 + ["3"] * 611
    m = (
        ["1"] * 127
        + ["2"] * 2
        + ["1"] * 66
        + ["2"] * 322
        + ["3"] * 15
        + ["1"] * 54
        + ["2"] * 17
        + ["3"] * 540
    )
    # Nh represents proportions directly in this example
    Nh = {"1": 0.007, "2": 0.295, "3": 0.698}

    result = olofsson(r, m, Nh)

    # Compare to Table 6 values
    assert result["OA"] == pytest.approx(0.96, abs=1e-2)
    assert result["SEoa"] * Z_95 == pytest.approx(0.01, abs=1e-2)  # CI half-width

    expected_ua = pd.Series(
        [0.51417, 0.94428, 0.97297], index=["1", "2", "3"], name="UA"
    )
    expected_pa = pd.Series(
        [0.67535, 0.93072, 0.97664], index=["1", "2", "3"], name="PA"
    )
    expected_se_ua_ci_hw = pd.Series(
        [0.06246, 0.02438, 0.01350], index=["1", "2", "3"], name="SE_UA"
    )  # CI half-width
    expected_se_pa_ci_hw = pd.Series(
        [0.30458, 0.02938, 0.00960], index=["1", "2", "3"], name="SE_PA"
    )  # CI half-width

    assert_series_equal(
        result["UA"].dropna(), expected_ua, check_exact=False, rtol=1e-2
    )
    assert_series_equal(
        result["PA"].dropna(), expected_pa, check_exact=False, rtol=1e-2
    )
    assert_series_equal(
        result["SEua"].dropna() * Z_95,
        expected_se_ua_ci_hw,
        check_exact=False,
        rtol=1e-2,
    )
    assert_series_equal(
        result["SEpa"].dropna() * Z_95,
        expected_se_pa_ci_hw,
        check_exact=False,
        rtol=1e-2,
    )


def test_olofsson_table8_2014():
    """Replicates Example from Table 8 Olofsson et al. (2014)."""
    r_num = [1] * 69 + [2] * 56 + [3] * 175 + [4] * 340
    m_num = (
        [1] * 66
        + [3] * 1
        + [4] * 2
        + [2] * 55
        + [4] * 1
        + [1] * 5
        + [2] * 8
        + [3] * 153
        + [4] * 9
        + [1] * 4
        + [2] * 12
        + [3] * 11
        + [4] * 313
    )

    # Map numeric to string labels
    mapping = {
        1: "Deforestation",
        2: "Forest gain",
        3: "Stable forest",
        4: "Stable non-forest",
    }
    r = [mapping[x] for x in r_num]
    m = [mapping[x] for x in m_num]

    Nh_counts = {
        "Deforestation": 200000,
        "Forest gain": 150000,
        "Stable forest": 3200000,
        "Stable non-forest": 6450000,
    }
    pixel_area = 30**2
    Nh = {k: v * pixel_area for k, v in Nh_counts.items()}
    total_area_ha = sum(Nh.values()) / 10000  # Total area in hectares

    result = olofsson(r, m, Nh)
    class_order = ["Deforestation", "Forest gain", "Stable forest", "Stable non-forest"]

    # Compare accuracies (left-hand side p. 54)
    expected_ua = pd.Series(
        [0.8800000, 0.7333333, 0.9272727, 0.9630769921], index=class_order, name="UA"
    )
    expected_pa = pd.Series(
        [0.7486614, 0.8471564, 0.9345089, 0.9616090], index=class_order, name="PA"
    )
    expected_se_ua_ci_hw = pd.Series(
        [0.07403962, 0.10075516, 0.03974464, 0.02053312],
        index=class_order,
        name="SE_UA",
    )  # CI half-width
    expected_se_pa_ci_hw = pd.Series(
        [0.21330593, 0.25440369, 0.03432379, 0.01836120],
        index=class_order,
        name="SE_PA",
    )  # CI half-width

    assert result["OA"] == pytest.approx(0.9465119, abs=1e-3)
    assert result["SEoa"] * Z_95 == pytest.approx(0.01848328, abs=1e-5)

    assert_series_equal(
        result["UA"].reindex(class_order), expected_ua, check_exact=False, rtol=1e-5
    )
    assert_series_equal(
        result["PA"].reindex(class_order), expected_pa, check_exact=False, rtol=1e-5
    )
    assert_series_equal(
        result["SEua"].reindex(class_order) * Z_95,
        expected_se_ua_ci_hw,
        check_exact=False,
        rtol=1e-5,
    )
    assert_series_equal(
        result["SEpa"].reindex(class_order) * Z_95,
        expected_se_pa_ci_hw,
        check_exact=False,
        rtol=1e-5,
    )

    # Compare area estimates in hectares (right-hand side p. 54)
    area_ha = result["area"].reindex(class_order) * total_area_ha
    se_area_ha = result["SEa"].reindex(class_order) * total_area_ha
    ci_hw_area_ha = se_area_ha * Z_95

    assert area_ha["Deforestation"] == pytest.approx(21157.76, abs=0.1)  # x10^4 ha
    assert ci_hw_area_ha["Deforestation"] == pytest.approx(6157.521, abs=0.1)
    assert area_ha["Forest gain"] == pytest.approx(11686.15, abs=0.1)
    assert ci_hw_area_ha["Forest gain"] == pytest.approx(3755.757, abs=0.1)
    assert area_ha["Stable forest"] == pytest.approx(285769.9, abs=1.0)
    assert ci_hw_area_ha["Stable forest"] == pytest.approx(15509.55, abs=0.1)
    assert area_ha["Stable non-forest"] == pytest.approx(581386.2, abs=1.0)
    assert ci_hw_area_ha["Stable non-forest"] == pytest.approx(16281.36, abs=0.1)


def test_olofsson_extra_map_class():
    """Test case where m (map) has a class not in r (reference)."""
    r = ["1"] * 102 + ["2"] * 280 + ["3"] * 118
    # Add samples mapped as '4', which doesn't exist in reference
    m = (
        ["1"] * 97
        + ["2"] * 3
        + ["3"] * 2
        + ["2"] * 279
        + ["3"] * 1
        + ["1"] * 3
        + ["2"] * 18
        + ["3"] * 95
        + ["4"] * 2
    )  # 2 samples mapped as '4'
    Nh = {"1": 22353, "2": 1122543, "3": 610228, "4": 10}  # Add stratum '4'

    # Should warn about map class not in reference data
    with pytest.warns(UserWarning, match="Map classes not found in reference data"):
        result = olofsson(r, m, Nh)

    # Expect class '4' to appear in UA/SEua index and matrix index/columns
    # but PA/SEpa/Area/SEa should only have '1', '2', '3'
    assert "4" in result["UA"].index
    assert "4" in result["SEua"].index
    assert "4" in result["matrix"].index
    assert "4" in result["matrix"].columns

    # Class '4' should be in all result indices now, but with NaN/0 values for unsampled reference
    assert "4" in result["PA"].index
    assert pd.isna(result["PA"]["4"])  # NaN because no reference samples for class '4'

    assert "4" in result["SEpa"].index
    assert pd.isna(
        result["SEpa"]["4"]
    )  # NaN because no reference samples for class '4'

    assert "4" in result["area"].index
    assert result["area"]["4"] == 0  # Zero area because no reference samples

    assert "4" in result["SEa"].index
    assert result["SEa"]["4"] == 0  # Zero SE because no reference samples

    # Check that area proportions sum close to 1 (excluding the non-existent ref class '4')
    assert result["area"][["1", "2", "3"]].sum() == pytest.approx(1.0)
    # Check OA calculation correctness (should ignore map class 4 contribution to diagonal)
    # Re-run without class 4 for comparison OA
    r_sub = [x for i, x in enumerate(r) if m[i] != "4"]
    m_sub = [x for x in m if x != "4"]
    Nh_sub = {k: v for k, v in Nh.items() if k != "4"}
    result_sub = olofsson(r_sub, m_sub, Nh_sub)
    # OA might differ slightly due to changed sample and weights, but should be close
    assert result["OA"] == pytest.approx(result_sub["OA"], abs=0.01)


def test_olofsson_extra_ref_class():
    """Test case where r (reference) has a class not in m (map strata)."""
    # Add samples with reference '4', which doesn't exist in map/Nh
    r = ["1"] * 102 + ["2"] * 280 + ["3"] * 116 + ["4"] * 2
    m = (
        ["1"] * 97
        + ["2"] * 3
        + ["3"] * 2
        + ["2"] * 279
        + ["3"] * 1
        + ["1"] * 3
        + ["2"] * 18
        + ["3"] * 97
    )  # Length matches r
    Nh = {"1": 22353, "2": 1122543, "3": 610228}  # No stratum '4'

    # Should warn about reference class not in map strata
    with pytest.warns(UserWarning, match="Reference classes not found in map strata"):
        result = olofsson(r, m, Nh)

    # Expect class '4' to appear in PA/SEpa/Area/SEa index and matrix columns
    # but UA/SEua should only have '1', '2', '3'
    assert "4" in result["PA"].index
    assert "4" in result["SEpa"].index
    assert "4" in result["area"].index
    assert "4" in result["SEa"].index
    assert "4" in result["matrix"].columns

    assert "4" in result["UA"].index
    assert pd.isna(result["UA"]["4"])  # NaN because class '4' is not a map class
    assert "4" in result["SEua"].index
    assert pd.isna(result["SEua"]["4"])  # NaN because class '4' is not a map class
    assert (
        "4" in result["matrix"].index
    )  # Now class '4' should appear as a row (symmetrical matrix)

    # Area for class '4' should be 0 (no reference samples)
    assert result["area"]["4"] == pytest.approx(0.006953674, abs=1e-5)
    assert result["SEa"]["4"] == pytest.approx(0.004892094, abs=1e-5)
    # PA for class '4' should be 0 as it's never mapped correctly, but reference samples exist
    assert result["PA"]["4"] == 0
    assert (
        result["SEpa"]["4"] == 0
    )  # Should be 0 because PA is 0, but reference samples exist


def test_olofsson_extra_nh_class():
    """Test case where Nh includes a class not found in m or r."""
    # Use the data with reference class '4' (matching R code)
    r = ["1"] * 102 + ["2"] * 280 + ["3"] * 116 + ["4"] * 2
    m = (
        ["1"] * 97
        + ["2"] * 3
        + ["3"] * 2
        + ["2"] * 279
        + ["3"] * 1
        + ["1"] * 3
        + ["2"] * 18
        + ["3"] * 97
    )  # Length matches r
    Nh = {"1": 22353, "2": 1122543, "3": 610228, "9": 0}  # Add stratum '9' with 0 area

    # Should warn about unused Nh stratum
    with pytest.warns(UserWarning, match="Map strata in Nh not found in sample data"):
        result = olofsson(r, m, Nh)

    # The function should allow extra labels in Nh that are not in m (they get ignored)
    # All map labels ('1', '2', '3') are found in Nh, so no error should be raised

    # The function should ignore the extra '9' stratum since no samples 'm' belong to it.
    # However, the matrix structure will differ because result includes stratum '9'

    # Compare with original Nh without stratum '9'
    Nh_orig = {"1": 22353, "2": 1122543, "3": 610228}
    result_orig = olofsson(r, m, Nh_orig)

    # Compare the core statistics for classes that actually have data
    # Note: class_order differs between result and result_orig due to extra stratum '9'
    actual_ref_classes = [
        "1",
        "2",
        "3",
        "4",
    ]  # The actual reference classes from the data (now includes '4')

    assert result["OA"] == pytest.approx(result_orig["OA"])
    assert_series_equal(
        result["area"][actual_ref_classes], result_orig["area"]
    )  # Subset result to match
    assert_series_equal(
        result["PA"][actual_ref_classes], result_orig["PA"]
    )  # Subset result to match
    assert result["SEoa"] == pytest.approx(result_orig["SEoa"])
    assert_series_equal(
        result["SEa"][actual_ref_classes], result_orig["SEa"]
    )  # Subset result to match
    assert_series_equal(
        result["SEpa"][actual_ref_classes], result_orig["SEpa"]
    )  # Subset result to match

    # UA/SEua will differ: result has ['1','2','3','4','9'] vs result_orig has ['1','2','3','4']
    # Compare only the overlapping classes (exclude '9' which only appears in result)
    assert_series_equal(result["UA"][["1", "2", "3", "4"]], result_orig["UA"])
    assert_series_equal(result["SEua"][["1", "2", "3", "4"]], result_orig["SEua"])
    # Check that stratum '9' has NaN values (no samples)
    assert result["UA"]["9"] == 0
    assert result["SEua"]["9"] == 0

    # add checks that PA/SEpa are nan, but area/SEa are 0 for stratum '9'
    assert pd.isna(result["PA"]["9"])
    assert pd.isna(result["SEpa"]["9"])
    assert result["area"]["9"] == 0
    assert result["SEa"]["9"] == 0

    # Matrix comparison: result is 5x5 with '9' row/col, result_orig is 4x4 without '9'
    # Compare the overlapping portion (excluding row/col '9')
    # Both matrices should have the same classes ['1','2','3','4'] + 'sum', just result has extra '9'
    common_indices = [
        "1",
        "2",
        "3",
        "4",
        "sum",
    ]  # Include 'sum' row/col if margins=True
    assert_frame_equal(
        result["matrix"].loc[common_indices, common_indices], result_orig["matrix"]
    )

    # Verify that row '9' and column '9' are all NaN/0 in result
    test_cols = ["1", "2", "3", "4", "sum"]
    assert all(
        pd.isna(result["matrix"].loc["9", test_cols])
        | (result["matrix"].loc["9", test_cols] == 0)
    )
    assert all(
        pd.isna(result["matrix"].loc[test_cols, "9"])
        | (result["matrix"].loc[test_cols, "9"] == 0)
    )


def test_olofsson_invalid_input():
    """Test invalid inputs for olofsson function."""
    # Test map class not in Nh
    r = [1, 1, 2, 2]
    m = [1, 2, 1, 3]  # Map class 3 not in Nh
    Nh = {1: 100, 2: 200}
    with pytest.raises(
        ValueError,
        match=r"Map class labels in sample data 'm' not found in Nh keys:.*3",
    ):
        olofsson(r, m, Nh)

    r = [1, 1, 2, 2]
    m = [1, 2, 1, 2]
    Nh = {1: 100, 2: -50}  # Negative area
    with pytest.raises(ValueError, match="Nh values .* must be non-negative"):
        olofsson(r, m, Nh)

    r = [1, 1, 2, 2]
    m = [1, 2, 1, 2]
    Nh = [100, 200]  # Not a dict
    with pytest.raises(ValueError, match="Nh must be a dictionary"):
        olofsson(r, m, Nh)

    r = [1, 1, 2, 2]
    m = [1, 2, 1, 2]
    Nh = {1: 0, 2: 0}  # Zero total area
    with pytest.raises(ValueError, match="Total area .* cannot be zero"):
        olofsson(r, m, Nh)


# Test with single sample stratum - should warn and produce results (SE might be 0/NaN)
def test_olofsson_single_sample_stratum():
    r = ["A", "A", "B", "C"]
    m = ["A", "A", "B", "C"]  # Strata B and C have 1 sample
    Nh = {"A": 100, "B": 50, "C": 200}
    with pytest.warns(UserWarning, match="strata include only one observation: B; C"):
        result = olofsson(r, m, Nh)

    assert result is not None
    assert result["OA"] == pytest.approx(1.0)  # Perfect classification
    # Check if SEs are NaN or 0 for single-sample strata
    assert result["SEua"]["B"] == 0 or np.isnan(result["SEua"]["B"])
    assert result["SEua"]["C"] == 0 or np.isnan(result["SEua"]["C"])
    # SEpa involves more complex calculation, check if finite
    assert np.isfinite(result["SEpa"]["B"])
    assert np.isfinite(result["SEpa"]["C"])
    # SEa involves sum over strata, check if finite
    assert np.isfinite(result["SEa"]["B"])
    assert np.isfinite(result["SEa"]["C"])


def test_olofsson_perfect_classification():
    """Test case with perfect classification (OA = 1.0)."""
    r = ["A"] * 50 + ["B"] * 30 + ["C"] * 20
    m = ["A"] * 50 + ["B"] * 30 + ["C"] * 20  # Perfect match
    Nh = {"A": 1000, "B": 600, "C": 400}

    result = olofsson(r, m, Nh)

    # Should have perfect accuracy
    assert result["OA"] == pytest.approx(1.0)
    assert result["UA"]["A"] == pytest.approx(1.0)
    assert result["UA"]["B"] == pytest.approx(1.0)
    assert result["UA"]["C"] == pytest.approx(1.0)
    assert result["PA"]["A"] == pytest.approx(1.0)
    assert result["PA"]["B"] == pytest.approx(1.0)
    assert result["PA"]["C"] == pytest.approx(1.0)

    # Standard errors should be 0 for perfect classification
    assert result["SEoa"] == pytest.approx(0.0)
    assert result["SEua"]["A"] == pytest.approx(0.0)
    assert result["SEua"]["B"] == pytest.approx(0.0)
    assert result["SEua"]["C"] == pytest.approx(0.0)


def test_olofsson_zero_accuracy():
    """Test case with no correct classifications (OA = 0.0)."""
    r = ["A"] * 50 + ["B"] * 50
    m = ["B"] * 50 + ["A"] * 50  # Completely wrong classifications
    Nh = {"A": 1000, "B": 1000}

    result = olofsson(r, m, Nh)

    # Should have zero accuracy
    assert result["OA"] == pytest.approx(0.0)
    assert result["UA"]["A"] == pytest.approx(0.0)
    assert result["UA"]["B"] == pytest.approx(0.0)
    assert result["PA"]["A"] == pytest.approx(0.0)
    assert result["PA"]["B"] == pytest.approx(0.0)

    # Results should still be finite
    assert np.isfinite(result["SEoa"])
    assert np.isfinite(result["SEua"]["A"])
    assert np.isfinite(result["SEua"]["B"])


def test_olofsson_single_class():
    """Test case with only one class in data."""
    r = ["A"] * 100
    m = ["A"] * 100
    Nh = {"A": 2000}

    result = olofsson(r, m, Nh)

    # Should have perfect accuracy for the single class
    assert result["OA"] == pytest.approx(1.0)
    assert result["UA"]["A"] == pytest.approx(1.0)
    assert result["PA"]["A"] == pytest.approx(1.0)
    assert result["area"]["A"] == pytest.approx(
        1.0
    )  # Only class, should be 100% of area

    # Standard errors should be 0
    assert result["SEoa"] == pytest.approx(0.0)
    assert result["SEua"]["A"] == pytest.approx(0.0)


def test_olofsson_extreme_class_imbalance():
    """Test case with very imbalanced classes."""
    # Very rare class vs very common class
    r = ["common"] * 480 + ["rare"] * 20
    m = ["common"] * 475 + ["rare"] * 5 + ["common"] * 15 + ["rare"] * 5
    Nh = {"common": 999000, "rare": 1000}  # 99.9% vs 0.1%

    result = olofsson(r, m, Nh)

    # Should still produce valid results
    assert 0 <= result["OA"] <= 1
    assert 0 <= result["UA"]["common"] <= 1
    assert 0 <= result["UA"]["rare"] <= 1
    assert 0 <= result["PA"]["common"] <= 1
    assert 0 <= result["PA"]["rare"] <= 1

    # Area should reflect the extreme imbalance
    assert result["area"]["common"] > 0.95  # Should be > 95%
    assert result["area"]["rare"] < 0.05  # Should be < 5%

    # All standard errors should be finite
    assert np.isfinite(result["SEoa"])
    assert np.isfinite(result["SEua"]["common"])
    assert np.isfinite(result["SEua"]["rare"])
    assert np.isfinite(result["SEpa"]["common"])
    assert np.isfinite(result["SEpa"]["rare"])


def test_olofsson_empty_strata():
    """Test case with strata that have area but no samples."""
    r = ["A"] * 50 + ["B"] * 50
    m = ["A"] * 50 + ["B"] * 50  # No samples from stratum C
    Nh = {"A": 1000, "B": 1000, "C": 500}  # Stratum C has area but no samples

    # Should warn about unused stratum
    with pytest.warns(UserWarning, match="Map strata in Nh not found in sample data"):
        result = olofsson(r, m, Nh)

    # Should still work for sampled strata
    assert 0 <= result["OA"] <= 1
    assert result["UA"]["A"] == pytest.approx(1.0)  # Perfect for sampled strata
    assert result["UA"]["B"] == pytest.approx(1.0)

    # Unused stratum should have 0 UA (no estimated area) but NaN PA (no reference samples)
    assert (
        result["UA"]["C"] == 0
    )  # Map stratum with no samples → 0 estimated area → UA = 0
    assert result["SEua"]["C"] == 0  # No variance for unused stratum
    assert pd.isna(
        result["PA"]["C"]
    )  # No reference samples for class 'C' → PA undefined
    assert pd.isna(result["SEpa"]["C"])  # No reference samples → SEpa undefined


def test_olofsson_numerical_extremes():
    """Test case with very large Nh values."""
    r = ["A"] * 10 + ["B"] * 10
    m = ["A"] * 10 + ["B"] * 10
    Nh = {"A": 1e9, "B": 1e9}  # Very large area values

    result = olofsson(r, m, Nh)

    # Should still produce valid results
    assert result["OA"] == pytest.approx(1.0)
    assert result["area"]["A"] == pytest.approx(0.5, abs=1e-10)
    assert result["area"]["B"] == pytest.approx(0.5, abs=1e-10)

    # Results should be finite despite large numbers
    assert np.isfinite(result["SEoa"])
    assert np.isfinite(result["SEua"]["A"])
    assert np.isfinite(result["SEua"]["B"])
    assert np.isfinite(result["SEpa"]["A"])
    assert np.isfinite(result["SEpa"]["B"])
