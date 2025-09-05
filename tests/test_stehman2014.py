import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal, assert_series_equal

from pymapaccuracy.estimators import stehman2014  # Corrected import path


def test_stehman2014_valid_input():
    # Proper Stehman scenario: strata (s) differ from map classes (m)
    s = [
        "elevation_low",
        "elevation_low",
        "elevation_med",
        "elevation_med",
        "elevation_high",
        "elevation_high",
    ]  # Elevation strata
    r = [
        "forest",
        "grassland",
        "grassland",
        "water",
        "water",
        "water",
    ]  # Reference classes
    m = [
        "forest",
        "forest",
        "grassland",
        "grassland",
        "water",
        "water",
    ]  # Map classes (different from strata)
    Nh_strata = {
        "elevation_low": 1000,
        "elevation_med": 800,
        "elevation_high": 600,
    }  # Areas by elevation stratum

    result = stehman2014(s, r, m, Nh_strata)

    assert isinstance(result, dict)
    assert "OA" in result
    assert "UA" in result
    assert "PA" in result
    assert "area" in result
    assert "matrix" in result

    # Check confidence intervals are present and have correct structure
    assert "CIoa" in result
    assert "CIua" in result
    assert "CIpa" in result
    assert "CIa" in result
    assert isinstance(result["CIoa"], tuple) and len(result["CIoa"]) == 2
    assert isinstance(result["CIua"], tuple) and len(result["CIua"]) == 2

    # Check CI bounds are reasonable (lower ≤ estimate ≤ upper)
    oa_lower, oa_upper = result["CIoa"]
    assert oa_lower <= result["OA"] <= oa_upper


def test_stehman2014_strata_equal_map_classes():
    """Test Stehman method when strata equal map classes (should work like Olofsson but using Stehman equations)."""
    # Valid scenario: strata same as map classes - this should work fine
    s = ["forest", "forest", "water", "water", "urban", "urban"]  # Strata = map classes
    r = ["forest", "water", "water", "urban", "urban", "urban"]  # Reference classes
    m = [
        "forest",
        "forest",
        "water",
        "water",
        "urban",
        "urban",
    ]  # Map classes (same as strata)
    Nh_strata = {
        "forest": 1000,
        "water": 800,
        "urban": 600,
    }  # Areas by stratum (which = map class)

    result = stehman2014(s, r, m, Nh_strata)

    # Should work without error and return valid results
    assert isinstance(result, dict)
    assert "OA" in result
    assert "UA" in result
    assert "PA" in result
    assert "area" in result
    assert "matrix" in result

    # Results should be reasonable
    assert 0 <= result["OA"] <= 1
    assert all(0 <= ua <= 1 or pd.isna(ua) for ua in result["UA"])
    assert all(0 <= pa <= 1 or pd.isna(pa) for pa in result["PA"])


def test_stehman2014_different_lengths_handled_gracefully():
    """Test that Stehman properly rejects different input lengths."""
    # Example where strata (s) differ from map classes (m) - this is the point of Stehman method!
    s = ["stratum_1", "stratum_1", "stratum_2"]  # Strata labels (length 3)
    r = ["class_A", "class_B"]  # Reference (length 2 - shorter)
    m = ["class_A", "class_A", "class_B"]  # Map classes (length 3)
    Nh_strata = {
        "stratum_1": 100,
        "stratum_2": 200,
    }  # Areas by stratum, not by map class

    # Should raise a descriptive error about length mismatch
    with pytest.raises(ValueError, match="Input vectors must have the same length"):
        stehman2014(s, r, m, Nh_strata)


def test_stehman2014_invalid_Nh_strata():
    # Test various invalid Nh_strata inputs
    s = [
        "geographic_north",
        "geographic_north",
        "geographic_south",
        "geographic_south",
    ]  # Geographic strata
    r = ["forest", "water", "water", "grassland"]  # Reference classes
    m = ["forest", "forest", "water", "water"]  # Map classes (different from strata)

    # Test negative area
    Nh_strata_negative = {
        "geographic_north": 1000,
        "geographic_south": -500,
    }  # Invalid negative area
    with pytest.raises(
        ValueError,
        match="Nh_strata values must be non-negative numbers representing stratum areas",
    ):
        stehman2014(s, r, m, Nh_strata_negative)

    # Test non-dictionary
    Nh_strata_list = [1000, 500]  # Should be dict, not list
    with pytest.raises(
        ValueError,
        match="Nh_strata must be a dictionary with stratum labels as keys and areas as values",
    ):
        stehman2014(s, r, m, Nh_strata_list)


def test_stehman2014_missing_strata():
    # Proper Stehman scenario: strata differ from map classes
    s = ["admin_zone_A", "admin_zone_A", "admin_zone_missing"]  # Administrative strata
    r = ["urban", "rural", "rural"]  # Reference classes
    m = ["urban", "urban", "rural"]  # Map classes (different from strata)
    Nh_strata = {
        "admin_zone_A": 500,
        "admin_zone_B": 300,
    }  # Missing 'admin_zone_missing'

    with pytest.raises(
        ValueError,
        match="Stratum labels in 's' not found in Nh_strata keys: .*admin_zone_missing.*All strata in sample data must have corresponding area values",
    ):
        stehman2014(s, r, m, Nh_strata)


def test_stehman2014_correctness_perfect():
    """Test with perfect classification where strata differ from map classes (proper Stehman scenario)."""
    # Strata based on administrative boundaries, classes based on land cover
    s_perf = ["admin_A", "admin_A", "admin_B", "admin_B"]  # Administrative strata
    r_perf = [
        "forest",
        "forest",
        "water",
        "water",
    ]  # Reference classes (perfect agreement)
    m_perf = [
        "forest",
        "forest",
        "water",
        "water",
    ]  # Map classes (perfect agreement with reference)
    Nh_perf = {
        "admin_A": 100,
        "admin_B": 100,
    }  # Areas by administrative stratum (not by land cover class)
    result = stehman2014(s_perf, r_perf, m_perf, Nh_perf, order=["forest", "water"])

    expected_OA = 1.0
    expected_UA = pd.Series([1.0, 1.0], index=["forest", "water"], name="UA")
    expected_PA = pd.Series([1.0, 1.0], index=["forest", "water"], name="PA")
    expected_area = pd.Series([0.5, 0.5], index=["forest", "water"], name="Area")
    expected_SEoa = 0.0
    expected_SEua = pd.Series([0.0, 0.0], index=["forest", "water"], name="SE_UA")
    expected_SEpa = pd.Series([0.0, 0.0], index=["forest", "water"], name="SE_PA")
    expected_SEa = pd.Series([0.0, 0.0], index=["forest", "water"], name="SE_Area")
    expected_matrix_data = {
        "forest": [0.5, np.nan, 0.5],
        "water": [np.nan, 0.5, 0.5],
        "sum": [0.5, 0.5, 1.0],
    }
    expected_matrix = pd.DataFrame(
        expected_matrix_data, index=["forest", "water", "sum"]
    )
    expected_matrix.columns.name = None  # Match output format
    expected_matrix.index.name = None

    assert result["OA"] == pytest.approx(expected_OA)
    assert_series_equal(result["UA"], expected_UA, check_exact=False, rtol=1e-6)
    assert_series_equal(result["PA"], expected_PA, check_exact=False, rtol=1e-6)
    assert_series_equal(result["area"], expected_area, check_exact=False, rtol=1e-6)
    assert result["SEoa"] == pytest.approx(expected_SEoa)
    assert_series_equal(result["SEua"], expected_SEua, check_exact=False, atol=1e-6)
    assert_series_equal(result["SEpa"], expected_SEpa, check_exact=False, atol=1e-6)
    assert_series_equal(result["SEa"], expected_SEa, check_exact=False, atol=1e-6)
    assert_frame_equal(result["matrix"], expected_matrix, check_exact=False, rtol=1e-6)

    # Test confidence intervals for perfect case (should have zero width)
    assert "CIoa" in result
    assert "CIua" in result
    assert "CIpa" in result
    assert "CIa" in result

    # Perfect classification: CI should be (1.0, 1.0) for OA since SE=0
    oa_lower, oa_upper = result["CIoa"]
    assert oa_lower == pytest.approx(1.0, abs=1e-10)
    assert oa_upper == pytest.approx(1.0, abs=1e-10)

    # CI half-width should be 0 for perfect case
    assert result["CI_halfwidth_oa"] == pytest.approx(0.0, abs=1e-10)

    # UA and PA CIs should also be (1.0, 1.0) for each class
    ua_lower, ua_upper = result["CIua"]
    pa_lower, pa_upper = result["CIpa"]
    for class_name in ["forest", "water"]:
        assert ua_lower[class_name] == pytest.approx(1.0, abs=1e-10)
        assert ua_upper[class_name] == pytest.approx(1.0, abs=1e-10)
        assert pa_lower[class_name] == pytest.approx(1.0, abs=1e-10)
        assert pa_upper[class_name] == pytest.approx(1.0, abs=1e-10)


def test_stehman2014_paper_example():
    """Replicate the numerical example from Stehman (2014), IJRS."""
    # Data from the R example matching the paper
    s = ["A"] * 10 + ["B"] * 10 + ["C"] * 10 + ["D"] * 10
    m = (
        ["A"] * 7
        + ["B"] * 3
        + ["A"] * 1
        + ["B"] * 11
        + ["C"] * 6
        + ["B"] * 2
        + ["D"] * 10
    )
    r = (
        ["A"] * 5
        + ["C"] * 1
        + ["B"] * 1
        + ["A"] * 1
        + ["B"] * 1
        + ["C"] * 1  # Stratum A (10)
        + ["A"] * 1
        + ["B"] * 5
        + ["A"] * 2
        + ["B"] * 2  # Stratum B (10)
        + ["C"] * 5
        + ["D"] * 2
        + ["B"] * 2
        + ["A"] * 1  # Stratum C (10)
        + ["D"] * 7
        + ["C"] * 2
        + ["B"] * 1
    )  # Stratum D (10)
    Nh_strata = {"A": 40000, "B": 30000, "C": 20000, "D": 10000}
    order = ["A", "B", "C", "D"]  # Explicitly set order for clarity

    # Expected results from R implementation (considered canonical)
    # R results are more precise than paper values due to rounding in paper
    expected_area_A = 0.35  # R: e$area[1] = 0.35
    expected_area_C = 0.20  # R: e$area[3] = 0.2
    expected_OA = 0.63  # R: e$OA = 0.63
    expected_UA_B = 0.5744681  # R: e$UA[2] = 0.5744681
    expected_PA_B = 0.7941176  # R: e$PA[2] = 0.7941176
    expected_P_BC = 0.08  # R: e$matrix[2,3] = 0.08

    expected_SEa_A = 0.0822478  # R: e$SEa[1] = 0.0822478
    expected_SEa_C = 0.06427977  # R: e$SEa[3] = 0.06427977
    expected_SEoa = 0.08464219  # R: e$SEoa = 0.08464219
    expected_SEua_B = 0.1247822  # R: e$SEua[2] = 0.1247822
    expected_SEpa_B = 0.1165479  # R: e$SEpa[2] = 0.1165479
    # SE for P_BC is not explicitly given in the paper text provided.

    result = stehman2014(
        s, r, m, Nh_strata, order=order, margins=False
    )  # Match R example default

    # Check point estimates (high precision to match R results exactly)
    assert result["area"]["A"] == pytest.approx(expected_area_A, abs=1e-6)
    assert result["area"]["C"] == pytest.approx(expected_area_C, abs=1e-6)
    assert result["OA"] == pytest.approx(expected_OA, abs=1e-6)
    assert result["UA"]["B"] == pytest.approx(expected_UA_B, abs=1e-6)
    assert result["PA"]["B"] == pytest.approx(expected_PA_B, abs=1e-6)
    assert result["matrix"].loc["B", "C"] == pytest.approx(expected_P_BC, abs=1e-6)

    # Check standard errors (high precision to match R results exactly)
    assert result["SEa"]["A"] == pytest.approx(expected_SEa_A, abs=1e-6)
    assert result["SEa"]["C"] == pytest.approx(expected_SEa_C, abs=1e-6)
    assert result["SEoa"] == pytest.approx(expected_SEoa, abs=1e-6)
    assert result["SEua"]["B"] == pytest.approx(expected_SEua_B, abs=1e-6)
    assert result["SEpa"]["B"] == pytest.approx(expected_SEpa_B, abs=1e-6)

    # Check confidence intervals are present and consistent with standard errors
    assert "CIoa" in result
    assert "CIua" in result
    assert "CIpa" in result
    assert "CIa" in result

    # Verify CI structure and bounds for paper example
    oa_lower, oa_upper = result["CIoa"]
    assert oa_lower <= result["OA"] <= oa_upper
    assert 0 <= oa_lower <= 1 and 0 <= oa_upper <= 1

    # Check that CI width is approximately 2 * 1.96 * SE for overall accuracy
    expected_ci_width = 2 * 1.96 * expected_SEoa
    actual_ci_width = oa_upper - oa_lower
    assert actual_ci_width == pytest.approx(expected_ci_width, rel=0.01)


def test_stehman2014_invalid_order():
    """Test providing an order list that misses classes."""
    # Re-use perfect data for simplicity
    s_perf = [1, 1, 2, 2]
    r_perf = [1, 1, 2, 2]
    m_perf = [1, 1, 2, 2]
    Nh_perf = {1: 100, 2: 100}
    order = [1]  # Missing class 2

    with pytest.raises(
        ValueError,
        match=r"Argument 'order' must include all class labels found in reference and map data.*Missing.*2",
    ):
        stehman2014(s_perf, r_perf, m_perf, Nh_perf, order=order)


def test_stehman2014_zero_accuracy():
    """Test Stehman with zero accuracy (all misclassified)."""
    s = ["region_A", "region_A", "region_B", "region_B"]  # Geographic strata
    r = ["forest", "forest", "water", "water"]  # Reference classes
    m = ["water", "water", "forest", "forest"]  # Map classes (completely wrong)
    Nh_strata = {"region_A": 100, "region_B": 100}

    result = stehman2014(s, r, m, Nh_strata)

    # Overall accuracy should be 0
    assert result["OA"] == pytest.approx(0.0, abs=1e-6)

    # All individual accuracies should be 0
    for class_name in result["UA"].index:
        assert result["UA"][class_name] == pytest.approx(0.0, abs=1e-6)
    for class_name in result["PA"].index:
        assert result["PA"][class_name] == pytest.approx(0.0, abs=1e-6)

    # Standard errors should be computable and non-negative
    for class_name in result["SEua"].index:
        assert result["SEua"][class_name] >= 0.0
    for class_name in result["SEpa"].index:
        assert result["SEpa"][class_name] >= 0.0
    assert result["SEoa"] >= 0.0

    # Check confidence intervals for zero accuracy case
    assert "CIoa" in result
    oa_lower, oa_upper = result["CIoa"]
    assert oa_lower <= result["OA"] <= oa_upper
    assert oa_lower >= 0.0  # Lower bound should be non-negative


def test_stehman2014_single_class():
    """Test Stehman with only one class present."""
    s = ["stratum_1", "stratum_1", "stratum_2", "stratum_2"]
    r = ["forest", "forest", "forest", "forest"]  # Only one reference class
    m = ["forest", "forest", "forest", "forest"]  # Only one map class
    Nh_strata = {"stratum_1": 150, "stratum_2": 50}

    result = stehman2014(s, r, m, Nh_strata)

    # Overall accuracy should be 1.0 (perfect match)
    assert result["OA"] == pytest.approx(1.0, abs=1e-6)

    # Should have results for the single class
    assert "forest" in result["UA"].index
    assert "forest" in result["PA"].index
    assert result["UA"]["forest"] == pytest.approx(1.0, abs=1e-6)
    assert result["PA"]["forest"] == pytest.approx(1.0, abs=1e-6)


def test_stehman2014_extreme_class_imbalance():
    """Test Stehman with extreme class imbalance."""
    s = ["urban_core"] * 19 + ["rural_edge"]  # Unequal strata sizes
    r = ["building"] * 19 + ["forest"]  # 95% vs 5%
    m = ["building"] * 19 + ["forest"]  # Perfect classification
    Nh_strata = {"urban_core": 9500, "rural_edge": 500}  # 95% vs 5% areas

    result = stehman2014(s, r, m, Nh_strata)

    # Should handle extreme imbalance gracefully
    assert 0 <= result["OA"] <= 1
    assert all(0 <= ua <= 1 or pd.isna(ua) for ua in result["UA"])
    assert all(0 <= pa <= 1 or pd.isna(pa) for pa in result["PA"])

    # Areas should reflect the imbalance
    assert result["area"]["building"] > result["area"]["forest"]
    assert result["area"]["building"] == pytest.approx(
        0.95, abs=0.1
    )  # Approximately 95%


def test_stehman2014_empty_strata():
    """Test Stehman when some strata have no samples."""
    s = ["coast", "coast", "mountain", "mountain"]  # No 'desert' stratum in sample
    r = ["water", "sand", "rock", "snow"]
    m = ["water", "sand", "rock", "snow"]
    Nh_strata = {
        "coast": 100,
        "mountain": 100,
        "desert": 200,
    }  # 'desert' has area but no samples

    # Should handle gracefully with warning about unused stratum
    with pytest.warns(
        UserWarning, match="Strata in Nh_strata not found in sample data"
    ):
        result = stehman2014(s, r, m, Nh_strata)

    # Should still return valid results
    assert isinstance(result, dict)
    assert "OA" in result
    assert "matrix" in result


def test_stehman2014_single_sample_stratum():
    """Test Stehman when some strata have only one sample."""
    s = [
        "high_elev",
        "low_elev",
        "low_elev",
        "low_elev",
    ]  # Unequal stratum sample sizes
    r = ["alpine", "grassland", "forest", "water"]
    m = ["alpine", "grassland", "forest", "water"]
    Nh_strata = {"high_elev": 300, "low_elev": 700}

    # Should issue warning about single observation stratum
    with pytest.warns(UserWarning, match="only one observation"):
        result = stehman2014(s, r, m, Nh_strata)

    # Should still compute results
    assert isinstance(result, dict)
    assert "OA" in result
    assert result["SEoa"] >= 0  # Standard errors should be non-negative


def test_stehman2014_extra_map_class():
    """Test Stehman when map has classes not in reference."""
    s = ["north", "north", "south", "south"]
    r = ["forest", "water", "forest", "water"]  # Only forest and water in reference
    m = ["forest", "urban", "forest", "water"]  # 'urban' only in map, not reference
    Nh_strata = {"north": 100, "south": 100}

    result = stehman2014(s, r, m, Nh_strata)

    # Should handle gracefully and include all classes in output
    assert "urban" in result["UA"].index  # Should have UA for map class 'urban'
    assert "forest" in result["PA"].index
    assert "water" in result["PA"].index

    # Matrix should be symmetrical
    expected_classes = sorted(["forest", "water", "urban"])
    assert list(result["matrix"].index[:-1]) == expected_classes  # Exclude 'sum' row
    assert (
        list(result["matrix"].columns[:-1]) == expected_classes
    )  # Exclude 'sum' column


def test_stehman2014_extra_ref_class():
    """Test Stehman when reference has classes not in map."""
    s = ["east", "east", "west", "west"]
    r = ["forest", "water", "grassland", "forest"]  # 'grassland' only in reference
    m = ["forest", "water", "forest", "forest"]  # No 'grassland' in map
    Nh_strata = {"east": 100, "west": 100}

    result = stehman2014(s, r, m, Nh_strata)

    # Should handle gracefully and include all classes
    assert (
        "grassland" in result["PA"].index
    )  # Should have PA for reference class 'grassland'
    assert "forest" in result["UA"].index
    assert "water" in result["UA"].index

    # Matrix should be symmetrical
    expected_classes = sorted(["forest", "water", "grassland"])
    assert list(result["matrix"].index[:-1]) == expected_classes
    assert list(result["matrix"].columns[:-1]) == expected_classes


def test_stehman2014_unused_stratum():
    """Test Stehman when Nh_strata has strata not in sample."""
    s = ["valley", "valley", "plateau", "plateau"]  # No 'peak' stratum in sample
    r = ["grass", "shrub", "rock", "snow"]
    m = ["grass", "shrub", "rock", "snow"]
    Nh_strata = {"valley": 100, "plateau": 100, "peak": 50}  # 'peak' not in sample

    # Should handle gracefully (this is already filtered in the function)
    result = stehman2014(s, r, m, Nh_strata)

    # Should work normally, ignoring unused stratum
    assert isinstance(result, dict)
    assert "OA" in result
    assert 0 <= result["OA"] <= 1


def test_stehman2014_all_classes_in_one_stratum():
    """Test Stehman when all classes appear in a single stratum."""
    s = ["central", "central", "central", "central"]  # All samples in one stratum
    r = ["forest", "water", "urban", "grassland"]  # Multiple classes
    m = ["forest", "water", "urban", "grassland"]  # Perfect classification
    Nh_strata = {
        "central": 1000,
        "peripheral": 500,
    }  # Two strata, but samples only in one

    result = stehman2014(s, r, m, Nh_strata)

    # Should handle this edge case
    assert result["OA"] == pytest.approx(1.0, abs=1e-6)  # Perfect classification
    assert all(result["UA"] == 1.0)
    assert all(result["PA"] == 1.0)

    # Should include all classes in output
    expected_classes = sorted(["forest", "water", "urban", "grassland"])
    assert sorted(result["UA"].index) == expected_classes
    assert sorted(result["PA"].index) == expected_classes


def test_stehman2014_class_appears_in_multiple_strata():
    """Test Stehman when same class appears across multiple strata."""
    s = ["humid", "humid", "arid", "arid", "temperate", "temperate"]
    r = [
        "forest",
        "forest",
        "forest",
        "desert",
        "forest",
        "grassland",
    ]  # 'forest' in multiple strata
    m = [
        "forest",
        "forest",
        "forest",
        "desert",
        "forest",
        "grassland",
    ]  # Perfect classification
    Nh_strata = {"humid": 300, "arid": 200, "temperate": 500}

    result = stehman2014(s, r, m, Nh_strata)

    # Should handle this correctly (this is the main use case for Stehman)
    assert result["OA"] == pytest.approx(1.0, abs=1e-6)  # Perfect classification

    # Forest appears in multiple strata - should aggregate correctly
    assert "forest" in result["UA"].index
    assert "forest" in result["PA"].index
    assert result["UA"]["forest"] == pytest.approx(1.0, abs=1e-6)
    assert result["PA"]["forest"] == pytest.approx(1.0, abs=1e-6)

    # Area estimates should reflect aggregation across strata
    assert result["area"]["forest"] > result["area"]["desert"]  # More forest samples


def test_stehman2014_confidence_intervals_structure():
    """Test that confidence intervals are properly structured and contain expected keys."""
    s = ["region_A", "region_A", "region_B", "region_B"]
    r = ["forest", "forest", "water", "water"]
    m = ["forest", "forest", "water", "water"]
    Nh_strata = {"region_A": 100, "region_B": 100}

    result = stehman2014(s, r, m, Nh_strata)

    # Check that all CI keys are present
    ci_keys = [
        "CIoa",
        "CIua",
        "CIpa",
        "CIa",
        "CI_halfwidth_oa",
        "CI_halfwidth_ua",
        "CI_halfwidth_pa",
        "CI_halfwidth_a",
    ]
    for key in ci_keys:
        assert key in result, f"Missing confidence interval key: {key}"

    # Check basic structure
    assert isinstance(result["CIoa"], tuple) and len(result["CIoa"]) == 2
    assert isinstance(result["CIua"], tuple) and len(result["CIua"]) == 2
    assert isinstance(result["CIpa"], tuple) and len(result["CIpa"]) == 2
    assert isinstance(result["CIa"], tuple) and len(result["CIa"]) == 2

    # Check that bounds are reasonable (lower ≤ estimate ≤ upper)
    oa_lower, oa_upper = result["CIoa"]
    assert oa_lower <= result["OA"] <= oa_upper

    # Check basic z-score relationship (half-width ≈ 1.96 * SE for 95% CI)
    expected_halfwidth = 1.96 * result["SEoa"]
    assert result["CI_halfwidth_oa"] == pytest.approx(expected_halfwidth, rel=0.01)


def test_stehman2014_confidence_intervals_perfect_classification():
    """Test confidence intervals for perfect classification case."""
    s = ["admin_A", "admin_A", "admin_B", "admin_B"]
    r = ["forest", "forest", "water", "water"]
    m = ["forest", "forest", "water", "water"]  # Perfect classification
    Nh_strata = {"admin_A": 100, "admin_B": 100}

    result = stehman2014(s, r, m, Nh_strata)

    # Perfect classification should have zero standard errors and zero-width CIs
    assert result["OA"] == pytest.approx(1.0)
    assert result["SEoa"] == pytest.approx(0.0, abs=1e-10)
    assert result["CI_halfwidth_oa"] == pytest.approx(0.0, abs=1e-10)

    oa_lower, oa_upper = result["CIoa"]
    assert oa_lower == pytest.approx(1.0, abs=1e-10)
    assert oa_upper == pytest.approx(1.0, abs=1e-10)
