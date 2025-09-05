import warnings
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm


def stehman2014(
    s: list[Any] | pd.Series,
    r: list[Any] | pd.Series,
    m: list[Any] | pd.Series,
    Nh_strata: dict[Any, int | float],
    margins: bool = True,
    order: list[Any] | None = None,
) -> dict[str, Any]:
    """
    Estimate thematic map accuracy and area under stratified random sampling where
    sampling strata differ from map classes.

    This function implements the statistical estimators described in Stehman (2014) for
    calculating overall accuracy, user's accuracy, producer's accuracy, and area
    proportions when the sampling design is stratified by criteria different from the
    map legend (e.g., geographic regions, administrative boundaries, elevation zones).

    The estimators account for the complex sampling design and provide unbiased estimates
    with appropriate standard errors and confidence intervals.

    Parameters
    ----------
    s : list or pd.Series
        Stratum labels for each sample unit. These represent the sampling strata
        (e.g., 'north_region', 'south_region') and must correspond to keys in Nh_strata.
    r : list or pd.Series
        Reference class labels for each sample unit. These are the "ground truth"
        land cover classes determined from field data or high-resolution imagery.
    m : list or pd.Series
        Map class labels for each sample unit. These are the predicted classes
        from the thematic map being assessed.
    Nh_strata : dict
        Population size (area) of each sampling stratum. Keys must be stratum labels
        matching values in 's', values are the pixel counts or area measurements
        for each stratum (e.g., {'north_region': 10000, 'south_region': 8000}).
    margins : bool, default True
        Whether to include sum margins (row/column totals) in the returned confusion matrix.
    order : list, optional
        Explicit ordering for classes in output matrices and series. If None,
        uses alphabetical ordering of all unique classes found in r and m.

    Returns
    -------
    dict
        Comprehensive results dictionary containing:

        - 'OA' : float
            Overall accuracy (proportion of correctly classified units)
        - 'UA' : pd.Series
            User's accuracy by map class (complement of commission error)
        - 'PA' : pd.Series
            Producer's accuracy by reference class (complement of omission error)
        - 'area' : pd.Series
            Estimated area proportion by reference class
        - 'SEoa' : float
            Standard error of overall accuracy
        - 'SEua' : pd.Series
            Standard errors of user's accuracies
        - 'SEpa' : pd.Series
            Standard errors of producer's accuracies
        - 'SEa' : pd.Series
            Standard errors of area estimates
        - 'CIoa' : tuple
            95% confidence interval for overall accuracy (lower, upper)
        - 'CIua' : tuple
            95% confidence intervals for user's accuracies (lower_series, upper_series)
        - 'CIpa' : tuple
            95% confidence intervals for producer's accuracies (lower_series, upper_series)
        - 'CIa' : tuple
            95% confidence intervals for area estimates (lower_series, upper_series)
        - 'CI_halfwidth_oa' : float
            Half-width of confidence interval for overall accuracy
        - 'CI_halfwidth_ua' : pd.Series
            Half-widths of confidence intervals for user's accuracies
        - 'CI_halfwidth_pa' : pd.Series
            Half-widths of confidence intervals for producer's accuracies
        - 'CI_halfwidth_a' : pd.Series
            Half-widths of confidence intervals for area estimates
        - 'matrix' : pd.DataFrame
            Area-weighted error matrix (rows=map classes, columns=reference classes)

    Raises
    ------
    ValueError
        If input vectors have different lengths, if stratum labels in 's' are not
        found in Nh_strata keys, if Nh_strata contains invalid values, or if
        required classes are missing from the 'order' parameter.

    Warns
    -----
    UserWarning
        When strata have area defined but no samples collected (unused strata),
        when strata contain only single observations, or when inclusion
        probabilities approach problematic values.

    Notes
    -----
    This estimator is appropriate when:

    - Sampling strata differ from map classes (e.g., geographic stratification)
    - Complex stratified sampling designs are used
    - Different sampling intensities across strata need to be accounted for

    The method uses inclusion probabilities and stratified estimators to provide
    design-unbiased estimates. Standard errors account for both within-stratum
    and between-stratum variability.

    References
    ----------
    Stehman, S.V. (2014). Estimating area and map accuracy for stratified random
    sampling when the strata are different from the map classes. International
    Journal of Remote Sensing, 35(13), 4923-4939. DOI: 10.1080/01431161.2014.930207

    Examples
    --------
    >>> import pandas as pd
    >>> from pymapaccuracy import stehman2014
    >>>
    >>> # Administrative regions as strata, land cover as classes
    >>> strata = ['region_A', 'region_A', 'region_B', 'region_B']
    >>> reference = ['forest', 'grassland', 'water', 'forest']
    >>> map_pred = ['forest', 'forest', 'water', 'grassland']
    >>> areas = {'region_A': 10000, 'region_B': 8000}
    >>>
    >>> results = stehman2014(strata, reference, map_pred, areas)
    >>> print(f"Overall Accuracy: {results['OA']:.3f}")
    >>> print(results['area'])  # Area estimates by reference class
    """

    # =========================================================================
    # INPUT VALIDATION AND PREPROCESSING
    # =========================================================================

    # Validate stratum area dictionary structure and content
    if not isinstance(Nh_strata, dict):
        raise ValueError(
            "Nh_strata must be a dictionary with stratum labels as keys and areas as values."
        )
    if not all(isinstance(k, str | int | float) for k in Nh_strata.keys()):
        raise ValueError(
            "Nh_strata keys must be valid stratum labels (str, int, or float)."
        )
    if not all(isinstance(v, int | float) and v >= 0 for v in Nh_strata.values()):
        raise ValueError(
            "Nh_strata values must be non-negative numbers representing stratum areas."
        )

    # Standardize input data as pandas Series for consistent indexing and operations
    s = pd.Series(s, name="stratum")
    r = pd.Series(r, name="reference")
    m = pd.Series(m, name="map")

    # Ensure all input vectors represent the same sample units (equal length requirement)
    lengths = [len(s), len(r), len(m)]
    if len(set(lengths)) > 1:
        raise ValueError(
            f"Input vectors must have the same length. Got: s={len(s)}, r={len(r)}, m={len(m)} elements. "
            f"Each element at position i should represent the same sample unit across all vectors."
        )

    # Extract unique identifiers for validation
    unique_s = s.unique()
    strata_keys = list(Nh_strata.keys())

    # Verify that all observed strata have corresponding area information
    if not set(unique_s).issubset(set(strata_keys)):
        missing_strata = set(unique_s) - set(strata_keys)
        raise ValueError(
            f"Stratum labels in 's' not found in Nh_strata keys: {missing_strata}. "
            f"All strata in sample data must have corresponding area values in Nh_strata."
        )

    # Issue informative warning for strata with defined area but no sample observations
    unused_strata = set(strata_keys) - set(unique_s)
    if unused_strata:
        warnings.warn(
            f"Strata in Nh_strata not found in sample data 's': {'; '.join(map(str, sorted(unused_strata)))}. "
            f"These strata have area defined but no samples collected and will be ignored in calculations.",
            UserWarning,
            stacklevel=2,
        )

    # Filter stratum areas to include only strata with sample observations
    Nh_strata_filt = {k: v for k, v in Nh_strata.items() if k in unique_s}
    strata_levels = list(
        Nh_strata_filt.keys()
    )  # Preserve stratum ordering for consistent indexing
    N = sum(Nh_strata_filt.values())  # Total population size across all sampled strata

    # Validate that total population size is meaningful
    if N == 0:
        raise ValueError(
            "Total population size (sum of Nh_strata values) cannot be zero. "
            "Check that your stratum area values are positive numbers."
        )

    # =========================================================================
    # CLASS ORDERING AND CATEGORICAL DATA PREPARATION
    # =========================================================================

    # Establish consistent class ordering for all output structures
    unique_rm = sorted(set(r.unique()) | set(m.unique()))
    if order is None:
        order = unique_rm  # Default to alphabetical ordering
    else:
        order = list(order)  # Ensure mutable list type
        missing_in_order = set(unique_rm) - set(order)
        if missing_in_order:
            raise ValueError(
                f"Argument 'order' must include all class labels found in reference and map data. "
                f"Missing: {'; '.join(map(str, sorted(missing_in_order)))}. "
                f"Found classes: {'; '.join(map(str, sorted(unique_rm)))}"
            )

    # Convert to categorical data with explicit factor levels for consistent cross-tabulation
    s_cat = pd.Categorical(s, categories=strata_levels)
    r_cat = pd.Categorical(r, categories=order)
    m_cat = pd.Categorical(m, categories=order)

    # Establish class dimensions for matrix operations
    map_classes = order
    ref_classes = order
    n_map_classes = len(map_classes)
    n_ref_classes = len(ref_classes)

    # =========================================================================
    # STATISTICAL ESTIMATION FUNCTIONS (Stehman 2014 Equations)
    # =========================================================================

    def eq2(yu_col: str) -> float:
        """
        Estimate population proportion using stratified sampling estimator.

        Implements Equation 2 from Stehman (2014): weighted average using
        inclusion probabilities to account for unequal sampling across strata.
        """
        return float((sample[yu_col] / sample["incl"]).sum() / N)

    def eq25(
        Nh_dict: dict[Any, float], nh_series: pd.Series, var_series: pd.Series
    ) -> float:
        """
        Variance estimator for population totals under stratified sampling.

        Implements Equation 25 from Stehman (2014) with finite population
        correction for improved precision in small populations.
        """
        term = 0
        for h in strata_levels:  # Iterate through strata names
            Nh = Nh_dict[h]
            nh = nh_series.get(h, 0)
            var_h = var_series.get(
                h, 0
            )  # Get variance for stratum h, default 0 if missing
            if nh > 0:  # Avoid division by zero
                # Finite population correction (1 - nh/Nh)
                fpc = (
                    (1 - nh / Nh) if Nh > 0 else 1
                )  # If Nh is 0, stratum shouldn't exist, but handle defensively
                term += (Nh**2 * fpc * var_h) / nh
        return term / (N**2)

    def eq28(
        R_ratio: float,
        X_est: float,
        Nh_dict: dict[Any, float],
        nh_series: pd.Series,
        vary_series: pd.Series,
        varx_series: pd.Series,
        cov_series: pd.Series,
    ) -> float:
        # Variance estimator for ratios (UA, PA)
        term = 0
        for h in strata_levels:
            Nh = Nh_dict[h]
            nh = nh_series.get(h, 0)
            vary_h = vary_series.get(h, 0)
            varx_h = varx_series.get(h, 0)
            cov_h = cov_series.get(h, 0)
            if nh > 0:
                fpc = (1 - nh / Nh) if Nh > 0 else 1
                variance_term = vary_h + (R_ratio**2 * varx_h) - (2 * R_ratio * cov_h)
                term += (Nh**2 * fpc * variance_term) / nh
        # Avoid division by zero if X_est is 0 (denominator of ratio)
        return term / (X_est**2) if X_est != 0 else np.nan

    # --- Inclusion Probabilities ---
    nh_series = s.value_counts().reindex(
        strata_levels, fill_value=0
    )  # Counts per stratum, ordered like Nh_strata_filt
    Nh_series = pd.Series(Nh_strata_filt).reindex(
        strata_levels, fill_value=0
    )  # Population per stratum, ordered

    if (nh_series < 2).any():
        single_obs_strata = nh_series[nh_series < 2].index.tolist()
        warnings.warn(
            f"The following strata include only one observation: {'; '.join(map(str, single_obs_strata))}. "
            f"Variance estimates for these strata will be zero, which may affect standard error calculations. "
            f"Consider increasing sample size in these strata if possible.",
            stacklevel=2,
        )

    # Calculate inclusion probability for each stratum
    # Inclusion probability = nh / Nh (samples in stratum / total area of stratum)
    with np.errstate(divide="ignore", invalid="ignore"):
        incl_prob_map = nh_series / Nh_series

    # Map inclusion probability to each sample unit
    incl = s.map(incl_prob_map)

    # Check for problematic inclusion probabilities
    # If a sample exists but has invalid inclusion probability, this indicates data inconsistency
    problematic_incl = incl.isna() | (incl <= 0) | ~np.isfinite(incl)
    if problematic_incl.any():
        problematic_strata = s[problematic_incl].unique()
        raise ValueError(
            f"Invalid inclusion probabilities detected for samples in strata: {list(problematic_strata)}. "
            f"This occurs when stratum areas (Nh_strata) are zero or negative, or when there's a "
            f"mismatch between sample strata and Nh_strata keys. All samples must come from strata "
            f"with positive areas."
        )

    # --- CASE 1: Area, Overall Accuracy, Error Matrix Cell Proportions ---
    sample = pd.DataFrame({"s": s_cat, "m": m_cat, "r": r_cat, "incl": incl})

    # yu for overall accuracy (eq 12)
    sample["yu_O"] = (sample["m"] == sample["r"]).astype(float)

    # yu for proportion of area of reference classes (eq 14)
    yu_r_cols = []
    for j, ref_class in enumerate(ref_classes):
        col_name = f"yu_r{j}"
        sample[col_name] = (sample["r"] == ref_class).astype(float)
        yu_r_cols.append(col_name)

    # yu for proportion of area with map class i and reference class j (eq 16)
    yu_m_r_cols: dict[Any, dict[Any, str]] = {}  # Store column names for matrix
    for i, map_class in enumerate(map_classes):
        yu_m_r_cols[map_class] = {}
        for j, ref_class in enumerate(ref_classes):
            col_name = f"yu_m{i}r{j}"
            sample[col_name] = (
                (sample["m"] == map_class) & (sample["r"] == ref_class)
            ).astype(float)
            yu_m_r_cols[map_class][ref_class] = col_name

    # Calculate Estimates using eq2
    OA = eq2("yu_O")

    Area_list = [eq2(f"yu_r{j}") for j in range(n_ref_classes)]
    Area = pd.Series(Area_list, index=ref_classes, name="Area")

    M_matrix = np.full((n_map_classes, n_ref_classes), np.nan)
    for i, map_class in enumerate(map_classes):
        for j, ref_class in enumerate(ref_classes):
            col_name = yu_m_r_cols[map_class][ref_class]
            prop = eq2(col_name)
            # Stehman uses NA for 0 proportion, we use np.nan
            M_matrix[i, j] = prop if prop > 0 else np.nan  # Match R code's NA for 0

    # Calculate Variances and Standard Errors (SE)
    # Aggregate mean and variance per stratum (h)
    # Need ddof=1 for unbiased sample variance (like R's var)
    # Group by the original stratum labels 's', not the categorical 's_cat' if levels differ
    grouped = sample.groupby(
        "s", observed=False
    )  # observed=False includes strata with 0 samples if they were in levels
    tmean = grouped.mean(numeric_only=True)
    tvar = grouped.var(numeric_only=True, ddof=1).fillna(
        0
    )  # Replace NaN variance (e.g., single sample strata) with 0

    # Reindex tvar and tmean to ensure all strata_levels are present, fill missing with 0
    tvar = tvar.reindex(strata_levels, fill_value=0)
    tmean = tmean.reindex(strata_levels, fill_value=0)

    # SE for Overall Accuracy (OA)
    Vo = eq25(Nh_strata_filt, nh_series, tvar["yu_O"])
    SEoa = np.sqrt(Vo) if Vo >= 0 else np.nan

    # SE for Area Proportions (A)
    Va_list = [
        eq25(Nh_strata_filt, nh_series, tvar[f"yu_r{j}"]) for j in range(n_ref_classes)
    ]
    SEa = pd.Series(
        np.sqrt(np.maximum(Va_list, 0)), index=ref_classes, name="SE_Area"
    )  # Use maximum to avoid sqrt(<0)

    # SE for Matrix Cells (Optional, not explicitly returned but calculated in R code)
    # Vm_matrix = np.zeros((n_map_classes, n_ref_classes))
    # for i, map_class in enumerate(map_classes):
    #     for j, ref_class in enumerate(ref_classes):
    #         col_name = yu_m_r_cols[map_class][ref_class]
    #         Vm_matrix[i, j] = eq25(Nh_strata_filt, nh_series, tvar[col_name])
    # SEm = np.sqrt(np.maximum(Vm_matrix, 0))

    # --- CASE 2: User's Accuracy (UA) and Producer's Accuracy (PA) ---

    # yu and xu for User's Accuracy (UA)
    yu_U_cols = []
    xu_U_cols = []
    for i, map_class in enumerate(map_classes):
        yu_col = f"yu_U{i}"
        xu_col = f"xu_U{i}"
        sample[yu_col] = (
            (sample["m"] == map_class) & (sample["m"] == sample["r"])
        ).astype(float)
        sample[xu_col] = (sample["m"] == map_class).astype(float)
        yu_U_cols.append(yu_col)
        xu_U_cols.append(xu_col)

    # yu and xu for Producer's Accuracy (PA)
    yu_P_cols = []
    xu_P_cols = []
    for j, ref_class in enumerate(ref_classes):
        yu_col = f"yu_P{j}"
        xu_col = f"xu_P{j}"
        sample[yu_col] = (
            (sample["r"] == ref_class) & (sample["r"] == sample["m"])
        ).astype(float)
        sample[xu_col] = (sample["r"] == ref_class).astype(float)
        yu_P_cols.append(yu_col)
        xu_P_cols.append(xu_col)

    # Recalculate aggregates including new columns
    grouped = sample.groupby("s", observed=False)
    tmean = grouped.mean(numeric_only=True).reindex(strata_levels, fill_value=0)
    tvar = (
        grouped.var(numeric_only=True, ddof=1)
        .fillna(0)
        .reindex(strata_levels, fill_value=0)
    )

    # Calculate UA estimates (Ratio R = Y_est / X_est)
    UA_list = []
    X_est_UA_list = []
    for i in range(n_map_classes):
        yu_col = yu_U_cols[i]
        xu_col = xu_U_cols[i]
        # Estimate totals Y and X using stratum means and population sizes
        Y_est = (tmean[yu_col] * Nh_series).sum()
        X_est = (tmean[xu_col] * Nh_series).sum()
        X_est_UA_list.append(X_est)
        with np.errstate(divide="ignore", invalid="ignore"):
            UA_list.append(Y_est / X_est if X_est != 0 else np.nan)
    UA = pd.Series(UA_list, index=map_classes, name="UA")

    # Calculate PA estimates
    PA_list = []
    X_est_PA_list = []
    for j in range(n_ref_classes):
        yu_col = yu_P_cols[j]
        xu_col = xu_P_cols[j]
        Y_est = (tmean[yu_col] * Nh_series).sum()
        X_est = (tmean[xu_col] * Nh_series).sum()
        X_est_PA_list.append(X_est)
        with np.errstate(divide="ignore", invalid="ignore"):
            PA_list.append(Y_est / X_est if X_est != 0 else np.nan)
    PA = pd.Series(PA_list, index=ref_classes, name="PA")

    # Calculate Covariances within strata needed for SE of UA and PA
    # This requires calculating cov(yu, xu) for each stratum h
    cov_U_dict = {}  # {stratum: {class_i: cov(yu_Ui, xu_Ui)}}
    cov_P_dict = {}  # {stratum: {class_j: cov(yu_Pj, xu_Pj)}}

    for h, group_df in sample.groupby("s", observed=False):
        if len(group_df) < 2:  # Covariance is NaN for < 2 samples
            cov_U_dict[h] = {i: 0.0 for i in range(n_map_classes)}
            cov_P_dict[h] = {j: 0.0 for j in range(n_ref_classes)}
            continue

        cov_U_dict[h] = {}
        for i in range(n_map_classes):
            # Ensure columns exist before calculating covariance
            if yu_U_cols[i] in group_df and xu_U_cols[i] in group_df:
                cov_val = group_df[yu_U_cols[i]].cov(group_df[xu_U_cols[i]], ddof=1)
                cov_U_dict[h][i] = cov_val if pd.notna(cov_val) else 0.0
            else:
                cov_U_dict[h][
                    i
                ] = 0.0  # Should not happen if columns were added correctly

        cov_P_dict[h] = {}
        for j in range(n_ref_classes):
            if yu_P_cols[j] in group_df and xu_P_cols[j] in group_df:
                cov_val = group_df[yu_P_cols[j]].cov(group_df[xu_P_cols[j]], ddof=1)
                cov_P_dict[h][j] = cov_val if pd.notna(cov_val) else 0.0
            else:
                cov_P_dict[h][j] = 0.0

    # Structure covariances similarly to variances (Series indexed by stratum)
    cov_U_series_list = []
    for i in range(n_map_classes):
        cov_series = pd.Series(
            {h: cov_U_dict.get(h, {}).get(i, 0.0) for h in strata_levels},
            name=f"cov_U{i}",
        )
        cov_U_series_list.append(cov_series)

    cov_P_series_list = []
    for j in range(n_ref_classes):
        cov_series = pd.Series(
            {h: cov_P_dict.get(h, {}).get(j, 0.0) for h in strata_levels},
            name=f"cov_P{j}",
        )
        cov_P_series_list.append(cov_series)

    # SE for User's Accuracy (UA) using eq28
    Vua_list = []
    for i in range(n_map_classes):
        R_ratio = UA.iloc[i]
        X_est = X_est_UA_list[i]
        if (
            pd.isna(R_ratio) or X_est == 0
        ):  # If UA is NaN or denominator is 0, SE is NaN
            Vua_list.append(np.nan)
            continue
        vary_series = tvar[yu_U_cols[i]]
        varx_series = tvar[xu_U_cols[i]]
        cov_series = cov_U_series_list[i]
        Vua = eq28(
            R_ratio,
            X_est,
            Nh_strata_filt,
            nh_series,
            vary_series,
            varx_series,
            cov_series,
        )
        Vua_list.append(Vua)
    SEua = pd.Series(
        np.sqrt(np.maximum(Vua_list, 0)), index=map_classes, name="SE_UA"
    )  # Use maximum to avoid sqrt(<0)

    # SE for Producer's Accuracy (PA) using eq28
    Vpa_list = []
    for j in range(n_ref_classes):
        R_ratio = PA.iloc[j]
        X_est = X_est_PA_list[j]
        if (
            pd.isna(R_ratio) or X_est == 0
        ):  # If PA is NaN or denominator is 0, SE is NaN
            Vpa_list.append(np.nan)
            continue
        vary_series = tvar[yu_P_cols[j]]
        varx_series = tvar[xu_P_cols[j]]
        cov_series = cov_P_series_list[j]
        Vpa = eq28(
            R_ratio,
            X_est,
            Nh_strata_filt,
            nh_series,
            vary_series,
            varx_series,
            cov_series,
        )
        Vpa_list.append(Vpa)
    SEpa = pd.Series(
        np.sqrt(np.maximum(Vpa_list, 0)), index=ref_classes, name="SE_PA"
    )  # Use maximum to avoid sqrt(<0)

    # --- Final Output Preparation ---
    M_df = pd.DataFrame(M_matrix, index=map_classes, columns=ref_classes)

    if margins:
        # Calculate sums, replacing NaN with 0 for summation
        row_sum = M_df.sum(axis=1, skipna=True)
        col_sum = M_df.sum(axis=0, skipna=True)
        total_sum = col_sum.sum()  # Sum of column sums

        # Add row sums as a new column
        M_df["sum"] = row_sum
        # Add column sums as a new row
        col_sum["sum"] = total_sum  # Add the total sum to the sum series
        M_df.loc["sum"] = col_sum

    # Add two-sided confidence intervals for OA, UA, PA, Area
    # Note: R code uses qnorm(0.975) for 95% CI, but we can use scipy.stats.norm.ppf(0.975)
    z_alpha = norm.ppf(0.975)

    # Confidence Interval half-widths
    CI_halfwidth_oa = z_alpha * SEoa
    CI_halfwidth_ua = z_alpha * SEua
    CI_halfwidth_pa = z_alpha * SEpa
    CI_halfwidth_a = z_alpha * SEa

    # Confidence Intervals (lower, upper)
    OA_CI = (OA - CI_halfwidth_oa, OA + CI_halfwidth_oa)
    UA_CI = (UA - CI_halfwidth_ua, UA + CI_halfwidth_ua)
    PA_CI = (PA - CI_halfwidth_pa, PA + CI_halfwidth_pa)
    Area_CI = (Area - CI_halfwidth_a, Area + CI_halfwidth_a)

    return {
        "OA": OA,
        "UA": UA,
        "PA": PA,
        "area": Area,
        "SEoa": SEoa,
        "SEua": SEua,
        "SEpa": SEpa,
        "SEa": SEa,
        "CIoa": OA_CI,
        "CIua": UA_CI,
        "CIpa": PA_CI,
        "CIa": Area_CI,
        "CI_halfwidth_oa": CI_halfwidth_oa,
        "CI_halfwidth_ua": CI_halfwidth_ua,
        "CI_halfwidth_pa": CI_halfwidth_pa,
        "CI_halfwidth_a": CI_halfwidth_a,
        "matrix": M_df,
    }


def olofsson(
    r: list[Any] | pd.Series,
    m: list[Any] | pd.Series,
    Nh: dict[Any, int | float],
    margins: bool = True,
) -> dict[str, Any]:
    """
    Estimate thematic map accuracy and area using stratified sampling where
    map classes serve as sampling strata.

    This function implements the statistical estimators described in Olofsson et al. (2014)
    for calculating overall accuracy, user's accuracy, producer's accuracy, and area
    proportions when sampling is stratified directly by the map legend classes. This is
    the standard approach for map accuracy assessment in remote sensing.

    The estimators provide design-unbiased estimates when samples are allocated
    proportionally or optimally across map class strata, with appropriate standard
    errors and confidence intervals.

    Parameters
    ----------
    r : list or pd.Series
        Reference class labels for each sample unit. These represent the "ground truth"
        land cover classes determined from field validation or high-resolution imagery.
    m : list or pd.Series
        Map class labels (strata) for each sample unit. These are both the predicted
        classes from the thematic map and the sampling strata used for sample allocation.
    Nh : dict
        Population size (area) of each map class stratum. Keys must be map class labels
        matching values in 'm', values are pixel counts or area measurements for each
        map class (e.g., {'forest': 15000, 'water': 3000, 'urban': 5000}).
    margins : bool, default True
        Whether to include sum margins (row/column totals) in the returned confusion matrix.

    Returns
    -------
    dict
        Comprehensive results dictionary containing:

        - 'OA' : float
            Overall accuracy (proportion of correctly classified units)
        - 'UA' : pd.Series
            User's accuracy by map class (1 - commission error rate)
        - 'PA' : pd.Series
            Producer's accuracy by reference class (1 - omission error rate)
        - 'area' : pd.Series
            Estimated area proportion by reference class
        - 'SEoa' : float
            Standard error of overall accuracy
        - 'SEua' : pd.Series
            Standard errors of user's accuracies
        - 'SEpa' : pd.Series
            Standard errors of producer's accuracies
        - 'SEa' : pd.Series
            Standard errors of area estimates
        - 'CIoa' : tuple
            95% confidence interval for overall accuracy (lower, upper)
        - 'CIua' : tuple
            95% confidence intervals for user's accuracies (lower_series, upper_series)
        - 'CIpa' : tuple
            95% confidence intervals for producer's accuracies (lower_series, upper_series)
        - 'CIa' : tuple
            95% confidence intervals for area estimates (lower_series, upper_series)
        - 'CI_halfwidth_oa' : float
            Half-width of confidence interval for overall accuracy
        - 'CI_halfwidth_ua' : pd.Series
            Half-widths of confidence intervals for user's accuracies
        - 'CI_halfwidth_pa' : pd.Series
            Half-widths of confidence intervals for producer's accuracies
        - 'CI_halfwidth_a' : pd.Series
            Half-widths of confidence intervals for area estimates
        - 'matrix' : pd.DataFrame
            Area-weighted error matrix (rows=map classes, columns=reference classes)

    Raises
    ------
    ValueError
        If input vectors have different lengths, if map class labels in 'm' are not
        found in Nh keys, if Nh contains invalid values, or if duplicate map class
        labels are detected.

    Warns
    -----
    UserWarning
        When reference classes are not found in map strata, when map classes are
        not found in reference data, or when map strata have no sample observations.

    Notes
    -----
    This estimator is appropriate when:

    - Map classes serve as sampling strata (standard stratified design)
    - Simple random sampling within each map class stratum
    - Proportional or optimal allocation across strata

    The method uses standard stratified sampling estimators and is equivalent to
    the approach recommended by Olofsson et al. (2014) for good practices in
    land change accuracy assessment.

    References
    ----------
    Olofsson, P., Foody, G.M., Stehman, S.V., & Woodcock, C.E. (2013). Making better
    use of accuracy data in land change studies: Estimating accuracy and area and
    quantifying uncertainty using stratified estimation. Remote Sensing of Environment,
    129, 122-131. https://doi.org/10.1016/j.rse.2012.10.031

    Olofsson, P., Foody, G.M., Herold, M., Stehman, S.V., Woodcock, C.E., &
    Wulder, M.A. (2014). Good practices for estimating area and assessing
    accuracy of land change. Remote Sensing of Environment, 148, 42-57.
    https://doi.org/10.1016/j.rse.2014.02.015

    Examples
    --------
    >>> import pandas as pd
    >>> from pymapaccuracy import olofsson
    >>>
    >>> # Map classes serve as sampling strata
    >>> reference = ['forest', 'forest', 'water', 'grassland', 'urban']
    >>> map_pred = ['forest', 'grassland', 'water', 'grassland', 'urban']
    >>> areas = {'forest': 15000, 'grassland': 8000, 'water': 3000, 'urban': 4000}
    >>>
    >>> results = olofsson(reference, map_pred, areas)
    >>> print(f"Overall Accuracy: {results['OA']:.3f}")
    >>> print(results['area'])  # Area estimates by reference class
    """

    # =========================================================================
    # INPUT VALIDATION AND PREPROCESSING
    # =========================================================================
    # --- Input Validation and Preparation ---
    if not isinstance(Nh, dict):
        raise ValueError(
            "Nh must be a dictionary with map class labels as keys and areas as values."
        )
    if not all(isinstance(k, str | int | float) for k in Nh.keys()):
        raise ValueError(
            "Nh keys (map class labels) must be valid labels (str, int, or float)."
        )
    if not all(isinstance(v, int | float) and v >= 0 for v in Nh.values()):
        raise ValueError("Nh values (map stratum areas) must be non-negative numbers.")
    if len(Nh) != len(set(Nh.keys())):
        raise ValueError(
            "Repeated map class names detected in Nh. Each map class must appear only once."
        )

    # Convert to Series for consistent handling
    r = pd.Series(r)
    m = pd.Series(m)

    # Check length consistency (like R's .check_length function)
    lengths = [len(r), len(m)]
    if len(set(lengths)) > 1:
        raise ValueError(
            f"Input vectors must have the same length. Got: r={len(r)}, m={len(m)} elements. "
            f"Each element at position i should represent the same sample unit across both vectors."
        )

    map_classes_nh = list(Nh.keys())
    unique_m = m.unique()

    # Check if all map classes in the sample 'm' are present in Nh
    missing_map_strata = set(unique_m) - set(map_classes_nh)
    if missing_map_strata:
        raise ValueError(
            f"Map class labels in sample data 'm' not found in Nh keys: {missing_map_strata}. "
            f"All map classes in sample data must have corresponding area values in Nh."
        )

    # Determine final class order: Start with Nh keys, add any extra reference classes
    ref_classes_unique = r.unique()
    extra_ref_classes = list(set(ref_classes_unique) - set(map_classes_nh))
    class_order = map_classes_nh + extra_ref_classes
    print("class order ", class_order)

    # Issue warnings for data mismatches (but don't alter behavior)
    if extra_ref_classes:
        warnings.warn(
            f"Reference classes not found in map strata (Nh): {extra_ref_classes}. "
            f"These classes will have zero estimated area but PA/SEpa may still be calculated.",
            UserWarning,
            stacklevel=2,
        )

    extra_map_classes = list(set(unique_m) - set(ref_classes_unique))
    if extra_map_classes:
        warnings.warn(
            f"Map classes not found in reference data: {extra_map_classes}. "
            f"These classes will have UA calculated but PA will be undefined.",
            UserWarning,
            stacklevel=2,
        )

    unused_nh_classes = list(set(map_classes_nh) - set(unique_m))
    if unused_nh_classes:
        warnings.warn(
            f"Map strata in Nh not found in sample data 'm': {unused_nh_classes}. "
            f"These strata have no samples and will be ignored in calculations.",
            UserWarning,
            stacklevel=2,
        )

    # Add extra reference classes to Nh with 0 area for consistent indexing
    Nh_full = Nh.copy()
    for cls in extra_ref_classes:
        Nh_full[cls] = 0

    # Convert to categorical with consistent levels based on the full class order
    r_cat = pd.Categorical(r, categories=class_order)
    m_cat = pd.Categorical(m, categories=map_classes_nh)  # Strata are only map classes

    # --- Calculations ---
    # Raw count confusion matrix (rows=map strata, columns=reference classes)
    matrix = pd.crosstab(m_cat, r_cat, dropna=False)  # Ensure all levels are kept

    # Sampling settings
    A = sum(Nh.values())  # Total area from Nh (map strata only)
    if A == 0:
        raise ValueError("Total area (sum of Nh) cannot be zero.")

    # Ensure Nh_series and Wh align with the matrix index (map classes)
    Nh_series = pd.Series(Nh).reindex(map_classes_nh)
    Wh = Nh_series / A  # Map class proportions (stratum weights)

    # Stratum sample sizes (number of samples in each map class stratum)
    nh = matrix.sum(axis=1).reindex(map_classes_nh, fill_value=0)

    # Check for strata with fewer than 2 samples for variance calculation warnings
    if (nh < 2).any():
        single_obs_strata = nh[nh < 2].index.tolist()
        warnings.warn(
            f"The following strata include only one observation: {'; '.join(map(str, single_obs_strata))}. Variance estimates for these strata might be zero or unstable.",
            stacklevel=2,
        )

    # Estimated area proportions matrix (p_ij)
    # Calculate cell proportions within each stratum (row)
    # Handle division by zero if nh is 0 for a stratum
    with np.errstate(divide="ignore", invalid="ignore"):
        stratum_props = matrix.div(nh, axis=0)
    stratum_props = stratum_props.fillna(0)  # If nh=0, proportion is 0

    # Multiply by stratum weights (Wh)
    props = stratum_props.mul(Wh, axis=0)
    # Ensure props DataFrame has all classes in columns, filling missing with 0
    props = props.reindex(columns=class_order, fill_value=0)

    # --- Accuracy and Area Estimates ---
    diag_props = np.diag(
        props.reindex(index=class_order, columns=class_order, fill_value=0)
    )  # Ensure square matrix for diag

    OA = diag_props.sum()  # Overall accuracy is a single value

    # User's Accuracy (UA) - indexed by map class
    map_totals_est = props.sum(axis=1).reindex(
        map_classes_nh, fill_value=0
    )  # Estimated proportion for each map class
    ua_diag = np.diag(
        props.reindex(index=map_classes_nh, columns=map_classes_nh, fill_value=0)
    )  # Diagonal elements corresponding to map classes

    with np.errstate(divide="ignore", invalid="ignore"):
        UA = pd.Series(
            ua_diag / map_totals_est, index=map_classes_nh, name="UA", dtype=float
        )
    UA = UA.fillna(0.0)  # If map_totals_est is 0, UA is 0 or NaN -> 0

    # Producer's Accuracy (PA) - indexed by reference class
    ref_totals_est = props.sum(axis=0).reindex(
        class_order, fill_value=0
    )  # Estimated area proportion for each reference class (Area estimate)
    pa_diag = np.diag(
        props.reindex(index=class_order, columns=class_order, fill_value=0)
    )  # Diagonal elements corresponding to all classes

    # Check which reference classes actually have samples in the reference data
    ref_sample_counts = (
        pd.Series(r_cat).value_counts().reindex(class_order, fill_value=0)
    )

    with np.errstate(divide="ignore", invalid="ignore"):
        PA = pd.Series(pa_diag / ref_totals_est, index=class_order, name="PA")

    # Set PA to NaN for classes with no reference samples, keep calculated values for classes with samples
    PA = PA.where(ref_sample_counts > 0, np.nan)

    # Area estimates (same as ref_totals_est)
    Area = ref_totals_est.rename("Area")

    # --- Standard Error Calculations ---

    # Standard Error of OA
    # var_oa_terms = Wh**2 * UA * (1 - UA) / (nh - 1) # R code version
    # Need UA aligned with Wh and nh (map classes)
    ua_for_var = UA.reindex(map_classes_nh).fillna(0)
    nh_safe = nh.replace(1, np.inf).replace(
        0, np.inf
    )  # Avoid division by zero or negative variance for nh=1
    var_oa_terms = Wh**2 * ua_for_var * (1 - ua_for_var) / (nh_safe - 1)
    VAR_OA = var_oa_terms.sum()
    SEoa = np.sqrt(VAR_OA) if VAR_OA >= 0 else np.nan

    # Standard Error of UA
    # var_ua_terms = UA * (1 - UA) / (nh - 1) # R code version
    var_ua_terms = ua_for_var * (1 - ua_for_var) / (nh_safe - 1)
    VAR_UA = var_ua_terms.fillna(0)  # If nh<=1, variance is 0/inf -> 0
    SEua = pd.Series(np.sqrt(np.maximum(VAR_UA, 0)), index=map_classes_nh, name="SE_UA")

    # Standard Error of Area
    # VAR_A_j = sum_i [ Wh_i^2 * (p_ij_stratum * (1 - p_ij_stratum)) / (nh_i - 1) ]
    # where p_ij_stratum = matrix[i, j] / nh[i]
    VAR_A = pd.Series(index=class_order, dtype=float, name="Var_Area")
    p_ij_stratum = stratum_props  # Already calculated: matrix.div(nh, axis=0).fillna(0)
    wh_squared = np.array((Wh**2).values)[:, np.newaxis]
    term_var_a = wh_squared * (p_ij_stratum * (1 - p_ij_stratum)).div(
        nh_safe - 1, axis=0
    )
    VAR_A = term_var_a.sum(axis=0).reindex(
        class_order, fill_value=0
    )  # Sum over strata (i) for each ref class (j)
    SEa = pd.Series(np.sqrt(np.maximum(VAR_A, 0)), index=class_order, name="SE_Area")

    # Standard Error of PA (Robust implementation using delta method) #TODO: This is a  variation from the original paper added because indexing required complex logic that was error-prone
    # VAR_PA_j = (1 / Area_j^2) * sum_i [ Wh_i^2 * var_component_i_j / (nh_i - 1) ]
    # where var_component accounts for the ratio structure properly
    VAR_PA = pd.Series(index=class_order, dtype=float, name="Var_PA")

    for _j_idx, j_class in enumerate(class_order):
        area_j = Area.get(j_class, 0)
        ref_samples_j = ref_sample_counts.get(j_class, 0)

        if ref_samples_j == 0:
            VAR_PA[j_class] = np.nan
            continue

        if area_j == 0:
            VAR_PA[j_class] = 0
            continue

        pa_j = PA.get(j_class, 0)

        # Calculate variance using proper delta method for ratio PA_j = numerator_j / area_j
        variance_sum = 0

        for _i_idx, i_class in enumerate(
            map_classes_nh
        ):  # Iterate through map strata i
            nh_i = nh.get(i_class, 0)
            if nh_i < 2:  # Skip strata with < 2 samples for variance calc
                continue

            wh_i = Wh.get(i_class, 0)
            p_ij_strat = p_ij_stratum.loc[
                i_class, j_class
            ]  # Proportion p_ij within stratum i

            # For PA_j = sum_i(p_ij) / sum_i(p_ij), we need var components
            if i_class == j_class:
                # Diagonal element: contributes to both numerator and denominator
                # Variance component: (1 - 2*PA_j + PA_j^2) * p_ij_strat * (1 - p_ij_strat)
                var_component = (1 - pa_j) ** 2 * p_ij_strat * (1 - p_ij_strat)
            else:
                # Off-diagonal element: contributes to denominator only
                # Variance component: PA_j^2 * p_ij_strat * (1 - p_ij_strat)
                var_component = pa_j**2 * p_ij_strat * (1 - p_ij_strat)

            # Add weighted variance component
            variance_sum += (wh_i**2 * var_component) / (nh_i - 1)

        VAR_PA[j_class] = variance_sum / (area_j**2) if area_j > 0 else 0

    SEpa = pd.Series(np.sqrt(np.maximum(VAR_PA, 0)), index=class_order, name="SE_PA")

    # add in two sided confidence intervals for OA, UA, PA, Area
    # Note: R code uses qnorm(0.975) for 95% CI, but we can use scipy.stats.norm.ppf(0.975)
    z_alpha = norm.ppf(0.975)

    # Confidence Interval half-widths
    CI_halfwidth_oa = z_alpha * SEoa
    CI_halfwidth_ua = pd.Series(
        z_alpha * SEua, index=SEua.index, name="CI_halfwidth_ua"
    )
    CI_halfwidth_pa = z_alpha * SEpa
    CI_halfwidth_a = z_alpha * SEa

    # Confidence Intervals (lower, upper)
    OA_CI = (OA - CI_halfwidth_oa, OA + CI_halfwidth_oa)
    UA_CI = (UA - CI_halfwidth_ua, UA + CI_halfwidth_ua)
    PA_CI = (PA - CI_halfwidth_pa, PA + CI_halfwidth_pa)
    Area_CI = (Area - CI_halfwidth_a, Area + CI_halfwidth_a)
    # Add confidence intervals to the output dictionary

    # --- Final Output Preparation ---
    # Use the area proportion matrix 'props', but ensure it includes all reference classes as rows
    output_matrix = props.copy()

    # Reindex to include all reference classes as rows (fill missing rows with NaN)
    output_matrix = output_matrix.reindex(
        index=class_order, columns=class_order, fill_value=np.nan
    )

    # Set cells to NaN where original matrix had 0 counts (to match R behavior)
    # Need to align matrix with the reindexed output_matrix
    matrix_aligned = matrix.reindex(
        index=class_order, columns=class_order, fill_value=0
    )
    output_matrix[matrix_aligned == 0] = np.nan

    if margins:
        # Calculate sums, replacing NaN with 0 for summation before adding margins
        row_sum = output_matrix.sum(axis=1, skipna=True).reindex(
            class_order, fill_value=0
        )
        col_sum = output_matrix.sum(axis=0, skipna=True).reindex(
            class_order, fill_value=0
        )
        total_sum = col_sum.sum()  # Sum of area proportions should be ~1.0

        # Add row sums as a new column
        output_matrix["sum"] = row_sum
        # Add column sums as a new row - ensure 'sum' column exists first
        col_sum_df = pd.DataFrame(col_sum).T  # Convert Series to DataFrame row
        col_sum_df.index = pd.Index(["sum"])
        col_sum_df["sum"] = total_sum  # Add the total sum to the sum series/row

        # Align columns before concatenating
        output_matrix, col_sum_df = output_matrix.align(
            col_sum_df, axis=1, join="outer", fill_value=np.nan
        )
        output_matrix = pd.concat([output_matrix, col_sum_df])
        # Reorder rows/cols potentially? R code adds sum last.
        final_row_order = class_order + [
            "sum"
        ]  # Use class_order instead of map_classes_nh
        final_col_order = class_order + ["sum"]
        output_matrix = output_matrix.reindex(
            index=final_row_order, columns=final_col_order
        )

    return {
        "OA": OA,
        "UA": UA.reindex(class_order).fillna(
            np.nan
        ),  # Show UA for all classes, NaN if not a map class
        "PA": PA,
        "area": Area,
        "SEoa": SEoa,
        "SEua": SEua.reindex(class_order).fillna(np.nan),  # Show SEua for all classes
        "SEpa": SEpa,
        "SEa": SEa,
        "CIoa": OA_CI,
        "CIua": UA_CI,
        "CIpa": PA_CI,
        "CIa": Area_CI,
        "CI_halfwidth_oa": CI_halfwidth_oa,
        "CI_halfwidth_ua": CI_halfwidth_ua.reindex(class_order).fillna(np.nan),
        "CI_halfwidth_pa": CI_halfwidth_pa,
        "CI_halfwidth_a": CI_halfwidth_a,
        "matrix": output_matrix,
    }
