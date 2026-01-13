"""Treatment effect estimation and partial identification.

This module provides functions for:
- ATE estimation (Naive, IPW, AIPW, OLS, DML)
- CATE estimation wrappers for econml models
- Partial identification bounds (Manski bounds)
- Positivity diagnostics and sensitivity analysis
- PS Trimming and Overlap Weighting (ATO)

For theory and background, see docs/positivity_assumption.md.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, NamedTuple
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, LassoCV
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier, RandomForestRegressor
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.metrics import roc_auc_score
import statsmodels.api as sm
import warnings


# =============================================================================
# Type Definitions
# =============================================================================

class ATEResult(NamedTuple):
    """ATE estimation result."""
    method: str
    estimate: float
    se: Optional[float] = None
    ci_lower: Optional[float] = None
    ci_upper: Optional[float] = None
    n_obs: Optional[int] = None


class CATEBounds(NamedTuple):
    """CATE partial identification bounds."""
    point: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    method: str = "manski"


class PositivityDiagnostics(NamedTuple):
    """Positivity assumption diagnostics."""
    ps_auc: float
    overlap_ratio: float
    overlap_ratio_strict: float
    ps_min_treated: float
    ps_max_treated: float
    ps_min_control: float
    ps_max_control: float
    n_extreme_low: int
    n_extreme_high: int


# =============================================================================
# Propensity Score Estimation
# =============================================================================

def estimate_propensity_score(
    X: np.ndarray,
    T: np.ndarray,
    model: str = "logistic",
    cv: int = 5,
    clip: Tuple[float, float] = (0.01, 0.99)
) -> np.ndarray:
    """Estimate propensity scores P(T=1|X).

    Args:
        X: Covariate matrix (n_samples, n_features)
        T: Treatment indicator (n_samples,)
        model: "logistic" or "gbm"
        cv: Cross-validation folds for regularization
        clip: Min/max bounds for propensity scores

    Returns:
        Propensity scores clipped to [clip[0], clip[1]]
    """
    if model == "logistic":
        clf = LogisticRegressionCV(cv=cv, max_iter=1000, n_jobs=-1)
    elif model == "gbm":
        clf = GradientBoostingClassifier(
            n_estimators=100, max_depth=3, min_samples_leaf=20,
            learning_rate=0.05, random_state=42
        )
    else:
        raise ValueError(f"Unknown model: {model}")

    clf.fit(X, T)
    ps = clf.predict_proba(X)[:, 1]

    return np.clip(ps, clip[0], clip[1])


def estimate_propensity_score_cv(
    X: np.ndarray,
    T: np.ndarray,
    cv: int = 5,
    clip: Tuple[float, float] = (0.01, 0.99)
) -> np.ndarray:
    """Estimate cross-fitted propensity scores (out-of-fold predictions).

    Args:
        X: Covariate matrix
        T: Treatment indicator
        cv: Number of folds
        clip: Min/max bounds

    Returns:
        Cross-fitted propensity scores
    """
    clf = LogisticRegression(max_iter=1000)
    ps = cross_val_predict(clf, X, T, cv=cv, method='predict_proba')[:, 1]
    return np.clip(ps, clip[0], clip[1])


# =============================================================================
# Positivity Diagnostics
# =============================================================================

def compute_positivity_diagnostics(
    ps: np.ndarray,
    T: np.ndarray,
    overlap_bounds: Tuple[float, float] = (0.1, 0.9),
    strict_bounds: Tuple[float, float] = (0.05, 0.95)
) -> PositivityDiagnostics:
    """Compute diagnostics for positivity assumption.

    Args:
        ps: Propensity scores
        T: Treatment indicator
        overlap_bounds: Bounds for overlap region (default: [0.1, 0.9])
        strict_bounds: Bounds for strict overlap (default: [0.05, 0.95])

    Returns:
        PositivityDiagnostics named tuple
    """
    ps_treated = ps[T == 1]
    ps_control = ps[T == 0]

    # PS AUC (discrimination ability)
    ps_auc = roc_auc_score(T, ps)

    # Overlap ratios
    in_overlap = (ps >= overlap_bounds[0]) & (ps <= overlap_bounds[1])
    in_strict_overlap = (ps >= strict_bounds[0]) & (ps <= strict_bounds[1])
    overlap_ratio = in_overlap.mean()
    overlap_ratio_strict = in_strict_overlap.mean()

    # Extreme values
    n_extreme_low = (ps < strict_bounds[0]).sum()
    n_extreme_high = (ps > strict_bounds[1]).sum()

    return PositivityDiagnostics(
        ps_auc=ps_auc,
        overlap_ratio=overlap_ratio,
        overlap_ratio_strict=overlap_ratio_strict,
        ps_min_treated=ps_treated.min(),
        ps_max_treated=ps_treated.max(),
        ps_min_control=ps_control.min(),
        ps_max_control=ps_control.max(),
        n_extreme_low=n_extreme_low,
        n_extreme_high=n_extreme_high
    )


def compute_covariate_balance(
    X: Union[np.ndarray, pd.DataFrame],
    T: np.ndarray,
    feature_names: Optional[List[str]] = None
) -> pd.DataFrame:
    """Compute Standardized Mean Difference (SMD) for covariate balance.

    Args:
        X: Covariate matrix or DataFrame
        T: Treatment indicator
        feature_names: Column names (required if X is ndarray)

    Returns:
        DataFrame with columns: feature, mean_treated, mean_control, smd, balanced
    """
    if isinstance(X, pd.DataFrame):
        feature_names = X.columns.tolist()
        X = X.values

    if feature_names is None:
        feature_names = [f"X{i}" for i in range(X.shape[1])]

    results = []
    for i, name in enumerate(feature_names):
        x = X[:, i]
        x_t = x[T == 1]
        x_c = x[T == 0]

        mean_t = np.nanmean(x_t)
        mean_c = np.nanmean(x_c)
        std_t = np.nanstd(x_t)
        std_c = np.nanstd(x_c)

        # Pooled standard deviation
        pooled_std = np.sqrt((std_t**2 + std_c**2) / 2)

        # SMD
        smd = (mean_t - mean_c) / pooled_std if pooled_std > 0 else 0

        results.append({
            'feature': name,
            'mean_treated': mean_t,
            'mean_control': mean_c,
            'std_treated': std_t,
            'std_control': std_c,
            'smd': smd,
            'abs_smd': abs(smd),
            'balanced': abs(smd) < 0.1
        })

    return pd.DataFrame(results).sort_values('abs_smd', ascending=False)


# =============================================================================
# ATE Estimators
# =============================================================================

def estimate_ate_naive(Y: np.ndarray, T: np.ndarray) -> ATEResult:
    """Naive ATE estimator (simple mean difference).

    Biased when treatment is not randomly assigned.

    Args:
        Y: Outcome variable
        T: Treatment indicator

    Returns:
        ATEResult with estimate and SE
    """
    y1 = Y[T == 1]
    y0 = Y[T == 0]

    estimate = y1.mean() - y0.mean()
    se = np.sqrt(y1.var() / len(y1) + y0.var() / len(y0))

    return ATEResult(
        method="naive",
        estimate=estimate,
        se=se,
        ci_lower=estimate - 1.96 * se,
        ci_upper=estimate + 1.96 * se,
        n_obs=len(Y)
    )


def estimate_ate_ipw(
    Y: np.ndarray,
    T: np.ndarray,
    ps: np.ndarray,
    normalize: bool = True
) -> ATEResult:
    """Inverse Propensity Weighting (IPW) ATE estimator.

    Args:
        Y: Outcome variable
        T: Treatment indicator
        ps: Propensity scores
        normalize: If True, use Hajek (normalized) estimator

    Returns:
        ATEResult with estimate
    """
    # IPW weights
    w1 = T / ps
    w0 = (1 - T) / (1 - ps)

    if normalize:
        # Hajek estimator (normalized weights)
        mu1 = (w1 * Y).sum() / w1.sum()
        mu0 = (w0 * Y).sum() / w0.sum()
    else:
        # Horvitz-Thompson estimator
        n = len(Y)
        mu1 = (w1 * Y).sum() / n
        mu0 = (w0 * Y).sum() / n

    estimate = mu1 - mu0

    # Bootstrap SE would be more accurate, but simple approximation:
    # Using sandwich variance estimator approximation
    psi = w1 * (Y - mu1) - w0 * (Y - mu0)
    se = np.sqrt(np.var(psi) / len(Y))

    return ATEResult(
        method="ipw",
        estimate=estimate,
        se=se,
        ci_lower=estimate - 1.96 * se,
        ci_upper=estimate + 1.96 * se,
        n_obs=len(Y)
    )


def estimate_ate_aipw(
    Y: np.ndarray,
    T: np.ndarray,
    X: np.ndarray,
    ps: Optional[np.ndarray] = None,
    outcome_model: str = "gbm"
) -> ATEResult:
    """Augmented IPW (Doubly Robust) ATE estimator.

    Consistent if either PS model or outcome model is correct.

    Args:
        Y: Outcome variable
        T: Treatment indicator
        X: Covariate matrix
        ps: Propensity scores (estimated if None)
        outcome_model: "gbm" or "linear"

    Returns:
        ATEResult with estimate
    """
    if ps is None:
        ps = estimate_propensity_score(X, T)

    # Outcome models (separate for T=1 and T=0)
    if outcome_model == "gbm":
        model_params = dict(
            n_estimators=100, max_depth=3, min_samples_leaf=20,
            learning_rate=0.05, random_state=42
        )
        model_1 = GradientBoostingRegressor(**model_params)
        model_0 = GradientBoostingRegressor(**model_params)
    else:
        model_1 = LassoCV(cv=5, n_jobs=-1)
        model_0 = LassoCV(cv=5, n_jobs=-1)

    # Fit outcome models
    model_1.fit(X[T == 1], Y[T == 1])
    model_0.fit(X[T == 0], Y[T == 0])

    # Predictions
    mu1 = model_1.predict(X)
    mu0 = model_0.predict(X)

    # AIPW estimator
    aipw_1 = mu1 + T / ps * (Y - mu1)
    aipw_0 = mu0 + (1 - T) / (1 - ps) * (Y - mu0)

    estimate = aipw_1.mean() - aipw_0.mean()

    # Influence function for SE
    psi = aipw_1 - aipw_0 - estimate
    se = np.sqrt(np.var(psi) / len(Y))

    return ATEResult(
        method="aipw",
        estimate=estimate,
        se=se,
        ci_lower=estimate - 1.96 * se,
        ci_upper=estimate + 1.96 * se,
        n_obs=len(Y)
    )


def estimate_ate_ols(
    Y: np.ndarray,
    T: np.ndarray,
    X: np.ndarray
) -> ATEResult:
    """OLS ATE estimator (regression adjustment).

    Args:
        Y: Outcome variable
        T: Treatment indicator
        X: Covariate matrix

    Returns:
        ATEResult with estimate and SE from OLS
    """
    # Design matrix: intercept, treatment, covariates
    X_design = np.column_stack([np.ones(len(Y)), T, X])

    # OLS with robust SE
    model = sm.OLS(Y, X_design).fit(cov_type='HC1')

    # Treatment effect is coefficient on T (index 1)
    estimate = model.params[1]
    se = model.bse[1]

    return ATEResult(
        method="ols",
        estimate=estimate,
        se=se,
        ci_lower=model.conf_int()[1, 0],
        ci_upper=model.conf_int()[1, 1],
        n_obs=len(Y)
    )


def estimate_ate_dml(
    Y: np.ndarray,
    T: np.ndarray,
    X: np.ndarray,
    cv: int = 5
) -> ATEResult:
    """Double Machine Learning (DML) ATE estimator.

    Cross-fitted residual-on-residual regression.

    Args:
        Y: Outcome variable
        T: Treatment indicator
        X: Covariate matrix
        cv: Number of cross-validation folds

    Returns:
        ATEResult with estimate and SE
    """
    try:
        from econml.dml import LinearDML

        dml = LinearDML(
            model_y=GradientBoostingRegressor(
                n_estimators=100, max_depth=4, min_samples_leaf=20,
                random_state=42
            ),
            model_t=LogisticRegression(C=1.0, max_iter=1000, random_state=42),
            discrete_treatment=True,  # Binary treatment
            cv=cv,
            random_state=42
        )
        dml.fit(Y, T, X=X)

        # Must pass X since model was fitted with X
        estimate = dml.ate(X=X)
        ci = dml.ate_interval(X=X, alpha=0.05)

        # Compute SE from CI: SE ≈ (CI_upper - CI_lower) / (2 * 1.96)
        se = (float(ci[1]) - float(ci[0])) / (2 * 1.96)

        return ATEResult(
            method="dml",
            estimate=float(estimate),
            se=se,
            ci_lower=float(ci[0]),
            ci_upper=float(ci[1]),
            n_obs=len(Y)
        )
    except ImportError:
        warnings.warn("econml not installed. Using manual DML implementation.")
        return _estimate_ate_dml_manual(Y, T, X, cv)


def _estimate_ate_dml_manual(
    Y: np.ndarray,
    T: np.ndarray,
    X: np.ndarray,
    cv: int = 5
) -> ATEResult:
    """Manual DML implementation without econml."""
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)

    Y_residual = np.zeros_like(Y, dtype=float)
    T_residual = np.zeros_like(T, dtype=float)

    for train_idx, test_idx in kf.split(X):
        # Outcome model
        y_model = GradientBoostingRegressor(
            n_estimators=100, max_depth=3, random_state=42
        ).fit(X[train_idx], Y[train_idx])
        Y_residual[test_idx] = Y[test_idx] - y_model.predict(X[test_idx])

        # Treatment model
        t_model = GradientBoostingClassifier(
            n_estimators=100, max_depth=3, random_state=42
        ).fit(X[train_idx], T[train_idx])
        T_residual[test_idx] = T[test_idx] - t_model.predict_proba(X[test_idx])[:, 1]

    # Residual-on-residual regression
    model = sm.OLS(Y_residual, sm.add_constant(T_residual)).fit(cov_type='HC1')

    estimate = model.params[1]
    se = model.bse[1]

    return ATEResult(
        method="dml",
        estimate=estimate,
        se=se,
        ci_lower=estimate - 1.96 * se,
        ci_upper=estimate + 1.96 * se,
        n_obs=len(Y)
    )


def estimate_all_ate(
    Y: np.ndarray,
    T: np.ndarray,
    X: np.ndarray,
    ps: Optional[np.ndarray] = None,
    methods: List[str] = None
) -> pd.DataFrame:
    """Run all ATE estimators and return comparison DataFrame.

    Args:
        Y: Outcome variable
        T: Treatment indicator
        X: Covariate matrix
        ps: Propensity scores (estimated if None)
        methods: List of methods to run (default: all)

    Returns:
        DataFrame comparing ATE estimates across methods
    """
    if methods is None:
        methods = ["naive", "ipw", "aipw", "ols", "dml"]

    if ps is None and any(m in methods for m in ["ipw", "aipw"]):
        ps = estimate_propensity_score(X, T)

    results = []
    for method in methods:
        if method == "naive":
            result = estimate_ate_naive(Y, T)
        elif method == "ipw":
            result = estimate_ate_ipw(Y, T, ps)
        elif method == "aipw":
            result = estimate_ate_aipw(Y, T, X, ps)
        elif method == "ols":
            result = estimate_ate_ols(Y, T, X)
        elif method == "dml":
            result = estimate_ate_dml(Y, T, X)
        else:
            continue

        results.append({
            'method': result.method,
            'estimate': result.estimate,
            'se': result.se,
            'ci_lower': result.ci_lower,
            'ci_upper': result.ci_upper,
            'n_obs': result.n_obs
        })

    return pd.DataFrame(results)


# =============================================================================
# Trimming and Overlap Weighting
# =============================================================================

def apply_ps_trimming(
    ps: np.ndarray,
    T: np.ndarray,
    lower: float = 0.1,
    upper: float = 0.9
) -> np.ndarray:
    """Return boolean mask for observations in overlap region.

    Args:
        ps: Propensity scores
        T: Treatment indicator (unused, kept for API consistency)
        lower: Lower bound for PS
        upper: Upper bound for PS

    Returns:
        Boolean mask (True = in overlap region)
    """
    return (ps >= lower) & (ps <= upper)


def compute_ato_weights(ps: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Compute Average Treatment on Overlap (ATO) weights.

    h(e) = e * (1 - e), highest weight at e = 0.5.

    Args:
        ps: Propensity scores
        T: Treatment indicator

    Returns:
        ATO weights
    """
    h = ps * (1 - ps)
    # Normalize within treatment groups
    w = np.where(T == 1, h / ps, h / (1 - ps))
    return w


def estimate_ate_ato(
    Y: np.ndarray,
    T: np.ndarray,
    ps: np.ndarray
) -> ATEResult:
    """ATE with Overlap Weighting (ATO estimator).

    Li, Morgan, Zaslavsky (2018) overlap weights.

    Args:
        Y: Outcome variable
        T: Treatment indicator
        ps: Propensity scores

    Returns:
        ATEResult with ATO estimate
    """
    # Overlap weights
    h = ps * (1 - ps)

    # Weighted outcomes
    w1 = h * T / ps
    w0 = h * (1 - T) / (1 - ps)

    mu1 = (w1 * Y).sum() / w1.sum()
    mu0 = (w0 * Y).sum() / w0.sum()

    estimate = mu1 - mu0

    # Approximate SE
    psi = w1 * (Y - mu1) / w1.sum() - w0 * (Y - mu0) / w0.sum()
    se = np.sqrt(len(Y) * np.var(psi))

    return ATEResult(
        method="ato",
        estimate=estimate,
        se=se,
        ci_lower=estimate - 1.96 * se,
        ci_upper=estimate + 1.96 * se,
        n_obs=len(Y)
    )


# =============================================================================
# Partial Identification Bounds
# =============================================================================

def compute_ate_manski_bounds(
    Y: np.ndarray,
    T: np.ndarray,
    Y_min: Optional[float] = None,
    Y_max: Optional[float] = None
) -> Tuple[float, float]:
    """Manski (1990) worst-case bounds for ATE.

    Without any assumptions, ATE is only partially identified.

    Args:
        Y: Outcome variable
        T: Treatment indicator
        Y_min: Minimum possible outcome (default: observed min)
        Y_max: Maximum possible outcome (default: observed max)

    Returns:
        Tuple of (lower_bound, upper_bound) for ATE
    """
    if Y_min is None:
        Y_min = Y.min()
    if Y_max is None:
        Y_max = Y.max()

    E_Y1 = Y[T == 1].mean()
    E_Y0 = Y[T == 0].mean()

    # Worst-case bounds
    # Lower: max benefit for T=0 (Y_max), min for T=1 (Y_min)
    # Upper: min benefit for T=0 (Y_min), max for T=1 (Y_max)
    lower_bound = (E_Y1 - Y_max)
    upper_bound = (E_Y1 - Y_min)

    return lower_bound, upper_bound


def compute_cate_bounds(
    Y: np.ndarray,
    T: np.ndarray,
    X: np.ndarray,
    Y_min: Optional[float] = None,
    Y_max: Optional[float] = None,
    mu1_model: Optional[object] = None
) -> CATEBounds:
    """Partial identification bounds for CATE tau(x).

    For each x, compute bounds on tau(x) = E[Y(1)|X=x] - E[Y(0)|X=x].

    When positivity is violated (no overlap), bounds are wide.
    When overlap exists, bounds can be tighter.

    Args:
        Y: Outcome variable
        T: Treatment indicator
        X: Covariate matrix
        Y_min: Minimum possible outcome
        Y_max: Maximum possible outcome
        mu1_model: Fitted model for E[Y|T=1, X] (estimated if None)

    Returns:
        CATEBounds with point estimate, lower, and upper bounds
    """
    if Y_min is None:
        Y_min = Y.min()
    if Y_max is None:
        Y_max = Y.max()

    # Fit outcome model for T=1 if not provided
    if mu1_model is None:
        mu1_model = GradientBoostingRegressor(
            n_estimators=100, max_depth=3, min_samples_leaf=20, random_state=42
        ).fit(X[T == 1], Y[T == 1])

    # mu1(x) = E[Y|T=1, X=x]
    mu1 = mu1_model.predict(X)

    # Manski-style bounds
    # tau(x) in [mu1(x) - Y_max, mu1(x) - Y_min]
    tau_lower = mu1 - Y_max
    tau_upper = mu1 - Y_min

    # Point estimate: midpoint (or use separate mu0 model if fitted)
    tau_point = (tau_lower + tau_upper) / 2

    return CATEBounds(
        point=tau_point,
        lower=tau_lower,
        upper=tau_upper,
        method="manski"
    )


def compute_cate_bounds_with_overlap(
    Y: np.ndarray,
    T: np.ndarray,
    X: np.ndarray,
    ps: np.ndarray,
    Y_min: Optional[float] = None,
    Y_max: Optional[float] = None,
    overlap_threshold: float = 0.1
) -> CATEBounds:
    """CATE bounds that are tighter in overlap regions.

    Uses model-based point estimates in overlap region,
    wider Manski bounds in non-overlap region.

    Args:
        Y: Outcome variable
        T: Treatment indicator
        X: Covariate matrix
        ps: Propensity scores
        Y_min: Minimum possible outcome
        Y_max: Maximum possible outcome
        overlap_threshold: PS threshold for overlap region

    Returns:
        CATEBounds with locally-adaptive bounds
    """
    if Y_min is None:
        Y_min = Y.min()
    if Y_max is None:
        Y_max = Y.max()

    # Fit separate outcome models
    mu1_model = GradientBoostingRegressor(
        n_estimators=100, max_depth=3, min_samples_leaf=20, random_state=42
    ).fit(X[T == 1], Y[T == 1])
    mu0_model = GradientBoostingRegressor(
        n_estimators=100, max_depth=3, min_samples_leaf=20, random_state=42
    ).fit(X[T == 0], Y[T == 0])

    mu1 = mu1_model.predict(X)
    mu0 = mu0_model.predict(X)

    # Point estimate from models
    tau_point = mu1 - mu0

    # Overlap indicator
    in_overlap = (ps >= overlap_threshold) & (ps <= 1 - overlap_threshold)

    # Bounds: tighter in overlap, wider outside
    tau_lower = np.zeros_like(tau_point)
    tau_upper = np.zeros_like(tau_point)

    # In overlap: use model predictions with small uncertainty
    # Approximate bounds based on model uncertainty
    model_uncertainty = np.abs(tau_point) * 0.3  # 30% uncertainty band
    tau_lower[in_overlap] = tau_point[in_overlap] - model_uncertainty[in_overlap]
    tau_upper[in_overlap] = tau_point[in_overlap] + model_uncertainty[in_overlap]

    # Outside overlap: wider Manski-style bounds
    tau_lower[~in_overlap] = mu1[~in_overlap] - Y_max
    tau_upper[~in_overlap] = mu1[~in_overlap] - Y_min

    return CATEBounds(
        point=tau_point,
        lower=tau_lower,
        upper=tau_upper,
        method="overlap_adaptive"
    )


def compute_monotone_cate_bounds(
    Y: np.ndarray,
    T: np.ndarray,
    X: np.ndarray,
    direction: str = "positive",
    Y_min: Optional[float] = None,
    Y_max: Optional[float] = None
) -> CATEBounds:
    """CATE bounds under monotonicity assumption.

    Assumes treatment effect is non-negative (or non-positive) for all x.
    This provides tighter bounds than Manski bounds.

    Args:
        Y: Outcome variable
        T: Treatment indicator
        X: Covariate matrix
        direction: "positive" (tau >= 0) or "negative" (tau <= 0)
        Y_min: Minimum possible outcome
        Y_max: Maximum possible outcome

    Returns:
        CATEBounds with tighter bounds under monotonicity
    """
    if Y_min is None:
        Y_min = Y.min()
    if Y_max is None:
        Y_max = Y.max()

    # Fit outcome model for T=1
    mu1_model = GradientBoostingRegressor(
        n_estimators=100, max_depth=3, min_samples_leaf=20, random_state=42
    ).fit(X[T == 1], Y[T == 1])
    mu1 = mu1_model.predict(X)

    if direction == "positive":
        # tau(x) >= 0 => Y(1) >= Y(0)
        # tau in [max(0, mu1 - Y_max), mu1 - Y_min]
        tau_lower = np.maximum(0, mu1 - Y_max)
        tau_upper = mu1 - Y_min
    else:
        # tau(x) <= 0 => Y(1) <= Y(0)
        # tau in [mu1 - Y_max, min(0, mu1 - Y_min)]
        tau_lower = mu1 - Y_max
        tau_upper = np.minimum(0, mu1 - Y_min)

    # Point estimate: constrained midpoint
    tau_point = (tau_lower + tau_upper) / 2

    return CATEBounds(
        point=tau_point,
        lower=tau_lower,
        upper=tau_upper,
        method=f"monotone_{direction}"
    )


# =============================================================================
# Sensitivity Analysis
# =============================================================================

def positivity_sensitivity(
    Y: np.ndarray,
    T: np.ndarray,
    X: np.ndarray,
    ps: np.ndarray,
    trim_levels: List[float] = None,
    estimator: str = "ipw"
) -> pd.DataFrame:
    """Sensitivity of ATE to trimming level.

    Args:
        Y: Outcome variable
        T: Treatment indicator
        X: Covariate matrix
        ps: Propensity scores
        trim_levels: List of trimming thresholds (default: [0.05, 0.1, 0.15, 0.2])
        estimator: ATE estimator to use ("ipw", "aipw", "dml")

    Returns:
        DataFrame with columns: trim_level, n_remaining, pct_remaining, ate, se
    """
    if trim_levels is None:
        trim_levels = [0.05, 0.1, 0.15, 0.2]

    results = []
    n_total = len(Y)

    for trim in trim_levels:
        mask = (ps >= trim) & (ps <= 1 - trim)
        n_remaining = mask.sum()

        if n_remaining < 100:
            continue

        Y_trim = Y[mask]
        T_trim = T[mask]
        X_trim = X[mask] if isinstance(X, np.ndarray) else X.values[mask]
        ps_trim = ps[mask]

        if estimator == "ipw":
            result = estimate_ate_ipw(Y_trim, T_trim, ps_trim)
        elif estimator == "aipw":
            result = estimate_ate_aipw(Y_trim, T_trim, X_trim, ps_trim)
        elif estimator == "dml":
            result = estimate_ate_dml(Y_trim, T_trim, X_trim)
        else:
            result = estimate_ate_ipw(Y_trim, T_trim, ps_trim)

        results.append({
            'trim_level': trim,
            'n_remaining': n_remaining,
            'pct_remaining': n_remaining / n_total,  # Ratio (0-1) for % formatting
            'ate': result.estimate,
            'se': result.se,
            'ci_lower': result.ci_lower,
            'ci_upper': result.ci_upper
        })

    return pd.DataFrame(results)


def compute_e_value(
    estimate: float,
    ci_lower: float,
    ci_upper: float,
    null_value: float = 0
) -> Dict[str, float]:
    """Compute E-value for unmeasured confounding sensitivity.

    VanderWeele & Ding (2017): How strong must unmeasured confounding
    be to explain away the observed effect?

    Args:
        estimate: Point estimate (on ratio scale for E-value)
        ci_lower: CI lower bound
        ci_upper: CI upper bound
        null_value: Null hypothesis value

    Returns:
        Dict with e_value (for point) and e_value_ci (for CI)
    """
    # Convert to risk ratio scale if needed
    # For continuous outcomes, this is an approximation

    # E-value formula: RR + sqrt(RR * (RR - 1))
    def e_value_formula(rr):
        if rr < 1:
            rr = 1 / rr
        return rr + np.sqrt(rr * (rr - 1))

    # For continuous outcomes, approximate RR from standardized effect
    # This is a rough approximation
    if estimate == 0:
        e_point = 1.0
    else:
        # Approximate RR using exp(estimate/SE) or similar
        # This is a simplification
        rr_approx = np.exp(np.abs(estimate) / (abs(ci_upper - ci_lower) / 3.92))
        e_point = e_value_formula(rr_approx)

    # E-value for CI bound closer to null
    ci_bound = ci_lower if estimate > null_value else ci_upper
    if (ci_bound - null_value) * (estimate - null_value) <= 0:
        # CI crosses null
        e_ci = 1.0
    else:
        rr_ci = np.exp(np.abs(ci_bound - null_value) / (abs(ci_upper - ci_lower) / 3.92))
        e_ci = e_value_formula(rr_ci)

    return {
        'e_value_point': e_point,
        'e_value_ci': e_ci
    }


# =============================================================================
# Confounder Analysis
# =============================================================================

def analyze_ps_feature_importance(
    X: Union[np.ndarray, pd.DataFrame],
    T: np.ndarray,
    Y: np.ndarray,
    feature_names: Optional[List[str]] = None,
    ps_threshold: float = 0.1,
    outcome_threshold: float = 0.1
) -> pd.DataFrame:
    """Analyze feature importance for propensity score and outcome.

    Classifies variables into:
    - Confounder: High PS importance + High outcome correlation
    - Selection-only: High PS importance + Low outcome correlation
    - Outcome-only: Low PS importance + High outcome correlation
    - Neither: Low both

    Args:
        X: Covariate matrix or DataFrame
        T: Treatment indicator
        Y: Outcome variable
        feature_names: Column names
        ps_threshold: Threshold for PS importance
        outcome_threshold: Threshold for outcome correlation

    Returns:
        DataFrame with feature roles and importance scores
    """
    if isinstance(X, pd.DataFrame):
        feature_names = X.columns.tolist()
        X = X.values

    if feature_names is None:
        feature_names = [f"X{i}" for i in range(X.shape[1])]

    # PS model (Logistic Regression for coefficient interpretation)
    ps_model = LogisticRegression(max_iter=1000, penalty='l2', C=1.0)
    ps_model.fit(X, T)
    ps_importance = np.abs(ps_model.coef_[0])

    # Normalize PS importance
    ps_importance = ps_importance / ps_importance.max()

    # Outcome correlation (absolute Pearson)
    outcome_corr = np.array([
        np.abs(np.corrcoef(X[:, i], Y)[0, 1]) for i in range(X.shape[1])
    ])
    outcome_corr = np.nan_to_num(outcome_corr, 0)

    # Classification
    def classify_role(ps_imp, out_corr):
        ps_high = ps_imp > ps_threshold
        out_high = out_corr > outcome_threshold

        if ps_high and out_high:
            return "Confounder"
        elif ps_high and not out_high:
            return "Selection-only"
        elif not ps_high and out_high:
            return "Outcome-only"
        else:
            return "Neither"

    results = pd.DataFrame({
        'feature': feature_names,
        'ps_importance': ps_importance,
        'outcome_correlation': outcome_corr,
        'role': [classify_role(ps_importance[i], outcome_corr[i])
                 for i in range(len(feature_names))]
    }).sort_values('ps_importance', ascending=False)

    return results


def run_covariate_experiment(
    Y: np.ndarray,
    T: np.ndarray,
    X_full: pd.DataFrame,
    var_sets: Dict[str, List[str]],
    estimator: str = "dml"
) -> pd.DataFrame:
    """Run ATE estimation with different covariate sets.

    Useful for assessing sensitivity to confounder inclusion.

    Args:
        Y: Outcome variable
        T: Treatment indicator
        X_full: Full covariate DataFrame
        var_sets: Dict mapping set name to list of column names
        estimator: ATE estimator ("dml", "aipw", "ipw")

    Returns:
        DataFrame comparing results across covariate sets
    """
    results = []

    for name, cols in var_sets.items():
        # Get subset of covariates
        available_cols = [c for c in cols if c in X_full.columns]
        if len(available_cols) == 0:
            continue

        X_subset = X_full[available_cols].values

        # Estimate PS for this set
        ps = estimate_propensity_score(X_subset, T)
        ps_diag = compute_positivity_diagnostics(ps, T)

        # Estimate ATE
        if estimator == "dml":
            ate_result = estimate_ate_dml(Y, T, X_subset)
        elif estimator == "aipw":
            ate_result = estimate_ate_aipw(Y, T, X_subset, ps)
        else:
            ate_result = estimate_ate_ipw(Y, T, ps)

        results.append({
            'var_set': name,
            'n_vars': len(available_cols),
            'ps_auc': ps_diag.ps_auc,
            'overlap_ratio': ps_diag.overlap_ratio,
            'ate': ate_result.estimate,
            'ate_se': ate_result.se,
            'ate_ci_lower': ate_result.ci_lower,
            'ate_ci_upper': ate_result.ci_upper
        })

    return pd.DataFrame(results)


# =============================================================================
# BLP Test (Corrected)
# =============================================================================

def blp_test(
    Y: np.ndarray,
    T: np.ndarray,
    X: np.ndarray,
    cate_predictions: np.ndarray,
    cv: int = 5
) -> Dict[str, float]:
    """Best Linear Predictor (BLP) test for HTE heterogeneity.

    Chernozhukov et al. (2018) methodology with cross-fitted residuals.

    Tests H0: tau(x) = tau (no heterogeneity)
    vs Ha: tau(x) varies with CATE predictions

    Args:
        Y: Outcome variable
        T: Treatment indicator
        X: Covariate matrix
        cate_predictions: CATE predictions from model
        cv: Number of cross-validation folds

    Returns:
        Dict with tau_1, tau_1_se, tau_1_pvalue, hte_exists
    """
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)

    # Cross-fitted residuals
    Y_residuals = np.zeros_like(Y, dtype=float)
    T_residuals = np.zeros_like(T, dtype=float)

    for train_idx, test_idx in kf.split(X):
        # Outcome model (cross-fitted)
        X_with_T = np.column_stack([X[train_idx], T[train_idx]])
        y_model = GradientBoostingRegressor(
            n_estimators=100, max_depth=3, random_state=42
        ).fit(X_with_T, Y[train_idx])

        X_test_with_T = np.column_stack([X[test_idx], T[test_idx]])
        Y_residuals[test_idx] = Y[test_idx] - y_model.predict(X_test_with_T)

        # Treatment model (cross-fitted)
        t_model = LogisticRegression(max_iter=1000).fit(X[train_idx], T[train_idx])
        ps = t_model.predict_proba(X[test_idx])[:, 1]
        T_residuals[test_idx] = T[test_idx] - ps

    # BLP regression
    cate_centered = cate_predictions - cate_predictions.mean()
    X_blp = np.column_stack([T_residuals, T_residuals * cate_centered])
    X_blp = sm.add_constant(X_blp)

    model = sm.OLS(Y_residuals, X_blp).fit(cov_type='HC1')

    return {
        'tau_0': model.params[1],  # Average effect
        'tau_0_se': model.bse[1],
        'tau_0_pvalue': model.pvalues[1],
        'tau_1': model.params[2],  # Heterogeneity loading
        'tau_1_se': model.bse[2],
        'tau_1_pvalue': model.pvalues[2],
        'hte_exists': model.pvalues[2] < 0.05
    }


# =============================================================================
# Refutation Tests
# =============================================================================

class PlaceboTestResult(NamedTuple):
    """Placebo treatment test result."""
    actual_cate_mean: float
    placebo_cate_mean: float
    placebo_cate_std: float
    ratio: float  # abs(placebo_mean) / abs(actual_mean)
    passed: bool
    threshold: float


class SubsetTestResult(NamedTuple):
    """Subset data test result."""
    correlation: float
    full_cate_mean: float
    subset_cate_mean: float
    full_cate_std: float
    subset_cate_std: float
    passed: bool
    threshold: float
    subset_fraction: float
    n_full: int
    n_subset: int


class RefutationTestResults(NamedTuple):
    """Combined refutation test results."""
    placebo: PlaceboTestResult
    subset: SubsetTestResult
    all_passed: bool


def placebo_treatment_test(
    Y: np.ndarray,
    T: np.ndarray,
    X: np.ndarray,
    actual_cate: np.ndarray,
    model_factory,
    X_predict: Optional[np.ndarray] = None,
    n_permutations: int = 1,
    threshold: float = 0.5,
    seed: int = 42
) -> PlaceboTestResult:
    """Placebo treatment test for CATE model validation.

    Tests whether randomly assigned treatment produces near-zero CATE.
    If the model correctly identifies treatment effects (not spurious
    correlations), random treatment should yield CATE close to 0.

    Args:
        Y: Outcome variable (n_samples,)
        T: Treatment indicator 0/1 (n_samples,)
        X: Covariate matrix for training (n_samples, n_features)
        actual_cate: CATE predictions from the actual model
        model_factory: Callable returning a fresh CATE model instance.
            The model must have .fit(Y, T, X=X) and .effect(X) methods.
        X_predict: Covariate matrix for prediction (default: X)
        n_permutations: Number of placebo runs (default: 1).
        threshold: Pass if |placebo_mean| < threshold * |actual_mean| (default: 0.5)
        seed: Random seed for reproducibility

    Returns:
        PlaceboTestResult with pass/fail and diagnostic metrics
    """
    np.random.seed(seed)

    if X_predict is None:
        X_predict = X

    treatment_rate = T.mean()
    placebo_cates = []

    for i in range(n_permutations):
        # Generate random (placebo) treatment
        T_placebo = np.random.binomial(1, treatment_rate, len(T))

        # Fit model on placebo treatment
        model = model_factory()
        model.fit(Y, T_placebo, X=X)
        cate_placebo = model.effect(X_predict).flatten()
        placebo_cates.append(cate_placebo.mean())

    placebo_mean = np.mean(placebo_cates)
    placebo_std = np.std(placebo_cates) if n_permutations > 1 else np.nan
    actual_mean = actual_cate.mean()

    # Pass criterion: placebo effect should be much smaller than actual
    ratio = abs(placebo_mean) / abs(actual_mean) if actual_mean != 0 else np.inf
    passed = ratio < threshold

    return PlaceboTestResult(
        actual_cate_mean=actual_mean,
        placebo_cate_mean=placebo_mean,
        placebo_cate_std=placebo_std,
        ratio=ratio,
        passed=passed,
        threshold=threshold
    )


def subset_data_test(
    Y: np.ndarray,
    T: np.ndarray,
    X: np.ndarray,
    full_cate: np.ndarray,
    model_factory,
    X_predict: Optional[np.ndarray] = None,
    subset_fraction: float = 0.5,
    threshold: float = 0.7,
    seed: int = 42
) -> SubsetTestResult:
    """Subset data test for CATE model stability.

    Tests whether training on a random subset produces CATE estimates
    correlated with the full model. This validates model stability.

    Args:
        Y: Outcome variable (n_samples,)
        T: Treatment indicator 0/1 (n_samples,)
        X: Covariate matrix for training (n_samples, n_features)
        full_cate: CATE predictions from model trained on full data
        model_factory: Callable returning a fresh CATE model instance.
        X_predict: Covariate matrix for prediction (default: X)
        subset_fraction: Fraction of data to use for subset (default: 0.5)
        threshold: Pass if correlation > threshold (default: 0.7)
        seed: Random seed for reproducibility

    Returns:
        SubsetTestResult with correlation and pass/fail status
    """
    np.random.seed(seed)

    if X_predict is None:
        X_predict = X

    n = len(Y)
    n_subset = int(n * subset_fraction)

    # Random subset indices
    subset_idx = np.random.choice(n, n_subset, replace=False)

    # Fit on subset
    model = model_factory()
    model.fit(Y[subset_idx], T[subset_idx], X=X[subset_idx])

    # Predict on X_predict (for comparison with full_cate)
    subset_cate = model.effect(X_predict).flatten()

    # Correlation between full and subset CATE
    correlation = np.corrcoef(full_cate, subset_cate)[0, 1]

    passed = correlation > threshold

    return SubsetTestResult(
        correlation=correlation,
        full_cate_mean=full_cate.mean(),
        subset_cate_mean=subset_cate.mean(),
        full_cate_std=full_cate.std(),
        subset_cate_std=subset_cate.std(),
        passed=passed,
        threshold=threshold,
        subset_fraction=subset_fraction,
        n_full=n,
        n_subset=n_subset
    )


def run_refutation_tests(
    Y: np.ndarray,
    T: np.ndarray,
    X: np.ndarray,
    actual_cate: np.ndarray,
    model_factory,
    X_predict: Optional[np.ndarray] = None,
    placebo_threshold: float = 0.5,
    subset_threshold: float = 0.7,
    subset_fraction: float = 0.5,
    seed: int = 42
) -> RefutationTestResults:
    """Run all refutation tests for CATE model validation.

    Args:
        Y: Outcome variable
        T: Treatment indicator
        X: Covariate matrix for training
        actual_cate: CATE from actual model (on X_predict)
        model_factory: Factory function returning fresh model instances
        X_predict: Covariate matrix for prediction (default: X)
        placebo_threshold: Pass if placebo_ratio < threshold
        subset_threshold: Pass if correlation > threshold
        subset_fraction: Fraction for subset test
        seed: Random seed

    Returns:
        RefutationTestResults with all test results
    """
    placebo_result = placebo_treatment_test(
        Y, T, X, actual_cate, model_factory,
        X_predict=X_predict,
        threshold=placebo_threshold, seed=seed
    )

    subset_result = subset_data_test(
        Y, T, X, actual_cate, model_factory,
        X_predict=X_predict,
        subset_fraction=subset_fraction,
        threshold=subset_threshold, seed=seed + 1
    )

    return RefutationTestResults(
        placebo=placebo_result,
        subset=subset_result,
        all_passed=placebo_result.passed and subset_result.passed
    )


# =============================================================================
# Improved Functions (v2) - Batch 1
# =============================================================================

class CovariateSelectionResult(NamedTuple):
    """Result of causal covariate selection."""
    selected_features: List[str]
    confounders: List[str]
    outcome_only: List[str]
    selection_only: List[str]
    role_df: pd.DataFrame
    original_ps_auc: float
    selected_ps_auc: float


def select_causal_covariates_v2(
    X: Union[np.ndarray, pd.DataFrame],
    T: np.ndarray,
    Y: np.ndarray,
    feature_names: Optional[List[str]] = None,
    ps_importance_method: str = "both",
    ps_threshold_pct: float = 75,
    outcome_threshold: float = 0.1,
    include_outcome_only: bool = True
) -> CovariateSelectionResult:
    """Select causal covariates by removing selection-only variables.

    This function identifies and removes variables that predict treatment
    but not outcome (selection-only), which can cause positivity violations
    without reducing confounding bias.

    Strategy:
    - Keep: Confounders (predict both T and Y)
    - Keep: Outcome-only (predict Y but not T) - improves efficiency
    - Remove: Selection-only (predict T but not Y) - causes positivity issues

    Args:
        X: Covariate matrix or DataFrame
        T: Treatment indicator
        Y: Outcome variable
        feature_names: Column names (required if X is ndarray)
        ps_importance_method: "logistic", "gbm", or "both" (average)
        ps_threshold_pct: Percentile threshold for high PS importance
        outcome_threshold: Absolute correlation threshold for outcome
        include_outcome_only: Whether to include outcome-only predictors

    Returns:
        CovariateSelectionResult with selected features and diagnostics
    """
    if isinstance(X, pd.DataFrame):
        feature_names = X.columns.tolist()
        X_arr = X.values
    else:
        X_arr = X
        if feature_names is None:
            feature_names = [f"X{i}" for i in range(X_arr.shape[1])]

    # Compute PS importance using multiple methods
    ps_importance = np.zeros(X_arr.shape[1])

    if ps_importance_method in ["logistic", "both"]:
        # Logistic regression coefficients
        lr = LogisticRegression(max_iter=1000, penalty='l2', C=1.0)
        lr.fit(X_arr, T)
        lr_imp = np.abs(lr.coef_[0])
        lr_imp = lr_imp / lr_imp.max()
        ps_importance += lr_imp

    if ps_importance_method in ["gbm", "both"]:
        # GBM feature importance
        gbm = GradientBoostingClassifier(
            n_estimators=100, max_depth=3, min_samples_leaf=20, random_state=42
        )
        gbm.fit(X_arr, T)
        gbm_imp = gbm.feature_importances_
        gbm_imp = gbm_imp / gbm_imp.max()
        ps_importance += gbm_imp

    if ps_importance_method == "both":
        ps_importance = ps_importance / 2

    # Outcome correlation (absolute Pearson)
    outcome_corr = np.array([
        np.abs(np.corrcoef(X_arr[:, i], Y)[0, 1])
        for i in range(X_arr.shape[1])
    ])
    outcome_corr = np.nan_to_num(outcome_corr, 0)

    # Dynamic threshold based on percentile
    ps_threshold = np.percentile(ps_importance, ps_threshold_pct)

    # Classification
    confounders = []
    outcome_only = []
    selection_only = []
    neither = []

    for i, name in enumerate(feature_names):
        ps_high = ps_importance[i] > ps_threshold
        out_high = outcome_corr[i] > outcome_threshold

        if ps_high and out_high:
            confounders.append(name)
        elif ps_high and not out_high:
            selection_only.append(name)
        elif not ps_high and out_high:
            outcome_only.append(name)
        else:
            neither.append(name)

    # Selected features
    selected = confounders.copy()
    if include_outcome_only:
        selected.extend(outcome_only)
    # Note: 'neither' variables are excluded as they don't help

    # Create role DataFrame
    role_df = pd.DataFrame({
        'feature': feature_names,
        'ps_importance': ps_importance,
        'outcome_correlation': outcome_corr,
        'role': [
            'Confounder' if f in confounders else
            'Outcome-only' if f in outcome_only else
            'Selection-only' if f in selection_only else
            'Neither'
            for f in feature_names
        ]
    }).sort_values('ps_importance', ascending=False)

    # Compute PS AUC before and after
    original_ps_auc = roc_auc_score(T, LogisticRegression(max_iter=1000).fit(X_arr, T).predict_proba(X_arr)[:, 1])

    if len(selected) > 0:
        selected_idx = [feature_names.index(f) for f in selected]
        X_selected = X_arr[:, selected_idx]
        selected_ps_auc = roc_auc_score(T, LogisticRegression(max_iter=1000).fit(X_selected, T).predict_proba(X_selected)[:, 1])
    else:
        selected_ps_auc = 0.5

    return CovariateSelectionResult(
        selected_features=selected,
        confounders=confounders,
        outcome_only=outcome_only,
        selection_only=selection_only,
        role_df=role_df,
        original_ps_auc=original_ps_auc,
        selected_ps_auc=selected_ps_auc
    )


class CATEEnsembleResult(NamedTuple):
    """Result of CATE ensemble."""
    cate: np.ndarray
    cate_std: np.ndarray
    weights: Dict[str, float]
    model_cates: Dict[str, np.ndarray]
    model_agreement: float


def create_cate_ensemble(
    cate_dict: Dict[str, np.ndarray],
    rscorer_dict: Optional[Dict[str, float]] = None,
    stable_models: Optional[List[str]] = None,
    weighting: str = "rscorer",
    agreement_threshold: float = 0.3
) -> CATEEnsembleResult:
    """Create weighted ensemble of CATE predictions.

    Combines multiple CATE models to reduce variance and improve
    robustness. Weights can be based on RScorer (causal loss) or
    equal weighting.

    Args:
        cate_dict: Dict mapping model name to CATE array
        rscorer_dict: Dict mapping model name to R-score (lower is better)
        stable_models: List of model names to include (default: auto-select)
        weighting: "rscorer" (inverse R-score) or "equal"
        agreement_threshold: Minimum pairwise correlation for inclusion

    Returns:
        CATEEnsembleResult with ensemble CATE and diagnostics
    """
    if stable_models is None:
        # Default stable models based on typical performance
        stable_models = ['s_learner', 'causal_forest_dml', 'x_learner', 't_learner']
        stable_models = [m for m in stable_models if m in cate_dict]

    if len(stable_models) == 0:
        stable_models = list(cate_dict.keys())

    # Filter to available models
    available_models = [m for m in stable_models if m in cate_dict]

    if len(available_models) == 0:
        raise ValueError("No valid models found in cate_dict")

    # Filter by agreement (correlation) if multiple models
    if len(available_models) > 2:
        # Compute pairwise correlations
        model_cates = [cate_dict[m] for m in available_models]
        n_models = len(model_cates)
        keep_mask = np.ones(n_models, dtype=bool)

        for i in range(n_models):
            if not keep_mask[i]:
                continue
            correlations = []
            for j in range(n_models):
                if i != j and keep_mask[j]:
                    corr = np.corrcoef(model_cates[i], model_cates[j])[0, 1]
                    correlations.append(corr if not np.isnan(corr) else 0)

            # Exclude if low agreement with all others
            if len(correlations) > 0 and np.mean(correlations) < agreement_threshold:
                keep_mask[i] = False

        available_models = [m for m, keep in zip(available_models, keep_mask) if keep]

    # Compute weights
    if weighting == "rscorer" and rscorer_dict is not None:
        # Inverse R-score weighting (lower R-score = higher weight)
        weights = {}
        for m in available_models:
            if m in rscorer_dict and rscorer_dict[m] > 0:
                weights[m] = 1 / rscorer_dict[m]
            else:
                weights[m] = 1.0  # Default weight
    else:
        # Equal weighting
        weights = {m: 1.0 for m in available_models}

    # Normalize weights
    total = sum(weights.values())
    weights = {k: v / total for k, v in weights.items()}

    # Weighted average
    cate_ensemble = sum(weights[m] * cate_dict[m] for m in available_models)

    # Ensemble uncertainty (weighted std of model predictions)
    cate_stack = np.stack([cate_dict[m] for m in available_models])
    cate_std = np.sqrt(np.average(
        (cate_stack - cate_ensemble) ** 2,
        weights=[weights[m] for m in available_models],
        axis=0
    ))

    # Model agreement (average pairwise correlation)
    if len(available_models) > 1:
        corrs = []
        for i, m1 in enumerate(available_models):
            for m2 in available_models[i+1:]:
                corr = np.corrcoef(cate_dict[m1], cate_dict[m2])[0, 1]
                if not np.isnan(corr):
                    corrs.append(corr)
        model_agreement = np.mean(corrs) if corrs else 1.0
    else:
        model_agreement = 1.0

    return CATEEnsembleResult(
        cate=cate_ensemble,
        cate_std=cate_std,
        weights=weights,
        model_cates={m: cate_dict[m] for m in available_models},
        model_agreement=model_agreement
    )


# =============================================================================
# Improved Functions (v2) - Batch 2: GRF and DR-Learner
# =============================================================================

class GRFResult(NamedTuple):
    """Result of GRF CATE estimation with confidence intervals."""
    cate: np.ndarray
    cate_lower: np.ndarray
    cate_upper: np.ndarray
    ate: float
    ate_se: float
    model: object


def estimate_cate_grf_v2(
    Y: np.ndarray,
    T: np.ndarray,
    X: np.ndarray,
    X_predict: Optional[np.ndarray] = None,
    n_estimators: int = 500,
    min_samples_leaf: int = 20,
    honest: bool = True,
    alpha: float = 0.05,
    random_state: int = 42
) -> GRFResult:
    """Estimate CATE using Generalized Random Forest with local centering.

    Uses econml.grf.CausalForest which provides:
    - Honest splitting (separate samples for splitting and estimation)
    - Local centering for debiasing
    - Individual confidence intervals via forest variance estimation

    Args:
        Y: Outcome variable (n_samples,)
        T: Treatment indicator (n_samples,)
        X: Covariate matrix for training (n_samples, n_features)
        X_predict: Covariate matrix for prediction (default: X)
        n_estimators: Number of trees
        min_samples_leaf: Minimum samples per leaf
        honest: Use honest splitting
        alpha: Significance level for confidence intervals
        random_state: Random seed

    Returns:
        GRFResult with CATE, confidence intervals, and model
    """
    try:
        from econml.grf import CausalForest

        if X_predict is None:
            X_predict = X

        grf = CausalForest(
            n_estimators=n_estimators,
            min_samples_leaf=min_samples_leaf,
            honest=honest,
            inference=True,
            random_state=random_state,
            n_jobs=-1
        )

        grf.fit(X, T, Y)

        # CATE predictions (econml.grf uses predict instead of effect)
        cate = grf.predict(X_predict).flatten()

        # Confidence intervals
        cate_lower, cate_upper = grf.predict_interval(X_predict, alpha=alpha)
        cate_lower = cate_lower.flatten()
        cate_upper = cate_upper.flatten()

        # ATE (average of CATE)
        ate = cate.mean()
        # Approximate SE using bootstrap or asymptotic variance
        ate_se = cate.std() / np.sqrt(len(cate))

        return GRFResult(
            cate=cate,
            cate_lower=cate_lower,
            cate_upper=cate_upper,
            ate=ate,
            ate_se=ate_se,
            model=grf
        )

    except ImportError:
        warnings.warn(
            "econml.grf.CausalForest not available. "
            "Falling back to CausalForestDML."
        )
        return _estimate_cate_grf_fallback(Y, T, X, X_predict, n_estimators,
                                           min_samples_leaf, alpha, random_state)


def _estimate_cate_grf_fallback(
    Y: np.ndarray,
    T: np.ndarray,
    X: np.ndarray,
    X_predict: Optional[np.ndarray],
    n_estimators: int,
    min_samples_leaf: int,
    alpha: float,
    random_state: int
) -> GRFResult:
    """Fallback using CausalForestDML when GRF not available."""
    from econml.dml import CausalForestDML

    if X_predict is None:
        X_predict = X

    cf = CausalForestDML(
        model_y=GradientBoostingRegressor(
            n_estimators=100, max_depth=4, min_samples_leaf=20, random_state=random_state
        ),
        model_t=LogisticRegression(max_iter=1000, random_state=random_state),
        n_estimators=n_estimators,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
        n_jobs=-1
    )

    cf.fit(Y, T, X=X)

    cate = cf.effect(X_predict).flatten()
    ci = cf.effect_interval(X_predict, alpha=alpha)
    cate_lower = ci[0].flatten()
    cate_upper = ci[1].flatten()

    ate = cate.mean()
    ate_se = cate.std() / np.sqrt(len(cate))

    return GRFResult(
        cate=cate,
        cate_lower=cate_lower,
        cate_upper=cate_upper,
        ate=ate,
        ate_se=ate_se,
        model=cf
    )


class DRLearnerResult(NamedTuple):
    """Result of DR-Learner CATE estimation."""
    cate: np.ndarray
    cate_lower: Optional[np.ndarray]
    cate_upper: Optional[np.ndarray]
    ate: float
    ate_se: float
    model: object


def estimate_cate_dr_learner(
    Y: np.ndarray,
    T: np.ndarray,
    X: np.ndarray,
    X_predict: Optional[np.ndarray] = None,
    cv: int = 5,
    alpha: float = 0.05,
    model_propensity: Optional[object] = None,
    model_regression: Optional[object] = None,
    model_final: Optional[object] = None,
    random_state: int = 42
) -> DRLearnerResult:
    """Estimate CATE using Doubly Robust Learner.

    DR-Learner (Kennedy 2020) is doubly robust:
    - Consistent if either propensity OR outcome model is correct
    - Lower variance when both are correct

    Args:
        Y: Outcome variable (n_samples,)
        T: Treatment indicator (n_samples,)
        X: Covariate matrix for training (n_samples, n_features)
        X_predict: Covariate matrix for prediction (default: X)
        cv: Number of cross-validation folds
        alpha: Significance level for confidence intervals
        model_propensity: Custom propensity model
        model_regression: Custom outcome regression model
        model_final: Custom final stage model
        random_state: Random seed

    Returns:
        DRLearnerResult with CATE, confidence intervals, and model
    """
    from econml.dr import DRLearner
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    if X_predict is None:
        X_predict = X

    # Scale data for stable propensity estimation
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_predict_scaled = scaler.transform(X_predict)

    # Default models with better convergence settings
    if model_propensity is None:
        # Use Pipeline with scaler for robust propensity model
        model_propensity = LogisticRegression(
            C=0.1, max_iter=2000, solver='lbfgs', n_jobs=-1
        )

    if model_regression is None:
        model_regression = GradientBoostingRegressor(
            n_estimators=200, max_depth=4, min_samples_leaf=20,
            learning_rate=0.05, random_state=random_state
        )

    if model_final is None:
        model_final = GradientBoostingRegressor(
            n_estimators=100, max_depth=3, min_samples_leaf=20,
            learning_rate=0.05, random_state=random_state
        )

    dr = DRLearner(
        model_propensity=model_propensity,
        model_regression=model_regression,
        model_final=model_final,
        cv=cv,
        random_state=random_state
    )

    dr.fit(Y, T, X=X_scaled)

    # CATE predictions
    cate = dr.effect(X_predict_scaled).flatten()

    # Confidence intervals (if available)
    try:
        ci = dr.effect_interval(X_predict_scaled, alpha=alpha)
        cate_lower = ci[0].flatten()
        cate_upper = ci[1].flatten()
    except Exception:
        cate_lower = None
        cate_upper = None

    # ATE
    ate = dr.ate(X=X_scaled)
    try:
        ate_ci = dr.ate_interval(X=X_scaled, alpha=alpha)
        ate_se = (ate_ci[1] - ate_ci[0]) / (2 * 1.96)
    except Exception:
        ate_se = cate.std() / np.sqrt(len(cate))

    return DRLearnerResult(
        cate=cate,
        cate_lower=cate_lower,
        cate_upper=cate_upper,
        ate=float(ate),
        ate_se=float(ate_se),
        model=dr
    )


# =============================================================================
# Improved Functions (v2) - Batch 3: Refutation and Monotonicity
# =============================================================================

class ComprehensiveRefutationResult(NamedTuple):
    """Result of comprehensive refutation tests with multiple runs."""
    placebo_ratio_mean: float
    placebo_ratio_std: float
    placebo_ratios: List[float]
    subset_corr_mean: float
    subset_corr_std: float
    subset_correlations: List[float]
    random_cause_mean: float
    random_cause_std: float
    random_cause_effects: List[float]
    pass_placebo: bool
    pass_subset: bool
    pass_random_cause: bool
    all_passed: bool
    placebo_threshold: float
    subset_threshold: float


def comprehensive_refutation(
    Y: np.ndarray,
    T: np.ndarray,
    X: np.ndarray,
    actual_cate: np.ndarray,
    model_factory,
    X_predict: Optional[np.ndarray] = None,
    n_placebo_runs: int = 10,
    n_subset_runs: int = 5,
    subset_fraction: float = 0.5,
    placebo_threshold: float = 0.3,
    subset_threshold: float = 0.6,
    seed: int = 42
) -> ComprehensiveRefutationResult:
    """Comprehensive refutation tests with multiple runs for robustness.

    Runs multiple iterations of placebo and subset tests to get
    a more stable assessment of model validity.

    Args:
        Y: Outcome variable
        T: Treatment indicator
        X: Covariate matrix for training
        actual_cate: CATE from actual model
        model_factory: Factory function returning fresh model instances
        X_predict: Covariate matrix for prediction (default: X)
        n_placebo_runs: Number of placebo test iterations
        n_subset_runs: Number of subset test iterations
        subset_fraction: Fraction of data for subset test
        placebo_threshold: Pass if mean ratio < threshold
        subset_threshold: Pass if mean correlation > threshold
        seed: Random seed

    Returns:
        ComprehensiveRefutationResult with aggregated test results
    """
    np.random.seed(seed)

    if X_predict is None:
        X_predict = X

    treatment_rate = T.mean()
    actual_cate_std = np.std(actual_cate)
    actual_cate_mean = np.mean(actual_cate)

    # Placebo tests - random treatment assignment should show no effect
    placebo_ratios = []
    for i in range(n_placebo_runs):
        # Random treatment assignment (preserve dtype)
        T_placebo = np.random.binomial(1, treatment_rate, len(T)).astype(T.dtype)

        # Fit model with placebo treatment
        model = model_factory()
        model.fit(Y, T_placebo, X=X)
        cate_placebo = model.effect(X_predict).flatten()

        # Ratio of placebo CATE std to actual CATE std
        ratio = np.std(cate_placebo) / actual_cate_std if actual_cate_std > 0 else np.inf
        placebo_ratios.append(ratio)

    # Subset tests - model trained on subset should correlate with full model
    subset_correlations = []
    n = len(Y)
    n_subset = int(n * subset_fraction)

    for i in range(n_subset_runs):
        # Random subset
        subset_idx = np.random.choice(n, n_subset, replace=False)

        # Fit on subset
        model = model_factory()
        model.fit(Y[subset_idx], T[subset_idx], X=X[subset_idx])

        # Predict on full X_predict
        subset_cate = model.effect(X_predict).flatten()

        # Correlation with full model CATE
        corr = np.corrcoef(actual_cate, subset_cate)[0, 1]
        if not np.isnan(corr):
            subset_correlations.append(corr)

    # Random cause test - adding random covariate should have no effect
    random_cause_effects = []
    n_random_cause_runs = n_placebo_runs  # Use same number as placebo
    for i in range(n_random_cause_runs):
        # Add random feature
        X_random = np.column_stack([X, np.random.randn(len(X))])

        # Fit model with random feature
        model = model_factory()
        model.fit(Y, T, X=X_random)

        # Effect of random cause (should be ~0)
        # Compare mean CATE with and without random feature
        cate_with_random = model.effect(np.column_stack([X_predict, np.random.randn(len(X_predict))])).flatten()
        effect_diff = np.mean(cate_with_random) - actual_cate_mean
        random_cause_effects.append(effect_diff)

    # Aggregate results
    placebo_ratio_mean = np.mean(placebo_ratios)
    placebo_ratio_std = np.std(placebo_ratios)
    subset_corr_mean = np.mean(subset_correlations) if subset_correlations else 0
    subset_corr_std = np.std(subset_correlations) if len(subset_correlations) > 1 else 0
    random_cause_mean = np.mean(random_cause_effects)
    random_cause_std = np.std(random_cause_effects) if len(random_cause_effects) > 1 else 1.0

    pass_placebo = placebo_ratio_mean < placebo_threshold
    pass_subset = subset_corr_mean > subset_threshold
    pass_random_cause = abs(random_cause_mean) < 2 * random_cause_std

    return ComprehensiveRefutationResult(
        placebo_ratio_mean=placebo_ratio_mean,
        placebo_ratio_std=placebo_ratio_std,
        placebo_ratios=placebo_ratios,
        subset_corr_mean=subset_corr_mean,
        subset_corr_std=subset_corr_std,
        subset_correlations=subset_correlations,
        random_cause_mean=random_cause_mean,
        random_cause_std=random_cause_std,
        random_cause_effects=random_cause_effects,
        pass_placebo=pass_placebo,
        pass_subset=pass_subset,
        pass_random_cause=pass_random_cause,
        all_passed=pass_placebo and pass_subset and pass_random_cause,
        placebo_threshold=placebo_threshold,
        subset_threshold=subset_threshold
    )


def apply_monotonicity_constraints(
    cate: np.ndarray,
    constraint: str = 'non_negative',
    lower_bound: float = -100,
    upper_bound: float = 500,
    shrink_factor: float = 0.5
) -> np.ndarray:
    """Apply domain-informed constraints to CATE predictions.

    Useful when prior knowledge suggests treatment effects should
    satisfy certain constraints (e.g., first campaign shouldn't hurt).

    Args:
        cate: CATE predictions
        constraint: Type of constraint:
            - 'non_negative': Assume treatment can only help (CATE >= 0)
            - 'non_positive': Assume treatment can only hurt (CATE <= 0)
            - 'bounded': Clip to [lower_bound, upper_bound]
            - 'soft': Shrink extreme values towards bounds
            - 'none': No constraint
        lower_bound: Lower bound for 'bounded' and 'soft' constraints
        upper_bound: Upper bound for 'bounded' and 'soft' constraints
        shrink_factor: Factor for soft shrinkage (0 = clip, 1 = no change)

    Returns:
        Constrained CATE predictions
    """
    if constraint == 'non_negative':
        return np.maximum(cate, 0)
    elif constraint == 'non_positive':
        return np.minimum(cate, 0)
    elif constraint == 'bounded':
        return np.clip(cate, lower_bound, upper_bound)
    elif constraint == 'soft':
        # Soft constraint: shrink extreme values towards bounds
        result = cate.copy()
        # For values below lower_bound, shrink towards lower_bound
        below_mask = cate < lower_bound
        if below_mask.any():
            result[below_mask] = lower_bound + shrink_factor * (cate[below_mask] - lower_bound)
        # For values above upper_bound, shrink towards upper_bound
        above_mask = cate > upper_bound
        if above_mask.any():
            result[above_mask] = upper_bound + shrink_factor * (cate[above_mask] - upper_bound)
        return result
    elif constraint == 'none':
        return cate
    else:
        raise ValueError(f"Unknown constraint: {constraint}")


class OutlierFilteringResult(NamedTuple):
    """Result of outlier filtering."""
    cate_filtered: np.ndarray
    outlier_mask: np.ndarray
    n_outliers: int
    outlier_ratio: float
    lower_bound: float
    upper_bound: float


def apply_outlier_filtering(
    cate: np.ndarray,
    method: str = 'iqr',
    threshold: float = 1.5,
    percentile_bounds: Tuple[float, float] = (1, 99)
) -> OutlierFilteringResult:
    """Filter outlier CATE predictions.

    Identifies and clips extreme CATE values that may be due to
    extrapolation or model instability.

    Args:
        cate: CATE predictions
        method: Detection method:
            - 'iqr': Interquartile range (threshold is IQR multiplier)
            - 'zscore': Z-score based (threshold is number of std devs)
            - 'mad': Median absolute deviation (threshold is MAD multiplier)
            - 'percentile': Percentile-based clipping
        threshold: Detection threshold (meaning depends on method)
        percentile_bounds: (lower, upper) percentiles for 'percentile' method

    Returns:
        OutlierFilteringResult with filtered CATE and outlier info
    """
    if method == 'iqr':
        q1, q3 = np.percentile(cate, [25, 75])
        iqr = q3 - q1
        lower = q1 - threshold * iqr
        upper = q3 + threshold * iqr
    elif method == 'zscore':
        mean = np.mean(cate)
        std = np.std(cate)
        lower = mean - threshold * std
        upper = mean + threshold * std
    elif method == 'mad':
        median = np.median(cate)
        mad = np.median(np.abs(cate - median))
        # Scale factor for consistency with normal distribution
        mad_scaled = mad * 1.4826
        lower = median - threshold * mad_scaled
        upper = median + threshold * mad_scaled
    elif method == 'percentile':
        lower, upper = np.percentile(cate, list(percentile_bounds))
    else:
        raise ValueError(f"Unknown method: {method}")

    outlier_mask = (cate < lower) | (cate > upper)
    filtered_cate = np.clip(cate, lower, upper)
    n_outliers = outlier_mask.sum()
    outlier_ratio = n_outliers / len(cate)

    return OutlierFilteringResult(
        cate_filtered=filtered_cate,
        outlier_mask=outlier_mask,
        n_outliers=n_outliers,
        outlier_ratio=outlier_ratio,
        lower_bound=lower,
        upper_bound=upper
    )


class CredibilityScoreResult(NamedTuple):
    """Result of credibility scoring."""
    credibility_score: np.ndarray
    overlap_score: np.ndarray
    precision_score: np.ndarray
    sign_stability: np.ndarray


def compute_cate_credibility_score(
    cate: np.ndarray,
    ps: np.ndarray,
    cate_std: Optional[np.ndarray] = None,
    cate_lower: Optional[np.ndarray] = None,
    cate_upper: Optional[np.ndarray] = None,
    overlap_bounds: Tuple[float, float] = (0.1, 0.9),
    model_agreement: Optional[float] = None
) -> CredibilityScoreResult:
    """Compute credibility score for each CATE prediction.

    Higher scores indicate more reliable predictions based on:
    - Propensity score (in overlap region)
    - CI width (narrower = more precise)
    - Sign stability (CI not crossing zero)

    Args:
        cate: CATE predictions
        ps: Propensity scores
        cate_std: Standard deviation of CATE (used if bounds not provided)
        cate_lower: Lower CI bounds (optional, computed from cate_std)
        cate_upper: Upper CI bounds (optional, computed from cate_std)
        overlap_bounds: PS bounds for overlap region
        model_agreement: Optional agreement score from ensemble

    Returns:
        CredibilityScoreResult with component scores
    """
    # Compute bounds from cate_std if not provided
    if cate_lower is None or cate_upper is None:
        if cate_std is not None:
            cate_lower = cate - 1.96 * cate_std
            cate_upper = cate + 1.96 * cate_std
        else:
            # Default: use overall std
            std = np.std(cate)
            cate_lower = cate - 1.96 * std
            cate_upper = cate + 1.96 * std

    # Overlap score: 1 if in overlap region, decreasing outside
    overlap_center = (overlap_bounds[0] + overlap_bounds[1]) / 2
    overlap_width = overlap_bounds[1] - overlap_bounds[0]
    overlap_score = 1 - np.minimum(
        np.abs(ps - overlap_center) / (overlap_width / 2),
        1.0
    )
    overlap_score = np.maximum(overlap_score, 0)

    # Precision score: inverse of CI width (normalized)
    ci_width = cate_upper - cate_lower
    ci_width_normalized = ci_width / (np.percentile(ci_width, 95) + 1e-6)
    precision_score = 1 / (1 + ci_width_normalized)

    # Sign stability: 1 if CI doesn't cross zero, 0 if it does
    # Higher score if sign is more certain
    crosses_zero = (cate_lower < 0) & (cate_upper > 0)
    # Distance from zero normalized by CI width
    dist_to_zero = np.minimum(np.abs(cate_lower), np.abs(cate_upper))
    sign_stability = np.where(
        crosses_zero,
        0.5 * dist_to_zero / (ci_width + 1e-6),  # Partial credit if close to boundary
        1.0
    )
    sign_stability = np.clip(sign_stability, 0, 1)

    # Combine scores
    if model_agreement is not None:
        credibility = (0.35 * overlap_score + 0.35 * precision_score +
                       0.15 * sign_stability + 0.15 * model_agreement)
    else:
        credibility = 0.4 * overlap_score + 0.35 * precision_score + 0.25 * sign_stability

    return CredibilityScoreResult(
        credibility_score=credibility,
        overlap_score=overlap_score,
        precision_score=precision_score,
        sign_stability=sign_stability
    )


# =============================================================================
# Improved Functions (v2) - Batch 4: TMLE and PS Matching
# =============================================================================

class TMLEResult(NamedTuple):
    """Result of TMLE ATE estimation."""
    ate: float
    ate_se: float
    ci_lower: float
    ci_upper: float
    p_value: float
    method: str


def estimate_ate_tmle(
    Y: np.ndarray,
    T: np.ndarray,
    X: np.ndarray,
    cv: int = 5,
    alpha: float = 0.05
) -> TMLEResult:
    """Estimate ATE using Targeted Minimum Loss Estimation (TMLE).

    TMLE is a doubly robust estimator with:
    - Automatic debiasing through targeting step
    - Finite sample bias correction
    - Valid inference even with ML nuisance models

    Uses zepid library if available, otherwise falls back to manual implementation.

    Args:
        Y: Outcome variable
        T: Treatment indicator
        X: Covariate matrix
        cv: Cross-validation folds for nuisance models
        alpha: Significance level for CI

    Returns:
        TMLEResult with ATE and inference
    """
    try:
        from zepid.causal.doublyrobust import TMLE as ZepidTMLE

        # Create DataFrame for zepid
        df = pd.DataFrame(X)
        df.columns = [f'X{i}' for i in range(X.shape[1])]
        df['treatment'] = T
        df['outcome'] = Y

        # Fit TMLE
        tmle = ZepidTMLE(df, exposure='treatment', outcome='outcome')

        # Specify models (using all covariates)
        covariate_formula = ' + '.join(df.columns[:-2])
        tmle.exposure_model(covariate_formula, print_results=False)
        tmle.outcome_model(covariate_formula, print_results=False)
        tmle.fit()

        # Extract results
        ate = tmle.average_treatment_effect
        ci = tmle.average_treatment_effect_ci
        se = (ci[1] - ci[0]) / (2 * 1.96)

        # Compute p-value
        z_stat = ate / se if se > 0 else 0
        p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))

        return TMLEResult(
            ate=float(ate),
            ate_se=float(se),
            ci_lower=float(ci[0]),
            ci_upper=float(ci[1]),
            p_value=float(p_value),
            method='zepid'
        )

    except ImportError:
        warnings.warn("zepid not installed. Using manual TMLE implementation.")
        return _estimate_ate_tmle_manual(Y, T, X, cv, alpha)


def _estimate_ate_tmle_manual(
    Y: np.ndarray,
    T: np.ndarray,
    X: np.ndarray,
    cv: int = 5,
    alpha: float = 0.05
) -> TMLEResult:
    """Manual TMLE implementation without zepid.

    Implements the standard TMLE algorithm:
    1. Estimate outcome model E[Y|T, X]
    2. Estimate propensity score P(T=1|X)
    3. Compute clever covariate H = T/PS - (1-T)/(1-PS)
    4. Update outcome model via logistic regression on clever covariate
    5. Compute targeted predictions and ATE
    """
    from scipy import stats as scipy_stats

    # Step 1: Cross-fitted propensity scores
    ps = estimate_propensity_score_cv(X, T, cv=cv)
    ps = np.clip(ps, 0.01, 0.99)

    # Step 2: Cross-fitted outcome predictions
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    Q1 = np.zeros(len(Y))  # E[Y|T=1, X]
    Q0 = np.zeros(len(Y))  # E[Y|T=0, X]

    for train_idx, test_idx in kf.split(X):
        # Fit outcome model on training set
        model = GradientBoostingRegressor(
            n_estimators=100, max_depth=3, min_samples_leaf=20, random_state=42
        )
        model.fit(
            np.column_stack([X[train_idx], T[train_idx]]),
            Y[train_idx]
        )

        # Predict for T=1 and T=0
        X_test = X[test_idx]
        Q1[test_idx] = model.predict(np.column_stack([X_test, np.ones(len(test_idx))]))
        Q0[test_idx] = model.predict(np.column_stack([X_test, np.zeros(len(test_idx))]))

    # Step 3: Clever covariate
    H1 = T / ps
    H0 = -(1 - T) / (1 - ps)

    # Step 4: Targeting step (update predictions)
    # Using linear regression to find epsilon
    Q_star = T * Q1 + (1 - T) * Q0

    # Fluctuation model: logit(Q*) + epsilon * H
    # For continuous Y, use linear regression
    residuals = Y - Q_star
    X_target = np.column_stack([H1 + H0])
    epsilon = np.linalg.lstsq(X_target, residuals, rcond=None)[0][0]

    # Update predictions
    Q1_star = Q1 + epsilon / ps
    Q0_star = Q0 - epsilon / (1 - ps)

    # Step 5: Compute ATE
    ate = np.mean(Q1_star - Q0_star)

    # Influence function for variance
    IC = (T / ps) * (Y - Q1_star) - ((1 - T) / (1 - ps)) * (Y - Q0_star) + (Q1_star - Q0_star) - ate
    se = np.sqrt(np.var(IC) / len(Y))

    # Confidence interval
    z = scipy_stats.norm.ppf(1 - alpha / 2)
    ci_lower = ate - z * se
    ci_upper = ate + z * se

    # P-value
    z_stat = ate / se if se > 0 else 0
    p_value = 2 * (1 - scipy_stats.norm.cdf(abs(z_stat)))

    return TMLEResult(
        ate=float(ate),
        ate_se=float(se),
        ci_lower=float(ci_lower),
        ci_upper=float(ci_upper),
        p_value=float(p_value),
        method='manual'
    )


class PSMatchingResult(NamedTuple):
    """Result of propensity score matching."""
    treated_idx: np.ndarray
    control_idx: np.ndarray
    n_matched: int
    match_quality: Dict[str, float]
    ate_matched: float
    ate_se: float


def propensity_score_matching(
    ps: np.ndarray,
    T: np.ndarray,
    Y: Optional[np.ndarray] = None,
    X: Optional[np.ndarray] = None,
    method: str = 'nearest',
    caliper: float = 0.1,
    replacement: bool = False,
    n_neighbors: int = 1
) -> PSMatchingResult:
    """Propensity score matching for causal inference.

    Matches treated units to control units based on propensity score
    to create a balanced dataset for ATE estimation.

    Args:
        ps: Propensity scores
        T: Treatment indicator
        Y: Outcome variable (optional, for ATE estimation)
        X: Covariates (optional, for balance checking)
        method: 'nearest' (1:1 matching) or 'caliper' (with caliper)
        caliper: Maximum PS distance for matching
        replacement: Allow matching with replacement
        n_neighbors: Number of neighbors (for k:1 matching)

    Returns:
        PSMatchingResult with matched indices and quality metrics
    """
    from sklearn.neighbors import NearestNeighbors

    treated_idx = np.where(T == 1)[0]
    control_idx = np.where(T == 0)[0]

    ps_treated = ps[treated_idx]
    ps_control = ps[control_idx]

    # Fit nearest neighbor on control PS
    nn = NearestNeighbors(n_neighbors=n_neighbors, metric='euclidean')
    nn.fit(ps_control.reshape(-1, 1))

    # Find nearest control for each treated
    distances, indices = nn.kneighbors(ps_treated.reshape(-1, 1))
    distances = distances.flatten()
    indices = indices.flatten()

    # Apply caliper
    if method == 'caliper':
        matched_mask = distances < caliper
    else:
        matched_mask = np.ones(len(treated_idx), dtype=bool)

    # Get matched pairs
    matched_treated_idx = treated_idx[matched_mask]
    matched_control_idx = control_idx[indices[matched_mask]]

    # Remove duplicates if no replacement
    if not replacement:
        seen_controls = set()
        final_treated = []
        final_control = []
        for t, c in zip(matched_treated_idx, matched_control_idx):
            if c not in seen_controls:
                final_treated.append(t)
                final_control.append(c)
                seen_controls.add(c)
        matched_treated_idx = np.array(final_treated)
        matched_control_idx = np.array(final_control)

    n_matched = len(matched_treated_idx)

    # Match quality metrics
    match_quality = {
        'n_treated_original': len(treated_idx),
        'n_control_original': len(control_idx),
        'n_matched': n_matched,
        'pct_matched': n_matched / len(treated_idx) if len(treated_idx) > 0 else 0,
        'mean_ps_distance': float(distances[matched_mask].mean()) if matched_mask.sum() > 0 else np.nan
    }

    # Check covariate balance if X provided
    if X is not None and n_matched > 0:
        X_matched_treated = X[matched_treated_idx]
        X_matched_control = X[matched_control_idx]

        # SMD after matching
        mean_t = X_matched_treated.mean(axis=0)
        mean_c = X_matched_control.mean(axis=0)
        std_pooled = np.sqrt((X_matched_treated.var(axis=0) + X_matched_control.var(axis=0)) / 2)
        smd = np.abs(mean_t - mean_c) / (std_pooled + 1e-6)
        match_quality['mean_smd_after'] = float(smd.mean())
        match_quality['max_smd_after'] = float(smd.max())

    # ATE on matched sample
    if Y is not None and n_matched > 0:
        Y_treated = Y[matched_treated_idx]
        Y_control = Y[matched_control_idx]
        ate_matched = Y_treated.mean() - Y_control.mean()
        ate_se = np.sqrt(Y_treated.var() / n_matched + Y_control.var() / n_matched)
    else:
        ate_matched = np.nan
        ate_se = np.nan

    return PSMatchingResult(
        treated_idx=matched_treated_idx,
        control_idx=matched_control_idx,
        n_matched=n_matched,
        match_quality=match_quality,
        ate_matched=float(ate_matched),
        ate_se=float(ate_se)
    )


# Add scipy.stats import at the top level if needed
try:
    from scipy import stats
except ImportError:
    pass
