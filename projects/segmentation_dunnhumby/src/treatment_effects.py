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
