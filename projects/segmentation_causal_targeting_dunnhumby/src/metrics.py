"""Uplift model evaluation metrics.

Provides functions for:
- AUUC (Area Under Uplift Curve)
- Qini coefficient
- Cumulative gain metrics
"""

from typing import Dict, NamedTuple, Optional
import numpy as np
import pandas as pd


class UpliftMetrics(NamedTuple):
    """Uplift evaluation metrics for a model."""
    auuc: float
    qini_coef: float
    random_auuc: float
    lift_at_10pct: float
    lift_at_20pct: float


def compute_auuc(
    Y: np.ndarray,
    T: np.ndarray,
    cate_pred: np.ndarray,
    n_bins: int = 10
) -> float:
    """Compute Area Under Uplift Curve (AUUC).

    AUUC measures how well a CATE model ranks customers by their
    treatment effect. Higher AUUC = better ranking.

    Args:
        Y: Outcome variable (n_samples,)
        T: Treatment indicator 0/1 (n_samples,)
        cate_pred: CATE predictions (n_samples,)
        n_bins: Number of bins for curve

    Returns:
        AUUC value (higher = better ranking)
    """
    order = np.argsort(-cate_pred)
    Y_sorted = Y[order]
    T_sorted = T[order]

    n = len(Y)
    fractions = []
    uplifts = []

    for i in range(n_bins):
        end = int(n * (i + 1) / n_bins)
        Y_k = Y_sorted[:end]
        T_k = T_sorted[:end]

        if T_k.sum() > 0 and (1 - T_k).sum() > 0:
            uplift = Y_k[T_k == 1].mean() - Y_k[T_k == 0].mean()
        else:
            uplift = 0

        fractions.append((i + 1) / n_bins)
        uplifts.append(uplift)

    # AUUC = area under the curve (trapezoidal)
    auuc = np.trapz(uplifts, fractions)
    return auuc


def compute_qini_coef(
    Y: np.ndarray,
    T: np.ndarray,
    cate_pred: np.ndarray,
    n_bins: int = 10
) -> float:
    """Compute Qini coefficient.

    Qini coefficient measures the area between the model's cumulative
    gain curve and the random baseline. Positive = better than random.

    Args:
        Y: Outcome variable (n_samples,)
        T: Treatment indicator 0/1 (n_samples,)
        cate_pred: CATE predictions (n_samples,)
        n_bins: Number of bins

    Returns:
        Qini coefficient (area between model curve and random baseline)
    """
    n = len(Y)
    n_t = T.sum()
    n_c = n - n_t

    order = np.argsort(-cate_pred)
    Y_sorted = Y[order]
    T_sorted = T[order]

    fractions = []
    qini_values = []

    for i in range(n_bins):
        end = int(n * (i + 1) / n_bins)
        Y_k = Y_sorted[:end]
        T_k = T_sorted[:end]

        cum_Y_t = (Y_k * T_k).sum()
        cum_Y_c = (Y_k * (1 - T_k)).sum()
        cum_n_t = T_k.sum()
        cum_n_c = (1 - T_k).sum()

        if cum_n_c > 0:
            qini = cum_Y_t - cum_Y_c * cum_n_t / cum_n_c
        else:
            qini = 0

        fractions.append((i + 1) / n_bins)
        qini_values.append(qini)

    # Random baseline
    random_qini = [f * (Y[T == 1].sum() - Y[T == 0].sum() * n_t / n_c) for f in fractions]

    # Qini coefficient = area between model and random
    model_area = np.trapz(qini_values, fractions)
    random_area = np.trapz(random_qini, fractions)
    qini_coef = model_area - random_area

    return qini_coef


def compute_uplift_at_k(
    Y: np.ndarray,
    T: np.ndarray,
    cate_pred: np.ndarray,
    k_pct: float = 0.1
) -> float:
    """Compute uplift at top k%.

    Args:
        Y: Outcome variable
        T: Treatment indicator
        cate_pred: CATE predictions
        k_pct: Fraction of top customers (default 0.1 = 10%)

    Returns:
        Uplift (mean(Y|T=1) - mean(Y|T=0)) in top k%
    """
    n = len(Y)
    k = int(n * k_pct)

    order = np.argsort(-cate_pred)
    Y_top = Y[order[:k]]
    T_top = T[order[:k]]

    if T_top.sum() > 0 and (1 - T_top).sum() > 0:
        return Y_top[T_top == 1].mean() - Y_top[T_top == 0].mean()
    return 0.0


def compute_all_uplift_metrics(
    Y: np.ndarray,
    T: np.ndarray,
    cate_pred: np.ndarray,
    n_bins: int = 10
) -> UpliftMetrics:
    """Compute all uplift metrics for a CATE model.

    Args:
        Y: Outcome variable
        T: Treatment indicator
        cate_pred: CATE predictions
        n_bins: Number of bins for curves

    Returns:
        UpliftMetrics namedtuple with all metrics
    """
    auuc = compute_auuc(Y, T, cate_pred, n_bins)
    qini_coef = compute_qini_coef(Y, T, cate_pred, n_bins)

    # Random baseline AUUC (random ordering)
    random_auuc = compute_auuc(Y, T, np.random.randn(len(Y)), n_bins)

    # Lift at 10% and 20%
    lift_10 = compute_uplift_at_k(Y, T, cate_pred, k_pct=0.1)
    lift_20 = compute_uplift_at_k(Y, T, cate_pred, k_pct=0.2)

    return UpliftMetrics(
        auuc=auuc,
        qini_coef=qini_coef,
        random_auuc=random_auuc,
        lift_at_10pct=lift_10,
        lift_at_20pct=lift_20
    )


def compare_models_uplift(
    Y: np.ndarray,
    T: np.ndarray,
    cate_dict: Dict[str, np.ndarray],
    n_bins: int = 10
) -> pd.DataFrame:
    """Compare multiple models using uplift metrics.

    Args:
        Y: Outcome variable
        T: Treatment indicator
        cate_dict: Dictionary {model_name: cate_predictions}
        n_bins: Number of bins

    Returns:
        DataFrame with columns: Model, AUUC, Qini_Coef, Lift@10%, Lift@20%
    """
    results = []

    for model_name, cate_pred in cate_dict.items():
        metrics = compute_all_uplift_metrics(Y, T, cate_pred, n_bins)
        results.append({
            'Model': model_name,
            'AUUC': metrics.auuc,
            'Qini_Coef': metrics.qini_coef,
            'Lift@10%': metrics.lift_at_10pct,
            'Lift@20%': metrics.lift_at_20pct,
        })

    df = pd.DataFrame(results)
    return df.sort_values('AUUC', ascending=False).reset_index(drop=True)
