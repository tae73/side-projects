"""Visualization functions for HTE analysis.

This module provides standardized plotting functions for:
- Propensity score distributions
- Covariate balance plots (Love plot)
- Uplift and Qini curves
- CATE comparisons and bounds visualization
- Sensitivity analysis plots

All functions follow functional programming style with NamedTuple configs.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Union, NamedTuple


# =============================================================================
# Plot Configuration
# =============================================================================

class PlotConfig(NamedTuple):
    """Standard plot configuration."""
    figsize: Tuple[int, int] = (10, 6)
    dpi: int = 100
    style: str = 'whitegrid'
    palette: str = 'Set2'
    title_fontsize: int = 14
    label_fontsize: int = 12
    tick_fontsize: int = 10


DEFAULT_CONFIG = PlotConfig()


def setup_style(config: PlotConfig = DEFAULT_CONFIG):
    """Apply standard plot styling."""
    sns.set_style(config.style)
    plt.rcParams['figure.figsize'] = config.figsize
    plt.rcParams['figure.dpi'] = config.dpi
    plt.rcParams['axes.titlesize'] = config.title_fontsize
    plt.rcParams['axes.labelsize'] = config.label_fontsize
    plt.rcParams['xtick.labelsize'] = config.tick_fontsize
    plt.rcParams['ytick.labelsize'] = config.tick_fontsize


# =============================================================================
# Propensity Score Plots
# =============================================================================

def plot_ps_distribution(
    ps: np.ndarray,
    T: np.ndarray,
    title: str = "Propensity Score Distribution by Treatment Group",
    ax: Optional[plt.Axes] = None,
    config: PlotConfig = DEFAULT_CONFIG,
    show_overlap: bool = True,
    overlap_bounds: Tuple[float, float] = (0.1, 0.9)
) -> plt.Axes:
    """Plot propensity score distribution by treatment group.

    Args:
        ps: Propensity scores
        T: Treatment indicator
        title: Plot title
        ax: Matplotlib axes (created if None)
        config: PlotConfig instance
        show_overlap: If True, shade overlap region
        overlap_bounds: Bounds for overlap region

    Returns:
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=config.figsize, dpi=config.dpi)

    # Plot distributions
    sns.kdeplot(ps[T == 1], ax=ax, label='Treated', color='tab:blue', fill=True, alpha=0.3)
    sns.kdeplot(ps[T == 0], ax=ax, label='Control', color='tab:orange', fill=True, alpha=0.3)

    # Overlap region
    if show_overlap:
        ax.axvspan(overlap_bounds[0], overlap_bounds[1], alpha=0.1, color='green',
                   label=f'Overlap [{overlap_bounds[0]}, {overlap_bounds[1]}]')

    # Formatting
    ax.set_xlabel('Propensity Score')
    ax.set_ylabel('Density')
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.legend()

    return ax


def plot_ps_overlap(
    ps: np.ndarray,
    T: np.ndarray,
    title: str = "Propensity Score Overlap Analysis",
    ax: Optional[plt.Axes] = None,
    config: PlotConfig = DEFAULT_CONFIG
) -> plt.Axes:
    """Plot PS overlap with histograms and statistics.

    Args:
        ps: Propensity scores
        T: Treatment indicator
        title: Plot title
        ax: Matplotlib axes
        config: PlotConfig instance

    Returns:
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=config.figsize, dpi=config.dpi)

    # Histograms (mirrored)
    bins = np.linspace(0, 1, 51)

    ax.hist(ps[T == 1], bins=bins, alpha=0.5, label='Treated', color='tab:blue', density=True)
    ax.hist(ps[T == 0], bins=bins, alpha=0.5, label='Control', color='tab:orange', density=True)

    # Statistics text
    overlap_01_09 = ((ps >= 0.1) & (ps <= 0.9)).mean()
    overlap_005_095 = ((ps >= 0.05) & (ps <= 0.95)).mean()

    stats_text = (
        f"Overlap [0.1, 0.9]: {overlap_01_09:.1%}\n"
        f"Overlap [0.05, 0.95]: {overlap_005_095:.1%}\n"
        f"Treated range: [{ps[T==1].min():.3f}, {ps[T==1].max():.3f}]\n"
        f"Control range: [{ps[T==0].min():.3f}, {ps[T==0].max():.3f}]"
    )
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, verticalalignment='top',
            fontsize=10, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax.set_xlabel('Propensity Score')
    ax.set_ylabel('Density')
    ax.set_title(title)
    ax.legend()

    return ax


def plot_ps_train_test(
    ps_train: np.ndarray,
    ps_test: np.ndarray,
    T_train: np.ndarray,
    T_test: np.ndarray,
    config: PlotConfig = DEFAULT_CONFIG
) -> plt.Figure:
    """Plot PS distribution for train and test sets side by side.

    Args:
        ps_train: Training propensity scores
        ps_test: Test propensity scores
        T_train: Training treatment indicator
        T_test: Test treatment indicator
        config: PlotConfig instance

    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=config.dpi)

    plot_ps_distribution(ps_train, T_train, "Train Set PS Distribution", axes[0], config)
    plot_ps_distribution(ps_test, T_test, "Test Set PS Distribution", axes[1], config)

    plt.tight_layout()
    return fig


# =============================================================================
# Balance Plots
# =============================================================================

def plot_love_plot(
    balance_df: pd.DataFrame,
    threshold: float = 0.1,
    title: str = "Covariate Balance (Love Plot)",
    ax: Optional[plt.Axes] = None,
    config: PlotConfig = DEFAULT_CONFIG
) -> plt.Axes:
    """Love plot for covariate balance.

    Args:
        balance_df: DataFrame with columns: feature, smd (or abs_smd)
        threshold: SMD threshold for balance (default: 0.1)
        title: Plot title
        ax: Matplotlib axes
        config: PlotConfig instance

    Returns:
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, max(6, len(balance_df) * 0.3)), dpi=config.dpi)

    # Sort by absolute SMD
    smd_col = 'abs_smd' if 'abs_smd' in balance_df.columns else 'smd'
    df = balance_df.sort_values(smd_col, ascending=True).copy()
    df['abs_smd'] = df[smd_col].abs() if smd_col == 'smd' else df[smd_col]

    # Colors based on balance
    colors = ['tab:green' if x < threshold else 'tab:red' for x in df['abs_smd']]

    # Plot
    y_pos = range(len(df))
    ax.barh(y_pos, df['abs_smd'], color=colors, alpha=0.7)
    ax.axvline(threshold, color='black', linestyle='--', label=f'Threshold = {threshold}')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(df['feature'])
    ax.set_xlabel('Absolute Standardized Mean Difference (SMD)')
    ax.set_title(title)
    ax.legend()

    # Add count annotation
    n_balanced = (df['abs_smd'] < threshold).sum()
    ax.text(0.95, 0.02, f"Balanced: {n_balanced}/{len(df)}", transform=ax.transAxes,
            ha='right', fontsize=10, bbox=dict(facecolor='white', alpha=0.8))

    return ax


def plot_balance_comparison(
    balance_before: pd.DataFrame,
    balance_after: pd.DataFrame,
    threshold: float = 0.1,
    title: str = "Balance Before vs After Weighting",
    config: PlotConfig = DEFAULT_CONFIG
) -> plt.Figure:
    """Compare covariate balance before and after adjustment.

    Args:
        balance_before: Balance DataFrame before adjustment
        balance_after: Balance DataFrame after adjustment
        threshold: SMD threshold
        title: Plot title
        config: PlotConfig instance

    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, max(6, len(balance_before) * 0.3)), dpi=config.dpi)

    plot_love_plot(balance_before, threshold, "Before Adjustment", axes[0], config)
    plot_love_plot(balance_after, threshold, "After Adjustment", axes[1], config)

    fig.suptitle(title, fontsize=config.title_fontsize)
    plt.tight_layout()
    return fig


# =============================================================================
# Uplift / CATE Plots
# =============================================================================

def plot_uplift_curve(
    Y: np.ndarray,
    T: np.ndarray,
    cate_predictions: np.ndarray,
    n_bins: int = 10,
    title: str = "Uplift Curve",
    ax: Optional[plt.Axes] = None,
    config: PlotConfig = DEFAULT_CONFIG,
    show_random: bool = True
) -> plt.Axes:
    """Plot uplift curve (cumulative gain).

    Args:
        Y: Outcome variable
        T: Treatment indicator
        cate_predictions: CATE predictions
        n_bins: Number of bins for grouping
        title: Plot title
        ax: Matplotlib axes
        config: PlotConfig instance
        show_random: If True, show random targeting line

    Returns:
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=config.figsize, dpi=config.dpi)

    n = len(Y)

    # Sort by CATE (descending = target high CATE first)
    sorted_idx = np.argsort(-cate_predictions)
    Y_sorted = Y[sorted_idx]
    T_sorted = T[sorted_idx]

    # Cumulative outcomes
    cum_treated_outcomes = np.cumsum(Y_sorted * T_sorted)
    cum_control_outcomes = np.cumsum(Y_sorted * (1 - T_sorted))
    cum_n_treated = np.cumsum(T_sorted)
    cum_n_control = np.cumsum(1 - T_sorted)

    # Avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        cum_treated_mean = np.where(cum_n_treated > 0, cum_treated_outcomes / cum_n_treated, 0)
        cum_control_mean = np.where(cum_n_control > 0, cum_control_outcomes / cum_n_control, 0)

    # Cumulative uplift
    cum_uplift = cum_treated_mean - cum_control_mean
    x_axis = np.arange(1, n + 1) / n

    # Plot
    ax.plot(x_axis, cum_uplift, label='Model', linewidth=2)

    if show_random:
        # Random baseline: constant uplift
        random_uplift = np.full(n, Y[T == 1].mean() - Y[T == 0].mean())
        ax.plot(x_axis, random_uplift, '--', color='gray', label='Random', alpha=0.7)

    ax.set_xlabel('Proportion of Population Targeted')
    ax.set_ylabel('Cumulative Uplift')
    ax.set_title(title)
    ax.legend()
    ax.axhline(0, color='black', linestyle='-', alpha=0.3)

    return ax


def plot_uplift_with_auuc(
    Y: np.ndarray,
    T: np.ndarray,
    cate_dict: Dict[str, np.ndarray],
    auuc_df: pd.DataFrame,
    outcome_label: str = "",
    n_bins: int = 10,
    figsize: Tuple[float, float] = (14, 5),
) -> Tuple[plt.Figure, Tuple[plt.Axes, plt.Axes]]:
    """Plot 2-panel uplift visualization: uplift curve + AUUC bar chart.

    Left panel: Binned uplift curve with all models overlaid.
    Right panel: AUUC horizontal bar chart (green=positive, red=negative).

    Args:
        Y: Outcome variable array
        T: Treatment indicator array (0/1)
        cate_dict: Dictionary mapping model names to CATE predictions
        auuc_df: DataFrame with 'Model' and 'AUUC' columns
        outcome_label: Label for the outcome (used in titles)
        n_bins: Number of bins for uplift curve
        figsize: Figure size tuple

    Returns:
        Tuple of (figure, (ax_uplift, ax_auuc))
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Left: Uplift curve
    ax = axes[0]
    for model_name, cate in cate_dict.items():
        order = np.argsort(-cate)
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

        ax.plot(fractions, uplifts, 'o-', label=model_name, alpha=0.7)

    ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
    ax.set_xlabel('Fraction of Population (sorted by CATE)')
    ax.set_ylabel('Uplift')
    ax.set_title(f'Uplift Curve - {outcome_label}')
    ax.legend(fontsize=8)

    # Right: AUUC bar chart
    ax = axes[1]
    df_plot = auuc_df.sort_values('AUUC', ascending=True)
    colors = ['green' if x > 0 else 'red' for x in df_plot['AUUC']]
    ax.barh(df_plot['Model'], df_plot['AUUC'], color=colors, alpha=0.7)
    ax.axvline(x=0, color='gray', linestyle='--', linewidth=0.5)
    ax.set_xlabel('AUUC')
    ax.set_title(f'Model Comparison by AUUC - {outcome_label}')

    plt.tight_layout()
    return fig, (axes[0], axes[1])


def plot_qini_curve(
    Y: np.ndarray,
    T: np.ndarray,
    cate_predictions: np.ndarray,
    n_bins: int = 10,
    title: str = "Qini Curve",
    ax: Optional[plt.Axes] = None,
    config: PlotConfig = DEFAULT_CONFIG
) -> plt.Axes:
    """Plot Qini curve for uplift evaluation.

    Args:
        Y: Outcome variable
        T: Treatment indicator
        cate_predictions: CATE predictions
        n_bins: Number of bins
        title: Plot title
        ax: Matplotlib axes
        config: PlotConfig instance

    Returns:
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=config.figsize, dpi=config.dpi)

    n = len(Y)
    n_t = T.sum()
    n_c = n - n_t

    # Sort by CATE (descending)
    sorted_idx = np.argsort(-cate_predictions)
    Y_sorted = Y[sorted_idx]
    T_sorted = T[sorted_idx]

    # Cumulative
    cum_Y_t = np.cumsum(Y_sorted * T_sorted)
    cum_Y_c = np.cumsum(Y_sorted * (1 - T_sorted))
    cum_n_t = np.cumsum(T_sorted)
    cum_n_c = np.cumsum(1 - T_sorted)

    # Qini: Y_t(k) - Y_c(k) * n_t(k)/n_c(k)
    with np.errstate(divide='ignore', invalid='ignore'):
        qini = np.where(cum_n_c > 0, cum_Y_t - cum_Y_c * cum_n_t / cum_n_c, 0)

    x_axis = np.arange(1, n + 1) / n

    # Random baseline (diagonal)
    random_qini = x_axis * (Y[T == 1].sum() - Y[T == 0].sum() * n_t / n_c)

    ax.plot(x_axis, qini, label='Model', linewidth=2)
    ax.plot(x_axis, random_qini, '--', color='gray', label='Random', alpha=0.7)
    ax.fill_between(x_axis, random_qini, qini, alpha=0.2)

    ax.set_xlabel('Proportion of Population Targeted')
    ax.set_ylabel('Qini')
    ax.set_title(title)
    ax.legend()

    return ax


def plot_qini_with_coef(
    Y: np.ndarray,
    T: np.ndarray,
    cate_dict: Dict[str, np.ndarray],
    qini_df: pd.DataFrame,
    outcome_label: str = "",
    n_bins: int = 10,
    figsize: Tuple[float, float] = (14, 5),
) -> Tuple[plt.Figure, Tuple[plt.Axes, plt.Axes]]:
    """Plot 2-panel Qini visualization: Qini curve + Qini coefficient bar chart.

    Left panel: Binned Qini curves with all models overlaid.
    Right panel: Qini coefficient horizontal bar chart (green=positive, red=negative).

    Args:
        Y: Outcome variable array
        T: Treatment indicator array (0/1)
        cate_dict: Dictionary mapping model names to CATE predictions
        qini_df: DataFrame with 'Model' and 'Qini_Coef' columns
        outcome_label: Label for the outcome (used in titles)
        n_bins: Number of bins for Qini curve
        figsize: Figure size tuple

    Returns:
        Tuple of (figure, (ax_qini, ax_coef))
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    n = len(Y)
    n_t = T.sum()
    n_c = n - n_t

    # Left: Qini curves
    ax = axes[0]
    for model_name, cate in cate_dict.items():
        order = np.argsort(-cate)
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

        ax.plot(fractions, qini_values, 'o-', label=model_name, alpha=0.7)

    # Random baseline
    random_qini = [f * (Y[T == 1].sum() - Y[T == 0].sum() * n_t / n_c) for f in fractions]
    ax.plot(fractions, random_qini, '--', color='gray', label='Random', alpha=0.5)

    ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
    ax.set_xlabel('Fraction of Population (sorted by CATE)')
    ax.set_ylabel('Qini')
    ax.set_title(f'Qini Curve - {outcome_label}')
    ax.legend(fontsize=8)

    # Right: Qini coefficient bar chart
    ax = axes[1]
    df_plot = qini_df.sort_values('Qini_Coef', ascending=True)
    colors = ['green' if x > 0 else 'red' for x in df_plot['Qini_Coef']]
    ax.barh(df_plot['Model'], df_plot['Qini_Coef'], color=colors, alpha=0.7)
    ax.axvline(x=0, color='gray', linestyle='--', linewidth=0.5)
    ax.set_xlabel('Qini Coefficient')
    ax.set_title(f'Model Comparison by Qini - {outcome_label}')

    plt.tight_layout()
    return fig, (axes[0], axes[1])


def plot_cate_comparison(
    cate_dict: Dict[str, np.ndarray],
    title: str = "CATE Distribution by Model",
    ax: Optional[plt.Axes] = None,
    config: PlotConfig = DEFAULT_CONFIG
) -> plt.Axes:
    """Compare CATE distributions across models.

    Args:
        cate_dict: Dict mapping model name to CATE predictions
        title: Plot title
        ax: Matplotlib axes
        config: PlotConfig instance

    Returns:
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=config.figsize, dpi=config.dpi)

    for name, cate in cate_dict.items():
        sns.kdeplot(cate, ax=ax, label=f"{name} (mean={cate.mean():.2f})", fill=True, alpha=0.3)

    ax.axvline(0, color='black', linestyle='--', alpha=0.5)
    ax.set_xlabel('CATE')
    ax.set_ylabel('Density')
    ax.set_title(title)
    ax.legend()

    return ax


def plot_cate_by_segment(
    cate: np.ndarray,
    segments: np.ndarray,
    segment_names: Optional[Dict[int, str]] = None,
    title: str = "CATE Distribution by Segment",
    ax: Optional[plt.Axes] = None,
    config: PlotConfig = DEFAULT_CONFIG
) -> plt.Axes:
    """Plot CATE distribution by customer segment.

    Args:
        cate: CATE predictions
        segments: Segment labels
        segment_names: Dict mapping segment ID to name
        title: Plot title
        ax: Matplotlib axes
        config: PlotConfig instance

    Returns:
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=config.figsize, dpi=config.dpi)

    unique_segments = sorted(np.unique(segments))

    for seg in unique_segments:
        mask = segments == seg
        label = segment_names.get(seg, f"Segment {seg}") if segment_names else f"Segment {seg}"
        cate_seg = cate[mask]
        sns.kdeplot(cate_seg, ax=ax, label=f"{label} (n={mask.sum()}, mean={cate_seg.mean():.2f})")

    ax.axvline(0, color='black', linestyle='--', alpha=0.5)
    ax.set_xlabel('CATE')
    ax.set_ylabel('Density')
    ax.set_title(title)
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left')

    return ax


def plot_cate_boxplot_by_segment(
    cate: np.ndarray,
    segments: np.ndarray,
    segment_names: Optional[Dict[int, str]] = None,
    title: str = "CATE by Segment",
    ax: Optional[plt.Axes] = None,
    config: PlotConfig = DEFAULT_CONFIG
) -> plt.Axes:
    """Boxplot of CATE by segment.

    Args:
        cate: CATE predictions
        segments: Segment labels
        segment_names: Dict mapping segment ID to name
        title: Plot title
        ax: Matplotlib axes
        config: PlotConfig instance

    Returns:
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=config.figsize, dpi=config.dpi)

    df = pd.DataFrame({'CATE': cate, 'Segment': segments})
    if segment_names:
        df['Segment'] = df['Segment'].map(segment_names)

    # Order by median CATE
    order = df.groupby('Segment')['CATE'].median().sort_values(ascending=False).index

    sns.boxplot(data=df, x='Segment', y='CATE', order=order, ax=ax, palette=config.palette)
    ax.axhline(0, color='red', linestyle='--', alpha=0.5)
    ax.set_xlabel('Segment')
    ax.set_ylabel('CATE')
    ax.set_title(title)
    ax.tick_params(axis='x', rotation=45)

    return ax


# =============================================================================
# Bounds Plots
# =============================================================================

def plot_cate_bounds(
    cate_point: np.ndarray,
    cate_lower: np.ndarray,
    cate_upper: np.ndarray,
    x: Optional[np.ndarray] = None,
    title: str = "CATE Point Estimates with Partial ID Bounds",
    ax: Optional[plt.Axes] = None,
    config: PlotConfig = DEFAULT_CONFIG,
    sample_size: int = 200
) -> plt.Axes:
    """Plot CATE point estimates with partial identification bounds.

    Args:
        cate_point: Point estimates
        cate_lower: Lower bounds
        cate_upper: Upper bounds
        x: X-axis variable (index if None)
        title: Plot title
        ax: Matplotlib axes
        config: PlotConfig instance
        sample_size: Number of points to plot (for large data)

    Returns:
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=config.figsize, dpi=config.dpi)

    n = len(cate_point)
    if n > sample_size:
        idx = np.random.choice(n, sample_size, replace=False)
        idx = np.sort(idx)
    else:
        idx = np.arange(n)

    if x is None:
        x = np.arange(n)

    x_plot = x[idx] if len(x) == n else np.arange(len(idx))
    point_plot = cate_point[idx]
    lower_plot = cate_lower[idx]
    upper_plot = cate_upper[idx]

    # Sort by point estimate for better visualization
    sort_idx = np.argsort(point_plot)
    x_sorted = np.arange(len(sort_idx))
    point_sorted = point_plot[sort_idx]
    lower_sorted = lower_plot[sort_idx]
    upper_sorted = upper_plot[sort_idx]

    # Plot bounds as shaded region
    ax.fill_between(x_sorted, lower_sorted, upper_sorted, alpha=0.3, color='tab:blue', label='Bounds')
    ax.plot(x_sorted, point_sorted, 'o', markersize=2, color='tab:blue', label='Point estimate')
    ax.axhline(0, color='red', linestyle='--', alpha=0.5, label='Zero effect')

    ax.set_xlabel('Observation (sorted by CATE)')
    ax.set_ylabel('CATE')
    ax.set_title(title)
    ax.legend()

    return ax


def plot_bounds_by_ps(
    cate_lower: np.ndarray,
    cate_upper: np.ndarray,
    ps: np.ndarray,
    title: str = "CATE Bounds Width by Propensity Score",
    ax: Optional[plt.Axes] = None,
    config: PlotConfig = DEFAULT_CONFIG
) -> plt.Axes:
    """Show how CATE bounds widen in low-overlap PS regions.

    Args:
        cate_lower: Lower bounds
        cate_upper: Upper bounds
        ps: Propensity scores
        title: Plot title
        ax: Matplotlib axes
        config: PlotConfig instance

    Returns:
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=config.figsize, dpi=config.dpi)

    bounds_width = cate_upper - cate_lower

    ax.scatter(ps, bounds_width, alpha=0.3, s=10)

    # Add binned means
    ps_bins = pd.cut(ps, bins=10)
    df = pd.DataFrame({'ps': ps, 'width': bounds_width, 'ps_bin': ps_bins})
    binned_mean = df.groupby('ps_bin')['width'].mean()

    # Plot binned means
    bin_centers = [(b.left + b.right) / 2 for b in binned_mean.index]
    ax.plot(bin_centers, binned_mean.values, 'ro-', markersize=8, linewidth=2, label='Binned mean')

    # Overlap region
    ax.axvspan(0.1, 0.9, alpha=0.1, color='green', label='Overlap region')

    ax.set_xlabel('Propensity Score')
    ax.set_ylabel('Bounds Width')
    ax.set_title(title)
    ax.legend()

    return ax


# =============================================================================
# Sensitivity Plots
# =============================================================================

def plot_trimming_sensitivity(
    sensitivity_df: pd.DataFrame,
    title: str = "ATE Sensitivity to PS Trimming",
    ax: Optional[plt.Axes] = None,
    config: PlotConfig = DEFAULT_CONFIG
) -> plt.Axes:
    """Plot ATE sensitivity to trimming level.

    Args:
        sensitivity_df: DataFrame from positivity_sensitivity()
            Columns: trim_level, ate, ci_lower, ci_upper, n_remaining
        title: Plot title
        ax: Matplotlib axes
        config: PlotConfig instance

    Returns:
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=config.figsize, dpi=config.dpi)

    x = sensitivity_df['trim_level']
    y = sensitivity_df['ate']
    ci_lower = sensitivity_df['ci_lower']
    ci_upper = sensitivity_df['ci_upper']

    ax.errorbar(x, y, yerr=[y - ci_lower, ci_upper - y], fmt='o-', capsize=5,
                markersize=8, linewidth=2)

    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Trimming Threshold')
    ax.set_ylabel('ATE Estimate')
    ax.set_title(title)

    # Add sample size annotation
    for i, row in sensitivity_df.iterrows():
        ax.annotate(f"n={int(row['n_remaining'])}", (row['trim_level'], row['ate']),
                    textcoords="offset points", xytext=(0, 10), ha='center', fontsize=8)

    return ax


def plot_model_comparison_heatmap(
    comparison_df: pd.DataFrame,
    title: str = "CATE Model Correlation",
    ax: Optional[plt.Axes] = None,
    config: PlotConfig = DEFAULT_CONFIG
) -> plt.Axes:
    """Heatmap of CATE correlations across models.

    Args:
        comparison_df: DataFrame with CATE predictions (columns = models)
        title: Plot title
        ax: Matplotlib axes
        config: PlotConfig instance

    Returns:
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6), dpi=config.dpi)

    corr = comparison_df.corr()

    sns.heatmap(corr, annot=True, cmap='RdYlGn', center=0, ax=ax,
                fmt='.2f', square=True, linewidths=0.5)
    ax.set_title(title)

    return ax


# =============================================================================
# ATE Comparison Plots
# =============================================================================

def plot_ate_comparison(
    ate_df: pd.DataFrame,
    title: str = "ATE Estimates by Method",
    ax: Optional[plt.Axes] = None,
    config: PlotConfig = DEFAULT_CONFIG
) -> plt.Axes:
    """Bar plot comparing ATE estimates across methods.

    Args:
        ate_df: DataFrame from estimate_all_ate()
            Columns: method, estimate, ci_lower, ci_upper
        title: Plot title
        ax: Matplotlib axes
        config: PlotConfig instance

    Returns:
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=config.figsize, dpi=config.dpi)

    methods = ate_df['method']
    estimates = ate_df['estimate']

    # Error bars if available
    if 'ci_lower' in ate_df.columns and 'ci_upper' in ate_df.columns:
        yerr_lower = estimates - ate_df['ci_lower']
        yerr_upper = ate_df['ci_upper'] - estimates
        yerr = [yerr_lower, yerr_upper]
    else:
        yerr = None

    colors = plt.cm.Set2(np.linspace(0, 1, len(methods)))
    bars = ax.bar(methods, estimates, yerr=yerr, capsize=5, color=colors, alpha=0.8)

    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.set_ylabel('ATE Estimate')
    ax.set_title(title)
    ax.tick_params(axis='x', rotation=45)

    # Add value labels
    for bar, est in zip(bars, estimates):
        ax.annotate(f'{est:.2f}', (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    ha='center', va='bottom', fontsize=9)

    return ax


def plot_ate_forest(
    ate_df: pd.DataFrame,
    title: str = "ATE Forest Plot",
    ax: Optional[plt.Axes] = None,
    config: PlotConfig = DEFAULT_CONFIG
) -> plt.Axes:
    """Forest plot for ATE estimates.

    Args:
        ate_df: DataFrame with columns: method, estimate, ci_lower, ci_upper
        title: Plot title
        ax: Matplotlib axes
        config: PlotConfig instance

    Returns:
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, max(4, len(ate_df) * 0.6)), dpi=config.dpi)

    y_pos = range(len(ate_df))
    methods = ate_df['method']
    estimates = ate_df['estimate']

    # Plot CIs
    if 'ci_lower' in ate_df.columns and 'ci_upper' in ate_df.columns:
        for i, (_, row) in enumerate(ate_df.iterrows()):
            ax.plot([row['ci_lower'], row['ci_upper']], [i, i], 'b-', linewidth=2)

    # Plot point estimates
    ax.plot(estimates, y_pos, 'ko', markersize=8)

    # Reference line at 0
    ax.axvline(0, color='red', linestyle='--', alpha=0.5)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(methods)
    ax.set_xlabel('ATE Estimate')
    ax.set_title(title)

    return ax


# =============================================================================
# Covariate Experiment Plots
# =============================================================================

def plot_covariate_experiment(
    experiment_df: pd.DataFrame,
    title: str = "Covariate Set Comparison",
    config: PlotConfig = DEFAULT_CONFIG
) -> plt.Figure:
    """Plot results from covariate set experiments.

    Args:
        experiment_df: DataFrame from run_covariate_experiment()
        title: Plot title
        config: PlotConfig instance

    Returns:
        Matplotlib figure with multiple panels
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=config.dpi)

    # Panel 1: PS AUC by covariate set
    ax = axes[0]
    x = range(len(experiment_df))
    ax.bar(x, experiment_df['ps_auc'], color='steelblue', alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(experiment_df['var_set'], rotation=45, ha='right')
    ax.set_ylabel('PS AUC')
    ax.set_title('PS AUC by Covariate Set')
    ax.axhline(0.8, color='orange', linestyle='--', alpha=0.7, label='Warning threshold')
    ax.axhline(0.9, color='red', linestyle='--', alpha=0.7, label='Critical threshold')
    ax.legend()

    # Panel 2: Overlap ratio
    ax = axes[1]
    ax.bar(x, experiment_df['overlap_ratio'] * 100, color='seagreen', alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(experiment_df['var_set'], rotation=45, ha='right')
    ax.set_ylabel('Overlap Ratio (%)')
    ax.set_title('Overlap Ratio by Covariate Set')

    # Panel 3: ATE with CI
    ax = axes[2]
    if 'ate_ci_lower' in experiment_df.columns:
        yerr = [experiment_df['ate'] - experiment_df['ate_ci_lower'],
                experiment_df['ate_ci_upper'] - experiment_df['ate']]
        ax.errorbar(x, experiment_df['ate'], yerr=yerr, fmt='o', capsize=5,
                    markersize=8, color='coral')
    else:
        ax.bar(x, experiment_df['ate'], color='coral', alpha=0.7)

    ax.set_xticks(x)
    ax.set_xticklabels(experiment_df['var_set'], rotation=45, ha='right')
    ax.set_ylabel('ATE')
    ax.set_title('ATE by Covariate Set')
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)

    fig.suptitle(title, fontsize=config.title_fontsize)
    plt.tight_layout()
    return fig


# =============================================================================
# Summary Plots
# =============================================================================

def plot_positivity_summary(
    ps: np.ndarray,
    T: np.ndarray,
    balance_df: pd.DataFrame,
    config: PlotConfig = DEFAULT_CONFIG
) -> plt.Figure:
    """Summary plot for positivity diagnostics.

    Args:
        ps: Propensity scores
        T: Treatment indicator
        balance_df: Covariate balance DataFrame
        config: PlotConfig instance

    Returns:
        Matplotlib figure with PS and balance panels
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=config.dpi)

    plot_ps_overlap(ps, T, "Propensity Score Distribution", axes[0], config)
    plot_love_plot(balance_df, 0.1, "Covariate Balance", axes[1], config)

    plt.tight_layout()
    return fig


# =============================================================================
# ROI Plots
# =============================================================================

def plot_roi_curves(
    roi_df: pd.DataFrame,
    optimal_pct: float,
    figsize: Tuple[float, float] = (14, 5),
) -> Tuple[plt.Figure, Tuple[plt.Axes, plt.Axes]]:
    """Plot 2-panel ROI visualization: Profit curve + ROI curve.

    Left panel: Profit by targeting percentage with green/red shading.
    Right panel: ROI by targeting percentage.

    Args:
        roi_df: DataFrame with columns 'pct_targeted', 'profit', 'roi'
        optimal_pct: Optimal targeting percentage (for vertical line)
        figsize: Figure size tuple

    Returns:
        Tuple of (figure, (ax_profit, ax_roi))
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Left: Profit curve
    ax = axes[0]
    ax.plot(roi_df['pct_targeted'], roi_df['profit'], 'b-o', markersize=4)
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
    ax.axvline(x=optimal_pct, color='red', linestyle='--', label=f'Optimal: {optimal_pct}%')
    ax.fill_between(
        roi_df['pct_targeted'], 0, roi_df['profit'],
        where=roi_df['profit'] > 0, alpha=0.3, color='green'
    )
    ax.fill_between(
        roi_df['pct_targeted'], 0, roi_df['profit'],
        where=roi_df['profit'] < 0, alpha=0.3, color='red'
    )
    ax.set_xlabel('% of Customers Targeted')
    ax.set_ylabel('Profit ($)')
    ax.set_title('Profit by Targeting Percentage')
    ax.legend()

    # Right: ROI curve
    ax = axes[1]
    ax.plot(roi_df['pct_targeted'], roi_df['roi'] * 100, 'g-o', markersize=4)
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
    ax.axvline(x=optimal_pct, color='red', linestyle='--')
    ax.set_xlabel('% of Customers Targeted')
    ax.set_ylabel('ROI (%)')
    ax.set_title('ROI by Targeting Percentage')

    plt.tight_layout()
    return fig, (axes[0], axes[1])


# =============================================================================
# Refutation Test Plots
# =============================================================================

def plot_placebo_comparison(
    actual_cate: np.ndarray,
    placebo_cate: np.ndarray,
    title: str = "CATE Distribution: Actual vs Placebo",
    ax: Optional[plt.Axes] = None,
    config: PlotConfig = DEFAULT_CONFIG
) -> plt.Axes:
    """Side-by-side CATE distribution for actual vs placebo treatment.

    Args:
        actual_cate: CATE from real treatment
        placebo_cate: CATE from placebo (random) treatment
        title: Plot title
        ax: Matplotlib axes
        config: PlotConfig instance

    Returns:
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=config.figsize, dpi=config.dpi)

    # KDE plots
    sns.kdeplot(actual_cate, ax=ax, label='Actual Treatment',
                color='tab:blue', fill=True, alpha=0.3)
    sns.kdeplot(placebo_cate, ax=ax, label='Placebo Treatment',
                color='tab:red', fill=True, alpha=0.3)

    # Reference lines
    ax.axvline(x=0, color='gray', linestyle='--', linewidth=1, label='Zero Effect')
    ax.axvline(x=actual_cate.mean(), color='tab:blue', linestyle=':',
               linewidth=2, label=f'Actual Mean: {actual_cate.mean():.2f}')
    ax.axvline(x=placebo_cate.mean(), color='tab:red', linestyle=':',
               linewidth=2, label=f'Placebo Mean: {placebo_cate.mean():.2f}')

    ax.set_xlabel('CATE')
    ax.set_ylabel('Density')
    ax.set_title(title)
    ax.legend(loc='upper right')

    return ax


def plot_subset_correlation(
    full_cate: np.ndarray,
    subset_cate: np.ndarray,
    correlation: float,
    title: str = "CATE Correlation: Full vs Subset Model",
    ax: Optional[plt.Axes] = None,
    config: PlotConfig = DEFAULT_CONFIG
) -> plt.Axes:
    """Scatter plot of full vs subset model CATE predictions.

    Args:
        full_cate: CATE from full model
        subset_cate: CATE from subset model
        correlation: Pearson correlation coefficient
        title: Plot title
        ax: Matplotlib axes
        config: PlotConfig instance

    Returns:
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=config.figsize, dpi=config.dpi)

    ax.scatter(full_cate, subset_cate, alpha=0.5, s=30)

    # Identity line
    lims = [min(full_cate.min(), subset_cate.min()),
            max(full_cate.max(), subset_cate.max())]
    ax.plot(lims, lims, 'r--', linewidth=1, label='Identity')

    # Regression line
    z = np.polyfit(full_cate, subset_cate, 1)
    p = np.poly1d(z)
    ax.plot(lims, p(lims), 'b-', linewidth=1, label='Regression')

    ax.set_xlabel('Full Model CATE')
    ax.set_ylabel('Subset Model CATE')
    ax.set_title(f'{title}\nCorrelation: r = {correlation:.3f}')
    ax.legend()

    return ax


# =============================================================================
# Policy Learning Plots
# =============================================================================

def plot_policy_comparison(
    results_df: pd.DataFrame,
    metric: str = 'expected_profit',
    title: str = "Policy Comparison",
    ax: Optional[plt.Axes] = None,
    config: PlotConfig = DEFAULT_CONFIG
) -> plt.Axes:
    """Bar chart comparing policies by specified metric.

    Args:
        results_df: DataFrame with policy comparison results
        metric: Column name to plot (e.g., 'expected_profit', 'value_dr')
        title: Plot title
        ax: Matplotlib axes
        config: PlotConfig instance

    Returns:
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=config.figsize, dpi=config.dpi)

    # Sort by metric
    df_sorted = results_df.sort_values(metric, ascending=True)

    # Color based on positive/negative
    colors = ['tab:green' if v > 0 else 'tab:red' for v in df_sorted[metric]]

    ax.barh(df_sorted['name'], df_sorted[metric], color=colors, alpha=0.7)
    ax.axvline(0, color='black', linestyle='--', linewidth=1)

    ax.set_xlabel(metric.replace('_', ' ').title())
    ax.set_ylabel('Policy')
    ax.set_title(title)

    # Add value labels
    for i, (name, val) in enumerate(zip(df_sorted['name'], df_sorted[metric])):
        offset = 5 if val >= 0 else -5
        ha = 'left' if val >= 0 else 'right'
        ax.annotate(f'${val:,.0f}' if 'profit' in metric.lower() else f'{val:.2f}',
                   xy=(val, i), xytext=(offset, 0),
                   textcoords='offset points', ha=ha, va='center',
                   fontsize=config.tick_fontsize)

    return ax


def plot_policy_comparison_dual(
    results_df: pd.DataFrame,
    title: str = "Policy Comparison",
    figsize: Tuple[int, int] = (14, 6),
    config: PlotConfig = DEFAULT_CONFIG
) -> plt.Figure:
    """Two-panel policy comparison: value and profit.

    Args:
        results_df: DataFrame with policy comparison results
        title: Plot title
        figsize: Figure size
        config: PlotConfig instance

    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize, dpi=config.dpi)

    # Left: DR Policy Value
    df_sorted = results_df.sort_values('value_dr', ascending=True)
    colors = ['tab:blue' for _ in df_sorted['value_dr']]
    axes[0].barh(df_sorted['name'], df_sorted['value_dr'], color=colors, alpha=0.7)
    axes[0].set_xlabel('Policy Value (DR)')
    axes[0].set_ylabel('Policy')
    axes[0].set_title('Policy Value Comparison')

    # Right: Expected Profit
    df_sorted = results_df.sort_values('expected_profit', ascending=True)
    colors = ['tab:green' if v > 0 else 'tab:red' for v in df_sorted['expected_profit']]
    axes[1].barh(df_sorted['name'], df_sorted['expected_profit'], color=colors, alpha=0.7)
    axes[1].axvline(0, color='black', linestyle='--', linewidth=1)
    axes[1].set_xlabel('Expected Profit ($)')
    axes[1].set_ylabel('')
    axes[1].set_title('Expected Profit Comparison')

    fig.suptitle(title, fontsize=config.title_fontsize, y=1.02)
    fig.tight_layout()

    return fig


def plot_policy_regions(
    X: np.ndarray,
    policy: np.ndarray,
    feature_names: List[str],
    top_k_features: int = 2,
    importances: Optional[np.ndarray] = None,
    title: str = "Policy Decision Regions",
    ax: Optional[plt.Axes] = None,
    config: PlotConfig = DEFAULT_CONFIG
) -> plt.Axes:
    """2D visualization of policy decision regions.

    Args:
        X: Covariate matrix
        policy: Policy assignments 0/1
        feature_names: Feature names
        top_k_features: Number of top features to plot
        importances: Feature importances (to select top features)
        title: Plot title
        ax: Matplotlib axes
        config: PlotConfig instance

    Returns:
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=config.figsize, dpi=config.dpi)

    # Select top features
    if importances is not None:
        top_idx = np.argsort(importances)[-top_k_features:]
    else:
        top_idx = [0, 1]  # Default to first two

    x_idx, y_idx = top_idx[-1], top_idx[-2]
    x_name = feature_names[x_idx]
    y_name = feature_names[y_idx]

    # Scatter plot
    scatter = ax.scatter(
        X[:, x_idx], X[:, y_idx],
        c=policy, cmap='RdYlGn',
        alpha=0.6, s=50,
        edgecolors='gray', linewidths=0.5
    )

    ax.set_xlabel(x_name)
    ax.set_ylabel(y_name)
    ax.set_title(title)

    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Policy (0=Control, 1=Target)')
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(['Control', 'Target'])

    return ax


def plot_cate_confidence_scatter(
    cate: np.ndarray,
    cate_lower: np.ndarray,
    cate_upper: np.ndarray,
    breakeven: float,
    title: str = "CATE vs Uncertainty",
    ax: Optional[plt.Axes] = None,
    config: PlotConfig = DEFAULT_CONFIG
) -> plt.Axes:
    """Scatter plot of CATE point estimate vs bounds width.

    Args:
        cate: CATE point estimates
        cate_lower: Lower bounds
        cate_upper: Upper bounds
        breakeven: Breakeven CATE threshold
        title: Plot title
        ax: Matplotlib axes
        config: PlotConfig instance

    Returns:
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=config.figsize, dpi=config.dpi)

    bounds_width = cate_upper - cate_lower

    # Color by confident positive / negative / uncertain
    colors = []
    for i in range(len(cate)):
        if cate_lower[i] > breakeven:
            colors.append('tab:green')  # Confident positive
        elif cate_upper[i] < 0:
            colors.append('tab:red')  # Confident negative
        else:
            colors.append('tab:gray')  # Uncertain

    ax.scatter(cate, bounds_width, c=colors, alpha=0.6, s=50)

    # Reference lines
    ax.axvline(breakeven, color='orange', linestyle='--', linewidth=2,
               label=f'Breakeven: ${breakeven:.2f}')
    ax.axvline(0, color='gray', linestyle=':', linewidth=1, label='Zero')

    ax.set_xlabel('CATE Point Estimate')
    ax.set_ylabel('Bounds Width (Uncertainty)')
    ax.set_title(title)
    ax.legend()

    # Add legend for colors
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='tab:green', alpha=0.6, label='Confident Positive'),
        Patch(facecolor='tab:red', alpha=0.6, label='Confident Negative'),
        Patch(facecolor='tab:gray', alpha=0.6, label='Uncertain'),
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    return ax


def plot_segment_policy_heatmap(
    segment_policy_df: pd.DataFrame,
    segment_col: str = 'segment_name',
    policy_cols: Optional[List[str]] = None,
    title: str = "Targeting Rate by Segment and Policy",
    ax: Optional[plt.Axes] = None,
    config: PlotConfig = DEFAULT_CONFIG
) -> plt.Axes:
    """Heatmap of targeting rates by segment and policy.

    Args:
        segment_policy_df: DataFrame with segment and policy columns
        segment_col: Column name for segments
        policy_cols: List of policy column names
        title: Plot title
        ax: Matplotlib axes
        config: PlotConfig instance

    Returns:
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=config.figsize, dpi=config.dpi)

    if policy_cols is None:
        policy_cols = [c for c in segment_policy_df.columns if c != segment_col]

    # Pivot data if needed
    if segment_col in segment_policy_df.columns:
        plot_df = segment_policy_df.set_index(segment_col)[policy_cols]
    else:
        plot_df = segment_policy_df[policy_cols]

    sns.heatmap(plot_df, annot=True, fmt='.1%', cmap='YlGn',
                ax=ax, cbar_kws={'label': 'Targeting Rate'})

    ax.set_xlabel('Policy')
    ax.set_ylabel('Segment')
    ax.set_title(title)

    return ax


def plot_sensitivity_heatmap(
    sensitivity_df: pd.DataFrame,
    x_col: str = 'cost',
    y_col: str = 'margin',
    value_col: str = 'profit',
    title: str = "Sensitivity Analysis: Profit by Cost and Margin",
    ax: Optional[plt.Axes] = None,
    config: PlotConfig = DEFAULT_CONFIG
) -> plt.Axes:
    """Heatmap of sensitivity analysis results.

    Args:
        sensitivity_df: DataFrame with sensitivity results
        x_col: Column for x-axis
        y_col: Column for y-axis
        value_col: Column for cell values
        title: Plot title
        ax: Matplotlib axes
        config: PlotConfig instance

    Returns:
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=config.figsize, dpi=config.dpi)

    # Pivot data
    pivot_df = sensitivity_df.pivot(index=y_col, columns=x_col, values=value_col)

    # Center colormap at 0 for profit
    vmax = max(abs(pivot_df.values.min()), abs(pivot_df.values.max()))
    vmin = -vmax

    sns.heatmap(pivot_df, annot=True, fmt=',.0f', cmap='RdYlGn',
                center=0, vmin=vmin, vmax=vmax,
                ax=ax, cbar_kws={'label': value_col.replace('_', ' ').title()})

    ax.set_xlabel(x_col.replace('_', ' ').title())
    ax.set_ylabel(y_col.replace('_', ' ').title())
    ax.set_title(title)

    return ax


def plot_cv_policy_value(
    cv_result: Dict,
    title: str = "Cross-Validated Policy Value",
    ax: Optional[plt.Axes] = None,
    config: PlotConfig = DEFAULT_CONFIG
) -> plt.Axes:
    """Plot cross-validation results with CI.

    Args:
        cv_result: Dict with 'mean', 'std', 'ci_lower', 'ci_upper', 'fold_values'
        title: Plot title
        ax: Matplotlib axes
        config: PlotConfig instance

    Returns:
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=config.figsize, dpi=config.dpi)

    fold_values = cv_result.get('fold_values', [])
    mean_val = cv_result.get('mean', np.mean(fold_values))
    ci_lower = cv_result.get('ci_lower', np.percentile(fold_values, 2.5))
    ci_upper = cv_result.get('ci_upper', np.percentile(fold_values, 97.5))

    # Bar plot for each fold
    x = range(len(fold_values))
    ax.bar(x, fold_values, color='tab:blue', alpha=0.7, label='Fold Values')

    # Mean line
    ax.axhline(mean_val, color='red', linestyle='-', linewidth=2,
               label=f'Mean: {mean_val:.2f}')

    # CI band
    ax.axhspan(ci_lower, ci_upper, alpha=0.2, color='red',
               label=f'95% CI: [{ci_lower:.2f}, {ci_upper:.2f}]')

    ax.set_xlabel('Fold')
    ax.set_ylabel('Policy Value')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels([f'Fold {i+1}' for i in x])
    ax.legend()

    return ax


def plot_tree_depth_sensitivity(
    depth_df: pd.DataFrame,
    title: str = "Policy Sensitivity to Tree Depth",
    ax: Optional[plt.Axes] = None,
    config: PlotConfig = DEFAULT_CONFIG
) -> plt.Figure:
    """Plot sensitivity to tree depth.

    Args:
        depth_df: DataFrame with columns 'depth', 'n_targeted', 'policy_value'
        title: Plot title
        ax: Matplotlib axes (ignored, creates new figure)
        config: PlotConfig instance

    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=config.dpi)

    # Left: N Targeted
    axes[0].plot(depth_df['depth'], depth_df['n_targeted'], 'o-',
                 color='tab:blue', linewidth=2, markersize=8)
    axes[0].set_xlabel('Tree Depth')
    axes[0].set_ylabel('N Targeted')
    axes[0].set_title('Targeting Volume by Depth')
    axes[0].set_xticks(depth_df['depth'])

    # Right: Policy Value
    axes[1].plot(depth_df['depth'], depth_df['policy_value'], 's-',
                 color='tab:green', linewidth=2, markersize=8)
    axes[1].set_xlabel('Tree Depth')
    axes[1].set_ylabel('Policy Value')
    axes[1].set_title('Policy Value by Depth')
    axes[1].set_xticks(depth_df['depth'])

    fig.suptitle(title, fontsize=config.title_fontsize, y=1.02)
    fig.tight_layout()

    return fig


def plot_policy_tree_simple(
    tree_rules: List[Dict],
    title: str = "Policy Tree Rules",
    ax: Optional[plt.Axes] = None,
    config: PlotConfig = DEFAULT_CONFIG
) -> plt.Axes:
    """Simple visualization of extracted tree rules.

    Args:
        tree_rules: List of rule dictionaries from extract_tree_rules
        title: Plot title
        ax: Matplotlib axes
        config: PlotConfig instance

    Returns:
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, max(6, len(tree_rules) * 0.5)),
                               dpi=config.dpi)

    # Format rules
    y_positions = range(len(tree_rules))
    colors = ['tab:green' if r['action'] == 'TARGET' else 'tab:red'
              for r in tree_rules]

    for i, rule in enumerate(tree_rules):
        # Format conditions
        conditions = ' AND '.join([
            f"{feat} {op} {val:.2f}" if isinstance(val, float) else f"{feat} {op} {val}"
            for feat, op, val in rule['conditions']
        ])

        text = f"IF {conditions}\n→ {rule['action']} (n={rule['n_samples']})"

        ax.text(0.02, i, text, fontsize=config.tick_fontsize,
                verticalalignment='center',
                bbox=dict(boxstyle='round', facecolor=colors[i], alpha=0.3))

    ax.set_xlim(0, 1)
    ax.set_ylim(-0.5, len(tree_rules) - 0.5)
    ax.set_title(title)
    ax.axis('off')

    return ax
