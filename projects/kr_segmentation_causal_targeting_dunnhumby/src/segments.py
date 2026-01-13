"""Customer segmentation modeling module.

This module provides functions for NMF-based factor analysis and clustering
for customer segmentation (Track 1 Step 1.1).

Design Pattern:
    - Config objects (NamedTuple): Input parameters for models
    - Result objects (NamedTuple): Structured outputs from functions

Main Components:
    - NMF: train_nmf, compute_nmf_metrics, interpret_factors
    - Clustering: train_clustering, evaluate_clustering, compare_clustering_methods
    - Profiling: profile_segments, test_segment_differences
    - Stability: bootstrap_stability, split_half_stability
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import NamedTuple, Any
from sklearn.decomposition import NMF
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    adjusted_rand_score,
)
from sklearn.utils import resample
from scipy.stats import f_oneway


# =============================================================================
# Configuration Objects
# =============================================================================

class NMFConfig(NamedTuple):
    """Configuration for NMF model."""
    n_components: int
    init: str = 'nndsvda'
    max_iter: int = 1000
    random_state: int = 42
    alpha_W: float = 0.0
    alpha_H: float = 0.0
    l1_ratio: float = 0.0


class ClusterConfig(NamedTuple):
    """Configuration for clustering model."""
    n_clusters: int
    method: str = 'kmeans'
    random_state: int = 42
    n_init: int = 10


# =============================================================================
# Result Objects
# =============================================================================

class NMFResult(NamedTuple):
    """Result from NMF training."""
    model: NMF
    W: np.ndarray              # Customer factor scores (n_samples, n_components)
    H: np.ndarray              # Factor loadings (n_components, n_features)
    reconstruction_error: float
    n_iter: int


class ClusterResult(NamedTuple):
    """Result from clustering."""
    model: Any                 # KMeans or GaussianMixture
    labels: np.ndarray
    silhouette: float
    calinski_harabasz: float
    davies_bouldin: float


class StabilityResult(NamedTuple):
    """Result from stability analysis."""
    mean_ari: float
    std_ari: float
    n_bootstrap: int


class PipelineResult(NamedTuple):
    """Result from NMF + Clustering pipeline."""
    nmf_result: NMFResult
    cluster_result: ClusterResult
    factor_scores: pd.DataFrame
    factor_loadings: pd.DataFrame
    segment_df: pd.DataFrame


# =============================================================================
# NMF Functions
# =============================================================================

def train_nmf(X: np.ndarray, config: NMFConfig) -> NMFResult:
    """Train NMF model and return factor matrices.

    Args:
        X: Feature matrix (n_samples, n_features), must be non-negative
        config: NMF configuration

    Returns:
        NMFResult containing model, W, H, reconstruction_error, n_iter
    """
    nmf = NMF(
        n_components=config.n_components,
        init=config.init,
        max_iter=config.max_iter,
        random_state=config.random_state,
        alpha_W=config.alpha_W,
        alpha_H=config.alpha_H,
        l1_ratio=config.l1_ratio,
    )
    W = nmf.fit_transform(X)
    H = nmf.components_

    return NMFResult(
        model=nmf,
        W=W,
        H=H,
        reconstruction_error=nmf.reconstruction_err_,
        n_iter=nmf.n_iter_,
    )


def compute_nmf_metrics(X: np.ndarray, n_range, random_state: int = 42) -> pd.DataFrame:
    """Evaluate NMF across a range of component numbers.

    Args:
        X: Feature matrix (n_samples, n_features), must be non-negative
        n_range: Range of n_components to evaluate (e.g., range(2, 12))
        random_state: Random seed

    Returns:
        DataFrame with columns:
        - n_components: Number of components
        - reconstruction_error: Frobenius norm of reconstruction error
        - explained_variance: 1 - (residual_var / total_var)
        - sparsity_W: L1/L2 ratio of W matrix
        - sparsity_H: L1/L2 ratio of H matrix
    """
    def _sparsity(M):
        l1 = np.abs(M).sum()
        l2 = np.sqrt((M ** 2).sum())
        return l1 / l2 if l2 > 0 else 0

    results = []
    total_var = np.var(X)

    for n in n_range:
        config = NMFConfig(n_components=n, random_state=random_state)
        nmf_result = train_nmf(X, config)

        # Reconstruction error (Frobenius norm)
        X_reconstructed = nmf_result.W @ nmf_result.H
        residual_var = np.var(X - X_reconstructed)
        explained_var = 1 - (residual_var / total_var) if total_var > 0 else 0

        results.append({
            'n_components': n,
            'reconstruction_error': nmf_result.reconstruction_error,
            'explained_variance': explained_var,
            'sparsity_W': _sparsity(nmf_result.W),
            'sparsity_H': _sparsity(nmf_result.H),
        })

    return pd.DataFrame(results)


def interpret_factors(loadings_df: pd.DataFrame, top_n: int = 5) -> dict:
    """Get top features for each factor.

    Args:
        loadings_df: DataFrame with features as index and factors as columns
        top_n: Number of top features to return per factor

    Returns:
        Dict mapping factor name to Series of top features with loadings
    """
    return {
        factor: loadings_df[factor].sort_values(ascending=False).head(top_n)
        for factor in loadings_df.columns
    }


def create_factor_dataframes(
    nmf_result: NMFResult,
    customer_index,
    feature_names: list
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create DataFrames for factor scores and loadings.

    Args:
        nmf_result: Result from train_nmf
        customer_index: Index for customers (e.g., household_key)
        feature_names: List of feature names

    Returns:
        Tuple of (df_factor_scores, df_factor_loadings)
    """
    n_components = nmf_result.W.shape[1]
    factor_names = [f'factor_{i+1}' for i in range(n_components)]

    df_factor_scores = pd.DataFrame(
        nmf_result.W,
        columns=factor_names,
        index=customer_index
    )

    df_factor_loadings = pd.DataFrame(
        nmf_result.H.T,
        columns=factor_names,
        index=feature_names
    )

    return df_factor_scores, df_factor_loadings


# =============================================================================
# Clustering Functions
# =============================================================================

def train_clustering(X: np.ndarray, config: ClusterConfig) -> ClusterResult:
    """Train clustering model and return labels with metrics.

    Args:
        X: Feature matrix (n_samples, n_features)
        config: Cluster configuration

    Returns:
        ClusterResult containing model, labels, and metrics
    """
    if config.method == 'kmeans':
        model = KMeans(
            n_clusters=config.n_clusters,
            n_init=config.n_init,
            random_state=config.random_state,
        )
        labels = model.fit_predict(X)
    elif config.method == 'gmm':
        model = GaussianMixture(
            n_components=config.n_clusters,
            n_init=config.n_init,
            random_state=config.random_state,
        )
        labels = model.fit_predict(X)
    else:
        raise ValueError(f"Unknown method: {config.method}. Use 'kmeans' or 'gmm'.")

    return ClusterResult(
        model=model,
        labels=labels,
        silhouette=silhouette_score(X, labels),
        calinski_harabasz=calinski_harabasz_score(X, labels),
        davies_bouldin=davies_bouldin_score(X, labels),
    )


def evaluate_clustering(
    X: np.ndarray,
    k_range,
    method: str = 'kmeans',
    random_state: int = 42
) -> pd.DataFrame:
    """Evaluate clustering metrics across a range of cluster numbers.

    Args:
        X: Feature matrix (n_samples, n_features)
        k_range: Range of n_clusters to evaluate (e.g., range(2, 11))
        method: 'kmeans' or 'gmm'
        random_state: Random seed

    Returns:
        DataFrame with columns:
        - k: Number of clusters
        - inertia: Sum of squared distances (K-Means only)
        - silhouette: Silhouette score (-1 to 1, higher is better)
        - calinski_harabasz: Calinski-Harabasz index (higher is better)
        - davies_bouldin: Davies-Bouldin index (lower is better)
        - bic: Bayesian Information Criterion (GMM only, lower is better)
    """
    results = []

    for k in k_range:
        config = ClusterConfig(n_clusters=k, method=method, random_state=random_state)
        result = train_clustering(X, config)

        row = {
            'k': k,
            'silhouette': result.silhouette,
            'calinski_harabasz': result.calinski_harabasz,
            'davies_bouldin': result.davies_bouldin,
        }

        if method == 'kmeans':
            row['inertia'] = result.model.inertia_
        elif method == 'gmm':
            row['bic'] = result.model.bic(X)

        results.append(row)

    return pd.DataFrame(results)


def compare_clustering_methods(
    X: np.ndarray,
    k: int,
    random_state: int = 42
) -> dict[str, ClusterResult]:
    """Compare K-Means and GMM clustering for a given k.

    Args:
        X: Feature matrix (n_samples, n_features)
        k: Number of clusters
        random_state: Random seed

    Returns:
        Dict with keys 'kmeans' and 'gmm', each containing ClusterResult
    """
    return {
        method: train_clustering(
            X,
            ClusterConfig(n_clusters=k, method=method, random_state=random_state)
        )
        for method in ['kmeans', 'gmm']
    }


# =============================================================================
# Segment Profiling Functions
# =============================================================================

def profile_segments(
    df: pd.DataFrame,
    features: list,
    segment_col: str = 'segment'
) -> pd.DataFrame:
    """Compute feature statistics by segment.

    Args:
        df: DataFrame with segment column and features
        features: List of feature column names
        segment_col: Name of segment column

    Returns:
        DataFrame with segments as index and mean/std for each feature
    """
    profiles = df.groupby(segment_col)[features].agg(['mean', 'std'])
    profiles.columns = ['_'.join(col) for col in profiles.columns]
    return profiles


def compute_segment_sizes(
    df: pd.DataFrame,
    segment_col: str = 'segment'
) -> pd.DataFrame:
    """Compute segment sizes and proportions.

    Args:
        df: DataFrame with segment column
        segment_col: Name of segment column

    Returns:
        DataFrame with segment, count, and proportion columns
    """
    sizes = df.groupby(segment_col).size().reset_index(name='count')
    sizes['proportion'] = sizes['count'] / sizes['count'].sum()
    return sizes.sort_values('count', ascending=False)


def test_segment_differences(
    df: pd.DataFrame,
    features: list,
    segment_col: str = 'segment'
) -> pd.DataFrame:
    """Test if features differ significantly across segments using ANOVA.

    Args:
        df: DataFrame with segment column and features
        features: List of feature column names
        segment_col: Name of segment column

    Returns:
        DataFrame with columns:
        - feature: Feature name
        - f_stat: F-statistic
        - p_value: p-value
        - significant: Boolean (p < 0.05)
    """
    results = []

    for feature in features:
        groups = [group[feature].dropna().values
                  for _, group in df.groupby(segment_col)]

        if any(len(g) == 0 for g in groups):
            continue

        try:
            stat, pvalue = f_oneway(*groups)
            results.append({
                'feature': feature,
                'f_stat': stat,
                'p_value': pvalue,
                'significant': pvalue < 0.05,
            })
        except Exception:
            continue

    return pd.DataFrame(results).sort_values('p_value')


def standardize_profiles(profiles_df: pd.DataFrame) -> pd.DataFrame:
    """Standardize profile means to z-scores for visualization.

    Args:
        profiles_df: Output from profile_segments()

    Returns:
        DataFrame with z-scores for mean columns
    """
    mean_cols = [c for c in profiles_df.columns if c.endswith('_mean')]
    z_scores = profiles_df[mean_cols].apply(
        lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0,
        axis=0
    )
    z_scores.columns = [c.replace('_mean', '') for c in mean_cols]
    return z_scores


# =============================================================================
# Stability Analysis Functions
# =============================================================================

def bootstrap_stability(
    X: np.ndarray,
    config: ClusterConfig,
    n_bootstrap: int = 100,
    sample_frac: float = 0.8
) -> StabilityResult:
    """Assess clustering stability via bootstrap resampling.

    Args:
        X: Feature matrix (n_samples, n_features)
        config: Cluster configuration
        n_bootstrap: Number of bootstrap iterations
        sample_frac: Fraction of samples to use in each bootstrap

    Returns:
        StabilityResult containing mean_ari, std_ari, n_bootstrap
    """
    # Fit on full data
    full_result = train_clustering(X, config)
    labels_full = full_result.labels

    ari_scores = []
    n_samples = int(len(X) * sample_frac)

    for i in range(n_bootstrap):
        # Bootstrap sample
        idx = resample(
            range(len(X)),
            n_samples=n_samples,
            random_state=config.random_state + i
        )

        X_boot = X[idx] if isinstance(X, np.ndarray) else X.iloc[idx].values

        # Fit on bootstrap sample
        boot_config = config._replace(random_state=config.random_state + i)
        boot_result = train_clustering(X_boot, boot_config)

        # Compare with full model on same samples
        labels_full_sub = labels_full[idx]
        ari = adjusted_rand_score(labels_full_sub, boot_result.labels)
        ari_scores.append(ari)

    return StabilityResult(
        mean_ari=np.mean(ari_scores),
        std_ari=np.std(ari_scores),
        n_bootstrap=n_bootstrap,
    )


def split_half_stability(
    X_h1: np.ndarray,
    X_h2: np.ndarray,
    nmf_config: NMFConfig,
    cluster_config: ClusterConfig
) -> float:
    """Assess stability by comparing segments from two time periods.

    Args:
        X_h1: Feature matrix for first half (n_samples, n_features)
        X_h2: Feature matrix for second half (same samples, different period)
        nmf_config: NMF configuration
        cluster_config: Cluster configuration

    Returns:
        Adjusted Rand Index between segments from two halves
    """
    # NMF on each half
    nmf_h1 = train_nmf(X_h1, nmf_config)
    nmf_h2 = train_nmf(X_h2, nmf_config)

    # Clustering on each half
    cluster_h1 = train_clustering(nmf_h1.W, cluster_config)
    cluster_h2 = train_clustering(nmf_h2.W, cluster_config)

    return adjusted_rand_score(cluster_h1.labels, cluster_h2.labels)


# =============================================================================
# Pipeline Functions
# =============================================================================

def run_nmf_clustering_pipeline(
    X: np.ndarray,
    feature_names: list,
    customer_index,
    nmf_config: NMFConfig,
    cluster_config: ClusterConfig
) -> PipelineResult:
    """Run complete NMF + Clustering pipeline.

    Args:
        X: Feature matrix (n_samples, n_features), must be non-negative
        feature_names: List of feature names
        customer_index: Index for customers (e.g., household_key values)
        nmf_config: NMF configuration
        cluster_config: Cluster configuration

    Returns:
        PipelineResult containing all outputs
    """
    # NMF
    nmf_result = train_nmf(X, nmf_config)

    # Create factor DataFrames
    df_factor_scores, df_factor_loadings = create_factor_dataframes(
        nmf_result, customer_index, feature_names
    )

    # Clustering
    cluster_result = train_clustering(nmf_result.W, cluster_config)

    # Create segment DataFrame
    df_segments = df_factor_scores.reset_index()
    df_segments.columns = ['household_key'] + list(df_factor_scores.columns)
    df_segments['segment'] = cluster_result.labels

    return PipelineResult(
        nmf_result=nmf_result,
        cluster_result=cluster_result,
        factor_scores=df_factor_scores,
        factor_loadings=df_factor_loadings,
        segment_df=df_segments,
    )


# =============================================================================
# Visualization Functions
# =============================================================================

def plot_cluster_bubble(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    segment_col: str = 'segment',
    segment_names: dict = None,
    size_scale: float = 3000,
    title: str = None,
    xlabel: str = None,
    ylabel: str = None,
    ax: plt.Axes = None,
    figsize: tuple = (10, 8),
    colors: list = None,
    alpha: float = 0.6,
    show_legend: bool = True,
) -> plt.Axes:
    """Create bubble chart for cluster comparison.

    Each bubble represents a segment with:
    - Position (x, y): Segment mean of x_col and y_col
    - Size: Segment count (number of customers)
    - Label: Segment number + name (if provided)
    - Color: Distinct color per segment

    Args:
        df: DataFrame with segment column and feature columns
        x_col: Column name for x-axis (will compute segment mean)
        y_col: Column name for y-axis (will compute segment mean)
        segment_col: Name of segment column
        segment_names: Dict mapping segment number to name (e.g., {0: 'VIP'})
        size_scale: Scaling factor for bubble sizes
        title: Chart title (auto-generated if None)
        xlabel: X-axis label (uses x_col if None)
        ylabel: Y-axis label (uses y_col if None)
        ax: Matplotlib axes (creates new figure if None)
        figsize: Figure size if creating new figure
        colors: List of colors for segments (uses tab10 if None)
        alpha: Bubble transparency
        show_legend: Whether to show size legend

    Returns:
        Matplotlib Axes object
    """
    # Compute segment statistics
    segment_stats = df.groupby(segment_col).agg(
        x_mean=(x_col, 'mean'),
        y_mean=(y_col, 'mean'),
        count=(segment_col, 'size'),
    ).reset_index()

    # Create figure if needed
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    # Colors
    if colors is None:
        cmap = plt.cm.tab10
        colors = [cmap(i) for i in range(len(segment_stats))]

    # Plot bubbles
    for i, row in segment_stats.iterrows():
        seg = row[segment_col]
        size = row['count'] / df.shape[0] * size_scale

        ax.scatter(
            row['x_mean'],
            row['y_mean'],
            s=size,
            c=[colors[i % len(colors)]],
            alpha=alpha,
            edgecolors='white',
            linewidths=1.5,
        )

        # Label
        label = f"Seg {seg}"
        if segment_names and seg in segment_names:
            label = f"{seg}: {segment_names[seg]}"

        ax.annotate(
            label,
            (row['x_mean'], row['y_mean']),
            ha='center',
            va='center',
            fontsize=9,
            fontweight='bold',
        )

    # Labels and title
    ax.set_xlabel(xlabel or x_col, fontsize=11)
    ax.set_ylabel(ylabel or y_col, fontsize=11)
    ax.set_title(title or f'{y_col} vs {x_col}', fontsize=12, fontweight='bold')

    # Grid
    ax.grid(True, alpha=0.3)
    ax.axhline(y=segment_stats['y_mean'].mean(), color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=segment_stats['x_mean'].mean(), color='gray', linestyle='--', alpha=0.5)

    # Size legend
    if show_legend:
        # Create legend entries for size reference
        sizes_pct = [0.10, 0.15, 0.20]
        legend_handles = [
            ax.scatter([], [], s=pct * size_scale, c='gray', alpha=0.5,
                       label=f'{pct*100:.0f}%')
            for pct in sizes_pct
        ]
        ax.legend(
            handles=legend_handles,
            title='Segment Size',
            loc='upper right',
            framealpha=0.9,
        )

    plt.tight_layout()
    return ax


def plot_cluster_bubble_grid(
    df: pd.DataFrame,
    axis_pairs: list[tuple[str, str]],
    segment_col: str = 'segment',
    segment_names: dict = None,
    titles: list[str] = None,
    figsize: tuple = (15, 10),
    ncols: int = 2,
    size_scale: float = 2000,
) -> plt.Figure:
    """Create grid of bubble charts for multiple axis combinations.

    Args:
        df: DataFrame with segment column and feature columns
        axis_pairs: List of (x_col, y_col) tuples
        segment_col: Name of segment column
        segment_names: Dict mapping segment number to name
        titles: List of titles for each chart (auto-generated if None)
        figsize: Overall figure size
        ncols: Number of columns in grid
        size_scale: Scaling factor for bubble sizes

    Returns:
        Matplotlib Figure object
    """
    n_charts = len(axis_pairs)
    nrows = (n_charts + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = np.array(axes).flatten() if n_charts > 1 else [axes]

    for i, (x_col, y_col) in enumerate(axis_pairs):
        title = titles[i] if titles and i < len(titles) else None
        plot_cluster_bubble(
            df=df,
            x_col=x_col,
            y_col=y_col,
            segment_col=segment_col,
            segment_names=segment_names,
            title=title,
            ax=axes[i],
            size_scale=size_scale,
            show_legend=(i == 0),  # Only show legend on first chart
        )

    # Hide unused axes
    for j in range(n_charts, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    return fig
