"""
Reusable plot functions with Config -> plot pattern.
"""
from typing import NamedTuple, Optional, List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# =============================================================================
# Config Classes
# =============================================================================

class KDEConfig(NamedTuple):
    """Configuration for KDE plot."""
    col: str
    title: str
    color: str


class HistConfig(NamedTuple):
    """Configuration for histogram plot."""
    col: str
    title: str
    xlabel: str
    color: str
    fmt: str  # mean label format, e.g., '{:.1f}', '${:.0f}'


class TrendConfig(NamedTuple):
    """Configuration for single-axis trend plot."""
    x_col: str
    y_col: str
    title: str
    xlabel: str
    ylabel: str
    color: str = 'tab:blue'
    marker: str = 'o'
    linewidth: int = 2


class DualAxisConfig(NamedTuple):
    """Configuration for dual-axis trend plot."""
    x_col: str
    left_col: str
    right_col: str
    title: str
    left_label: str = 'Transaction Count'
    right_label: str = 'Total Spend'
    left_color: str = 'tab:blue'
    right_color: str = 'tab:red'
    figsize: Tuple[int, int] = (15, 8)


class BarHConfig(NamedTuple):
    """Configuration for horizontal bar chart."""
    title: str
    xlabel: str
    color: str
    n: int = 20


class ParetoConfig(NamedTuple):
    """Configuration for Pareto (concentration) curve."""
    title: str
    xlabel: str
    ylabel: str
    thresholds: Tuple[float, ...] = (50, 80)
    threshold_colors: Tuple[str, ...] = ('orange', 'red')


class ScatterConfig(NamedTuple):
    """Configuration for scatter plot."""
    x_col: str
    y_col: str
    title: str
    xlabel: str
    ylabel: str
    color: str = 'steelblue'
    alpha: float = 0.5
    s: int = 30
    label_col: Optional[str] = None
    n_labels: int = 5


class HeatmapConfig(NamedTuple):
    """Configuration for heatmap plot."""
    title: str
    fmt: str = 'd'
    cmap: str = 'Blues'
    annot: bool = True
    xlabel: Optional[str] = None
    ylabel: Optional[str] = None


# =============================================================================
# Plot Functions
# =============================================================================

def plot_kde(ax, df: pd.DataFrame, config: KDEConfig):
    """
    Plot KDE distribution with mean line.

    Args:
        ax: matplotlib axes object
        df: DataFrame containing the data
        config: KDEConfig with col, title, color

    Returns:
        ax: matplotlib axes object
    """
    sns.kdeplot(data=df, x=config.col, ax=ax, fill=True, color=config.color, alpha=0.7)
    mean_val = df[config.col].mean()
    ax.axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.1f}')
    ax.set_title(f'Distribution of {config.title}')
    ax.legend()
    return ax


def plot_hist(ax, df: pd.DataFrame, config: HistConfig):
    """
    Plot histogram with mean line.

    Args:
        ax: matplotlib axes object
        df: DataFrame containing the data
        config: HistConfig with col, title, xlabel, color, fmt

    Returns:
        ax: matplotlib axes object
    """
    ax.hist(df[config.col], bins=50, color=config.color, edgecolor='white')
    mean_val = df[config.col].mean()
    ax.axvline(mean_val, color='red', linestyle='--', label=f"Mean: {config.fmt.format(mean_val)}")
    ax.set_xlabel(config.xlabel)
    ax.set_ylabel('Count')
    ax.set_title(f'Distribution of {config.title}')
    ax.legend()
    return ax


def plot_trend(ax, df: pd.DataFrame, config: TrendConfig):
    """
    Plot single-axis trend line.

    Args:
        ax: matplotlib axes object
        df: DataFrame containing the data
        config: TrendConfig with x_col, y_col, title, labels, styling

    Returns:
        ax: matplotlib axes object
    """
    sns.lineplot(
        data=df.reset_index(), x=config.x_col, y=config.y_col,
        ax=ax, color=config.color, marker=config.marker, linewidth=config.linewidth
    )
    ax.set_xlabel(config.xlabel, fontsize=12)
    ax.set_ylabel(config.ylabel, color=config.color, fontsize=12)
    ax.tick_params(axis='y', labelcolor=config.color)
    ax.set_title(config.title, fontsize=14, fontweight='bold')
    return ax


def plot_dual_axis_trend(df: pd.DataFrame, config: DualAxisConfig):
    """
    Plot dual-axis time series trend. Wraps plot_trend for both axes.

    Args:
        df: DataFrame with index as x-axis values
        config: DualAxisConfig with axis columns and labels

    Returns:
        fig, ax1, ax2: figure and both axes
    """
    fig, ax1 = plt.subplots(figsize=config.figsize)

    # Left axis
    left_config = TrendConfig(
        x_col=config.x_col, y_col=config.left_col, title='',
        xlabel=config.x_col, ylabel=config.left_label, color=config.left_color
    )
    plot_trend(ax1, df, left_config)

    # Right axis
    ax2 = ax1.twinx()
    right_config = TrendConfig(
        x_col=config.x_col, y_col=config.right_col, title='',
        xlabel='', ylabel=config.right_label, color=config.right_color, marker='s'
    )
    plot_trend(ax2, df, right_config)

    plt.title(config.title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig, ax1, ax2


def plot_top_n_barh(ax, series: pd.Series, config: BarHConfig):
    """
    Plot horizontal bar chart for top N items.

    Args:
        ax: matplotlib axes object
        series: Series with index as labels, values for bar lengths
        config: BarHConfig with title, xlabel, color, n

    Returns:
        ax: matplotlib axes object
    """
    series.head(config.n).plot(kind='barh', ax=ax, color=config.color, alpha=0.8)
    ax.set_title(config.title, fontsize=12, fontweight='bold')
    ax.set_xlabel(config.xlabel)
    ax.invert_yaxis()
    return ax


def plot_pareto_curve(ax, cumsum_pct: pd.Series, config: ParetoConfig):
    """
    Plot Pareto (concentration) curve with threshold lines.

    Args:
        ax: matplotlib axes object
        cumsum_pct: Series of cumulative percentages (0-100)
        config: ParetoConfig with title, labels, thresholds

    Returns:
        ax: matplotlib axes object
    """
    x_vals = np.arange(1, len(cumsum_pct) + 1) / len(cumsum_pct) * 100
    ax.plot(x_vals, cumsum_pct.values, 'b-', linewidth=2)

    for threshold, color in zip(config.thresholds, config.threshold_colors):
        ax.axhline(y=threshold, color=color, linestyle='--', label=f'{threshold}% threshold')

    ax.set_xlabel(config.xlabel)
    ax.set_ylabel(config.ylabel)
    ax.set_title(config.title, fontsize=12, fontweight='bold')
    ax.legend()
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)
    return ax


def plot_scatter(ax, df: pd.DataFrame, config: ScatterConfig):
    """
    Plot scatter plot with optional labels.

    Args:
        ax: matplotlib axes object
        df: DataFrame containing x and y columns
        config: ScatterConfig with columns, labels, styling

    Returns:
        ax: matplotlib axes object
    """
    ax.scatter(
        df[config.x_col], df[config.y_col],
        s=config.s, alpha=config.alpha, c=config.color
    )

    # Labels for top items (optional)
    if config.label_col:
        for _, row in df.nlargest(config.n_labels, config.y_col).iterrows():
            ax.annotate(
                row[config.label_col],
                (row[config.x_col], row[config.y_col]),
                fontsize=8, alpha=0.8
            )

    ax.set_xlabel(config.xlabel)
    ax.set_ylabel(config.ylabel)
    ax.set_title(config.title, fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)
    return ax


def plot_heatmap(ax, data: pd.DataFrame, config: HeatmapConfig):
    """
    Plot heatmap with annotations.

    Args:
        ax: matplotlib axes object
        data: DataFrame (2D matrix) for heatmap
        config: HeatmapConfig with title, format, colormap

    Returns:
        ax: matplotlib axes object
    """
    sns.heatmap(data, annot=config.annot, fmt=config.fmt, cmap=config.cmap, ax=ax)
    ax.set_title(config.title, fontsize=12, fontweight='bold')
    if config.xlabel:
        ax.set_xlabel(config.xlabel)
    if config.ylabel:
        ax.set_ylabel(config.ylabel)
    return ax


# =============================================================================
# Usage Examples
# =============================================================================

if __name__ == '__main__':
    # Sample data
    np.random.seed(42)
    n = 100

    df_sample = pd.DataFrame({
        'value': np.random.normal(100, 20, n),
        'category': np.random.choice(['A', 'B', 'C'], n),
        'week': np.repeat(range(1, 11), 10),
        'sales': np.random.exponential(50, n),
        'transactions': np.random.poisson(10, n),
    })

    df_weekly = df_sample.groupby('week').agg({
        'sales': 'sum',
        'transactions': 'sum'
    })

    # 1. KDE Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    config = KDEConfig(col='value', title='Value', color='skyblue')
    plot_kde(ax, df_sample, config)
    plt.tight_layout()
    plt.show()

    # 2. Histogram
    fig, ax = plt.subplots(figsize=(8, 5))
    config = HistConfig(col='sales', title='Sales', xlabel='Sales ($)',
                        color='coral', fmt='${:.0f}')
    plot_hist(ax, df_sample, config)
    plt.tight_layout()
    plt.show()

    # 3. Single Trend
    fig, ax = plt.subplots(figsize=(10, 5))
    config = TrendConfig(x_col='week', y_col='sales', title='Weekly Sales',
                         xlabel='Week', ylabel='Sales ($)')
    plot_trend(ax, df_weekly, config)
    plt.tight_layout()
    plt.show()

    # 4. Dual Axis Trend
    config = DualAxisConfig(x_col='week', left_col='transactions', right_col='sales',
                            title='Weekly Trends', left_label='Transactions',
                            right_label='Sales ($)')
    fig, ax1, ax2 = plot_dual_axis_trend(df_weekly, config)
    plt.show()

    # 5. Horizontal Bar Chart
    category_sales = df_sample.groupby('category')['sales'].sum().sort_values()
    fig, ax = plt.subplots(figsize=(8, 4))
    config = BarHConfig(title='Sales by Category', xlabel='Sales ($)',
                        color='teal', n=10)
    plot_top_n_barh(ax, category_sales, config)
    plt.tight_layout()
    plt.show()

    # 6. Pareto Curve
    sales_sorted = df_sample['sales'].sort_values(ascending=False)
    cumsum_pct = sales_sorted.cumsum() / sales_sorted.sum() * 100
    fig, ax = plt.subplots(figsize=(8, 6))
    config = ParetoConfig(title='Sales Concentration',
                          xlabel='% of Items', ylabel='% of Sales')
    plot_pareto_curve(ax, cumsum_pct, config)
    plt.tight_layout()
    plt.show()

    # 7. Scatter Plot (with correlation in title)
    fig, ax = plt.subplots(figsize=(8, 6))
    config = ScatterConfig(x_col='value', y_col='sales', title='Value vs Sales',
                           xlabel='Value', ylabel='Sales ($)')
    plot_scatter(ax, df_sample, config)
    corr = df_sample[['value', 'sales']].corr().iloc[0, 1]
    ax.set_title(f'Value vs Sales (r = {corr:.3f})')
    plt.tight_layout()
    plt.show()

    # 8. Heatmap
    crosstab = pd.crosstab(df_sample['category'],
                           pd.cut(df_sample['value'], bins=3, labels=['Low', 'Mid', 'High']))
    fig, ax = plt.subplots(figsize=(8, 5))
    config = HeatmapConfig(title='Category vs Value Range', fmt='d', cmap='YlOrRd')
    plot_heatmap(ax, crosstab, config)
    plt.tight_layout()
    plt.show()
