"""General utility functions."""

from typing import Optional, List
import pandas as pd
import numpy as np


def safe_qcut(
    series: pd.Series,
    q: int = 4,
    labels: Optional[List[str]] = None
) -> pd.Series:
    """Create quantile bins with handling for low-variance data.

    Handles edge cases where:
    - Data has too few unique values
    - Data is constant
    - Standard qcut would fail

    Args:
        series: Numeric series to bin
        q: Number of quantiles
        labels: Optional labels for bins

    Returns:
        Categorical series with quantile bins
    """
    try:
        return pd.qcut(series, q=q, labels=labels, duplicates='drop')
    except ValueError:
        unique_vals = series.nunique()
        if unique_vals <= 1:
            return pd.Series(['All'] * len(series), index=series.index)
        elif unique_vals < q:
            return pd.cut(series, bins=unique_vals, labels=labels[:unique_vals] if labels else None)
        else:
            return pd.qcut(series, q=q, labels=labels, duplicates='drop')


def format_currency(value: float, decimals: int = 2) -> str:
    """Format number as currency string.

    Args:
        value: Numeric value
        decimals: Number of decimal places

    Returns:
        Formatted string (e.g., "$1,234.56")

    Examples:
        >>> format_currency(1234.5678)
        '$1,234.57'
        >>> format_currency(-500)
        '-$500.00'
    """
    if value < 0:
        return f"-${abs(value):,.{decimals}f}"
    return f"${value:,.{decimals}f}"


def format_percentage(value: float, decimals: int = 1) -> str:
    """Format number as percentage string.

    Args:
        value: Numeric value (0.5 = 50%)
        decimals: Number of decimal places

    Returns:
        Formatted string (e.g., "50.0%")

    Examples:
        >>> format_percentage(0.5)
        '50.0%'
        >>> format_percentage(1.234, decimals=2)
        '123.40%'
    """
    return f"{value * 100:.{decimals}f}%"


def create_ps_region_labels(
    ps: np.ndarray,
    thresholds: tuple = (0.1, 0.9),
    labels: Optional[List[str]] = None
) -> pd.Series:
    """Create propensity score region labels.

    Args:
        ps: Propensity scores
        thresholds: (lower, upper) threshold for overlap region
        labels: Custom labels [low, overlap, high]

    Returns:
        Series with region labels
    """
    if labels is None:
        labels = [
            f'Extreme Low (<{thresholds[0]})',
            f'Overlap ({thresholds[0]}-{thresholds[1]})',
            f'Extreme High (>{thresholds[1]})'
        ]

    return pd.cut(
        ps,
        bins=[0, thresholds[0], thresholds[1], 1.0],
        labels=labels,
        include_lowest=True
    )


def is_in_overlap(ps: np.ndarray, thresholds: tuple = (0.1, 0.9)) -> np.ndarray:
    """Check if propensity scores are in overlap region.

    Args:
        ps: Propensity scores
        thresholds: (lower, upper) threshold for overlap

    Returns:
        Boolean array (True = in overlap)
    """
    return (ps >= thresholds[0]) & (ps <= thresholds[1])


def summary_stats(
    data: np.ndarray,
    name: str = 'value'
) -> pd.Series:
    """Compute summary statistics for an array.

    Args:
        data: Numeric array
        name: Name for the series

    Returns:
        Series with mean, std, min, 25%, 50%, 75%, max
    """
    return pd.Series({
        'count': len(data),
        'mean': np.mean(data),
        'std': np.std(data),
        'min': np.min(data),
        '25%': np.percentile(data, 25),
        '50%': np.percentile(data, 50),
        '75%': np.percentile(data, 75),
        'max': np.max(data),
    }, name=name)
