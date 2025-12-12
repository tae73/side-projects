"""
Correlation analysis utilities for heterogeneous variable types.

Supports: numeric (continuous), categorical (nominal), ordinal variables.
Automatically selects appropriate correlation method based on variable types.
"""

from typing import NamedTuple, Callable, Literal
from itertools import combinations
import numpy as np
import pandas as pd
from scipy import stats


VariableType = Literal['numeric', 'categorical', 'ordinal']


class CorrelationResult(NamedTuple):
    """Result of a pairwise correlation computation."""
    var1: str
    var2: str
    correlation: float
    p_value: float | None
    method: str


class VariableSpec(NamedTuple):
    """Specification for a variable's type and optional ordering."""
    name: str
    var_type: VariableType
    order: list | None = None  # For ordinal variables


def _pearson(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Pearson correlation for numeric-numeric."""
    mask = ~(np.isnan(x) | np.isnan(y))
    if mask.sum() < 3:
        return np.nan, np.nan
    r, p = stats.pearsonr(x[mask], y[mask])
    return r, p


def _spearman(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Spearman correlation for ordinal-ordinal or ordinal-numeric."""
    mask = ~(np.isnan(x) | np.isnan(y))
    if mask.sum() < 3:
        return np.nan, np.nan
    rho, p = stats.spearmanr(x[mask], y[mask])
    return rho, p


def _point_biserial(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Point-biserial correlation for binary categorical-numeric."""
    mask = ~(np.isnan(x) | np.isnan(y))
    if mask.sum() < 3:
        return np.nan, np.nan
    r, p = stats.pointbiserialr(x[mask].astype(int), y[mask])
    return r, p


def _cramers_v(x: np.ndarray, y: np.ndarray) -> tuple[float, None]:
    """Cramér's V for categorical-categorical."""
    mask = ~(pd.isna(x) | pd.isna(y))
    x_clean, y_clean = x[mask], y[mask]

    if len(x_clean) < 2:
        return np.nan, None

    contingency = pd.crosstab(x_clean, y_clean)
    chi2, _, _, _ = stats.chi2_contingency(contingency)
    n = contingency.sum().sum()
    min_dim = min(contingency.shape) - 1

    if min_dim == 0 or n == 0:
        return np.nan, None

    v = np.sqrt(chi2 / (n * min_dim))
    return v, None


def _correlation_ratio(categories: np.ndarray, values: np.ndarray) -> tuple[float, None]:
    """
    Correlation ratio (eta) for categorical-numeric.
    Measures the proportion of variance in numeric explained by categorical.
    """
    mask = ~(pd.isna(categories) | np.isnan(values))
    categories_clean = categories[mask]
    values_clean = values[mask]

    if len(values_clean) < 2:
        return np.nan, None

    groups = pd.Series(values_clean).groupby(categories_clean)
    ss_between = sum(
        len(group) * (group.mean() - values_clean.mean()) ** 2
        for _, group in groups
    )
    ss_total = ((values_clean - values_clean.mean()) ** 2).sum()

    if ss_total == 0:
        return np.nan, None

    eta = np.sqrt(ss_between / ss_total)
    return eta, None


def _rank_biserial(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Rank-biserial correlation for binary categorical-ordinal."""
    mask = ~(pd.isna(x) | np.isnan(y))
    x_clean, y_clean = x[mask], y[mask]

    unique_vals = np.unique(x_clean)
    if len(unique_vals) != 2:
        return np.nan, None

    group1 = y_clean[x_clean == unique_vals[0]]
    group2 = y_clean[x_clean == unique_vals[1]]

    if len(group1) < 1 or len(group2) < 1:
        return np.nan, None

    stat, p = stats.mannwhitneyu(group1, group2, alternative='two-sided')
    n1, n2 = len(group1), len(group2)
    r = 1 - (2 * stat) / (n1 * n2)
    return r, p


def _encode_ordinal(series: pd.Series, order: list | None = None) -> np.ndarray:
    """Encode ordinal variable to numeric ranks."""
    if order is not None:
        mapping = {val: idx for idx, val in enumerate(order)}
        return series.map(mapping).to_numpy(dtype=float)
    # Infer order from sorted unique values
    unique_sorted = sorted(series.dropna().unique())
    mapping = {val: idx for idx, val in enumerate(unique_sorted)}
    return series.map(mapping).to_numpy(dtype=float)


def _encode_categorical(series: pd.Series) -> np.ndarray:
    """Encode categorical variable, preserving original values for contingency."""
    return series.to_numpy()


def _is_binary(arr: np.ndarray) -> bool:
    """Check if array has exactly 2 unique non-null values."""
    unique = pd.Series(arr).dropna().unique()
    return len(unique) == 2


def _select_method(
    type1: VariableType,
    type2: VariableType,
    is_binary1: bool,
    is_binary2: bool
) -> tuple[Callable, str]:
    """Select appropriate correlation method based on variable types."""

    # Numeric - Numeric
    if type1 == 'numeric' and type2 == 'numeric':
        return _pearson, 'pearson'

    # Ordinal - Ordinal
    if type1 == 'ordinal' and type2 == 'ordinal':
        return _spearman, 'spearman'

    # Numeric - Ordinal (either order)
    if {type1, type2} == {'numeric', 'ordinal'}:
        return _spearman, 'spearman'

    # Categorical - Categorical
    if type1 == 'categorical' and type2 == 'categorical':
        return _cramers_v, 'cramers_v'

    # Binary Categorical - Numeric
    if type1 == 'categorical' and type2 == 'numeric' and is_binary1:
        return _point_biserial, 'point_biserial'
    if type2 == 'categorical' and type1 == 'numeric' and is_binary2:
        return lambda x, y: _point_biserial(y, x), 'point_biserial'

    # Non-binary Categorical - Numeric
    if type1 == 'categorical' and type2 == 'numeric':
        return lambda x, y: _correlation_ratio(x, y), 'correlation_ratio'
    if type2 == 'categorical' and type1 == 'numeric':
        return lambda x, y: _correlation_ratio(y, x), 'correlation_ratio'

    # Binary Categorical - Ordinal
    if type1 == 'categorical' and type2 == 'ordinal' and is_binary1:
        return _rank_biserial, 'rank_biserial'
    if type2 == 'categorical' and type1 == 'ordinal' and is_binary2:
        return lambda x, y: _rank_biserial(y, x), 'rank_biserial'

    # Non-binary Categorical - Ordinal: treat ordinal as categorical
    if {type1, type2} == {'categorical', 'ordinal'}:
        return _cramers_v, 'cramers_v'

    # Fallback
    return _spearman, 'spearman'


def compute_pairwise_correlation(
    df: pd.DataFrame,
    var1_spec: VariableSpec,
    var2_spec: VariableSpec
) -> CorrelationResult:
    """Compute correlation between two variables with specified types."""

    # Encode variables based on type
    if var1_spec.var_type == 'numeric':
        arr1 = df[var1_spec.name].to_numpy(dtype=float)
    elif var1_spec.var_type == 'ordinal':
        arr1 = _encode_ordinal(df[var1_spec.name], var1_spec.order)
    else:
        arr1 = _encode_categorical(df[var1_spec.name])

    if var2_spec.var_type == 'numeric':
        arr2 = df[var2_spec.name].to_numpy(dtype=float)
    elif var2_spec.var_type == 'ordinal':
        arr2 = _encode_ordinal(df[var2_spec.name], var2_spec.order)
    else:
        arr2 = _encode_categorical(df[var2_spec.name])

    # Select method and compute
    is_binary1 = _is_binary(arr1)
    is_binary2 = _is_binary(arr2)

    method_fn, method_name = _select_method(
        var1_spec.var_type, var2_spec.var_type,
        is_binary1, is_binary2
    )

    corr, p_val = method_fn(arr1, arr2)

    return CorrelationResult(
        var1=var1_spec.name,
        var2=var2_spec.name,
        correlation=corr,
        p_value=p_val,
        method=method_name
    )


def compute_correlation_matrix(
    df: pd.DataFrame,
    var_specs: list[VariableSpec]
) -> pd.DataFrame:
    """
    Compute correlation matrix for variables of mixed types.

    Args:
        df: DataFrame containing the variables
        var_specs: List of VariableSpec defining each variable's type

    Returns:
        DataFrame with correlation values (lower triangle) and methods (as attributes)
    """
    n_vars = len(var_specs)
    var_names = [spec.name for spec in var_specs]

    # Initialize matrices
    corr_matrix = np.eye(n_vars)
    p_matrix = np.zeros((n_vars, n_vars))
    method_matrix = np.empty((n_vars, n_vars), dtype=object)
    np.fill_diagonal(method_matrix, '-')

    # Compute all pairwise correlations
    results = [
        compute_pairwise_correlation(df, var_specs[i], var_specs[j])
        for i, j in combinations(range(n_vars), 2)
    ]

    # Fill matrices
    for result in results:
        i = var_names.index(result.var1)
        j = var_names.index(result.var2)
        corr_matrix[i, j] = corr_matrix[j, i] = result.correlation
        p_matrix[i, j] = p_matrix[j, i] = result.p_value if result.p_value is not None else np.nan
        method_matrix[i, j] = method_matrix[j, i] = result.method

    # Create DataFrame
    corr_df = pd.DataFrame(corr_matrix, index=var_names, columns=var_names)
    corr_df.attrs['p_values'] = pd.DataFrame(p_matrix, index=var_names, columns=var_names)
    corr_df.attrs['methods'] = pd.DataFrame(method_matrix, index=var_names, columns=var_names)

    return corr_df


def infer_variable_type(series: pd.Series, ordinal_threshold: int = 10) -> VariableType:
    """
    Infer variable type from pandas Series.

    Heuristics:
    - Numeric dtype with many unique values -> 'numeric'
    - Numeric dtype with few unique values -> 'ordinal'
    - Object/category dtype -> 'categorical'
    """
    if pd.api.types.is_numeric_dtype(series):
        n_unique = series.nunique()
        if n_unique <= ordinal_threshold:
            return 'ordinal'
        return 'numeric'
    return 'categorical'


def auto_correlation_matrix(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    ordinal_overrides: dict[str, list] | None = None,
    type_overrides: dict[str, VariableType] | None = None
) -> pd.DataFrame:
    """
    Automatically compute correlation matrix with type inference.

    Args:
        df: Input DataFrame
        columns: Columns to include (default: all)
        ordinal_overrides: Dict mapping column names to their ordinal order
        type_overrides: Dict mapping column names to forced variable types

    Returns:
        Correlation matrix DataFrame with methods stored in attrs
    """
    columns = columns or df.columns.tolist()
    ordinal_overrides = ordinal_overrides or {}
    type_overrides = type_overrides or {}

    var_specs = []
    for col in columns:
        if col in type_overrides:
            var_type = type_overrides[col]
        else:
            var_type = infer_variable_type(df[col])

        order = ordinal_overrides.get(col)
        var_specs.append(VariableSpec(name=col, var_type=var_type, order=order))

    return compute_correlation_matrix(df, var_specs)


def get_significant_correlations(
    corr_df: pd.DataFrame,
    threshold: float = 0.3,
    p_threshold: float = 0.05
) -> pd.DataFrame:
    """
    Extract significant correlations from a correlation matrix.

    Args:
        corr_df: Correlation matrix from compute_correlation_matrix
        threshold: Minimum absolute correlation value
        p_threshold: Maximum p-value (ignored if p-value not available)

    Returns:
        DataFrame with significant correlations sorted by absolute value
    """
    p_values = corr_df.attrs.get('p_values')
    methods = corr_df.attrs.get('methods')

    results = []
    n = len(corr_df)

    for i in range(n):
        for j in range(i + 1, n):
            var1, var2 = corr_df.index[i], corr_df.columns[j]
            corr = corr_df.iloc[i, j]

            if abs(corr) < threshold:
                continue

            p_val = p_values.iloc[i, j] if p_values is not None else None
            if p_val is not None and not np.isnan(p_val) and p_val >= p_threshold:
                continue

            method = methods.iloc[i, j] if methods is not None else None

            results.append({
                'var1': var1,
                'var2': var2,
                'correlation': corr,
                'p_value': p_val,
                'method': method
            })

    return (
        pd.DataFrame(results)
        .sort_values('correlation', key=abs, ascending=False)
        .reset_index(drop=True)
    )


# =============================================================================
# Usage Examples
# =============================================================================

if __name__ == '__main__':
    # Sample data with mixed variable types
    np.random.seed(42)
    n = 200

    df_sample = pd.DataFrame({
        'age': np.random.normal(45, 15, n),                           # numeric
        'income': np.random.exponential(50000, n),                    # numeric
        'satisfaction': np.random.choice([1, 2, 3, 4, 5], n),         # ordinal
        'education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n),  # ordinal
        'gender': np.random.choice(['M', 'F'], n),                    # categorical (binary)
        'region': np.random.choice(['North', 'South', 'East', 'West'], n),  # categorical
    })

    # Ordinal ordering
    education_order = ['High School', 'Bachelor', 'Master', 'PhD']

    # -------------------------------------------------------------------------
    # Example 1: Auto correlation matrix (automatic type inference)
    # -------------------------------------------------------------------------
    print("=" * 60)
    print("Example 1: Auto Correlation Matrix")
    print("=" * 60)

    corr_matrix = auto_correlation_matrix(
        df_sample,
        ordinal_overrides={'education': education_order}
    )

    print("\nCorrelation Matrix:")
    print(corr_matrix.round(3))

    print("\nMethods Used:")
    print(corr_matrix.attrs['methods'])

    # -------------------------------------------------------------------------
    # Example 2: Manual variable specification
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Example 2: Manual Variable Specification")
    print("=" * 60)

    var_specs = [
        VariableSpec('age', 'numeric'),
        VariableSpec('income', 'numeric'),
        VariableSpec('satisfaction', 'ordinal'),
        VariableSpec('education', 'ordinal', order=education_order),
        VariableSpec('gender', 'categorical'),
    ]

    corr_matrix_manual = compute_correlation_matrix(df_sample, var_specs)
    print("\nCorrelation Matrix (manual specs):")
    print(corr_matrix_manual.round(3))

    # -------------------------------------------------------------------------
    # Example 3: Pairwise correlation
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Example 3: Pairwise Correlation")
    print("=" * 60)

    result = compute_pairwise_correlation(
        df_sample,
        VariableSpec('age', 'numeric'),
        VariableSpec('income', 'numeric')
    )
    print(f"\n{result.var1} vs {result.var2}:")
    print(f"  Method: {result.method}")
    print(f"  Correlation: {result.correlation:.3f}")
    print(f"  P-value: {result.p_value:.4f}")

    # -------------------------------------------------------------------------
    # Example 4: Get significant correlations
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Example 4: Significant Correlations (|r| >= 0.1)")
    print("=" * 60)

    significant = get_significant_correlations(corr_matrix, threshold=0.1)
    print("\n")
    print(significant.to_string(index=False))

    # -------------------------------------------------------------------------
    # Example 5: Type inference
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Example 5: Variable Type Inference")
    print("=" * 60)

    for col in df_sample.columns:
        inferred = infer_variable_type(df_sample[col])
        print(f"  {col}: {inferred}")