"""
Table One utility for case-control studies.

Generates descriptive statistics tables with effect sizes for group comparisons.
Designed for large datasets where p-values alone are insufficient.
"""

from typing import NamedTuple, Literal, Callable
from functools import reduce
from itertools import starmap
import numpy as np
import pandas as pd
from scipy import stats


VariableType = Literal['continuous', 'categorical', 'binary']


class EffectSize(NamedTuple):
    """Effect size result with interpretation."""
    value: float
    ci_lower: float | None
    ci_upper: float | None
    interpretation: str


class TestResult(NamedTuple):
    """Statistical test result."""
    statistic: float
    p_value: float
    test_name: str
    effect_size: EffectSize


class VariableStats(NamedTuple):
    """Summary statistics for a variable."""
    name: str
    var_type: VariableType
    overall: str
    group_stats: dict[str, str]
    test_result: TestResult | None


# Effect size interpretation thresholds (Cohen's conventions)
COHENS_D_THRESHOLDS = [(0.2, 'negligible'), (0.5, 'small'), (0.8, 'medium'), (np.inf, 'large')]
CRAMERS_V_THRESHOLDS = [(0.1, 'negligible'), (0.3, 'small'), (0.5, 'medium'), (np.inf, 'large')]


def _interpret_effect_size(value: float, thresholds: list[tuple[float, str]]) -> str:
    """Interpret effect size magnitude."""
    abs_val = abs(value)
    return next(
        (label for threshold, label in thresholds if abs_val < threshold),
        'large'
    )


def _cohens_d(group1: np.ndarray, group2: np.ndarray) -> EffectSize:
    """
    Calculate Cohen's d for two independent groups.

    Uses pooled standard deviation with Hedges' correction for small samples.
    """
    n1, n2 = len(group1), len(group2)

    if n1 < 2 or n2 < 2:
        return EffectSize(np.nan, None, None, 'insufficient data')

    mean1, mean2 = np.nanmean(group1), np.nanmean(group2)
    var1, var2 = np.nanvar(group1, ddof=1), np.nanvar(group2, ddof=1)

    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

    if pooled_std == 0:
        return EffectSize(np.nan, None, None, 'zero variance')

    d = (mean1 - mean2) / pooled_std

    # Hedges' correction for small samples
    correction = 1 - (3 / (4 * (n1 + n2) - 9))
    d_corrected = d * correction

    # Approximate 95% CI
    se = np.sqrt((n1 + n2) / (n1 * n2) + d_corrected**2 / (2 * (n1 + n2)))
    ci_lower = d_corrected - 1.96 * se
    ci_upper = d_corrected + 1.96 * se

    interpretation = _interpret_effect_size(d_corrected, COHENS_D_THRESHOLDS)

    return EffectSize(d_corrected, ci_lower, ci_upper, interpretation)


def _compute_cramers_v_value(contingency_table: np.ndarray) -> float:
    """Compute Cramér's V value from contingency table array."""
    try:
        chi2, _, _, _ = stats.chi2_contingency(contingency_table)
    except ValueError:
        return np.nan
    n = contingency_table.sum()
    r, k = contingency_table.shape
    if n <= 1:
        return np.nan
    phi2 = chi2 / n
    phi2_corrected = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    r_corrected = r - ((r - 1)**2) / (n - 1)
    k_corrected = k - ((k - 1)**2) / (n - 1)
    denom = min(k_corrected - 1, r_corrected - 1)
    if denom <= 0:
        return np.nan
    return np.sqrt(phi2_corrected / denom)


def _cramers_v(contingency_table: pd.DataFrame, bootstrap_ci: bool = False, n_bootstrap: int = 1000) -> EffectSize:
    """
    Calculate Cramér's V for categorical variables.

    Includes bias correction for small samples.
    Optionally computes bootstrap CI.
    """
    v = _compute_cramers_v_value(contingency_table.values)

    if np.isnan(v):
        return EffectSize(np.nan, None, None, 'insufficient categories')

    interpretation = _interpret_effect_size(v, CRAMERS_V_THRESHOLDS)

    if not bootstrap_ci:
        return EffectSize(v, None, None, interpretation)

    # Bootstrap CI
    table_flat = contingency_table.values
    n = table_flat.sum()
    r, k = table_flat.shape

    # Create row/col indices for resampling
    rows, cols = [], []
    for i in range(r):
        for j in range(k):
            rows.extend([i] * int(table_flat[i, j]))
            cols.extend([j] * int(table_flat[i, j]))
    rows, cols = np.array(rows), np.array(cols)

    bootstrap_vs = []
    rng = np.random.default_rng(42)
    for _ in range(n_bootstrap):
        idx = rng.choice(len(rows), size=len(rows), replace=True)
        boot_table = np.zeros((r, k))
        for ri, ci in zip(rows[idx], cols[idx]):
            boot_table[ri, ci] += 1
        boot_v = _compute_cramers_v_value(boot_table)
        if not np.isnan(boot_v):
            bootstrap_vs.append(boot_v)

    if len(bootstrap_vs) < 100:
        return EffectSize(v, None, None, interpretation)

    ci_lower, ci_upper = np.percentile(bootstrap_vs, [2.5, 97.5])
    return EffectSize(v, ci_lower, ci_upper, interpretation)


def _odds_ratio(contingency_table: pd.DataFrame) -> EffectSize:
    """
    Calculate odds ratio for 2x2 tables.
    """
    if contingency_table.shape != (2, 2):
        return EffectSize(np.nan, None, None, 'not 2x2 table')

    table = contingency_table.values
    a, b = table[0, 0], table[0, 1]
    c, d = table[1, 0], table[1, 1]

    # Add 0.5 for zero cells (Haldane-Anscombe correction)
    if 0 in [a, b, c, d]:
        a, b, c, d = a + 0.5, b + 0.5, c + 0.5, d + 0.5

    odds_ratio = (a * d) / (b * c)
    log_or = np.log(odds_ratio)

    # 95% CI on log scale
    se_log = np.sqrt(1/a + 1/b + 1/c + 1/d)
    ci_lower = np.exp(log_or - 1.96 * se_log)
    ci_upper = np.exp(log_or + 1.96 * se_log)

    # Interpretation based on OR
    if odds_ratio < 0.5:
        interpretation = 'strong protective'
    elif odds_ratio < 0.8:
        interpretation = 'moderate protective'
    elif odds_ratio <= 1.25:
        interpretation = 'negligible'
    elif odds_ratio <= 2.0:
        interpretation = 'moderate risk'
    else:
        interpretation = 'strong risk'

    return EffectSize(odds_ratio, ci_lower, ci_upper, interpretation)


def _rank_biserial(group1: np.ndarray, group2: np.ndarray) -> EffectSize:
    """
    Calculate rank-biserial correlation (effect size for Mann-Whitney U).
    """
    n1, n2 = len(group1), len(group2)

    if n1 < 1 or n2 < 1:
        return EffectSize(np.nan, None, None, 'insufficient data')

    u_stat, _ = stats.mannwhitneyu(group1, group2, alternative='two-sided')
    r = 1 - (2 * u_stat) / (n1 * n2)

    interpretation = _interpret_effect_size(r, COHENS_D_THRESHOLDS)

    return EffectSize(r, None, None, interpretation)


def _format_continuous(values: np.ndarray, use_median: bool = False) -> str:
    """Format continuous variable summary."""
    valid = values[~np.isnan(values)]

    if len(valid) == 0:
        return 'N/A'

    if use_median:
        median = np.median(valid)
        q1, q3 = np.percentile(valid, [25, 75])
        return f'{median:.1f} [{q1:.1f}, {q3:.1f}]'
    else:
        mean = np.mean(valid)
        std = np.std(valid, ddof=1)
        return f'{mean:.1f} ({std:.1f})'


def _format_categorical(values: pd.Series) -> str:
    """Format categorical variable summary."""
    counts = values.value_counts()
    total = len(values.dropna())

    if total == 0:
        return 'N/A'

    return '; '.join(
        f'{cat}: {count} ({100*count/total:.1f}%)'
        for cat, count in counts.items()
    )


def _test_continuous(
    group1: np.ndarray,
    group2: np.ndarray,
    normal_test: bool = True
) -> TestResult:
    """Perform appropriate test for continuous variable."""
    # Check normality if requested
    is_normal = True
    if normal_test and len(group1) >= 8 and len(group2) >= 8:
        _, p1 = stats.shapiro(group1[:5000])  # Limit for large samples
        _, p2 = stats.shapiro(group2[:5000])
        is_normal = p1 > 0.05 and p2 > 0.05

    if is_normal:
        stat, p = stats.ttest_ind(group1, group2, nan_policy='omit')
        effect = _cohens_d(group1, group2)
        test_name = "Student's t-test"
    else:
        stat, p = stats.mannwhitneyu(group1, group2, alternative='two-sided')
        effect = _rank_biserial(group1, group2)
        test_name = 'Mann-Whitney U'

    return TestResult(stat, p, test_name, effect)


def _test_categorical(
    contingency_table: pd.DataFrame,
    bootstrap_ci: bool = False
) -> TestResult:
    """Perform appropriate test for categorical variable."""
    # Check expected frequencies
    chi2, p, dof, expected = stats.chi2_contingency(contingency_table)

    if contingency_table.shape == (2, 2):
        # Use Fisher's exact for 2x2 with small expected counts
        if (expected < 5).any():
            _, p = stats.fisher_exact(contingency_table.values)
            test_name = "Fisher's exact"
        else:
            test_name = 'Chi-squared'
        effect = _odds_ratio(contingency_table)
    else:
        if (expected < 5).sum() > 0.2 * expected.size:
            test_name = 'Chi-squared (warning: expected < 5)'
        else:
            test_name = 'Chi-squared'
        effect = _cramers_v(contingency_table, bootstrap_ci=bootstrap_ci)

    return TestResult(chi2, p, test_name, effect)


def infer_variable_type(
    series: pd.Series,
    categorical_threshold: int = 10
) -> VariableType:
    """Infer variable type from data."""
    if pd.api.types.is_numeric_dtype(series):
        n_unique = series.nunique()
        if n_unique == 2:
            return 'binary'
        elif n_unique <= categorical_threshold:
            return 'categorical'
        return 'continuous'
    return 'categorical'


def _format_category_row(
    col: str,
    category: str,
    df: pd.DataFrame,
    group_dfs: dict,
    group_names: list,
    is_first: bool,
    test_result: TestResult | None
) -> dict:
    """Format a single category row for expanded categorical output."""
    total = len(df[col].dropna())
    count = (df[col] == category).sum()
    overall = f'{count} ({100*count/total:.1f}%)'

    group_stats = {}
    for g in group_names:
        g_total = len(group_dfs[g][col].dropna())
        g_count = (group_dfs[g][col] == category).sum()
        group_stats[g] = f'{g_count} ({100*g_count/g_total:.1f}%)' if g_total > 0 else 'N/A'

    row = {
        'Variable': col if is_first else '',
        'Category': category,
        'Overall': overall,
    }

    for g in group_names:
        row[f'{g} (n={len(group_dfs[g])})'] = group_stats[g]

    # Test results only on first row
    if is_first and test_result:
        row.update({
            'Test': test_result.test_name,
            'Statistic': f'{test_result.statistic:.2f}',
            'P-value': _format_pvalue(test_result.p_value),
            'Effect Size': f'{test_result.effect_size.value:.3f}' if not np.isnan(test_result.effect_size.value) else 'N/A',
            'Effect CI': _format_ci(test_result.effect_size),
            'Interpretation': test_result.effect_size.interpretation,
        })
    else:
        row.update({
            'Test': '',
            'Statistic': '',
            'P-value': '',
            'Effect Size': '',
            'Effect CI': '',
            'Interpretation': '',
        })

    return row


def _format_category_row_with_options(
    col: str,
    category: str,
    df: pd.DataFrame,
    group_dfs: dict,
    group_names: list,
    is_first: bool,
    test_result: TestResult | None,
    include_effect_size: bool
) -> dict:
    """Format a single category row for expanded categorical output with effect size option."""
    total = len(df[col].dropna())
    count = (df[col] == category).sum()
    overall = f'{count} ({100*count/total:.1f}%)'

    group_stats = {}
    for g in group_names:
        g_total = len(group_dfs[g][col].dropna())
        g_count = (group_dfs[g][col] == category).sum()
        group_stats[g] = f'{g_count} ({100*g_count/g_total:.1f}%)' if g_total > 0 else 'N/A'

    row = {
        'Variable': col if is_first else '',
        'Category': category,
        'Overall': overall,
    }

    for g in group_names:
        row[f'{g} (n={len(group_dfs[g])})'] = group_stats[g]

    # Test results only on first row
    if is_first and test_result:
        row.update({
            'Test': test_result.test_name,
            'Statistic': f'{test_result.statistic:.2f}',
            'P-value': _format_pvalue(test_result.p_value),
        })
        if include_effect_size:
            row.update({
                'Effect Size': f'{test_result.effect_size.value:.3f}' if not np.isnan(test_result.effect_size.value) else 'N/A',
                'Effect CI': _format_ci(test_result.effect_size),
                'Interpretation': test_result.effect_size.interpretation,
            })
    else:
        row.update({
            'Test': '',
            'Statistic': '',
            'P-value': '',
        })
        if include_effect_size:
            row.update({
                'Effect Size': '',
                'Effect CI': '',
                'Interpretation': '',
            })

    return row


def compute_tableone(
    df: pd.DataFrame,
    groupby: str,
    columns: list[str] | None = None,
    categorical: list[str] | None = None,
    continuous: list[str] | None = None,
    normal_test: bool = True,
    categorical_threshold: int = 10,
    expand_categorical: bool = False,
    effect_size: bool = True,
    bootstrap_ci: bool = False
) -> pd.DataFrame:
    """
    Generate Table One with effect sizes for case-control comparison.

    Args:
        df: Input DataFrame
        groupby: Column name for group comparison (case vs control)
        columns: Columns to include (default: all except groupby)
        categorical: Columns to treat as categorical
        continuous: Columns to treat as continuous
        normal_test: Whether to test normality for continuous variables
        categorical_threshold: Max unique values for auto-categorical detection
        expand_categorical: If True, expand each category into separate rows
        effect_size: If True, include effect size columns (default True)
        bootstrap_ci: If True, compute bootstrap CI for Cramér's V (slower)

    Returns:
        DataFrame with descriptive statistics, p-values, and optionally effect sizes
    """
    columns = columns or [c for c in df.columns if c != groupby]
    categorical = set(categorical or [])
    continuous = set(continuous or [])

    groups = df[groupby].dropna().unique()
    if len(groups) != 2:
        raise ValueError(f'Expected 2 groups, got {len(groups)}: {groups}')

    group_names = sorted(groups, key=str)
    group_dfs = {g: df[df[groupby] == g] for g in group_names}

    results = []

    for col in columns:
        # Determine variable type
        if col in categorical:
            var_type = 'categorical'
        elif col in continuous:
            var_type = 'continuous'
        else:
            var_type = infer_variable_type(df[col], categorical_threshold)

        if var_type == 'continuous':
            overall_fmt = _format_continuous(df[col].values)
            group_stats = {g: _format_continuous(group_dfs[g][col].values) for g in group_names}

            g1_vals = group_dfs[group_names[0]][col].dropna().values
            g2_vals = group_dfs[group_names[1]][col].dropna().values
            test_result = _test_continuous(g1_vals, g2_vals, normal_test)

            row = {
                'Variable': col,
                'Overall': overall_fmt,
            }
            if expand_categorical:
                row['Category'] = ''

            for g in group_names:
                row[f'{g} (n={len(group_dfs[g])})'] = group_stats[g]

            row.update({
                'Test': test_result.test_name,
                'Statistic': f'{test_result.statistic:.2f}',
                'P-value': _format_pvalue(test_result.p_value),
            })

            if effect_size:
                row.update({
                    'Effect Size': f'{test_result.effect_size.value:.3f}' if not np.isnan(test_result.effect_size.value) else 'N/A',
                    'Effect CI': _format_ci(test_result.effect_size),
                    'Interpretation': test_result.effect_size.interpretation,
                })

            results.append(row)

        else:  # categorical or binary
            contingency = pd.crosstab(df[col], df[groupby])
            test_result = _test_categorical(contingency, bootstrap_ci=bootstrap_ci)

            if expand_categorical:
                # Expand each category into separate rows
                categories = df[col].dropna().unique()
                for i, category in enumerate(sorted(categories, key=str)):
                    results.append(_format_category_row_with_options(
                        col, category, df, group_dfs, group_names,
                        is_first=(i == 0), test_result=test_result,
                        include_effect_size=effect_size
                    ))
            else:
                # Compact format
                overall_fmt = _format_categorical(df[col])
                group_stats = {g: _format_categorical(group_dfs[g][col]) for g in group_names}

                row = {
                    'Variable': col,
                    'Overall': overall_fmt,
                }
                for g in group_names:
                    row[f'{g} (n={len(group_dfs[g])})'] = group_stats[g]

                row.update({
                    'Test': test_result.test_name,
                    'Statistic': f'{test_result.statistic:.2f}',
                    'P-value': _format_pvalue(test_result.p_value),
                })

                if effect_size:
                    row.update({
                        'Effect Size': f'{test_result.effect_size.value:.3f}' if not np.isnan(test_result.effect_size.value) else 'N/A',
                        'Effect CI': _format_ci(test_result.effect_size),
                        'Interpretation': test_result.effect_size.interpretation,
                    })

                results.append(row)

    result_df = pd.DataFrame(results)

    # Reorder columns
    base_cols = ['Variable']
    if expand_categorical:
        base_cols.append('Category')
    base_cols.append('Overall')
    group_cols = [c for c in result_df.columns if '(n=' in c]
    stat_cols = ['Test', 'Statistic', 'P-value']
    if effect_size:
        stat_cols.extend(['Effect Size', 'Effect CI', 'Interpretation'])

    return result_df[base_cols + group_cols + stat_cols]


def _format_pvalue(p: float) -> str:
    """Format p-value for display."""
    if p < 0.001:
        return '<0.001'
    elif p < 0.01:
        return f'{p:.3f}'
    else:
        return f'{p:.2f}'


def _format_ci(effect: EffectSize) -> str:
    """Format confidence interval."""
    if effect.ci_lower is None or effect.ci_upper is None:
        return 'N/A'
    return f'[{effect.ci_lower:.2f}, {effect.ci_upper:.2f}]'


def tableone_summary(
    df: pd.DataFrame,
    groupby: str,
    columns: list[str] | None = None,
    categorical: list[str] | None = None,
    continuous: list[str] | None = None
) -> pd.DataFrame:
    """
    Simplified Table One output focused on key metrics.

    Returns a compact table with:
    - Variable name and type
    - Group-wise statistics
    - P-value with significance stars
    - Effect size with interpretation
    """
    full_table = compute_tableone(
        df, groupby, columns, categorical, continuous
    )

    # Add significance stars
    def add_stars(p_str: str) -> str:
        try:
            p = float(p_str.replace('<', ''))
            if p < 0.001:
                return f'{p_str}***'
            elif p < 0.01:
                return f'{p_str}**'
            elif p < 0.05:
                return f'{p_str}*'
            return p_str
        except ValueError:
            return p_str

    full_table['P-value'] = full_table['P-value'].apply(add_stars)

    # Combine effect size and interpretation
    full_table['Effect'] = full_table.apply(
        lambda row: f"{row['Effect Size']} ({row['Interpretation']})"
        if row['Effect Size'] != 'N/A' else 'N/A',
        axis=1
    )

    # Select key columns
    group_cols = [c for c in full_table.columns if c.startswith(('0', '1', 'True', 'False')) or '(n=' in c]

    return full_table[['Variable', 'Type'] + group_cols + ['P-value', 'Effect']]


def _parse_pvalue(p_str: str) -> float:
    """Parse p-value string to float. Handles '<0.001' format."""
    if not p_str or p_str == 'N/A':
        return np.nan
    clean = p_str.replace('<', '').replace('*', '').strip()
    try:
        return float(clean)
    except ValueError:
        return np.nan


def _parse_effect_size(es_str: str) -> float:
    """Parse effect size string to float."""
    if not es_str or es_str == 'N/A':
        return np.nan
    try:
        return float(es_str)
    except ValueError:
        return np.nan


def _assign_variable_groups(df: pd.DataFrame) -> pd.Series:
    """Assign each row to its parent variable for expanded categorical tables."""
    var_groups = []
    current_var = None
    for _, row in df.iterrows():
        if row['Variable'] != '':
            current_var = row['Variable']
        var_groups.append(current_var)
    return pd.Series(var_groups, index=df.index)


def filter_tableone(
    table_df: pd.DataFrame,
    p_threshold: float | None = None,
    effect_threshold: float | None = None,
    p_col: str = 'P-value',
    effect_col: str = 'Effect Size'
) -> pd.DataFrame:
    """
    Filter Table One results by p-value and/or effect size.

    Args:
        table_df: DataFrame from compute_tableone()
        p_threshold: Keep rows with p-value < threshold (e.g., 0.05)
        effect_threshold: Keep rows with |effect size| >= threshold (e.g., 0.3)
        p_col: Column name for p-value
        effect_col: Column name for effect size

    Returns:
        Filtered DataFrame
    """
    result = table_df.copy()
    is_expanded = 'Category' in result.columns

    # Assign variable groups for expanded format
    if is_expanded:
        result['_var_group'] = _assign_variable_groups(result)

    passing_vars = set(result['Variable'].unique()) - {''}

    # Filter by p-value
    if p_threshold is not None:
        p_values = result[p_col].apply(_parse_pvalue)
        if is_expanded:
            # Get variables where p-value passes (only check first row of each variable)
            first_rows = result[result['Variable'] != '']
            p_passing = first_rows.loc[
                first_rows[p_col].apply(_parse_pvalue) < p_threshold, 'Variable'
            ].unique()
            passing_vars &= set(p_passing)
        else:
            passing_vars &= set(result.loc[p_values < p_threshold, 'Variable'])

    # Filter by effect size
    if effect_threshold is not None:
        effect_values = result[effect_col].apply(_parse_effect_size)
        if is_expanded:
            first_rows = result[result['Variable'] != '']
            effect_passing = first_rows.loc[
                first_rows[effect_col].apply(_parse_effect_size).abs() >= effect_threshold,
                'Variable'
            ].unique()
            passing_vars &= set(effect_passing)
        else:
            passing_vars &= set(result.loc[effect_values.abs() >= effect_threshold, 'Variable'])

    # Apply filter
    if is_expanded:
        result = result[result['_var_group'].isin(passing_vars)].drop(columns=['_var_group'])
    else:
        result = result[result['Variable'].isin(passing_vars)]

    return result.reset_index(drop=True)


def get_significant_variables(
    df: pd.DataFrame,
    groupby: str,
    columns: list[str] | None = None,
    categorical: list[str] | None = None,
    continuous: list[str] | None = None,
    p_threshold: float = 0.05,
    effect_threshold: float | None = None,
    expand_categorical: bool = False
) -> pd.DataFrame:
    """
    Convenience function: compute tableone and filter in one step.

    Args:
        df: Input DataFrame
        groupby: Column name for group comparison
        columns: Columns to include
        categorical: Columns to treat as categorical
        continuous: Columns to treat as continuous
        p_threshold: Keep rows with p-value < threshold
        effect_threshold: Keep rows with |effect size| >= threshold
        expand_categorical: If True, expand each category into separate rows

    Returns:
        Filtered Table One DataFrame
    """
    table = compute_tableone(
        df, groupby, columns, categorical, continuous,
        expand_categorical=expand_categorical
    )
    return filter_tableone(table, p_threshold, effect_threshold)


# =============================================================================
# Usage Examples
# =============================================================================

if __name__ == '__main__':
    # Sample case-control data
    np.random.seed(42)
    n = 300

    # Create sample data with differences between groups
    group = np.random.choice([0, 1], n, p=[0.6, 0.4])

    df_sample = pd.DataFrame({
        'group': group,
        'age': np.where(group == 1, np.random.normal(55, 12, n), np.random.normal(50, 10, n)),
        'bmi': np.where(group == 1, np.random.normal(28, 5, n), np.random.normal(25, 4, n)),
        'blood_pressure': np.where(group == 1, np.random.normal(140, 20, n), np.random.normal(120, 15, n)),
        'cholesterol': np.random.normal(200, 40, n),  # No difference
        'smoking': np.where(group == 1,
                           np.random.choice(['Never', 'Former', 'Current'], n, p=[0.3, 0.3, 0.4]),
                           np.random.choice(['Never', 'Former', 'Current'], n, p=[0.5, 0.3, 0.2])),
        'diabetes': np.where(group == 1,
                             np.random.choice(['Yes', 'No'], n, p=[0.35, 0.65]),
                             np.random.choice(['Yes', 'No'], n, p=[0.15, 0.85])),
        'exercise': np.random.choice(['Low', 'Medium', 'High'], n),  # No difference
    })

    # -------------------------------------------------------------------------
    # Example 1: Basic Table One
    # -------------------------------------------------------------------------
    print("=" * 80)
    print("Example 1: Basic Table One (Compact Format)")
    print("=" * 80)

    table1 = compute_tableone(
        df_sample,
        groupby='group',
        columns=['age', 'bmi', 'blood_pressure', 'cholesterol', 'smoking', 'diabetes'],
    )
    print("\n")
    print(table1.to_string(index=False))

    # -------------------------------------------------------------------------
    # Example 2: Expanded Categorical Format
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("Example 2: Expanded Categorical Format")
    print("=" * 80)

    table2 = compute_tableone(
        df_sample,
        groupby='group',
        columns=['age', 'smoking', 'diabetes'],
        expand_categorical=True
    )
    print("\n")
    print(table2.to_string(index=False))

    # -------------------------------------------------------------------------
    # Example 3: Without Effect Size
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("Example 3: Without Effect Size Columns")
    print("=" * 80)

    table3 = compute_tableone(
        df_sample,
        groupby='group',
        columns=['age', 'bmi', 'diabetes'],
        effect_size=False
    )
    print("\n")
    print(table3.to_string(index=False))

    # -------------------------------------------------------------------------
    # Example 4: Filter by Significance
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("Example 4: Filter Significant Variables (p < 0.05)")
    print("=" * 80)

    significant = get_significant_variables(
        df_sample,
        groupby='group',
        p_threshold=0.05
    )
    print("\n")
    print(significant.to_string(index=False))

    # -------------------------------------------------------------------------
    # Example 5: Filter by Effect Size
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("Example 5: Filter by Effect Size (|d| >= 0.3)")
    print("=" * 80)

    large_effect = filter_tableone(
        table1,
        effect_threshold=0.3
    )
    print("\n")
    print(large_effect.to_string(index=False))

    # -------------------------------------------------------------------------
    # Example 6: Specify Variable Types
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("Example 6: Manual Variable Type Specification")
    print("=" * 80)

    table6 = compute_tableone(
        df_sample,
        groupby='group',
        columns=['age', 'bmi', 'smoking'],
        continuous=['age', 'bmi'],
        categorical=['smoking']
    )
    print("\n")
    print(table6.to_string(index=False))

    # -------------------------------------------------------------------------
    # Example 7: Effect Size Interpretation
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("Example 7: Effect Size Interpretation Guide")
    print("=" * 80)
    print("""
    Cohen's d (continuous variables):
      |d| < 0.2  : negligible
      |d| < 0.5  : small
      |d| < 0.8  : medium
      |d| >= 0.8 : large

    Cramér's V (categorical variables):
      V < 0.1  : negligible
      V < 0.3  : small
      V < 0.5  : medium
      V >= 0.5 : large

    Odds Ratio (binary categorical):
      OR < 0.5  : strong protective
      OR < 0.8  : moderate protective
      OR ~ 1.0  : negligible
      OR > 1.25 : moderate risk
      OR > 2.0  : strong risk
    """)