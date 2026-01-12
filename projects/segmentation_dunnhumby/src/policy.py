"""Policy learning and evaluation functions.

This module provides functions for:
- Policy value estimation (IPW, Doubly Robust)
- Policy creation (threshold, conservative, budget-constrained)
- Policy comparison and evaluation
- Robustness analysis (cross-validation, sensitivity, bootstrap)

For theory and background:
- Athey & Wager (2021). Policy Learning With Observational Data. Econometrica.
- Zhou, Athey, Wager (2023). Offline Multi-Action Policy Learning.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, NamedTuple, Callable
from sklearn.model_selection import KFold
from sklearn.tree import export_text
import warnings


# =============================================================================
# Type Definitions
# =============================================================================

class PolicyConfig(NamedTuple):
    """Configuration for policy learning."""
    cost_per_contact: float = 12.73
    margin_rate: float = 0.30
    max_depth: int = 4
    min_samples_leaf: int = 50
    budget_fraction: Optional[float] = None


class PolicyResult(NamedTuple):
    """Result from policy evaluation."""
    name: str
    n_targeted: int
    pct_targeted: float
    value_ipw: float
    value_dr: float
    expected_profit: float
    expected_roi: float


class PolicyValueResult(NamedTuple):
    """Result from policy value estimation."""
    value: float
    se: Optional[float] = None
    ci_lower: Optional[float] = None
    ci_upper: Optional[float] = None


class CVResult(NamedTuple):
    """Cross-validation result."""
    mean: float
    std: float
    ci_lower: float
    ci_upper: float
    fold_values: List[float]


# =============================================================================
# Policy Value Estimation
# =============================================================================

def estimate_policy_value_ipw(
    Y: np.ndarray,
    T: np.ndarray,
    policy: np.ndarray,
    ps: np.ndarray,
    normalize: bool = True
) -> float:
    """Inverse Propensity Weighted policy value estimator.

    Estimates E[Y(policy)] = E[Y * I(T = policy) / P(T|X)]

    Args:
        Y: Outcome variable (n_samples,)
        T: Treatment indicator 0/1 (n_samples,)
        policy: Policy assignments 0/1 (n_samples,)
        ps: Propensity scores P(T=1|X) (n_samples,)
        normalize: If True, use Hajek (normalized) estimator

    Returns:
        Estimated policy value
    """
    # Indicator that policy matches treatment
    matches = (policy == T).astype(float)

    # IPW weights
    weights = np.where(T == 1, 1 / ps, 1 / (1 - ps))

    if normalize:
        # Hajek estimator: sum(w * Y * match) / sum(w * match)
        numerator = (weights * Y * matches).sum()
        denominator = (weights * matches).sum()
        value = numerator / denominator if denominator > 0 else 0.0
    else:
        # Horvitz-Thompson: mean(w * Y * match)
        value = (weights * Y * matches).mean()

    return value


def estimate_policy_value_dr(
    Y: np.ndarray,
    T: np.ndarray,
    policy: np.ndarray,
    ps: np.ndarray,
    mu0: np.ndarray,
    mu1: np.ndarray
) -> float:
    """Doubly Robust policy value estimator.

    More robust than IPW - consistent if either PS or outcome model is correct.

    V(policy) = E[mu_policy(X) + I(T=policy)/P(T|X) * (Y - mu_T(X))]

    Args:
        Y: Outcome variable (n_samples,)
        T: Treatment indicator 0/1 (n_samples,)
        policy: Policy assignments 0/1 (n_samples,)
        ps: Propensity scores P(T=1|X) (n_samples,)
        mu0: Predicted outcome under control E[Y|X, T=0] (n_samples,)
        mu1: Predicted outcome under treatment E[Y|X, T=1] (n_samples,)

    Returns:
        Estimated policy value
    """
    n = len(Y)
    value = 0.0

    for i in range(n):
        if policy[i] == 1:  # Policy says treat
            # mu1(x) + I(T=1)/ps * (Y - mu1(x))
            if T[i] == 1:
                value += mu1[i] + (1 / ps[i]) * (Y[i] - mu1[i])
            else:
                value += mu1[i]  # Impute with model prediction
        else:  # Policy says control
            # mu0(x) + I(T=0)/(1-ps) * (Y - mu0(x))
            if T[i] == 0:
                value += mu0[i] + (1 / (1 - ps[i])) * (Y[i] - mu0[i])
            else:
                value += mu0[i]  # Impute with model prediction

    return value / n


def estimate_policy_value_dr_vectorized(
    Y: np.ndarray,
    T: np.ndarray,
    policy: np.ndarray,
    ps: np.ndarray,
    mu0: np.ndarray,
    mu1: np.ndarray
) -> float:
    """Vectorized version of DR policy value estimator (faster)."""
    # Outcome model prediction based on policy
    mu_policy = np.where(policy == 1, mu1, mu0)
    mu_T = np.where(T == 1, mu1, mu0)

    # IPW correction term
    ps_T = np.where(T == 1, ps, 1 - ps)
    matches = (policy == T).astype(float)
    correction = matches / ps_T * (Y - mu_T)

    return (mu_policy + correction).mean()


# =============================================================================
# Policy Creation
# =============================================================================

def create_threshold_policy(
    cate: np.ndarray,
    threshold: float
) -> np.ndarray:
    """Create policy from CATE threshold.

    Treat if CATE > threshold.

    Args:
        cate: CATE predictions (n_samples,)
        threshold: Treatment threshold (often breakeven CATE)

    Returns:
        Policy array 0/1 (n_samples,)
    """
    return (cate > threshold).astype(int)


def create_conservative_policy(
    cate_lower: np.ndarray,
    threshold: float
) -> np.ndarray:
    """Conservative policy using CATE lower bounds.

    Only treat if lower bound of CATE > threshold.
    This ensures confident positive effect.

    Args:
        cate_lower: Lower bounds of CATE (n_samples,)
        threshold: Treatment threshold

    Returns:
        Policy array 0/1 (n_samples,)
    """
    return (cate_lower > threshold).astype(int)


def create_risk_adjusted_policy(
    cate: np.ndarray,
    cate_lower: np.ndarray,
    cate_upper: np.ndarray,
    threshold: float,
    risk_aversion: float = 0.5
) -> np.ndarray:
    """Risk-adjusted policy weighing point estimate and uncertainty.

    Risk-adjusted CATE = cate - risk_aversion * (cate - cate_lower)

    Args:
        cate: CATE point estimates (n_samples,)
        cate_lower: Lower bounds (n_samples,)
        cate_upper: Upper bounds (n_samples,)
        threshold: Treatment threshold
        risk_aversion: Weight on downside risk (0=risk-neutral, 1=fully conservative)

    Returns:
        Policy array 0/1 (n_samples,)
    """
    # Risk-adjusted CATE: penalize uncertainty
    downside = cate - cate_lower
    cate_adjusted = cate - risk_aversion * downside

    return (cate_adjusted > threshold).astype(int)


def create_budget_constrained_policy(
    cate: np.ndarray,
    budget_n: int
) -> np.ndarray:
    """Policy with budget constraint (top-k by CATE).

    Target top budget_n customers by predicted CATE.

    Args:
        cate: CATE predictions (n_samples,)
        budget_n: Number of customers to target

    Returns:
        Policy array 0/1 (n_samples,)
    """
    policy = np.zeros(len(cate), dtype=int)

    if budget_n <= 0:
        return policy

    # Top-k by CATE
    budget_n = min(budget_n, len(cate))
    top_k_idx = np.argsort(cate)[-budget_n:]
    policy[top_k_idx] = 1

    return policy


def create_segment_policy(
    cate: np.ndarray,
    segments: np.ndarray,
    segment_actions: Dict[int, str]
) -> np.ndarray:
    """Create policy based on segment-level rules.

    Args:
        cate: CATE predictions (n_samples,)
        segments: Segment assignments (n_samples,)
        segment_actions: Dict mapping segment ID to action
            'target_all': Target all in segment
            'target_positive': Target if CATE > 0
            'exclude': Exclude all

    Returns:
        Policy array 0/1 (n_samples,)
    """
    policy = np.zeros(len(cate), dtype=int)

    for seg_id, action in segment_actions.items():
        mask = (segments == seg_id)

        if action == 'target_all':
            policy[mask] = 1
        elif action == 'target_positive':
            policy[mask] = (cate[mask] > 0).astype(int)
        elif action == 'exclude':
            policy[mask] = 0
        # else: leave as default (0)

    return policy


# =============================================================================
# Rule Extraction
# =============================================================================

def extract_tree_rules(
    tree,
    feature_names: List[str],
    max_rules: int = 10,
    min_samples: int = 10
) -> List[Dict]:
    """Extract human-readable rules from policy tree.

    Args:
        tree: Fitted tree model (sklearn-like with tree_ attribute)
               - econml PolicyTree/DRPolicyTree: has policy_model_ attribute
               - sklearn DecisionTree: has tree_ attribute
        feature_names: Feature names
        max_rules: Maximum number of rules to extract
        min_samples: Minimum samples in leaf to include rule

    Returns:
        List of rule dictionaries with conditions and action
    """
    rules = []

    try:
        # Handle econml PolicyTree/DRPolicyTree (has policy_model_)
        if hasattr(tree, 'policy_model_'):
            tree_model = tree.policy_model_
        # Handle econml older versions or other structures
        elif hasattr(tree, 'tree_model_'):
            tree_model = tree.tree_model_
        # sklearn tree directly
        elif hasattr(tree, 'tree_'):
            tree_model = tree
        else:
            tree_model = tree

        # Extract from sklearn tree structure
        if hasattr(tree_model, 'tree_'):
            _tree = tree_model.tree_
        else:
            warnings.warn(f"Cannot extract rules from tree structure. "
                         f"Tree type: {type(tree_model)}, "
                         f"Available attributes: {[a for a in dir(tree_model) if not a.startswith('_')][:10]}")
            return rules

        feature = _tree.feature
        threshold = _tree.threshold
        children_left = _tree.children_left
        children_right = _tree.children_right
        n_samples = _tree.n_node_samples
        value = _tree.value

        def recurse(node, conditions):
            if children_left[node] == children_right[node]:  # Leaf
                if n_samples[node] >= min_samples:
                    # Determine action (0 or 1) from leaf value
                    # Handle various value shapes from different tree types
                    node_value = value[node]

                    # Flatten if needed
                    if hasattr(node_value, 'flatten'):
                        flat_value = node_value.flatten()
                    else:
                        flat_value = np.array([node_value])

                    # Determine action
                    if len(flat_value) >= 2:
                        # Classification tree: [count_class_0, count_class_1, ...]
                        action = int(np.argmax(flat_value))
                    elif len(flat_value) == 1:
                        # Regression tree: single value (positive = target)
                        action = int(flat_value[0] > 0)
                    else:
                        action = 0

                    # Compute confidence (probability of chosen action)
                    total = np.sum(np.abs(flat_value))
                    if total > 0 and len(flat_value) >= 2:
                        confidence = float(np.max(flat_value) / total)
                    else:
                        confidence = 1.0

                    rules.append({
                        'conditions': conditions.copy(),
                        'action': 'TARGET' if action == 1 else 'CONTROL',
                        'n_samples': int(n_samples[node]),
                        'confidence': confidence,
                        'value': float(flat_value[action]) if len(flat_value) > action else float(flat_value[0])
                    })
            else:
                feat_name = feature_names[feature[node]] if feature[node] < len(feature_names) else f"X{feature[node]}"
                thresh = threshold[node]

                # Left branch: feature <= threshold
                recurse(children_left[node],
                       conditions + [(feat_name, '<=', thresh)])

                # Right branch: feature > threshold
                recurse(children_right[node],
                       conditions + [(feat_name, '>', thresh)])

        recurse(0, [])

        # Sort by action (TARGET first) and n_samples
        rules = sorted(rules, key=lambda x: (-int(x['action'] == 'TARGET'), -x['n_samples']))

        return rules[:max_rules]

    except Exception as e:
        warnings.warn(f"Error extracting rules: {e}")
        return []


def format_rules_as_text(rules: List[Dict]) -> str:
    """Format extracted rules as human-readable text.

    Args:
        rules: List of rule dictionaries from extract_tree_rules

    Returns:
        Formatted string
    """
    if not rules:
        return "No rules extracted. Tree may not have valid leaf nodes with enough samples."

    lines = []

    for i, rule in enumerate(rules, 1):
        if rule['conditions']:
            conditions = ' AND '.join([
                f"{feat} {op} {val:.2f}" if isinstance(val, float) else f"{feat} {op} {val}"
                for feat, op, val in rule['conditions']
            ])
            condition_str = f"IF {conditions}"
        else:
            condition_str = "DEFAULT (root)"

        value_str = f", val={rule.get('value', 0):.2f}" if 'value' in rule else ""

        lines.append(
            f"Rule {i}: {condition_str} THEN {rule['action']} "
            f"(n={rule['n_samples']}, conf={rule['confidence']:.2f}{value_str})"
        )

    return '\n'.join(lines)


def export_policy_tree_text(
    tree,
    feature_names: List[str],
    max_depth: int = 10
) -> str:
    """Export policy tree as text using sklearn's export_text.

    This is a more robust alternative to extract_tree_rules.

    Args:
        tree: Fitted tree model (econml PolicyTree/DRPolicyTree or sklearn tree)
        feature_names: Feature names
        max_depth: Maximum depth to display

    Returns:
        Text representation of the tree
    """
    try:
        # Get the underlying sklearn tree
        if hasattr(tree, 'policy_model_'):
            sklearn_tree = tree.policy_model_
        elif hasattr(tree, 'tree_model_'):
            sklearn_tree = tree.tree_model_
        elif hasattr(tree, 'tree_'):
            sklearn_tree = tree
        else:
            sklearn_tree = tree

        # Use sklearn's export_text
        if hasattr(sklearn_tree, 'tree_'):
            return export_text(sklearn_tree, feature_names=feature_names, max_depth=max_depth)
        else:
            return f"Cannot export tree. Type: {type(sklearn_tree)}"

    except Exception as e:
        return f"Error exporting tree: {e}"


# =============================================================================
# Policy Comparison
# =============================================================================

def compare_policies(
    Y: np.ndarray,
    T: np.ndarray,
    policies: Dict[str, np.ndarray],
    ps: np.ndarray,
    mu0: np.ndarray,
    mu1: np.ndarray,
    config: PolicyConfig
) -> pd.DataFrame:
    """Compare multiple policies on various metrics.

    Args:
        Y: Outcome variable
        T: Treatment indicator
        policies: Dict mapping policy name to policy array
        ps: Propensity scores
        mu0: Predicted outcome under control
        mu1: Predicted outcome under treatment
        config: Policy configuration

    Returns:
        DataFrame with policy comparison metrics
    """
    results = []

    for name, policy in policies.items():
        n_targeted = int(policy.sum())
        pct_targeted = n_targeted / len(policy)

        # Policy values
        value_ipw = estimate_policy_value_ipw(Y, T, policy, ps)
        value_dr = estimate_policy_value_dr_vectorized(Y, T, policy, ps, mu0, mu1)

        # Expected profit (based on CATE predictions)
        cate = mu1 - mu0
        expected_incremental = (cate * policy).sum()
        expected_revenue = expected_incremental * config.margin_rate
        cost = n_targeted * config.cost_per_contact
        expected_profit = expected_revenue - cost
        expected_roi = expected_profit / cost if cost > 0 else 0

        results.append(PolicyResult(
            name=name,
            n_targeted=n_targeted,
            pct_targeted=pct_targeted,
            value_ipw=value_ipw,
            value_dr=value_dr,
            expected_profit=expected_profit,
            expected_roi=expected_roi
        ))

    return pd.DataFrame(results).sort_values('expected_profit', ascending=False)


def compute_policy_lift(
    policy: np.ndarray,
    cate: np.ndarray,
    baseline_policy: np.ndarray
) -> Dict[str, float]:
    """Compute lift of policy over baseline.

    Args:
        policy: Proposed policy
        cate: CATE predictions
        baseline_policy: Baseline policy to compare against

    Returns:
        Dict with lift metrics
    """
    # Expected value under each policy (using CATE)
    value_policy = (cate * policy).sum()
    value_baseline = (cate * baseline_policy).sum()

    # Lift
    absolute_lift = value_policy - value_baseline
    relative_lift = absolute_lift / abs(value_baseline) if value_baseline != 0 else np.inf

    # Targeting difference
    n_policy = policy.sum()
    n_baseline = baseline_policy.sum()

    return {
        'value_policy': value_policy,
        'value_baseline': value_baseline,
        'absolute_lift': absolute_lift,
        'relative_lift': relative_lift,
        'n_policy': int(n_policy),
        'n_baseline': int(n_baseline),
        'targeting_change': int(n_policy - n_baseline)
    }


# =============================================================================
# Robustness Analysis
# =============================================================================

def cross_validate_policy_value(
    X: np.ndarray,
    Y: np.ndarray,
    T: np.ndarray,
    ps: np.ndarray,
    mu0: np.ndarray,
    mu1: np.ndarray,
    policy_fn: Callable,
    n_splits: int = 5,
    seed: int = 42
) -> CVResult:
    """Cross-validate policy value estimation.

    Args:
        X: Covariate matrix
        Y: Outcome variable
        T: Treatment indicator
        ps: Propensity scores
        mu0: Predicted outcome under control
        mu1: Predicted outcome under treatment
        policy_fn: Function(X, Y, T) -> policy array
        n_splits: Number of CV folds
        seed: Random seed

    Returns:
        CVResult with mean, std, CI, and fold values
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    cv_values = []

    for train_idx, test_idx in kf.split(X):
        # Train policy on train fold
        policy_train = policy_fn(X[train_idx], Y[train_idx], T[train_idx])

        # For test fold, we need to predict policy
        # This requires the policy function to support prediction
        # If not, we just evaluate the same policy
        try:
            if hasattr(policy_fn, 'predict'):
                policy_test = policy_fn.predict(X[test_idx])
            else:
                policy_test = policy_fn(X[test_idx], Y[test_idx], T[test_idx])
        except:
            policy_test = policy_fn(X[test_idx], Y[test_idx], T[test_idx])

        # Evaluate on test fold
        value = estimate_policy_value_dr_vectorized(
            Y[test_idx], T[test_idx], policy_test,
            ps[test_idx], mu0[test_idx], mu1[test_idx]
        )
        cv_values.append(value)

    return CVResult(
        mean=float(np.mean(cv_values)),
        std=float(np.std(cv_values)),
        ci_lower=float(np.percentile(cv_values, 2.5)),
        ci_upper=float(np.percentile(cv_values, 97.5)),
        fold_values=cv_values
    )


def sensitivity_analysis(
    cate: np.ndarray,
    costs: List[float] = None,
    margins: List[float] = None
) -> pd.DataFrame:
    """Grid sensitivity analysis over cost and margin assumptions.

    Args:
        cate: CATE predictions
        costs: List of cost per contact values
        margins: List of margin rates

    Returns:
        DataFrame with sensitivity results
    """
    if costs is None:
        costs = [5, 10, 12.73, 15, 20]
    if margins is None:
        margins = [0.20, 0.25, 0.30, 0.35, 0.40]

    results = []
    n_total = len(cate)

    for cost in costs:
        for margin in margins:
            breakeven = cost / margin

            # Threshold policy at breakeven
            policy = (cate > breakeven).astype(int)
            n_target = int(policy.sum())

            # Expected metrics
            incremental = (cate * policy).sum()
            revenue = incremental * margin
            total_cost = n_target * cost
            profit = revenue - total_cost
            roi = profit / total_cost if total_cost > 0 else 0

            results.append({
                'cost': cost,
                'margin': margin,
                'breakeven': breakeven,
                'n_targeted': n_target,
                'pct_targeted': n_target / n_total * 100,
                'incremental_sales': incremental,
                'revenue': revenue,
                'cost_total': total_cost,
                'profit': profit,
                'roi': roi
            })

    return pd.DataFrame(results)


def tree_depth_sensitivity(
    Y: np.ndarray,
    T: np.ndarray,
    X: np.ndarray,
    depths: List[int] = None,
    min_samples_leaf: int = 50,
    seed: int = 42
) -> pd.DataFrame:
    """Assess policy stability across tree depths.

    Args:
        Y: Outcome variable
        T: Treatment indicator
        X: Covariate matrix
        depths: List of tree depths to test
        min_samples_leaf: Minimum samples per leaf
        seed: Random seed

    Returns:
        DataFrame with depth sensitivity results
    """
    if depths is None:
        depths = [3, 4, 5, 6]

    try:
        from econml.policy import DRPolicyTree
    except ImportError:
        warnings.warn("econml not available for tree depth sensitivity")
        return pd.DataFrame()

    results = []

    for depth in depths:
        try:
            dr_policy = DRPolicyTree(
                max_depth=depth,
                min_samples_leaf=min_samples_leaf,
                honest=True,
                cv=3,
                random_state=seed
            )
            dr_policy.fit(Y, T, X=X)

            policy = dr_policy.predict(X)

            # Get policy value if available
            try:
                value = dr_policy.predict_value(X)[:, 1].mean()
            except:
                value = np.nan

            results.append({
                'depth': depth,
                'n_targeted': int(policy.sum()),
                'pct_targeted': policy.mean() * 100,
                'policy_value': value,
            })
        except Exception as e:
            warnings.warn(f"Error at depth {depth}: {e}")
            results.append({
                'depth': depth,
                'n_targeted': np.nan,
                'pct_targeted': np.nan,
                'policy_value': np.nan,
            })

    return pd.DataFrame(results)


def bootstrap_policy_ci(
    cate: np.ndarray,
    config: PolicyConfig,
    n_bootstrap: int = 100,
    seed: int = 42
) -> Dict[str, float]:
    """Bootstrap confidence intervals for policy profit.

    Args:
        cate: CATE predictions
        config: Policy configuration
        n_bootstrap: Number of bootstrap samples
        seed: Random seed

    Returns:
        Dict with mean, std, and CI bounds
    """
    np.random.seed(seed)
    n = len(cate)
    breakeven = config.cost_per_contact / config.margin_rate

    profits = []

    for _ in range(n_bootstrap):
        # Bootstrap sample
        idx = np.random.choice(n, size=n, replace=True)
        cate_boot = cate[idx]

        # Threshold policy
        policy = (cate_boot > breakeven).astype(int)
        n_target = policy.sum()

        # Profit
        incremental = (cate_boot * policy).sum()
        revenue = incremental * config.margin_rate
        cost = n_target * config.cost_per_contact
        profit = revenue - cost

        profits.append(profit)

    return {
        'mean': float(np.mean(profits)),
        'std': float(np.std(profits)),
        'ci_lower': float(np.percentile(profits, 2.5)),
        'ci_upper': float(np.percentile(profits, 97.5)),
        'profits': profits
    }


# =============================================================================
# A/B Test Design
# =============================================================================

def calculate_ab_sample_size(
    effect_size: float,
    baseline_std: float,
    alpha: float = 0.05,
    power: float = 0.80,
    allocation_ratio: float = 1.0
) -> Dict[str, int]:
    """Calculate required sample size for A/B test.

    Args:
        effect_size: Expected treatment effect (CATE)
        baseline_std: Standard deviation of outcome
        alpha: Type I error rate
        power: Statistical power (1 - Type II error)
        allocation_ratio: Ratio of treatment to control (1.0 = balanced)

    Returns:
        Dict with sample sizes per group and total
    """
    from scipy import stats

    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)

    # Standard two-sample formula
    n_per_group = 2 * ((z_alpha + z_beta) * baseline_std / effect_size) ** 2

    # Adjust for allocation ratio
    n_treatment = int(np.ceil(n_per_group))
    n_control = int(np.ceil(n_per_group / allocation_ratio))

    return {
        'n_treatment': n_treatment,
        'n_control': n_control,
        'n_total': n_treatment + n_control,
        'effect_size': effect_size,
        'baseline_std': baseline_std,
        'alpha': alpha,
        'power': power
    }


def design_ab_test(
    cate: np.ndarray,
    Y: np.ndarray,
    segments: Optional[np.ndarray] = None,
    ps: Optional[np.ndarray] = None,
    alpha: float = 0.05,
    power: float = 0.80
) -> Dict:
    """Design A/B test based on CATE estimates.

    Args:
        cate: CATE predictions
        Y: Baseline outcome
        segments: Segment assignments (optional)
        ps: Propensity scores for stratification (optional)
        alpha: Significance level
        power: Desired power

    Returns:
        Dict with test design parameters
    """
    # Effect size: use mean CATE in overlap region if PS available
    if ps is not None:
        overlap_mask = (ps >= 0.1) & (ps <= 0.9)
        if overlap_mask.sum() > 10:
            effect_size = np.abs(cate[overlap_mask].mean())
        else:
            effect_size = np.abs(cate.mean())
    else:
        effect_size = np.abs(cate.mean())

    # Use conservative estimate (25th percentile of positive CATE)
    positive_cate = cate[cate > 0]
    if len(positive_cate) > 10:
        effect_size = max(effect_size, np.percentile(positive_cate, 25))

    baseline_std = Y.std()

    # Sample size
    sample_size = calculate_ab_sample_size(
        effect_size=effect_size,
        baseline_std=baseline_std,
        alpha=alpha,
        power=power
    )

    # Stratification recommendation
    strata = []
    if segments is not None:
        strata.append('segment')
    if ps is not None:
        strata.append('ps_region')

    return {
        'sample_size': sample_size,
        'effect_size_assumed': effect_size,
        'baseline_std': baseline_std,
        'stratification_vars': strata,
        'recommendation': (
            f"Minimum {sample_size['n_total']} customers needed "
            f"({sample_size['n_treatment']} treatment, {sample_size['n_control']} control) "
            f"to detect effect of ${effect_size:.2f} with {power*100:.0f}% power."
        )
    }
