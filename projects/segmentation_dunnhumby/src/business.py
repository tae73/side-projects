"""Business impact analysis for marketing campaigns.

Provides functions for:
- ROI curve computation
- Break-even analysis
- Campaign cost extraction
- Targeting optimization
"""

from typing import Dict, NamedTuple, Optional, List
import numpy as np
import pandas as pd


class ROIConfig(NamedTuple):
    """ROI simulation configuration."""
    cost_per_contact: float
    margin_rate: float
    min_targeting_pct: int = 5
    max_targeting_pct: int = 100
    step_pct: int = 5


class ROISummary(NamedTuple):
    """ROI analysis summary."""
    optimal_pct: float
    optimal_n: int
    max_profit: float
    max_roi: float
    breakeven_cate: float
    n_profitable: int
    pct_profitable: float


def compute_roi_curve(
    cate: np.ndarray,
    config: ROIConfig
) -> pd.DataFrame:
    """Compute ROI curve by targeting percentage.

    Simulates profit and ROI when targeting top k% customers
    ranked by CATE.

    Args:
        cate: CATE predictions for each customer
        config: ROI configuration with cost and margin

    Returns:
        DataFrame with columns:
        - pct_targeted, n_targeted
        - incremental_sales, revenue, cost, profit, roi
    """
    n = len(cate)
    sorted_idx = np.argsort(cate)[::-1]  # Sort descending by CATE
    sorted_cate = cate[sorted_idx]

    results = []
    for pct in range(config.min_targeting_pct, config.max_targeting_pct + 1, config.step_pct):
        k = int(n * pct / 100)

        incremental_sales = sorted_cate[:k].sum()
        revenue = incremental_sales * config.margin_rate
        cost = k * config.cost_per_contact
        profit = revenue - cost
        roi = (revenue - cost) / cost if cost > 0 else 0

        results.append({
            'pct_targeted': pct,
            'n_targeted': k,
            'incremental_sales': incremental_sales,
            'revenue': revenue,
            'cost': cost,
            'profit': profit,
            'roi': roi
        })

    return pd.DataFrame(results)


def compute_breakeven_analysis(
    cate: np.ndarray,
    config: ROIConfig,
    segments: Optional[np.ndarray] = None,
    segment_names: Optional[Dict[int, str]] = None
) -> pd.DataFrame:
    """Break-even analysis by segment.

    Args:
        cate: CATE predictions
        config: ROI configuration
        segments: Segment assignments (optional)
        segment_names: Segment name mapping (optional)

    Returns:
        DataFrame with columns:
        - segment, segment_name, n_customers
        - pct_above_breakeven, mean_cate, recommended_action
    """
    breakeven_cate = config.cost_per_contact / config.margin_rate

    if segments is None:
        # Single segment
        n = len(cate)
        pct_above = (cate > breakeven_cate).mean() * 100
        mean_cate = cate.mean()

        return pd.DataFrame([{
            'segment': 'All',
            'segment_name': 'All Customers',
            'n_customers': n,
            'breakeven_cate': breakeven_cate,
            'pct_above_breakeven': pct_above,
            'mean_cate': mean_cate,
            'recommended_action': _get_action(pct_above, mean_cate, breakeven_cate)
        }])

    # By segment
    unique_segments = np.unique(segments[~np.isnan(segments)]).astype(int)
    results = []

    for seg in unique_segments:
        mask = segments == seg
        seg_cate = cate[mask]
        n = mask.sum()
        pct_above = (seg_cate > breakeven_cate).mean() * 100
        mean_cate = seg_cate.mean()
        seg_name = segment_names.get(seg, f'Segment {seg}') if segment_names else f'Segment {seg}'

        results.append({
            'segment': seg,
            'segment_name': seg_name,
            'n_customers': n,
            'breakeven_cate': breakeven_cate,
            'pct_above_breakeven': pct_above,
            'mean_cate': mean_cate,
            'recommended_action': _get_action(pct_above, mean_cate, breakeven_cate)
        })

    return pd.DataFrame(results).sort_values('pct_above_breakeven', ascending=False)


def _get_action(pct_above: float, mean_cate: float, breakeven_cate: float) -> str:
    """Get recommended action based on metrics."""
    if pct_above >= 50 and mean_cate > breakeven_cate:
        return 'Increase Targeting'
    elif pct_above >= 30:
        return 'Maintain'
    elif mean_cate > 0:
        return 'Reduce Targeting'
    else:
        return 'Exclude'


def extract_campaign_cost(
    df_coupon: pd.DataFrame,
    df_coupon_redempt: pd.DataFrame,
    df_trans: pd.DataFrame,
    campaign_ids: Optional[List] = None,
    target_customers: Optional[np.ndarray] = None
) -> float:
    """Extract average campaign cost from actual data.

    Args:
        df_coupon: Coupon definitions (COUPON_UPC, CAMPAIGN, PRODUCT_ID)
        df_coupon_redempt: Coupon redemptions (household_key, COUPON_UPC, DAY)
        df_trans: Transaction data with COUPON_DISC
        campaign_ids: Campaign IDs to filter (optional)
        target_customers: Customer IDs to filter (optional)

    Returns:
        Average coupon discount per redeeming customer
    """
    # Filter coupons by campaign
    if campaign_ids is not None:
        coupon_upcs = df_coupon[df_coupon['CAMPAIGN'].isin(campaign_ids)]['COUPON_UPC'].unique()
        products = df_coupon[df_coupon['CAMPAIGN'].isin(campaign_ids)]['PRODUCT_ID'].unique()
    else:
        coupon_upcs = df_coupon['COUPON_UPC'].unique()
        products = df_coupon['PRODUCT_ID'].unique()

    # Get redemptions
    redemptions = df_coupon_redempt[df_coupon_redempt['COUPON_UPC'].isin(coupon_upcs)]
    if target_customers is not None:
        redemptions = redemptions[redemptions['household_key'].isin(target_customers)]

    # Prepare transaction data
    df_trans_work = df_trans.copy()

    # Negate COUPON_DISC if negative (raw data has negative discounts)
    if (df_trans_work['COUPON_DISC'] < 0).any():
        df_trans_work['COUPON_DISC'] = -df_trans_work['COUPON_DISC']

    # Filter transactions with coupon discounts
    mask = (
        (df_trans_work['PRODUCT_ID'].isin(products)) &
        (df_trans_work['COUPON_DISC'] > 0)
    )
    if target_customers is not None:
        mask &= df_trans_work['household_key'].isin(target_customers)

    coupon_transactions = df_trans_work[mask]

    if len(coupon_transactions) == 0:
        return 10.0  # Fallback default

    # Average discount per customer
    avg_discount = coupon_transactions.groupby('household_key')['COUPON_DISC'].sum().mean()

    return avg_discount


def compute_targeting_summary(
    cate: np.ndarray,
    ps: np.ndarray,
    config: ROIConfig
) -> ROISummary:
    """Compute comprehensive targeting summary.

    Args:
        cate: CATE predictions
        ps: Propensity scores
        config: ROI configuration

    Returns:
        ROISummary with optimal targeting and profitability metrics
    """
    roi_df = compute_roi_curve(cate, config)

    # Find optimal
    optimal_idx = roi_df['profit'].idxmax()
    optimal_pct = roi_df.loc[optimal_idx, 'pct_targeted']
    optimal_n = roi_df.loc[optimal_idx, 'n_targeted']
    max_profit = roi_df.loc[optimal_idx, 'profit']
    max_roi = roi_df.loc[optimal_idx, 'roi']

    # Break-even
    breakeven_cate = config.cost_per_contact / config.margin_rate
    n_profitable = (cate > breakeven_cate).sum()
    pct_profitable = (cate > breakeven_cate).mean()

    return ROISummary(
        optimal_pct=optimal_pct,
        optimal_n=optimal_n,
        max_profit=max_profit,
        max_roi=max_roi,
        breakeven_cate=breakeven_cate,
        n_profitable=n_profitable,
        pct_profitable=pct_profitable
    )


def compute_segment_targeting_priority(
    cate: np.ndarray,
    segments: np.ndarray,
    ps: np.ndarray,
    config: ROIConfig,
    segment_names: Optional[Dict[int, str]] = None,
    overlap_threshold: tuple = (0.1, 0.9)
) -> pd.DataFrame:
    """Compute targeting priority by segment.

    Args:
        cate: CATE predictions
        segments: Segment assignments
        ps: Propensity scores
        config: ROI configuration
        segment_names: Segment name mapping
        overlap_threshold: PS overlap range (default 0.1-0.9)

    Returns:
        DataFrame with segment targeting recommendations
    """
    breakeven_cate = config.cost_per_contact / config.margin_rate
    ps_min, ps_max = overlap_threshold

    unique_segments = np.unique(segments[~np.isnan(segments)]).astype(int)
    results = []

    for seg in unique_segments:
        mask = segments == seg
        seg_cate = cate[mask]
        seg_ps = ps[mask]
        n = mask.sum()

        # Metrics
        mean_cate = seg_cate.mean()
        std_cate = seg_cate.std()
        pct_above_breakeven = (seg_cate > breakeven_cate).mean() * 100
        pct_in_overlap = ((seg_ps >= ps_min) & (seg_ps <= ps_max)).mean() * 100

        # Priority assignment
        if mean_cate > breakeven_cate and pct_in_overlap > 20:
            priority = 'High'
        elif mean_cate > 0 and pct_in_overlap > 15:
            priority = 'Medium'
        else:
            priority = 'Low'

        seg_name = segment_names.get(seg, f'Segment {seg}') if segment_names else f'Segment {seg}'

        results.append({
            'segment': seg,
            'segment_name': seg_name,
            'n_customers': n,
            'mean_cate': mean_cate,
            'std_cate': std_cate,
            'pct_above_breakeven': pct_above_breakeven,
            'pct_in_overlap': pct_in_overlap,
            'priority': priority,
        })

    df = pd.DataFrame(results)

    # Sort by priority then mean_cate
    priority_order = {'High': 0, 'Medium': 1, 'Low': 2}
    df['priority_rank'] = df['priority'].map(priority_order)
    df = df.sort_values(['priority_rank', 'mean_cate'], ascending=[True, False])
    df = df.drop('priority_rank', axis=1)

    return df.reset_index(drop=True)
