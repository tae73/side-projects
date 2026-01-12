"""Customer feature engineering for segmentation.

This module provides feature engineering functions for both Track 1 (descriptive)
and Track 2 (causal) analyses.

Track 1 Features (33):
    - RFM: recency, frequency, monetary (19 features + 2 auxiliary)
    - Behavioral: price sensitivity, brand, basket (7 features)
    - Category: macro category shares (6 features)
    - Time: shopping regularity (1 feature)

Track 2 Features:
    - Exposure: marketing exposure from causal_data
    - Campaign: campaign targeting indicators
    - Outcome: redemption and purchase outcomes
    - Demographic: household demographics
"""

import numpy as np
import pandas as pd


# =============================================================================
# Constants
# =============================================================================

# Macro category mapping
MACRO_CATEGORY = {
    'grocery': ['GROCERY', 'FROZEN GROCERY', 'GRO BAKERY'],
    'fresh': ['PRODUCE', 'MEAT', 'MEAT-PCKGD', 'MEAT-WHSE', 'PORK',
              'SEAFOOD', 'SEAFOOD-PCKGD', 'DELI', 'DELI/SNACK BAR',
              'DAIRY DELI', 'CHEF SHOPPE', 'SALAD BAR'],
    'bakery': ['PASTRY'],
    'health_beauty': ['DRUG GM', 'NUTRITION', 'COSMETICS', 'RX',
                      'PHARMACY SUPPLY', 'HBC'],
    'alcohol': ['SPIRITS'],
}
DEPT_TO_MACRO = {dept: cat for cat, depts in MACRO_CATEGORY.items() for dept in depts}


# =============================================================================
# Track 1: Base Features (Descriptive)
# =============================================================================

def _compute_purchase_intervals(days):
    """Compute days between consecutive purchases."""
    sorted_days = np.sort(days.unique())
    if len(sorted_days) < 2:
        return pd.Series({'avg': np.nan, 'std': np.nan})
    intervals = np.diff(sorted_days)
    return pd.Series({
        'avg': intervals.mean(),
        'std': intervals.std() if len(intervals) > 1 else 0
    })


def build_rfm_features(df_trans):
    """Build RFM (Recency, Frequency, Monetary) features.

    Args:
        df_trans: Preprocessed transaction DataFrame

    Returns:
        DataFrame with household_key and 21 RFM features
    """
    max_day = df_trans['DAY'].max()
    max_week = df_trans['WEEK_NO'].max()

    # Basket-level aggregation
    df_basket = df_trans.groupby(['household_key', 'BASKET_ID']).agg(**{
        'basket_sales': ('SALES_VALUE', 'sum'),
        'basket_actual': ('ACTUAL_SPENT', 'sum'),
        'basket_day': ('DAY', 'first'),
        'basket_week': ('WEEK_NO', 'first'),
    }).reset_index()

    # Purchase intervals
    df_intervals = (
        df_basket.groupby('household_key')['basket_day']
        .apply(_compute_purchase_intervals)
        .unstack()
        .reset_index()
    )
    df_intervals.columns = ['household_key', 'days_between_purchases_avg', 'days_between_purchases_std']

    # Main aggregation
    df_rfm = df_trans.groupby('household_key').agg(**{
        'first_purchase_day': ('DAY', 'min'),
        'last_purchase_day': ('DAY', 'max'),
        'first_week': ('WEEK_NO', 'min'),
        'last_week': ('WEEK_NO', 'max'),
        'frequency': ('BASKET_ID', 'nunique'),
        'transaction_count': ('BASKET_ID', 'count'),
        'weeks_with_purchase': ('WEEK_NO', 'nunique'),
        'monetary_sales': ('SALES_VALUE', 'sum'),
        'monetary_actual': ('ACTUAL_SPENT', 'sum'),
        'total_coupon_disc': ('COUPON_DISC', 'sum'),
    }).reset_index()

    # Basket stats
    df_basket_stats = df_basket.groupby('household_key').agg(**{
        'monetary_avg_basket_sales': ('basket_sales', 'mean'),
        'monetary_avg_basket_actual': ('basket_actual', 'mean'),
        'monetary_std': ('basket_sales', 'std'),
    }).reset_index()

    # Merge and derive
    df_rfm = (
        df_rfm
        .merge(df_basket_stats, on='household_key', how='left')
        .merge(df_intervals, on='household_key', how='left')
        .assign(
            recency=lambda x: max_day - x['last_purchase_day'],
            recency_weeks=lambda x: max_week - x['last_week'],
            active_last_4w=lambda x: (x['last_week'] >= max_week - 4).astype(int),
            active_last_12w=lambda x: (x['last_week'] >= max_week - 12).astype(int),
            tenure=lambda x: max_day - x['first_purchase_day'],
            tenure_weeks=lambda x: max_week - x['first_week'] + 1,
            frequency_per_week=lambda x: x['frequency'] / x['tenure_weeks'],
            frequency_per_month=lambda x: x['frequency'] / x['tenure_weeks'] * 4.33,
            purchase_regularity=lambda x: x['weeks_with_purchase'] / x['tenure_weeks'],
            monetary_per_week=lambda x: x['monetary_sales'] / x['tenure_weeks'],
            coupon_savings_ratio=lambda x: x['total_coupon_disc'] / x['monetary_sales'],
        )
    )

    # Select columns
    cols = [
        'household_key',
        'recency', 'recency_weeks', 'active_last_4w', 'active_last_12w',
        'days_between_purchases_avg', 'days_between_purchases_std',
        'frequency', 'frequency_per_week', 'frequency_per_month',
        'transaction_count', 'weeks_with_purchase', 'purchase_regularity',
        'monetary_sales', 'monetary_actual', 'monetary_avg_basket_sales', 'monetary_avg_basket_actual',
        'monetary_std', 'monetary_per_week', 'coupon_savings_ratio',
        'tenure', 'tenure_weeks',
    ]
    return df_rfm[cols]


def build_behavioral_features(df_trans, df_product):
    """Build behavioral features (price sensitivity, brand, basket).

    Args:
        df_trans: Preprocessed transaction DataFrame
        df_product: Product DataFrame

    Returns:
        DataFrame with household_key and 7 behavioral features
    """
    df = df_trans.merge(
        df_product[['PRODUCT_ID', 'DEPARTMENT', 'BRAND']],
        on='PRODUCT_ID',
        how='left'
    )

    df_behavior = df.groupby('household_key').agg(**{
        'total_retail_disc': ('RETAIL_DISC', 'sum'),
        'total_coupon_disc': ('COUPON_DISC', 'sum'),
        'total_sales': ('SALES_VALUE', 'sum'),
        'discount_transactions': ('RETAIL_DISC', lambda x: (x > 0).sum()),
        'private_label_count': ('BRAND', lambda x: (x == 'Private').sum()),
        'national_brand_count': ('BRAND', lambda x: (x == 'National').sum()),
        'n_departments': ('DEPARTMENT', 'nunique'),
        'n_products': ('PRODUCT_ID', 'nunique'),
        'total_quantity': ('QUANTITY', 'sum'),
        'n_baskets': ('BASKET_ID', 'nunique'),
    }).reset_index()

    df_behavior = df_behavior.assign(
        discount_rate=lambda x: (x['total_retail_disc'] + x['total_coupon_disc']) /
                                (x['total_sales'] + x['total_retail_disc'] + x['total_coupon_disc']),
        discount_usage_pct=lambda x: x['discount_transactions'] /
                                     (x['private_label_count'] + x['national_brand_count']),
        private_label_ratio=lambda x: x['private_label_count'] /
                                      (x['private_label_count'] + x['national_brand_count']),
        avg_items_per_basket=lambda x: x['total_quantity'] / x['n_baskets'],
        avg_products_per_basket=lambda x: x['n_products'] / x['n_baskets'],
    )

    cols = [
        'household_key',
        'discount_rate', 'discount_usage_pct',
        'private_label_ratio',
        'n_departments', 'n_products',
        'avg_items_per_basket', 'avg_products_per_basket',
    ]
    return df_behavior[cols]


def build_category_features(df_trans, df_product):
    """Build macro category share features.

    Args:
        df_trans: Preprocessed transaction DataFrame
        df_product: Product DataFrame

    Returns:
        DataFrame with household_key and 6 category share features
    """
    df = df_trans.merge(
        df_product[['PRODUCT_ID', 'DEPARTMENT']],
        on='PRODUCT_ID',
        how='left'
    ).assign(
        macro_category=lambda x: x['DEPARTMENT'].map(DEPT_TO_MACRO).fillna('other')
    )

    df_macro = (
        df.groupby(['household_key', 'macro_category'])['SALES_VALUE']
        .sum()
        .unstack(fill_value=0)
    )

    df_share = df_macro.div(df_macro.sum(axis=1), axis=0)
    df_share.columns = [f'share_{col}' for col in df_share.columns]

    return df_share.reset_index()


def build_time_features(df_trans):
    """Build time pattern features.

    Args:
        df_trans: Preprocessed transaction DataFrame

    Returns:
        DataFrame with household_key and 1 time feature
    """
    df_time = df_trans.groupby('household_key').agg(**{
        'n_weeks_active': ('WEEK_NO', 'nunique'),
        'week_range': ('WEEK_NO', lambda x: x.max() - x.min()),
    }).reset_index()

    df_time = df_time.assign(
        week_coverage=lambda x: x['n_weeks_active'] / (x['week_range'] + 1)
    )

    return df_time[['household_key', 'week_coverage']]


def build_all_features(df_trans, df_product):
    """Build all customer features for segmentation.

    Args:
        df_trans: Preprocessed transaction DataFrame
        df_product: Product DataFrame

    Returns:
        DataFrame with household_key and 33 features
    """
    df_rfm = build_rfm_features(df_trans)
    df_behavior = build_behavioral_features(df_trans, df_product)
    df_category = build_category_features(df_trans, df_product)
    df_time = build_time_features(df_trans)

    df_features = (
        df_rfm
        .merge(df_behavior, on='household_key', how='left')
        .merge(df_category, on='household_key', how='left')
        .merge(df_time, on='household_key', how='left')
    )

    return df_features


# =============================================================================
# Track 2: Causal Features
# =============================================================================

def build_exposure_features(df_trans, df_causal):
    """Build marketing exposure features from causal_data.

    Aggregates product-store-week level exposure to household level
    by joining through transaction data.

    Args:
        df_trans: Preprocessed transaction DataFrame
        df_causal: Causal data DataFrame with display/mailer columns

    Returns:
        DataFrame with household_key and exposure features:
        - display_exposure_rate: proportion of transactions with display > 0
        - display_intensity_avg: mean display level when exposed (1-9)
        - mailer_exposure_rate: proportion of transactions with mailer != '0'
    """
    # Convert display to numeric (A -> 10)
    df_causal = df_causal.copy()
    df_causal['display_num'] = pd.to_numeric(
        df_causal['display'].replace('A', '10'),
        errors='coerce'
    ).fillna(0).astype(int)
    df_causal['mailer_exposed'] = (df_causal['mailer'] != '0').astype(int)

    # Join transaction with causal data
    df_merged = df_trans.merge(
        df_causal[['PRODUCT_ID', 'STORE_ID', 'WEEK_NO', 'display_num', 'mailer_exposed']],
        on=['PRODUCT_ID', 'STORE_ID', 'WEEK_NO'],
        how='left'
    )

    # Fill missing (no causal data) with 0
    df_merged['display_num'] = df_merged['display_num'].fillna(0)
    df_merged['mailer_exposed'] = df_merged['mailer_exposed'].fillna(0)

    # Aggregate to household level
    df_exposure = df_merged.groupby('household_key').agg(**{
        'n_transactions': ('BASKET_ID', 'count'),
        'display_exposed_count': ('display_num', lambda x: (x > 0).sum()),
        'display_sum': ('display_num', 'sum'),
        'display_exposed_sum': ('display_num', lambda x: x[x > 0].sum()),
        'mailer_exposed_count': ('mailer_exposed', 'sum'),
    }).reset_index()

    df_exposure = df_exposure.assign(
        display_exposure_rate=lambda x: x['display_exposed_count'] / x['n_transactions'],
        display_intensity_avg=lambda x: np.where(
            x['display_exposed_count'] > 0,
            x['display_exposed_sum'] / x['display_exposed_count'],
            0
        ),
        mailer_exposure_rate=lambda x: x['mailer_exposed_count'] / x['n_transactions'],
    )

    return df_exposure[['household_key', 'display_exposure_rate',
                        'display_intensity_avg', 'mailer_exposure_rate']]


def build_campaign_features(df_campaign_table, df_campaign_desc):
    """Build campaign targeting features.

    Args:
        df_campaign_table: Campaign targeting table (household_key, CAMPAIGN)
        df_campaign_desc: Campaign descriptions (CAMPAIGN, DESCRIPTION)

    Returns:
        DataFrame with household_key and campaign features:
        - targeted_typeA/B/C: binary indicators
        - n_campaigns_targeted: total campaigns
        - n_typeA_campaigns: TypeA campaign count
    """
    # Add campaign type
    campaign_types = df_campaign_desc.set_index('CAMPAIGN')['DESCRIPTION'].to_dict()
    df = df_campaign_table.assign(
        campaign_type=lambda x: x['CAMPAIGN'].map(campaign_types)
    )

    # Count by type
    df_counts = (
        df.groupby(['household_key', 'campaign_type'])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )

    # Ensure all type columns exist
    for ctype in ['TypeA', 'TypeB', 'TypeC']:
        if ctype not in df_counts.columns:
            df_counts[ctype] = 0

    df_counts = df_counts.assign(
        targeted_typeA=lambda x: (x['TypeA'] > 0).astype(int),
        targeted_typeB=lambda x: (x['TypeB'] > 0).astype(int),
        targeted_typeC=lambda x: (x['TypeC'] > 0).astype(int),
        n_campaigns_targeted=lambda x: x['TypeA'] + x['TypeB'] + x['TypeC'],
        n_typeA_campaigns=lambda x: x['TypeA'],
    )

    return df_counts[['household_key', 'targeted_typeA', 'targeted_typeB',
                      'targeted_typeC', 'n_campaigns_targeted', 'n_typeA_campaigns']]


def build_outcome_features(df_trans, df_coupon_redempt, period_start=32, period_end=102):
    """Build outcome features for campaign period.

    Args:
        df_trans: Preprocessed transaction DataFrame
        df_coupon_redempt: Coupon redemption DataFrame
        period_start: Campaign period start week (default: 32)
        period_end: Campaign period end week (default: 102)

    Returns:
        DataFrame with household_key and outcome features:
        - redemption_count: number of coupon redemptions
        - redemption_any: binary indicator
        - purchase_amount: total purchase in period
        - purchase_count: number of baskets in period
    """
    # Filter to campaign period
    df_period = df_trans[
        (df_trans['WEEK_NO'] >= period_start) &
        (df_trans['WEEK_NO'] <= period_end)
    ]

    # Purchase outcomes
    df_purchase = df_period.groupby('household_key').agg(**{
        'purchase_amount': ('SALES_VALUE', 'sum'),
        'purchase_count': ('BASKET_ID', 'nunique'),
    }).reset_index()

    # Redemption outcomes
    # Convert DAY to WEEK_NO approximation
    df_redempt = df_coupon_redempt.copy()
    df_redempt['WEEK_NO_approx'] = np.ceil(df_redempt['DAY'] / 7).astype(int)
    df_redempt_period = df_redempt[
        (df_redempt['WEEK_NO_approx'] >= period_start) &
        (df_redempt['WEEK_NO_approx'] <= period_end)
    ]

    df_redempt_agg = (
        df_redempt_period
        .groupby('household_key')
        .size()
        .reset_index(name='redemption_count')
    )

    # Merge
    df_outcome = (
        df_purchase
        .merge(df_redempt_agg, on='household_key', how='left')
        .fillna({'redemption_count': 0})
        .astype({'redemption_count': int})
        .assign(redemption_any=lambda x: (x['redemption_count'] > 0).astype(int))
    )

    return df_outcome[['household_key', 'redemption_count', 'redemption_any',
                       'purchase_amount', 'purchase_count']]


def build_demographic_features(df_demo):
    """Extract demographic features.

    Args:
        df_demo: Household demographic DataFrame

    Returns:
        DataFrame with household_key and demographic columns
    """
    demo_cols = ['household_key', 'AGE_DESC', 'MARITAL_STATUS_CODE',
                 'INCOME_DESC', 'HOMEOWNER_DESC', 'HH_COMP_DESC',
                 'HOUSEHOLD_SIZE_DESC', 'KID_CATEGORY_DESC']

    # Select available columns
    available_cols = [c for c in demo_cols if c in df_demo.columns]

    return df_demo[available_cols].copy()


# =============================================================================
# Track Integration Functions
# =============================================================================

def build_track1_features(df_trans, df_product):
    """Build all Track 1 features (33 base features).

    Alias for build_all_features() for consistency.
    """
    return build_all_features(df_trans, df_product)


def build_track2_features(df_trans, df_product, df_causal,
                          df_campaign_table, df_campaign_desc,
                          df_coupon_redempt, df_demo,
                          pre_period_end=31, campaign_period_start=32):
    """Build all Track 2 features (confounders + treatment + outcome).

    Uses pre-treatment period (Week 1-31) for base features,
    campaign period (Week 32-102) for outcomes.

    Args:
        df_trans: Preprocessed transaction DataFrame
        df_product: Product DataFrame
        df_causal: Causal data DataFrame
        df_campaign_table: Campaign targeting table
        df_campaign_desc: Campaign descriptions
        df_coupon_redempt: Coupon redemption DataFrame
        df_demo: Household demographic DataFrame
        pre_period_end: End week for pre-treatment features (default: 31)
        campaign_period_start: Start week for campaign period (default: 32)

    Returns:
        DataFrame with household_key and all Track 2 features
    """
    # Pre-treatment period transactions
    df_trans_pre = df_trans[df_trans['WEEK_NO'] <= pre_period_end]

    # Base features from pre-treatment period
    df_base = build_all_features(df_trans_pre, df_product)

    # Exposure features from pre-treatment period
    # (measures customer's tendency to purchase display/mailer-exposed products)
    df_causal_pre = df_causal[df_causal['WEEK_NO'] <= pre_period_end]
    df_exposure = build_exposure_features(df_trans_pre, df_causal_pre)

    # Campaign features
    df_campaign = build_campaign_features(df_campaign_table, df_campaign_desc)

    # Outcome features
    df_outcome = build_outcome_features(df_trans, df_coupon_redempt,
                                        period_start=campaign_period_start)

    # Demographics
    df_demo_features = build_demographic_features(df_demo)

    # Merge all
    df_track2 = (
        df_base
        .merge(df_exposure, on='household_key', how='left')
        .merge(df_campaign, on='household_key', how='left')
        .merge(df_outcome, on='household_key', how='left')
        .merge(df_demo_features, on='household_key', how='left')
    )

    # Fill campaign NaN (not targeted)
    campaign_cols = ['targeted_typeA', 'targeted_typeB', 'targeted_typeC',
                     'n_campaigns_targeted', 'n_typeA_campaigns']
    for col in campaign_cols:
        if col in df_track2.columns:
            df_track2[col] = df_track2[col].fillna(0).astype(int)

    return df_track2


# =============================================================================
# Scenario 1: Campaign-specific Feature Engineering
# =============================================================================

def get_campaign_periods(df_campaign_desc, campaign_type='TypeA', post_window=4):
    """Get campaign-specific pre-treatment and outcome periods.

    Args:
        df_campaign_desc: Campaign descriptions with START_DAY, END_DAY
        campaign_type: Filter by campaign type (default: 'TypeA')
        post_window: Weeks after campaign end for outcome measurement (default: 4)

    Returns:
        DataFrame with columns:
        - CAMPAIGN, campaign_type
        - start_week, end_week (campaign period)
        - pre_end_week (pre-treatment ends at start_week - 1)
        - outcome_start_week, outcome_end_week (campaign + post window, capped)
    """
    df = df_campaign_desc.copy()
    df['start_week'] = np.ceil(df['START_DAY'] / 7).astype(int)
    df['end_week'] = np.ceil(df['END_DAY'] / 7).astype(int)

    if campaign_type:
        df = df[df['DESCRIPTION'] == campaign_type].copy()

    df = df.sort_values('start_week').reset_index(drop=True)

    # Pre-treatment ends at campaign start - 1
    df['pre_end_week'] = df['start_week'] - 1

    # Outcome period: campaign start to end + post_window
    # But capped by next campaign start (to avoid contamination)
    df['next_campaign_start'] = df['start_week'].shift(-1)
    df['outcome_start_week'] = df['start_week']
    df['outcome_end_week'] = df.apply(
        lambda row: min(
            row['end_week'] + post_window,
            row['next_campaign_start'] - 1 if pd.notna(row['next_campaign_start']) else row['end_week'] + post_window
        ),
        axis=1
    ).astype(int)

    df['campaign_type'] = df['DESCRIPTION']

    return df[['CAMPAIGN', 'campaign_type', 'start_week', 'end_week',
               'pre_end_week', 'outcome_start_week', 'outcome_end_week']]


def get_campaign_periods_all_types(df_campaign_desc, post_window=4):
    """Get campaign periods for ALL campaign types (TypeA, TypeB, TypeC).

    Similar to get_campaign_periods() but includes all types.
    Outcome period is capped by next campaign of ANY type.

    Args:
        df_campaign_desc: Campaign descriptions with START_DAY, END_DAY, DESCRIPTION
        post_window: Weeks after campaign end for outcome measurement (default: 4)

    Returns:
        DataFrame with columns:
        - CAMPAIGN, campaign_type
        - start_week, end_week (campaign period)
        - pre_end_week (pre-treatment ends at start_week - 1)
        - outcome_start_week, outcome_end_week (campaign + post window, capped by next campaign)
    """
    df = df_campaign_desc.copy()
    df['start_week'] = np.ceil(df['START_DAY'] / 7).astype(int)
    df['end_week'] = np.ceil(df['END_DAY'] / 7).astype(int)

    # Sort by start_week (all campaigns together)
    df = df.sort_values('start_week').reset_index(drop=True)

    # Pre-treatment ends at campaign start - 1
    df['pre_end_week'] = df['start_week'] - 1

    # Outcome period: capped by next campaign start (ANY type)
    df['next_campaign_start'] = df['start_week'].shift(-1)
    df['outcome_start_week'] = df['start_week']
    df['outcome_end_week'] = df.apply(
        lambda row: min(
            row['end_week'] + post_window,
            row['next_campaign_start'] - 1 if pd.notna(row['next_campaign_start']) else row['end_week'] + post_window
        ),
        axis=1
    ).astype(int)

    df['campaign_type'] = df['DESCRIPTION']

    return df[['CAMPAIGN', 'campaign_type', 'start_week', 'end_week',
               'pre_end_week', 'outcome_start_week', 'outcome_end_week']]


def build_scenario1_features(df_trans, df_product, df_causal,
                              df_campaign_table, df_campaign_desc,
                              df_coupon_redempt, df_demo,
                              post_window=4,
                              first_campaign_only=True):
    """Build Scenario 1 features with campaign-specific periods.

    Args:
        df_trans: Preprocessed transaction DataFrame
        df_product: Product DataFrame
        df_causal: Causal data DataFrame
        df_campaign_table: Campaign targeting table
        df_campaign_desc: Campaign descriptions
        df_coupon_redempt: Coupon redemption DataFrame
        df_demo: Household demographic DataFrame
        post_window: Weeks after campaign end for outcome (default: 4)
        first_campaign_only: If True (default), use first TypeA campaign only
            for clean causal identification. If False, use all campaigns
            (legacy behavior, may have pre-treatment contamination).

    Returns:
        DataFrame with columns:
        - household_key, CAMPAIGN, campaign_type
        - targeted (treatment indicator)
        - pre_end_week, outcome_start_week, outcome_end_week (period info)
        - All base features (computed from pre-treatment period)
        - Exposure features (from pre-treatment period)
        - Outcome features (from campaign + post period)
        - Demographic features

    Note:
        When first_campaign_only=True (default):
        - Treatment: Each customer's first TypeA campaign (1,513 customers)
        - Control: Never-targeted customers, assigned to first active campaign (987 customers)
        - Total: 2,500 observations (1 per customer)
        - No pre-treatment contamination from previous campaigns
    """
    if first_campaign_only:
        from projects.segmentation_dunnhumby.src.cohorts import build_scenario1_first_campaign_cohort
        cohort = build_scenario1_first_campaign_cohort(
            df_trans, df_campaign_table, df_campaign_desc, post_window
        )
    else:
        from projects.segmentation_dunnhumby.src.cohorts import build_scenario1_cohort
        # Get campaign periods
        campaign_periods = get_campaign_periods(df_campaign_desc, 'TypeA', post_window)
        # Build base cohort (customer × campaign with treatment indicator)
        cohort = build_scenario1_cohort(df_trans, df_campaign_table, df_campaign_desc)
        # Add period information
        cohort = cohort.merge(
            campaign_periods[['CAMPAIGN', 'pre_end_week', 'outcome_start_week', 'outcome_end_week']],
            on='CAMPAIGN',
            how='left'
        )

    # Get unique customers and campaigns
    customers = cohort['household_key'].unique()
    unique_campaigns = cohort['CAMPAIGN'].unique()

    # Demographics (time-invariant, compute once)
    df_demo_features = build_demographic_features(df_demo)

    # Get campaign periods for feature computation
    campaign_periods = get_campaign_periods(df_campaign_desc, 'TypeA', post_window)
    campaign_periods = campaign_periods[campaign_periods['CAMPAIGN'].isin(unique_campaigns)]

    # Pre-compute features for each campaign's pre-treatment period
    campaign_features = {}

    for _, camp_row in campaign_periods.iterrows():
        campaign_id = camp_row['CAMPAIGN']
        pre_end = camp_row['pre_end_week']
        outcome_start = camp_row['outcome_start_week']
        outcome_end = camp_row['outcome_end_week']

        # Get customers assigned to this campaign
        campaign_customers = cohort[cohort['CAMPAIGN'] == campaign_id]['household_key'].values

        # Pre-treatment data
        df_trans_pre = df_trans[
            (df_trans['WEEK_NO'] <= pre_end) &
            (df_trans['household_key'].isin(campaign_customers))
        ]
        df_causal_pre = df_causal[df_causal['WEEK_NO'] <= pre_end]

        # Base features from pre-treatment
        df_base = build_all_features(df_trans_pre, df_product)

        # Exposure features from pre-treatment
        df_exposure = build_exposure_features(df_trans_pre, df_causal_pre)

        # Outcome features from campaign + post period
        df_outcome = build_outcome_features(
            df_trans, df_coupon_redempt,
            period_start=outcome_start,
            period_end=outcome_end
        )

        # Merge features for this campaign
        df_camp_features = (
            df_base
            .merge(df_exposure, on='household_key', how='left')
            .merge(df_outcome, on='household_key', how='left')
        )
        df_camp_features['CAMPAIGN'] = campaign_id

        campaign_features[campaign_id] = df_camp_features

    # Combine all campaign features
    df_all_features = pd.concat(campaign_features.values(), ignore_index=True)

    # Merge with cohort (to get treatment indicator and period info)
    result = cohort.merge(
        df_all_features,
        on=['household_key', 'CAMPAIGN'],
        how='left'
    )

    # Add demographics
    result = result.merge(df_demo_features, on='household_key', how='left')

    # Fill NaN for outcome features
    fill_cols = ['redemption_count', 'redemption_any', 'purchase_amount', 'purchase_count']
    for col in fill_cols:
        if col in result.columns:
            result[col] = result[col].fillna(0)

    return result


# =============================================================================
# Scenario 2: First Campaign Overall with Type as Attribute
# =============================================================================

def build_scenario2_features(df_trans, df_product, df_causal,
                              df_campaign_table, df_campaign_desc,
                              df_coupon_redempt, df_demo,
                              post_window=4):
    """Build Scenario 2 features with campaign-specific periods (all types).

    For clean causal identification with campaign type comparison:
    - Each customer appears exactly once using their first campaign (any type)
    - campaign_type is recorded as attribute for HTE analysis
    - Pre-treatment features computed from period before first campaign
    - Outcomes computed from campaign + post window period

    Args:
        df_trans: Preprocessed transaction DataFrame
        df_product: Product DataFrame
        df_causal: Causal data DataFrame
        df_campaign_table: Campaign targeting table
        df_campaign_desc: Campaign descriptions
        df_coupon_redempt: Coupon redemption DataFrame
        df_demo: Household demographic DataFrame
        post_window: Weeks after campaign end for outcome (default: 4)

    Returns:
        DataFrame with columns:
        - household_key, CAMPAIGN, campaign_type (TypeA/B/C/Control)
        - targeted (treatment indicator)
        - pre_end_week, outcome_start_week, outcome_end_week (period info)
        - All base features (computed from pre-treatment period)
        - Exposure features (from pre-treatment period)
        - Outcome features (from campaign + post period)
        - Demographic features
    """
    from projects.segmentation_dunnhumby.src.cohorts import build_scenario2_first_campaign_cohort

    # 1. Build cohort
    cohort = build_scenario2_first_campaign_cohort(
        df_trans, df_campaign_table, df_campaign_desc, post_window
    )

    # 2. Get campaign periods
    campaign_periods = get_campaign_periods_all_types(df_campaign_desc, post_window)
    unique_campaigns = cohort['CAMPAIGN'].unique()
    campaign_periods = campaign_periods[campaign_periods['CAMPAIGN'].isin(unique_campaigns)]

    # 3. Demographics (time-invariant, compute once)
    df_demo_features = build_demographic_features(df_demo)

    # 4. Pre-compute features for each campaign's pre-treatment period
    campaign_features = {}

    for _, camp_row in campaign_periods.iterrows():
        campaign_id = camp_row['CAMPAIGN']
        pre_end = camp_row['pre_end_week']
        outcome_start = camp_row['outcome_start_week']
        outcome_end = camp_row['outcome_end_week']

        # Get customers assigned to this campaign
        campaign_customers = cohort[cohort['CAMPAIGN'] == campaign_id]['household_key'].values

        if len(campaign_customers) == 0:
            continue

        # Pre-treatment data
        df_trans_pre = df_trans[
            (df_trans['WEEK_NO'] <= pre_end) &
            (df_trans['household_key'].isin(campaign_customers))
        ]
        df_causal_pre = df_causal[df_causal['WEEK_NO'] <= pre_end]

        # Base features from pre-treatment
        df_base = build_all_features(df_trans_pre, df_product)

        # Exposure features from pre-treatment
        df_exposure = build_exposure_features(df_trans_pre, df_causal_pre)

        # Outcome features from campaign + post period
        df_outcome = build_outcome_features(
            df_trans, df_coupon_redempt,
            period_start=outcome_start,
            period_end=outcome_end
        )

        # Merge features for this campaign
        df_camp_features = (
            df_base
            .merge(df_exposure, on='household_key', how='left')
            .merge(df_outcome, on='household_key', how='left')
        )
        df_camp_features['CAMPAIGN'] = campaign_id

        campaign_features[campaign_id] = df_camp_features

    # 5. Combine all campaign features
    df_all_features = pd.concat(campaign_features.values(), ignore_index=True)

    # 6. Merge with cohort (to get treatment indicator and period info)
    result = cohort.merge(
        df_all_features,
        on=['household_key', 'CAMPAIGN'],
        how='left'
    )

    # 7. Add demographics
    result = result.merge(df_demo_features, on='household_key', how='left')

    # 8. Fill NaN for outcome features
    fill_cols = ['redemption_count', 'redemption_any', 'purchase_amount', 'purchase_count']
    for col in fill_cols:
        if col in result.columns:
            result[col] = result[col].fillna(0)

    return result


# =============================================================================
# Feature Column Groups
# =============================================================================

FEATURE_COLS = {
    # Track 1 base features
    'recency': [
        'recency', 'recency_weeks', 'active_last_4w', 'active_last_12w',
        'days_between_purchases_avg', 'days_between_purchases_std',
    ],
    'frequency': [
        'frequency', 'frequency_per_week', 'frequency_per_month',
        'transaction_count', 'weeks_with_purchase', 'purchase_regularity',
    ],
    'monetary': [
        'monetary_sales', 'monetary_actual', 'monetary_avg_basket_sales', 'monetary_avg_basket_actual',
        'monetary_std', 'monetary_per_week', 'coupon_savings_ratio',
    ],
    'behavioral': [
        'discount_rate', 'discount_usage_pct',
        'private_label_ratio',
        'n_departments', 'n_products',
        'avg_items_per_basket', 'avg_products_per_basket',
    ],
    'category': [
        'share_grocery', 'share_fresh', 'share_bakery',
        'share_health_beauty', 'share_alcohol', 'share_other',
    ],
    'time': ['week_coverage'],

    # Track 2 features
    'exposure': [
        'display_exposure_rate', 'display_intensity_avg', 'mailer_exposure_rate',
    ],
    'campaign': [
        'targeted_typeA', 'targeted_typeB', 'targeted_typeC',
        'n_campaigns_targeted', 'n_typeA_campaigns',
    ],
    'outcome': [
        'redemption_count', 'redemption_any', 'purchase_amount', 'purchase_count',
    ],
    'demographic': [
        'AGE_DESC', 'MARITAL_STATUS_CODE', 'INCOME_DESC', 'HOMEOWNER_DESC',
        'HH_COMP_DESC', 'HOUSEHOLD_SIZE_DESC', 'KID_CATEGORY_DESC',
    ],
}

# Track 1 feature list (33 features)
TRACK1_FEATURE_COLS = [col for group in ['recency', 'frequency', 'monetary',
                                          'behavioral', 'category', 'time']
                       for col in FEATURE_COLS[group]]

# Track 1 reduced features (19 features) - removes redundant/highly correlated features
FEATURE_COLS_REDUCED = {
    'recency': ['recency', 'days_between_purchases_avg'],  # 2 (removed: recency_weeks, active_4w/12w, days_std)
    'frequency': ['frequency', 'frequency_per_week', 'purchase_regularity'],  # 3 (removed: freq_per_month, transaction_count, weeks_with_purchase)
    'monetary': ['monetary_sales', 'monetary_avg_basket_sales',
                 'monetary_std', 'coupon_savings_ratio'],  # 4 (removed: actual variants, per_week)
    'behavioral': ['discount_usage_pct', 'private_label_ratio',
                   'n_departments', 'n_products', 'avg_items_per_basket'],  # 5 (removed: discount_rate, avg_products)
    'category': ['share_grocery', 'share_fresh', 'share_bakery',
                 'share_health_beauty', 'share_alcohol'],  # 5 (removed: share_other - sum to 1)
}
TRACK1_REDUCED_COLS = [col for group in ['recency', 'frequency', 'monetary',
                                          'behavioral', 'category']
                       for col in FEATURE_COLS_REDUCED[group]]

# All base features (alias for backward compatibility)
ALL_FEATURE_COLS = TRACK1_FEATURE_COLS

# Value/Need separation for Track 1.2
VALUE_FEATURES = (FEATURE_COLS['recency'] + FEATURE_COLS['frequency'] +
                  FEATURE_COLS['monetary'] + FEATURE_COLS['time'])
NEED_FEATURES = FEATURE_COLS['behavioral'] + FEATURE_COLS['category']


# =============================================================================
# Improved Features - Batch 2
# =============================================================================

# Default interaction pairs based on domain knowledge
INTERACTION_FEATURES = [
    ('monetary_sales', 'discount_usage_pct'),   # High spender × Discount seeker
    ('frequency', 'private_label_ratio'),        # Frequent shopper × PL preference
    ('purchase_regularity', 'share_fresh'),      # Regular × Fresh food buyer
    ('n_products', 'share_health_beauty'),       # Variety seeker × H&B focus
    ('monetary_avg_basket_sales', 'n_departments'),  # Basket size × Dept diversity
    ('recency', 'frequency'),                    # Recent × Frequent (engagement)
]


def add_interaction_features(
    X_df: pd.DataFrame,
    interactions: list = None,
    prefix: str = 'int_'
) -> pd.DataFrame:
    """Add interaction terms to feature matrix.

    Creates multiplicative interactions between specified feature pairs.
    Useful for capturing non-linear relationships in treatment effect
    heterogeneity.

    Args:
        X_df: Feature DataFrame
        interactions: List of (feature1, feature2) tuples.
            Default: INTERACTION_FEATURES
        prefix: Prefix for new column names

    Returns:
        DataFrame with original features plus interaction terms
    """
    if interactions is None:
        interactions = INTERACTION_FEATURES

    X_new = X_df.copy()

    for f1, f2 in interactions:
        if f1 in X_df.columns and f2 in X_df.columns:
            col_name = f'{prefix}{f1}_x_{f2}'
            X_new[col_name] = X_df[f1] * X_df[f2]

    return X_new


def add_trend_features(
    df_trans: pd.DataFrame,
    pre_end_week: int,
    recent_weeks: int = 8,
    agg_cols: list = None
) -> pd.DataFrame:
    """Add trend features comparing recent vs past behavior.

    Creates features that capture behavioral changes over time,
    which may predict response to treatment.

    Args:
        df_trans: Transaction DataFrame with WEEK_NO, household_key
        pre_end_week: Last week of pre-treatment period
        recent_weeks: Number of weeks to consider as "recent"
        agg_cols: Columns to aggregate (default: SALES_VALUE, BASKET_ID)

    Returns:
        DataFrame indexed by household_key with trend features:
        - spending_trend: recent_sales / past_sales
        - freq_trend: recent_freq / past_freq
        - items_trend: recent_items / past_items
    """
    if agg_cols is None:
        agg_cols = ['SALES_VALUE', 'BASKET_ID']

    recent_start = pre_end_week - recent_weeks

    # Split into recent and past periods
    df_recent = df_trans[df_trans['WEEK_NO'] > recent_start]
    df_past = df_trans[df_trans['WEEK_NO'] <= recent_start]

    # Aggregate recent period
    recent_agg = df_recent.groupby('household_key').agg(
        recent_sales=('SALES_VALUE', 'sum'),
        recent_freq=('BASKET_ID', 'nunique'),
        recent_items=('QUANTITY', 'sum') if 'QUANTITY' in df_recent.columns else ('BASKET_ID', 'count')
    )

    # Aggregate past period
    past_agg = df_past.groupby('household_key').agg(
        past_sales=('SALES_VALUE', 'sum'),
        past_freq=('BASKET_ID', 'nunique'),
        past_items=('QUANTITY', 'sum') if 'QUANTITY' in df_past.columns else ('BASKET_ID', 'count')
    )

    # Combine and compute trends
    trend_df = recent_agg.join(past_agg, how='outer').fillna(0)

    # Compute ratio with smoothing to avoid division by zero
    epsilon = 1.0
    trend_df['spending_trend'] = trend_df['recent_sales'] / (trend_df['past_sales'] + epsilon)
    trend_df['freq_trend'] = trend_df['recent_freq'] / (trend_df['past_freq'] + epsilon)
    trend_df['items_trend'] = trend_df['recent_items'] / (trend_df['past_items'] + epsilon)

    # Log transform for better distribution
    trend_df['spending_trend_log'] = np.log1p(trend_df['spending_trend'])
    trend_df['freq_trend_log'] = np.log1p(trend_df['freq_trend'])

    return trend_df[['spending_trend', 'freq_trend', 'items_trend',
                     'spending_trend_log', 'freq_trend_log']]


def add_nonlinear_transforms(
    X_df: pd.DataFrame,
    log_cols: list = None,
    rank_cols: list = None
) -> pd.DataFrame:
    """Add non-linear transformations to features.

    Applies log transforms and percentile ranks to improve
    distribution and robustness to outliers.

    Args:
        X_df: Feature DataFrame
        log_cols: Columns to log-transform (default: monetary features)
        rank_cols: Columns to convert to percentile ranks

    Returns:
        DataFrame with additional transformed columns
    """
    if log_cols is None:
        log_cols = ['monetary_sales', 'frequency', 'n_products',
                    'monetary_avg_basket_sales']

    if rank_cols is None:
        rank_cols = ['monetary_sales', 'monetary_avg_basket_sales']

    X_new = X_df.copy()

    # Log transforms
    for col in log_cols:
        if col in X_df.columns:
            X_new[f'{col}_log'] = np.log1p(X_df[col])

    # Percentile ranks
    for col in rank_cols:
        if col in X_df.columns:
            X_new[f'{col}_rank'] = X_df[col].rank(pct=True)

    return X_new


# =============================================================================
# Improved Features - Batch 4: Expanded Control Design
# =============================================================================

def build_scenario1_features_expanded(
    df_features: pd.DataFrame,
    df_campaign: pd.DataFrame,
    df_campaign_desc: pd.DataFrame,
    control_definition: str = 'expanded'
) -> pd.DataFrame:
    """Build Scenario 1 features with expanded control definition.

    Original Scenario 1:
    - Treatment: First TypeA campaign targeting
    - Control: Never targeted by any campaign

    Expanded Scenario 1:
    - Treatment: First TypeA campaign targeting
    - Control: Never targeted OR only TypeB/C targeted (never TypeA)

    This increases the control group size and improves overlap with treatment.

    Args:
        df_features: Base customer features DataFrame
        df_campaign: Campaign targeting DataFrame (household_key, CAMPAIGN)
        df_campaign_desc: Campaign description DataFrame (CAMPAIGN, DESCRIPTION)
        control_definition: 'original' or 'expanded'

    Returns:
        DataFrame with treatment indicator and expanded control
    """
    # Handle case where df_campaign already has DESCRIPTION column
    if 'DESCRIPTION' in df_campaign.columns:
        df_targeting = df_campaign.copy()
    else:
        # Merge campaign info
        df_targeting = df_campaign.merge(
            df_campaign_desc[['CAMPAIGN', 'DESCRIPTION']],
            on='CAMPAIGN',
            how='left'
        )

    # Extract campaign type from DESCRIPTION (e.g., "TypeA", "TypeB", "TypeC")
    df_targeting['campaign_type'] = df_targeting['DESCRIPTION'].str.extract(r'(Type[ABC])')

    # Get first campaign per customer
    df_first = (
        df_targeting.sort_values(['household_key', 'CAMPAIGN'])
        .groupby('household_key')
        .first()
        .reset_index()
        [['household_key', 'CAMPAIGN', 'campaign_type']]
        .rename(columns={'CAMPAIGN': 'first_campaign', 'campaign_type': 'first_campaign_type'})
    )

    # Identify customers by targeting status
    all_customers = set(df_features['household_key'])
    ever_targeted = set(df_targeting['household_key'].unique())
    never_targeted = all_customers - ever_targeted

    # TypeA targeted customers
    typeA_customers = set(
        df_targeting[df_targeting['campaign_type'] == 'TypeA']['household_key'].unique()
    )

    # TypeB/C only customers (targeted but never by TypeA)
    typeBC_only = ever_targeted - typeA_customers

    # Build treatment/control indicators
    df_result = df_features.copy()
    df_result = df_result.merge(df_first, on='household_key', how='left')

    if control_definition == 'original':
        # Original: Control = never targeted
        df_result['treatment'] = df_result['household_key'].isin(typeA_customers).astype(int)
        df_result['in_sample'] = (
            df_result['household_key'].isin(typeA_customers) |
            df_result['household_key'].isin(never_targeted)
        )
        df_result['control_type'] = np.where(
            df_result['household_key'].isin(never_targeted),
            'never_targeted',
            np.where(df_result['household_key'].isin(typeA_customers), 'treated', 'excluded')
        )

    elif control_definition == 'expanded':
        # Expanded: Control = never targeted OR TypeB/C only
        df_result['treatment'] = (
            df_result['first_campaign_type'] == 'TypeA'
        ).fillna(False).astype(int)

        # In sample: TypeA first OR never targeted OR TypeB/C only
        df_result['in_sample'] = (
            (df_result['first_campaign_type'] == 'TypeA') |
            df_result['household_key'].isin(never_targeted) |
            df_result['household_key'].isin(typeBC_only)
        )

        df_result['control_type'] = np.where(
            df_result['household_key'].isin(never_targeted),
            'never_targeted',
            np.where(
                df_result['household_key'].isin(typeBC_only),
                'typeBC_only',
                np.where(df_result['treatment'] == 1, 'treated', 'excluded')
            )
        )

    else:
        raise ValueError(f"Unknown control_definition: {control_definition}")

    # Summary statistics
    summary = {
        'n_total': len(df_result),
        'n_in_sample': df_result['in_sample'].sum(),
        'n_treatment': (df_result['treatment'] == 1).sum(),
        'n_control': ((df_result['in_sample']) & (df_result['treatment'] == 0)).sum(),
        'n_never_targeted': len(never_targeted),
        'n_typeBC_only': len(typeBC_only),
        'control_definition': control_definition
    }

    # Add summary as attribute (accessible via df_result.attrs)
    df_result.attrs['sample_summary'] = summary

    return df_result


def get_expanded_control_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Get summary of expanded control design from built features.

    Args:
        df: DataFrame from build_scenario1_features_expanded

    Returns:
        Summary DataFrame
    """
    if 'control_type' not in df.columns:
        raise ValueError("DataFrame must have 'control_type' column from build_scenario1_features_expanded")

    summary = df.groupby('control_type').agg(
        n_customers=('household_key', 'count'),
        mean_monetary=('monetary_sales', 'mean') if 'monetary_sales' in df.columns else ('household_key', 'count'),
        mean_frequency=('frequency', 'mean') if 'frequency' in df.columns else ('household_key', 'count'),
    ).reset_index()

    # Add percentages
    total = summary['n_customers'].sum()
    summary['pct'] = summary['n_customers'] / total * 100

    return summary