"""Cohort definition functions for segmentation study."""

import numpy as np
import pandas as pd


# =============================================================================
# Data Validation
# =============================================================================

def _filter_valid_transactions(df_trans: pd.DataFrame) -> pd.DataFrame:
    """Filter out invalid transactions.

    Removes:
    - QUANTITY <= 0: Invalid transaction (no actual purchase)
    - RETAIL_DISC > 0: Floating point artifact (raw data stores discounts as negative)

    Args:
        df_trans: Transaction DataFrame

    Returns:
        Filtered DataFrame with valid transactions only
    """
    return df_trans[(df_trans['QUANTITY'] > 0) & (df_trans['RETAIL_DISC'] <= 0)]


# =============================================================================
# Track 1: Customer Understanding
# =============================================================================

def define_track1_cohort(df_trans: pd.DataFrame) -> np.ndarray:
    """Define Track 1 cohort: customers with at least 1 purchase in Week 1-102.

    Args:
        df_trans: Transaction DataFrame with 'household_key' column

    Returns:
        Array of unique household_key values
    """
    df_valid = _filter_valid_transactions(df_trans)
    return df_valid['household_key'].unique()


# =============================================================================
# Track 2: Causal Targeting - Base Cohort
# =============================================================================

def define_track2_base_cohort(df_trans: pd.DataFrame) -> np.ndarray:
    """Define Track 2 base cohort: pre-treatment history + campaign period active.

    Inclusion criteria:
    - At least 1 purchase in pre-treatment period (Week 1-31)
    - At least 1 purchase in campaign period (Week 32-102)

    Args:
        df_trans: Transaction DataFrame with 'household_key' and 'WEEK_NO'

    Returns:
        Array of household_key values meeting both criteria
    """
    df_valid = _filter_valid_transactions(df_trans)
    pre_treatment = df_valid[df_valid['WEEK_NO'] <= 31]['household_key'].unique()
    campaign_period = df_valid[df_valid['WEEK_NO'] >= 32]['household_key'].unique()
    return np.intersect1d(pre_treatment, campaign_period)


# =============================================================================
# Track 2: Scenario-specific Cohort Builders
# =============================================================================

def build_scenario1_cohort(
    df_trans: pd.DataFrame,
    df_campaign_table: pd.DataFrame,
    df_campaign_desc: pd.DataFrame
) -> pd.DataFrame:
    """Build Scenario 1 cohort: Customer x TypeA Campaign.

    Args:
        df_trans: Transaction DataFrame
        df_campaign_table: Campaign targeting table (household_key, CAMPAIGN)
        df_campaign_desc: Campaign descriptions (CAMPAIGN, DESCRIPTION, START_DAY, END_DAY)

    Returns:
        DataFrame with columns:
        - household_key, CAMPAIGN, campaign_type
        - targeted (binary treatment indicator)
    """
    # Get TypeA campaigns
    typeA = df_campaign_desc[df_campaign_desc['DESCRIPTION'] == 'TypeA']['CAMPAIGN'].values

    # Get base cohort
    base_cohort = define_track2_base_cohort(df_trans)

    # Create all customer x TypeA campaign combinations
    cohort = pd.DataFrame({
        'household_key': np.repeat(base_cohort, len(typeA)),
        'CAMPAIGN': np.tile(typeA, len(base_cohort))
    })

    # Add treatment indicator
    targeted = df_campaign_table[df_campaign_table['CAMPAIGN'].isin(typeA)][
        ['household_key', 'CAMPAIGN']
    ].assign(targeted=1)

    cohort = (
        cohort
        .merge(targeted, on=['household_key', 'CAMPAIGN'], how='left')
        .assign(
            targeted=lambda x: x['targeted'].fillna(0).astype(int),
            campaign_type='TypeA'
        )
    )

    return cohort


def build_scenario2_cohort(
    df_trans: pd.DataFrame,
    df_campaign_table: pd.DataFrame,
    df_campaign_desc: pd.DataFrame
) -> pd.DataFrame:
    """Build Scenario 2 cohort: Customer x All Campaigns with type as moderator.

    Args:
        df_trans: Transaction DataFrame
        df_campaign_table: Campaign targeting table
        df_campaign_desc: Campaign descriptions

    Returns:
        DataFrame with columns:
        - household_key, CAMPAIGN, campaign_type (moderator)
        - targeted (binary treatment indicator)
    """
    # Get all campaigns
    all_campaigns = df_campaign_desc['CAMPAIGN'].values
    campaign_types = df_campaign_desc.set_index('CAMPAIGN')['DESCRIPTION'].to_dict()

    # Get base cohort
    base_cohort = define_track2_base_cohort(df_trans)

    # Create all customer x campaign combinations
    cohort = pd.DataFrame({
        'household_key': np.repeat(base_cohort, len(all_campaigns)),
        'CAMPAIGN': np.tile(all_campaigns, len(base_cohort))
    })

    # Add treatment indicator
    targeted = df_campaign_table[['household_key', 'CAMPAIGN']].assign(targeted=1)

    cohort = (
        cohort
        .merge(targeted, on=['household_key', 'CAMPAIGN'], how='left')
        .assign(
            targeted=lambda x: x['targeted'].fillna(0).astype(int),
            campaign_type=lambda x: x['CAMPAIGN'].map(campaign_types)
        )
    )

    return cohort


def build_scenario3_cohort(
    df_trans: pd.DataFrame,
    df_campaign_table: pd.DataFrame,
    df_campaign_desc: pd.DataFrame
) -> pd.DataFrame:
    """Build Scenario 3 cohort: Customer-level aggregation.

    Args:
        df_trans: Transaction DataFrame
        df_campaign_table: Campaign targeting table
        df_campaign_desc: Campaign descriptions

    Returns:
        DataFrame with columns:
        - household_key
        - ever_targeted_typeA (binary)
        - n_typeA_campaigns (count)
        - ever_targeted_any (binary)
        - n_campaigns_total (count)
    """
    # Get TypeA campaigns
    typeA = df_campaign_desc[df_campaign_desc['DESCRIPTION'] == 'TypeA']['CAMPAIGN'].values

    # Get base cohort
    base_cohort = define_track2_base_cohort(df_trans)

    # Count TypeA targeting per customer
    typeA_targeting = (
        df_campaign_table[df_campaign_table['CAMPAIGN'].isin(typeA)]
        .groupby('household_key')['CAMPAIGN']
        .nunique()
        .reset_index()
        .rename(columns={'CAMPAIGN': 'n_typeA_campaigns'})
    )

    # Count all targeting per customer
    all_targeting = (
        df_campaign_table
        .groupby('household_key')['CAMPAIGN']
        .nunique()
        .reset_index()
        .rename(columns={'CAMPAIGN': 'n_campaigns_total'})
    )

    # Build cohort
    cohort = (
        pd.DataFrame({'household_key': base_cohort})
        .merge(typeA_targeting, on='household_key', how='left')
        .merge(all_targeting, on='household_key', how='left')
        .fillna(0)
        .astype({'n_typeA_campaigns': int, 'n_campaigns_total': int})
        .assign(
            ever_targeted_typeA=lambda x: (x['n_typeA_campaigns'] > 0).astype(int),
            ever_targeted_any=lambda x: (x['n_campaigns_total'] > 0).astype(int)
        )
    )

    return cohort[['household_key', 'ever_targeted_typeA', 'n_typeA_campaigns',
                   'ever_targeted_any', 'n_campaigns_total']]


# =============================================================================
# Cohort Comparison Utilities
# =============================================================================

def summarize_cohort(cohort: pd.DataFrame, name: str) -> dict:
    """Summarize a cohort for comparison.

    Args:
        cohort: Cohort DataFrame
        name: Cohort name for display

    Returns:
        Dictionary with summary statistics
    """
    if 'targeted' in cohort.columns:
        n_treatment = cohort['targeted'].sum()
        n_control = len(cohort) - n_treatment
        treatment_pct = n_treatment / len(cohort) * 100
    else:
        n_treatment = None
        n_control = None
        treatment_pct = None

    return {
        'name': name,
        'n_total': len(cohort),
        'n_customers': cohort['household_key'].nunique(),
        'n_treatment': n_treatment,
        'n_control': n_control,
        'treatment_pct': treatment_pct
    }


def compare_cohorts(cohorts: dict) -> pd.DataFrame:
    """Compare multiple cohorts.

    Args:
        cohorts: Dictionary of {name: cohort_df}

    Returns:
        DataFrame with comparison statistics
    """
    summaries = [summarize_cohort(df, name) for name, df in cohorts.items()]
    return pd.DataFrame(summaries).set_index('name')
