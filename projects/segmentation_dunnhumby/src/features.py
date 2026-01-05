"""Customer feature engineering for segmentation."""

import numpy as np
import pandas as pd


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


# Feature column groups for reference
FEATURE_COLS = {
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
}

ALL_FEATURE_COLS = [col for cols in FEATURE_COLS.values() for col in cols]