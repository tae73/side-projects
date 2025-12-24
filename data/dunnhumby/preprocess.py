"""Dunnhumby transaction data preprocessing utilities."""

import pandas as pd


def load_transactions(data_path, preprocess=True):
    """Load and optionally preprocess transaction data.

    Args:
        data_path: Path to raw data directory containing transaction_data.csv
        preprocess: If True, apply standard preprocessing (default: True)

    Returns:
        DataFrame with transaction data
    """
    df = pd.read_csv(data_path / 'transaction_data.csv')

    if preprocess:
        df = preprocess_transactions(df)

    return df


def preprocess_transactions(df):
    """Apply standard preprocessing to transaction data.

    Steps:
    1. Remove data artifacts (QUANTITY=0, positive RETAIL_DISC)
    2. Negate discount columns (raw data stores as negative)
    3. Add derived price columns (SHELF_PRICE, UNIT_PRICE, ACTUAL_SPENT)

    Args:
        df: Raw transaction DataFrame

    Returns:
        Preprocessed DataFrame
    """
    # Step 1: Remove artifacts
    # - QUANTITY=0: Invalid transaction (no purchase)
    # - RETAIL_DISC > 0: Floating point artifact (should be negative or zero)
    df = df[(df['QUANTITY'] > 0) & (df['RETAIL_DISC'] <= 0)].copy()

    # Step 2: Negate discount columns (stored as negative in raw data)
    # After this, positive values = discount amount
    df = df.assign(**{
        'RETAIL_DISC': -df['RETAIL_DISC'],
        'COUPON_DISC': -df['COUPON_DISC'],
        'COUPON_MATCH_DISC': -df['COUPON_MATCH_DISC'],
    })

    # Step 3: Derived price columns
    # Reference: dunnhumby User Guide p.3
    df = df.assign(**{
        # Original shelf price before any discount
        'SHELF_PRICE': (df['SALES_VALUE'] + df['RETAIL_DISC'] + df['COUPON_MATCH_DISC']) / df['QUANTITY'],
        # Price per unit charged to customer
        'UNIT_PRICE': df['SALES_VALUE'] / df['QUANTITY'],
        # What customer actually paid (after manufacturer coupon)
        'ACTUAL_SPENT': df['SALES_VALUE'] - df['COUPON_DISC'],
    })

    return df
