"""Analytical functions for pairwise correlation vs shared factor analysis."""

import numpy as np
import pandas as pd
import statsmodels.api as sm


def compute_rolling_correlation(
    series_a: pd.Series,
    series_b: pd.Series,
    window: int,
) -> pd.Series:
    """Rolling Pearson correlation between two return series.

    First (window - 1) values are NaN.
    """
    return series_a.rolling(window).corr(series_b)


def compute_rolling_beta(
    asset_returns: pd.Series,
    market_returns: pd.Series,
    window: int,
) -> pd.Series:
    """Rolling OLS beta (slope) of asset returns on market returns.

    For each day i >= window-1, fits OLS with intercept on the trailing
    window and records the slope. First (window - 1) values are NaN.
    """
    n = len(asset_returns)
    betas = pd.Series(np.nan, index=asset_returns.index, dtype=float)
    for i in range(window - 1, n):
        y = asset_returns.iloc[i - window + 1 : i + 1].values
        x = market_returns.iloc[i - window + 1 : i + 1].values
        X = sm.add_constant(x)
        model = sm.OLS(y, X).fit()
        betas.iloc[i] = model.params[1]
    return betas
