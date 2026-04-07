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


def compute_residuals(
    asset_returns: pd.Series,
    market_returns: pd.Series,
    window: int,
) -> pd.Series:
    """Rolling-window OLS residuals (last-day residual from each trailing window).

    For day i (i >= window-1), fits OLS on the trailing window, then
    residual[i] = asset_returns[i] - predicted[i]. First (window - 1) values are NaN.
    """
    n = len(asset_returns)
    residuals = pd.Series(np.nan, index=asset_returns.index, dtype=float)
    for i in range(window - 1, n):
        y = asset_returns.iloc[i - window + 1 : i + 1].values
        x = market_returns.iloc[i - window + 1 : i + 1].values
        X = sm.add_constant(x)
        model = sm.OLS(y, X).fit()
        residuals.iloc[i] = model.resid[-1]
    return residuals


def compute_rolling_residual_correlation(
    resid_a: pd.Series,
    resid_b: pd.Series,
    window: int,
) -> pd.Series:
    """Rolling Pearson correlation of two residual series."""
    return compute_rolling_correlation(resid_a, resid_b, window)


def compute_residual_stats(
    resid_a: pd.Series,
    resid_b: pd.Series,
) -> dict:
    """OLS regression of resid_a on resid_b. Returns R², slope, intercept.

    NaN values are dropped before fitting.
    """
    mask = resid_a.notna() & resid_b.notna()
    y = resid_a[mask].values
    x = resid_b[mask].values
    X = sm.add_constant(x)
    model = sm.OLS(y, X).fit()
    return {
        "r_squared": model.rsquared,
        "slope": model.params[1],
        "intercept": model.params[0],
    }
