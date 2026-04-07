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
