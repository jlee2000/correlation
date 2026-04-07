"""Tests for correlation.analysis functions."""

import numpy as np
import pandas as pd
import pytest

from correlation.analysis import compute_rolling_correlation

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(42)
SERIES_LEN = 100
WINDOW = 10


@pytest.fixture
def random_series():
    return pd.Series(RNG.standard_normal(SERIES_LEN))


# ---------------------------------------------------------------------------
# compute_rolling_correlation
# ---------------------------------------------------------------------------


class TestComputeRollingCorrelation:
    def test_self_is_one(self, random_series):
        result = compute_rolling_correlation(random_series, random_series, WINDOW)
        valid = result.dropna()
        np.testing.assert_allclose(valid.values, 1.0, atol=1e-10)

    def test_perfect_negative(self, random_series):
        result = compute_rolling_correlation(random_series, -random_series, WINDOW)
        valid = result.dropna()
        np.testing.assert_allclose(valid.values, -1.0, atol=1e-10)

    def test_nan_padding(self, random_series):
        result = compute_rolling_correlation(random_series, random_series, WINDOW)
        assert result.iloc[: WINDOW - 1].isna().all()
        assert result.iloc[WINDOW - 1 :].notna().all()

    def test_output_shape_dtype(self, random_series):
        other = pd.Series(RNG.standard_normal(SERIES_LEN))
        result = compute_rolling_correlation(random_series, other, WINDOW)
        assert len(result) == SERIES_LEN
        assert result.dtype == np.float64

    def test_constant_series(self):
        s = pd.Series(np.ones(SERIES_LEN))
        other = pd.Series(RNG.standard_normal(SERIES_LEN))
        result = compute_rolling_correlation(s, other, WINDOW)
        # Correlation with a constant series is undefined (NaN)
        assert result.iloc[WINDOW - 1 :].isna().all()
