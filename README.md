# correlation

A toolkit for testing whether two stocks have genuine pairwise correlation or if their co-movement is just shared exposure to a common factor (e.g. QQQ). Given any pair of tickers and a market factor, it strips out the factor via rolling OLS regression and checks whether the residuals are still correlated.

## How it works

1. Pull daily close prices via yfinance
2. Compute rolling pairwise correlation (raw)
3. Compute rolling OLS beta to the market factor for each stock
4. Regress both stocks on the market factor, extract residuals
5. Compute rolling correlation of the residuals — if near zero, the co-movement is entirely shared factor exposure

## Usage

The analytical functions in `correlation/analysis.py` are ticker-agnostic. To analyze a new pair, create a notebook in `notebooks/` and swap in your tickers:

```python
from correlation.analysis import (
    compute_rolling_correlation,
    compute_rolling_beta,
    compute_residuals,
    compute_rolling_residual_correlation,
    compute_residual_stats,
)

tickers = ["META", "AAPL", "QQQ"]  # [stock_a, stock_b, factor]
data = yf.download(tickers, period="1y")["Close"]
returns = data.pct_change().dropna()
```

See `notebooks/pltr_nvda_correlation.ipynb` for a complete example.

## Analyses

- [NVDA vs PLTR](analysis/nvda_pltr.md) — co-movement is entirely shared QQQ beta; transient beta overlap, not structural linkage

## Structure

- `correlation/analysis.py` — reusable analytical functions (rolling correlation, rolling beta, residual computation, residual stats)
- `tests/test_correlation.py` — 22 pytest tests covering edge cases, known outputs, NaN handling, and shape/dtype expectations
- `notebooks/` — analysis notebooks (not tracked in git)
- `analysis/` — writeups and findings for each pair analyzed

## Setup

```
pip install -e ".[dev]"
```

Run notebooks:

```
jupyter lab notebooks/
```

Run tests:

```
pytest -v
```
