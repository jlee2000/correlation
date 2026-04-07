# correlation

Investigating whether PLTR and NVDA have genuine pairwise correlation or if their recent co-movement is just shared QQQ beta exposure.

## Approach

Regress both PLTR and NVDA daily returns on QQQ returns using 63-day rolling OLS windows, extract the residuals (the part of each stock's return not explained by QQQ), then compute rolling correlation of those residuals. If residual correlation is near zero, the co-movement is entirely shared market factor. If elevated, there's genuine pairwise signal.

## Findings (Apr 2026, trailing 63-day window)

| Metric | Value |
|--------|-------|
| Raw Pairwise Correlation | 0.365 |
| PLTR Beta to QQQ | 1.526 |
| NVDA Beta to QQQ | 1.531 |
| Residual Correlation (63d) | -0.072 |
| Residual R² | 0.005 |

**Conclusion: the co-movement is entirely shared QQQ beta.** Both stocks have nearly identical betas (~1.53), meaning they amplify Nasdaq moves by the same factor. Once that exposure is stripped out, residual correlation drops to -0.07 with an R² of 0.5% — no meaningful idiosyncratic co-movement.

The more interesting finding is in the rolling beta chart: PLTR's beta peaked at ~2.5 in mid-January 2026 and has been declining toward ~1.5, where it happens to coincide with NVDA's beta today. The co-movement isn't "PLTR became an AI stock" — it's that PLTR's beta happened to land on NVDA's beta for a window, creating a temporary illusion of pairwise linkage. If PLTR continues mean-reverting toward its historical 1.0-1.2 government-contractor beta, the apparent correlation will weaken even though nothing fundamental changed between the two companies.

A 7-day rolling residual correlation was also computed. It spikes occasionally (up to ~0.75), but these spikes are not statistically significant — with n=7 the critical value for p=0.05 is ~0.75, and after accounting for multiple overlapping windows, nothing survives. The 7-day line oscillates symmetrically around zero with no persistent bias, consistent with noise around a true correlation of zero.

The 63-day window provides much stronger evidence: with n=63, correlations as low as ~0.25 would be detectable, yet the line stays in a tight band around zero. This is evidence of absence rather than absence of evidence.

## Structure

- `correlation/analysis.py` — reusable analytical functions (rolling correlation, rolling beta, residual computation, residual stats)
- `tests/test_correlation.py` — 22 pytest tests covering edge cases, known outputs, NaN handling, and shape/dtype expectations
- `notebooks/pltr_nvda_correlation.ipynb` — full analysis with charts (not tracked in git)

## Setup

```
pip install -e ".[dev]"
```

Run the notebook:

```
jupyter lab notebooks/
```
