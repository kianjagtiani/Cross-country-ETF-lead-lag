# Cross-Country ETF Lead-Lag

QuantSC · Single-Stock & ETF Desk, Project #1 \
**Desk Head:** Kian Jagtiani

---

Does price information travel between markets slowly enough to trade on? This project tests whether US-listed country ETFs and sector baskets lead or lag each other intraday, starting from daily correlations and working down to the minute.

## What the work included

- Daily lead-lag and correlation analysis on financial-sector country ETFs (XLF, EWU, EWJ, EWC, EIRL) from 2020 to 2025.
- A set of macro hypotheses about which ETF should follow which (country ETFs following QQQ, IWM following SPY, XLK following QQQ), tested rather than assumed.
- A rolling 30-minute, 1-minute-resolution engine that scans lags of ±5 minutes and records the correlation-maximizing lag for each window.
- Cross-correlation on log returns across QQQ, XLK, and the country ETFs (EWJ, EWT, EWC, EWU), plus a US-only variant on SPY, QQQ, and IWM.
- Per-window result tables and merged-timeframe plots for every hypothesis.

## Repository contents

- `leadLag.py` and `findCorrelationMatrix.py`: daily-frequency analysis on financial-sector country ETFs.
- `TestLeadLagHypotheses.py` and `TestLeadLagHypotheses_CrossCountry.py`: the 1-minute rolling-window tests that write the result CSVs.
- `ShowMergedTimeFrames*.py`: plotting, with figures saved under `lead_lag_plots_*/`.
- `*_results*.csv`: best lag and correlation per window. Prices come from Yahoo Finance at run time, so nothing is stored.
