import yfinance as yf
import pandas as pd
import numpy as np

tickers = ["XLK", "QQQ", "EWJ", "EWT", "EWC", "EWU"]

frequency = "m"

if frequency == "h":
   data = yf.download(
      tickers,
      start="2025-08-01",
      end="2025-10-27",
      interval="1h"
   )["Close"]

   # hourly log returns
   returns = np.log(data / data.shift(1)).dropna()

   def findLeadLagCorrs(etfx, etfy, maxLag=5):
      lagRange = range(-maxLag, maxLag + 1)
      correlations = {}
      for lag in lagRange:
         shiftedy = etfy.shift(-lag)  # negative shift => Y leads X
         corr = etfx.corr(shiftedy)
         correlations[lag] = corr
      return pd.Series(correlations)

   # compute correlations for all pairs
   for ticker1 in tickers:
      for ticker2 in tickers:
         if ticker1 != ticker2:
               corrs = findLeadLagCorrs(returns[ticker1], returns[ticker2], maxLag=5)
               best_lag = corrs.idxmax()
               print(f"Highest correlation at lag = {best_lag} hours → {ticker1} leads {ticker2} by {best_lag} hours")

elif frequency == "m":
    data = yf.download(
        tickers,
        period="7d",
        interval="1m",
        prepost=False,
        progress=False
    )["Close"]

    # minute-level log returns
    returns = np.log(data / data.shift(1)).dropna()

    def findLeadLagCorrs(etfx, etfy, maxLag=5):
        lagRange = range(-maxLag, maxLag + 1)
        correlations = {}
        for lag in lagRange:
            shiftedy = etfy.shift(-lag)  # negative lag => Y leads X
            corr = etfx.corr(shiftedy)
            correlations[lag] = corr
        return pd.Series(correlations)

    # compute correlations for all pairs
    for ticker1 in tickers:
        for ticker2 in tickers:
            if ticker1 != ticker2:
                corrs = findLeadLagCorrs(returns[ticker1], returns[ticker2], maxLag=5)
                best_lag = corrs.idxmax()
                print(f"Highest correlation at lag = {best_lag} minutes → {ticker1} leads {ticker2} by {best_lag} minutes")