import yfinance as yf
import pandas as pd
import numpy as np

# import data
tickers = ["XLF", "EWU", "EWJ", "EWC", "EIRL"]  # us, uk, japan, canada, ireland etfs for financial sector
data = yf.download(tickers, start="2020-01-01", end="2025-01-01")["Close"] # download data from 2020 to 2025

# daily returns; take log
returns = np.log(data / data.shift(1)).dropna()

# calculate lead lag correlation between etfx and etfy
# maxLag is set to 5 arbitrarily
# if - lag, etfy leads etfx
# if + lag, etfx leads etfy
# calculates correlation between etfx and etfy, trying different lag values
#     essentially: correlation between etfx(t) and etfy(t + lag)
def findLeadLagCorrs(etfx, etfy, maxLag=5):
    lagRange = range(-maxLag, maxLag + 1) # range of possible lags we're trying
    correlations = {}
    for lag in lagRange:
        shiftedy = etfy.shift(-lag)  # note: negative shift means Y leads X
        corr = etfx.corr(shiftedy)
        correlations[lag] = corr
    return pd.Series(correlations)

# test example: checking if japan leads the us then finding lag with max correlation
#corrs = findLeadLagCorrs(returns["EWJ"], returns["XLF"], maxLag=5)
#print(corrs)

#best_lag = corrs.idxmax()
#print(f"Highest correlation at lag = {best_lag} days → EWJ leads XLF by {best_lag} days")

# now try for all possible pairs of countries:
for ticker1 in tickers:
    for ticker2 in tickers:
      if (ticker1 != ticker2):
         corrs = findLeadLagCorrs(returns[ticker1], returns[ticker2], maxLag=5)
         
         best_lag = corrs.idxmax()
         print(f"Highest correlation at lag = {best_lag} days → {ticker1} leads {ticker2} by {best_lag} days")
         