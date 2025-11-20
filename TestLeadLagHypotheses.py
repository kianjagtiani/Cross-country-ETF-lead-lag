#Macro hypotheses tested:
#- Hypothesis 1: it's expected qqq follows spy since spy reflects broad-market shifts, which qqq then adjusts to
#- Hypothesis 2: xlk follows qqq because qqq contains mega-cap tech names w/ heavier weightings
#- Hypothesis 3: iwm follows spy, since small-caps are less liquid + incorporate info more slowly, while large-caps react faster to macro news


import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def find_lead_lag(etfx, etfy, maxLag=5):
    lagRange = range(-maxLag, maxLag+1)
    correlations = {}
    for lag in lagRange:
        shifted_y = etfy.shift(-lag)
        correlations[lag] = etfx.corr(shifted_y)
    return pd.Series(correlations)

LL_pairs = {
    "QQQ follows SPY": ("SPY", "QQQ"),
    "XLK follows QQQ": ("QQQ", "XLK"),
    "IWM follows SPY": ("SPY", "IWM")
}

tickers = list(set([t for pair in LL_pairs.values() for t in pair]))

end_date = datetime.now()
start_date = end_date - timedelta(days=30)

# Split into 7-day chunks for 1-minute data
chunks = []
current_start = start_date
while current_start < end_date:
    current_end = min(current_start + timedelta(days=7), end_date)
    chunk = yf.download(
        tickers=tickers,
        start=current_start.strftime("%Y-%m-%d"),
        end=current_end.strftime("%Y-%m-%d"),
        interval="1m"
    )["Close"]
    chunks.append(chunk)
    current_start = current_end

data = pd.concat(chunks)
data = data.dropna()

window_size = 30
results = []

for start_idx in range(len(data) - window_size + 1):
    end_idx = start_idx + window_size
    period_data = data.iloc[start_idx:end_idx]
    period_returns = np.log(period_data / period_data.shift(1)).dropna()
    period_label = f"{period_data.index[0]} to {period_data.index[-1]}"
    
    for desc, (x, y) in LL_pairs.items():
        corrs = find_lead_lag(period_returns[y], period_returns[x], maxLag=5)
        best_lag = corrs.idxmax()
        max_corr = corrs.max()
        results.append({
            "Period": period_label,
            "Hypothesis": desc,
            "Lag (minutes)": best_lag,
            "Correlation": max_corr
        })

results_df = pd.DataFrame(results)
nonzero_lags_df = results_df[results_df["Lag (minutes)"] != 0]
nonzero_lags_df.to_csv("nonzero_lead_lag_results.csv", index=False)