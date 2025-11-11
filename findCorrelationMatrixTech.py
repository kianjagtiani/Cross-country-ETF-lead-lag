import yfinance as yf
import pandas as pd

# download tech ETFs
tickers = {
    "XLK": "US Tech",
    "QQQ": "US Nasdaq",
    "EWJ": "Japan ETF",
    "EWT": "Taiwan ETF",
    "EWC": "Canada ETF",
    "EWU": "UK ETF"
}

tickers = ["XLK", "QQQ", "EWJ", "EWT", "EWC", "EWU"]
data = yf.download(tickers, start="2022-01-01", end="2025-01-01")["Close"] # data from 2022 to 2025

data = yf.download(tickers, start="2022-01-01", end="2025-01-01") # data from 2022 to 2025
print(data.columns)

# daily returns
returns = data.pct_change().dropna()

# correlation matrix (diagonal from upper left to lower right is 1.00, disregard)
corr_matrix = returns.corr()
print(corr_matrix)
