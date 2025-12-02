import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


# ================================================================
# Configuration
# ================================================================

TICKERS = ["EWJ", "EWT", "EWC", "EWU", "XLK", "QQQ"]

# X leads Y
LL_PAIRS = {
    "EWJ follows QQQ": ("QQQ", "EWJ"),
    "EWT follows QQQ": ("QQQ", "EWT"),
    "EWC follows QQQ": ("QQQ", "EWC"),
    "EWU follows QQQ": ("QQQ", "EWU"),
    "XLK follows QQQ": ("QQQ", "XLK")
}

MAX_LAG = 5               # +/- 5 minutes
WINDOW = 30               # 30-minute rolling window
LOOKBACK_DAYS = 5         # Keep small for reliable 1m data
OUTPUT_FILE = "clean_lead_lag_results_cross_country.csv"


# ================================================================
# Lead-lag correlation
# ================================================================

def compute_lead_lag(series_y, series_x, max_lag):
    corrs = {}
    for lag in range(-max_lag, max_lag + 1):
        shifted_y = series_y.shift(-lag)
        corrs[lag] = series_x.corr(shifted_y)
    return pd.Series(corrs)


# ================================================================
# Safe 1-minute downloader
# ================================================================

def get_data(tickers, lookback_days):
    end = datetime.now()
    start = end - timedelta(days=lookback_days)

    print(f"Downloading 1m data from {start.date()} to {end.date()}...")

    df = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        interval="1m",
        auto_adjust=False
    )

    if "Close" in df:
        df = df["Close"]

    df = df.dropna(how="all")
    print(f"Downloaded shape: {df.shape}")

    return df


# ================================================================
# Main execution
# ================================================================

def run_analysis():
    # Step 1: Get data
    prices = get_data(TICKERS, LOOKBACK_DAYS)

    # Step 2: Compute log returns
    returns = np.log(prices / prices.shift(1)).dropna()

    results = []

    print("Computing rolling lead-lag relationships...")

    # Step 3: Rolling window lead-lag analysis
    for start_idx in range(len(returns) - WINDOW + 1):
        end_idx = start_idx + WINDOW
        window = returns.iloc[start_idx:end_idx]
        period_label = f"{window.index[0]} → {window.index[-1]}"

        for desc, (leader, follower) in LL_PAIRS.items():
            corr_series = compute_lead_lag(
                series_y=window[follower],
                series_x=window[leader],
                max_lag=MAX_LAG
            )

            best_lag = corr_series.idxmax()
            best_corr = corr_series.max()

            results.append({
                "Period": period_label,
                "Hypothesis": desc,
                "Lag (min)": best_lag,
                "Correlation": best_corr
            })

    # Step 4: Save output
    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_FILE, index=False)

    print("\n✔ Done!")
    print(f"File written → {OUTPUT_FILE}")
    print(results_df.head())

    return results_df


# ================================================================
# Run when executed as a script
# ================================================================

if __name__ == "__main__":
    run_analysis()
