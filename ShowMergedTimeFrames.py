import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import timedelta, datetime
import os

results_df = pd.read_csv("nonzero_lead_lag_results.csv")

results_df["start"] = pd.to_datetime(results_df["Period"].apply(lambda x: x.split(" to ")[0]))
results_df["end"] = pd.to_datetime(results_df["Period"].apply(lambda x: x.split(" to ")[1]))
results_df = results_df.sort_values("start").reset_index(drop=True)

hypothesis_pairs = {
    "QQQ follows SPY": ("SPY", "QQQ"),
    "XLK follows QQQ": ("QQQ", "XLK"),
    "IWM follows SPY": ("SPY", "IWM")
}

tickers = list(set([t for pair in hypothesis_pairs.values() for t in pair]))

overall_start = results_df["start"].min()
overall_end = results_df["end"].max()

data_chunks = []
current_start = overall_start
while current_start < overall_end:
    current_end = min(current_start + timedelta(days=7), overall_end)
    chunk = yf.download(
        tickers=tickers,
        start=current_start.strftime("%Y-%m-%d"),
        end=current_end.strftime("%Y-%m-%d"),
        interval="1m"
    )["Close"]
    data_chunks.append(chunk)
    current_start = current_end

data = pd.concat(data_chunks).dropna()

groups = []
current_group = [0]
for i in range(1, len(results_df)):
    prev_end = results_df.loc[i-1, "end"]
    curr_start = results_df.loc[i, "start"]
    if curr_start <= prev_end + pd.Timedelta(minutes=1) and (curr_start - results_df.loc[current_group[0], "start"] <= pd.Timedelta(hours=1)):
        current_group.append(i)
    else:
        groups.append(current_group)
        current_group = [i]
groups.append(current_group)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
folder_name = f"lead_lag_plots_{timestamp}"
os.makedirs(folder_name, exist_ok=True)

for group in groups:
    start_idx = group[0]
    end_idx = group[-1]
    combined_start = results_df.loc[start_idx, "start"]
    combined_end = results_df.loc[end_idx, "end"]
    
    merged_data = data.loc[combined_start:combined_end]
    if merged_data.empty:
        print(f"No price data for {combined_start} to {combined_end}, skipping.")
        continue
    
    hypotheses_in_group = results_df.loc[group, ["Hypothesis", "Lag (minutes)"]].drop_duplicates()
    
    for _, row in hypotheses_in_group.iterrows():
        hypothesis = row["Hypothesis"]
        lag_minutes = int(row["Lag (minutes)"])
        lead, lag = hypothesis_pairs[hypothesis]
        
        lead_norm = merged_data[lead] / merged_data[lead].iloc[0]
        lag_norm = merged_data[lag] / merged_data[lag].iloc[0]
        
        fig, axs = plt.subplots(2, 1, figsize=(12,6), sharex=True)
        
        axs[0].plot(lead_norm, label=f"{lead} (lead)")
        axs[0].plot(lag_norm, label=f"{lag} (actual lag)")
        if lag_minutes != 0:
            pred_start = lead_norm.index[0]
            pred_end = lead_norm.index[-lag_minutes] if lag_minutes < len(lead_norm) else lead_norm.index[-1]
            axs[0].axvspan(pred_start, pred_end, color='yellow', alpha=0.2, label="Leader predictive region")
        axs[0].set_title(f"{hypothesis}: Actual Prices\n{combined_start} to {combined_end}")
        axs[0].set_ylabel("Normalized Price")
        axs[0].legend()
        axs[0].grid(True)
        
        axs[1].plot(lead_norm, label=f"{lead} (lead)")
        if lag_minutes != 0:
            shifted_lag = lag_norm.shift(-lag_minutes)
            axs[1].plot(shifted_lag, label=f"{lag} (shifted {lag_minutes} min)", alpha=0.5)
        else:
            axs[1].plot(lag_norm, label=f"{lag} (lag)", alpha=0.5)
        axs[1].set_title(f"{hypothesis}: Shifted Lag\n{combined_start} to {combined_end}")
        axs[1].set_xlabel("Time")
        axs[1].set_ylabel("Normalized Price")
        axs[1].legend()
        axs[1].grid(True)
        
        plt.tight_layout()
        
        safe_hypothesis = hypothesis.replace(" ", "_").replace("/", "_")
        filename = f"{folder_name}/{safe_hypothesis}_{combined_start.strftime('%Y%m%d_%H%M')}_to_{combined_end.strftime('%Y%m%d_%H%M')}.png"
        plt.savefig(filename)
        plt.close(fig) 