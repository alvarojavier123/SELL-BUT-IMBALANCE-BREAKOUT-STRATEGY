import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from colorama import init, Fore, Style
from tqdm import tqdm
import os

init()
os.makedirs("minute_backtest_outputs", exist_ok=True)

# Load data
data_original = pd.read_csv('aggTrades_aggregated_1h.csv')
data_original.index = pd.to_datetime(data_original['timestamp'])
data_original = data_original.drop('timestamp', axis=1)
data_original = data_original.loc[data_original.index < '2024-01-01']  # IN-SAMPLE

# Minute data
min_df_raw = pd.read_csv("BTC_USDT_1min_since_2020.csv")
min_df_raw.index = pd.to_datetime(min_df_raw['timestamp'])
min_df_raw = min_df_raw.drop('timestamp', axis=1)
min_df_raw = min_df_raw.loc[min_df_raw.index < '2024-01-01']  # IN-SAMPLE

# Parameters
lookback_list = [24, 48, 72, 96, 120, 180, 240, 480, 720]
holding_period_list = [24, 48, 72, 96, 120, 180, 480, 720]

stop_loss_pct = 0.01
take_profit_pct = 0.1
trading_cost = 0.001
max_slippage = 0.001

summary = []
# Results storage
all_curves = {}


for lookback in lookback_list:
    for holding_period in holding_period_list:
        print(Fore.YELLOW + f"\n==== L{lookback}-H{holding_period} ====" + Style.RESET_ALL)
        data = data_original.copy()

        # Signal generation
        mean_diff = data['volume_diff'].mean()
        std_diff = data['volume_diff'].std()
        volume_threshold = mean_diff + 3 * std_diff

        data['signal'] = 0
        data.loc[
            (data['volume_diff'] > volume_threshold) &
            (data['aggressor'] == 'Buy') &
            (data['price'] > data['price'].rolling(lookback).max().shift(1)),
            'signal'
        ] = 1

        data.loc[
            (data['volume_diff'] < -volume_threshold) &
            (data['aggressor'] == 'Sell') &
            (data['price'] < data['price'].rolling(lookback).min().shift(1)),
            'signal'
        ] = -1

        # Shift signals by 59 minutes to match close of hour
        shifted_signals = data[['signal']].copy()
        shifted_signals.index = shifted_signals.index + pd.Timedelta(minutes=59)

        # Backtest on minute data
        min_df = min_df_raw.copy()
        min_df['signal'] = shifted_signals['signal'].reindex(min_df.index)

        min_df['pnl'] = 0.0
        min_df['exit_reason'] = np.nan
        min_df['entry_time'] = pd.NaT
        min_df['exit_time'] = pd.NaT

        prices = min_df['close'].values
        signals = min_df['signal'].values

        holding_period_minutes = holding_period * 60  # convert hours to minutes
        last_exit = -1
        trades = []

        for i in tqdm(range(len(min_df) - holding_period_minutes), desc=f"L{lookback}-H{holding_period}"):
            if i <= last_exit or signals[i] == 0 or np.isnan(signals[i]):
                continue

            direction = signals[i]
            entry_price = prices[i] * (1 + direction * max_slippage)
            max_exit_idx = i + holding_period_minutes
            trade_slice = prices[i+1:max_exit_idx+1]
            returns = (trade_slice - entry_price) / entry_price * direction

            tp_hits = np.where(returns >= take_profit_pct)[0]
            sl_hits = np.where(returns <= -stop_loss_pct)[0]

            if len(tp_hits) > 0 and (len(sl_hits) == 0 or tp_hits[0] < sl_hits[0]):
                exit_idx = i + 1 + tp_hits[0]
                reason = 'TP'
                exit_price = entry_price * (1 + direction * take_profit_pct)
            elif len(sl_hits) > 0:
                exit_idx = i + 1 + sl_hits[0]
                reason = 'SL'
                exit_price = entry_price * (1 - direction * stop_loss_pct)
            else:
                exit_idx = max_exit_idx
                reason = 'Hold'
                exit_price = prices[exit_idx]

            gross_ret = direction * (exit_price - entry_price) / entry_price
            ret = gross_ret - 2 * trading_cost

            min_df.at[min_df.index[i], 'entry_time'] = min_df.index[i]
            min_df.at[min_df.index[exit_idx], 'exit_time'] = min_df.index[exit_idx]
            min_df.at[min_df.index[exit_idx], 'pnl'] = ret
            min_df.at[min_df.index[exit_idx], 'exit_reason'] = reason

            trades.append((min_df.index[i], min_df.index[exit_idx], direction, entry_price, exit_price, ret, reason))
            last_exit = exit_idx

        trade_log = pd.DataFrame(trades, columns=['entry_time', 'exit_time', 'direction', 'entry_price', 'exit_price', 'return', 'exit_reason'])

        # Metrics
        trade_pnls = min_df.loc[min_df['pnl'] != 0, 'pnl']
        if trade_pnls.empty or trade_pnls.std() == 0:
            sharpe = 0
            cum_return = 1.0
            max_dd = 0.0
            avg_holding_hours = 0.0
        else:
            cumprod = (1 + trade_pnls).cumprod()
            rolling_max = cumprod.cummax()
            drawdown = cumprod / rolling_max - 1
            max_dd = drawdown.min()

            avg_holding_hours = (trade_log['exit_time'] - trade_log['entry_time']).dt.total_seconds().mean() / 3600
            start_time = trade_log['entry_time'].iloc[0]
            end_time = trade_log['exit_time'].iloc[-1]
            total_days = (end_time - start_time).total_seconds() / (3600 * 24)
            annual_factor = np.sqrt(252 / (total_days / len(trade_pnls)))

            sharpe = trade_pnls.mean() / trade_pnls.std() * annual_factor

            cum_return = cumprod.iloc[-1]

        label = f"L{lookback}-H{holding_period}"
        all_curves[label] = cumprod
        summary.append({
            "lookback": lookback,
            "holding": holding_period,
            "sharpe": sharpe,
            "cum_return": cum_return,
            "max_drawdown": max_dd,
            "num_trades": len(trades),
            "avg_holding_hours": avg_holding_hours
        })

        print(Fore.CYAN + f"Sharpe: {sharpe:.2f} | Return: {cum_return:.2f} | Max DD: {max_dd:.2%} | Trades: {len(trades)}" + Style.RESET_ALL)

        # Optional: save trade log for each run
        trade_log.to_csv(f"minute_backtest_outputs/trades_L{lookback}_H{holding_period}.csv")

# Save summary
summary_df = pd.DataFrame(summary)
summary_df.to_csv("SUMMARY OF PARAMETER SWEEP(LOOKBACKS-HOLDING PERIODS) MINUTES.csv")
print(Fore.GREEN + "\n===== FINAL SUMMARY (MINUTE DATA) =====" + Style.RESET_ALL)
print(summary_df.sort_values(by="sharpe", ascending=False))

# Normal plot
plt.figure(figsize=(16, 8))
for label, curve in all_curves.items():
    plt.plot(curve.values, label=label)
plt.title("Cumulative PnL Curves by Strategy")
plt.xlabel("Trade Index")
plt.ylabel("Cumulative Return")
plt.legend()
plt.grid(True)
plt.show()

