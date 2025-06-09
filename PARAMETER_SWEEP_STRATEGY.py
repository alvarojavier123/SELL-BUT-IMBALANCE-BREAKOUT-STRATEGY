import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from colorama import init, Fore, Style
init()

# Load data
data_original = pd.read_csv('aggTrades_aggregated_1h.csv')
data_original.index = pd.to_datetime(data_original['timestamp'])
data_original = data_original.drop('timestamp', axis=1)
data_original = data_original.loc[data_original.index < '2024-01-01']  # IN-SAMPLE
#data_original = data_original.loc[data.index > '2024-01-01']  # OUT-OF-SAMPLE

# Parameter grids
lookback_list = [24, 48, 72, 96, 120, 180, 240, 480, 720]
holding_period_list = [24, 48, 72, 96, 120, 180, 480, 720]

# Constants
stop_loss_pct = 0.01
take_profit_pct = 0.1
trading_cost = 0.001
max_slippage = 0.001  # <- FIXED SLIPPAGE

# Results storage
all_curves = {}
summary = []

for lookback in lookback_list:
    for holding_period in holding_period_list:
        data = data_original.copy()

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

        data['pnl'] = 0.0
        data['exit_reason'] = ''
        data['entry_time'] = pd.NaT
        data['exit_time'] = pd.NaT

        prices = data['price'].values
        signals = data['signal'].values

        last_exit = -1
        trade_counter = 0
        trades_times = []

        for i in range(len(data) - holding_period):
            if i <= last_exit:
                continue
            if signals[i] == 0:
                continue

            direction = signals[i]
            entry_price = prices[i] * (1 + direction * max_slippage)  # <- FIXED slippage here
            max_exit_idx = i + holding_period
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

            data.at[data.index[i], 'entry_time'] = data.index[i]
            data.at[data.index[exit_idx], 'exit_time'] = data.index[exit_idx]
            data.at[data.index[exit_idx], 'pnl'] = ret
            data.at[data.index[exit_idx], 'exit_reason'] = reason

            trade_counter += 1
            last_exit = exit_idx

            # Store trade entry and exit times for holding period calc
            trades_times.append((data.index[i], data.index[exit_idx]))

        trade_pnls = data.loc[data['pnl'] != 0, 'pnl']
        if len(trade_pnls) == 0:
            continue
        cumprod = (1 + trade_pnls).cumprod()
        rolling_max = cumprod.cummax()
        drawdown = cumprod / rolling_max - 1
        max_dd = drawdown.min()

        # Calculate average holding period in hours from trades_times
        if len(trades_times) > 0:
            holding_hours_list = [(exit_time - entry_time).total_seconds() / 3600 for entry_time, exit_time in trades_times]
            avg_holding_hours = np.mean(holding_hours_list)
        else:
            avg_holding_hours = np.nan

        # Annualize Sharpe ratio using average holding period
        # Number of trades per year = 24 * 365 / avg_holding_hours (since 1 unit = 1 hour)
        if avg_holding_hours and avg_holding_hours > 0 and trade_pnls.std() > 0:
            trades_per_year = (24 * 365) / avg_holding_hours
            sharpe = trade_pnls.mean() / trade_pnls.std() * np.sqrt(trades_per_year)
        else:
            sharpe = 0

        label = f"L{lookback}-H{holding_period}"
        all_curves[label] = cumprod
        summary.append({
            "lookback": lookback,
            "holding": holding_period,
            "sharpe": sharpe,
            "cum_return": cumprod.iloc[-1],
            "max_drawdown": max_dd,
            "num_trades": trade_counter,
            "avg_holding_hours": avg_holding_hours
        })

        print(Fore.YELLOW + f"\n[{label}]" + Style.RESET_ALL)
        print(f"Trades: {trade_counter}")
        print(Fore.CYAN + f"Annualized Sharpe Ratio: {sharpe:.2f}")
        print(f"Cumulative Return: {cumprod.iloc[-1]:.2f}")
        print(f"Max Drawdown: {max_dd:.2%}")
        print(f"Avg Holding Period (hours): {avg_holding_hours:.2f}")

# Summary DataFrame
summary_df = pd.DataFrame(summary)
summary_df.to_csv("SUMMARY OF PARAMETER SWEEP(LOOKBACKS-HOLDING PERIODS).csv")
print(Fore.GREEN + "\n===== STRATEGY SUMMARY =====" + Style.RESET_ALL)
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
