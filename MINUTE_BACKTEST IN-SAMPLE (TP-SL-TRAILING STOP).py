import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from colorama import init, Fore, Style
from tqdm import tqdm
import os

init()
os.makedirs("minute_backtest_outputs", exist_ok=True)

# Load data
data = pd.read_csv('aggTrades_aggregated_1h.csv')
data.index = pd.to_datetime(data['timestamp'])
data = data.drop('timestamp', axis=1)
data = data.loc[data.index < '2024-01-01']  # IN-SAMPLE


mean_diff = data['volume_diff'].mean()
std_diff = data['volume_diff'].std()
volume_threshold = mean_diff + 3 * std_diff

lookback = 180
holding_period = 480
stop_loss_pct = 0.01
take_profit_pct = 0.1
trading_cost = 0.001

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


df = data
print(df.head(1000))

# Load minute data CSV and set datetime index
min_df = pd.read_csv("BTC_USDT_1min_since_2020.csv")
min_df.index = pd.to_datetime(min_df['timestamp'])
min_df = min_df.drop('timestamp', axis=1)

# Shift signals by 59 minutes to align with close of hour
shifted_signals = df.copy()
shifted_signals.index = shifted_signals.index + pd.Timedelta(minutes=59)
min_df['signal'] = shifted_signals['signal'].reindex(min_df.index).fillna(0)
min_df = min_df.loc[min_df.index < '2024-01-01']  # IN-SAMPLE


# Prepare columns for PnL and trade info
min_df['pnl'] = 0.0
min_df['exit_reason'] = np.nan
min_df['entry_time'] = pd.NaT
min_df['exit_time'] = pd.NaT

holding_period = holding_period * 60  # 120 HOURS * 60 MINUTES = 7200
stop_loss_pct = 0.01
take_profit_pct = 0.2
trailing_stop_pct = 0.05  

trading_cost = 0.001

prices = min_df['close'].values
signals = min_df['signal'].values

last_exit = -1
trade_counter = 0
trades = []

print("\nStarting backtest with progress bar...\n")

for i in tqdm(range(len(min_df) - holding_period), desc="Backtesting Progress"):
    print(i)
    if i <= last_exit:
        continue

    if signals[i] == 0:
        continue
    
    direction = signals[i]
    print("Direction = ", direction)
    entry_idx = i

    entry_price = prices[i]
    print("Entry Price = ", entry_price)
    max_slippage = 0.001
    entry_price *= (1 + direction * max_slippage)

    print("Entry Price After Slippage = ", entry_price)

    max_exit_idx = i + holding_period
    print("Max Exit Idx = ", max_exit_idx)
    trade_slice = prices[i+1:max_exit_idx+1]
    print("Trade Slice = ", trade_slice)
    returns = (trade_slice - entry_price) / entry_price * direction
    print("Returns = ", returns)

    best_price = entry_price
    print("Best Price = ", best_price)
    exit_idx = max_exit_idx
    print("Exit Idx = ", exit_idx)
    reason = 'Hold'
    print("Reason = ", reason)

    for j in range(i + 1, max_exit_idx + 1):
        print("Trade Slice = ", trade_slice)
        print("Returns = ", returns)
        price = prices[j]
        print("Price = ", price)
        move = (price - entry_price) / entry_price * direction
        print("Move = ", move)

        if move >= take_profit_pct:
            exit_idx = j
            print("Exit Idx = ", exit_idx)
            reason = 'TP'
            print("Reason = ", reason)
            break

        if direction == 1:
            if price > best_price:
                best_price = price
                print("Best Price = ", best_price)
        else:
            if price < best_price:
                best_price = price
                print("Best Price = ", best_price)

        trailing_level = best_price * (1 - trailing_stop_pct) if direction == 1 else best_price * (1 + trailing_stop_pct)
        print("TRAILING LEVEL = ", trailing_level)

        if (direction == 1 and price < trailing_level) or (direction == -1 and price > trailing_level):
            exit_idx = j
            print("Exit Idx = ", exit_idx)
            reason = 'Trailing Stop'
            print("Reason = ", reason)
            break

        if move <= -stop_loss_pct:
            exit_idx = j
            print("Exit Idx = ", exit_idx)
            reason = 'SL'
            print("Reason = ", reason)
            break
        
        """
        res = input("Continue ? ")
        if res == "":
            print("------------------------------------------------")
            continue
        """


    exit_price = prices[exit_idx]
    gross_ret = direction * (exit_price - entry_price) / entry_price
    ret = gross_ret - 2 * trading_cost  # fees applied on entry and exit

    min_df.at[min_df.index[i], 'entry_time'] = min_df.index[i]
    min_df.at[min_df.index[exit_idx], 'exit_time'] = min_df.index[exit_idx]
    min_df.at[min_df.index[exit_idx], 'pnl'] = ret
    min_df.at[min_df.index[exit_idx], 'exit_reason'] = reason

    trades.append([min_df.index[i], min_df.index[exit_idx], direction, entry_price, exit_price, ret, reason])

    trade_counter += 1
    last_exit = exit_idx

# Save trade log CSV
trade_log = pd.DataFrame(trades, columns=['entry_time', 'exit_time', 'direction', 'entry_price', 'exit_price', 'return', 'exit_reason'])
trade_log.to_csv("trades_minutes.csv")

pd.set_option("display.max_rows", None)
print(min_df.loc['2020-07-08 00:00:00':'2020-07-28 00:00:00'])


# Calculate metrics from pnl only on exit rows
trade_pnls = min_df.loc[min_df['pnl'] != 0, 'pnl']
cumprod = (1 + trade_pnls).cumprod()

# Actual duration of the backtest
start_time = trade_log['entry_time'].iloc[0]
end_time = trade_log['exit_time'].iloc[-1]
total_days = (end_time - start_time).total_seconds() / (3600 * 24)
annual_factor = np.sqrt(252 / (total_days / len(trade_pnls)))

sharpe = trade_pnls.mean() / trade_pnls.std() * annual_factor

rolling_max = cumprod.cummax()
drawdown = cumprod / rolling_max - 1
max_dd = drawdown.min()

print(Fore.YELLOW + f"\nTrade count: {trade_counter}" + Style.RESET_ALL)
print(Fore.CYAN + f"Sharpe Ratio: {sharpe:.2f}")
print(f"Cumulative Return (final): {cumprod.iloc[-1]:.2f}")
print(f"Max Drawdown: {max_dd:.2%}" + Style.RESET_ALL)

# Plot cumulative returns
plt.figure(figsize=(15, 7))
cumprod.plot(title='Strategy Cumulative Returns with SL/TP')
plt.xlabel('Time')
plt.ylabel('Cumulative Return')
plt.show()
