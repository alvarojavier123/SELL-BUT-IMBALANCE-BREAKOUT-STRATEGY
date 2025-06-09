import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from colorama import init, Fore, Style
init()

# Load data
data = pd.read_csv('aggTrades_aggregated_1h.csv')
data.index = pd.to_datetime(data['timestamp'])
data = data.drop('timestamp', axis=1)
#data = data.loc[data.index < '2024-01-01']  # IN-SAMPLE
data = data.loc[data.index > '2024-01-01']  # OUT-OF-SAMPLE

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

data['signal'].to_csv("SIGNALS 180 LOOKBACK OUT-SAMPLE.csv")


pd.set_option("display.max_rows", None)

data['pnl'] = 0.0
data['exit_reason'] = np.nan
data['entry_time'] = pd.NaT
data['exit_time'] = pd.NaT

print(data.head(1000))
print("len data = ", len(data))

prices = data['price'].values
print(prices)
print(len(prices))
signals = data['signal'].values
print(signals)
print(len(signals))

last_exit = -1
trade_counter = 0

trades = []

for i in range(len(data) - holding_period):
    print(i)
    if i <= last_exit:
        continue  # skip if still in a trade

    if signals[i] == 0:
        continue  # no signal here
    
    direction = signals[i]
    print("Direction = ", direction)

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

    tp_hits = np.where(returns >= take_profit_pct)[0]
    print("tp_hits = ", tp_hits)
    sl_hits = np.where(returns <= -stop_loss_pct)[0]
    print("sl_hits = ", sl_hits)

    if len(tp_hits) > 0 and (len(sl_hits) == 0 or tp_hits[0] < sl_hits[0]):
        exit_idx = i + 1 + tp_hits[0]
        print("Exit Idx = ", exit_idx)
        reason = 'TP'
        print("Reason = ", reason)
        exit_price = entry_price * (1 + direction * take_profit_pct)  # FIXED: Exact TP exit price
        print("Exit Price = ", exit_price)
    
    elif len(sl_hits) > 0:
        exit_idx = i + 1 + sl_hits[0]
        print("Exit Idx = ", exit_idx)
        reason = 'SL'
        print("Reason = ", reason)
        exit_price = entry_price * (1 - direction * stop_loss_pct)   # FIXED: Exact SL exit price
        print("Exit Price = ", exit_price)

    else:
        exit_idx = max_exit_idx
        print("Exit Idx = ", exit_idx)
        reason = 'Hold'
        print("Reason = ", reason)
        exit_price = prices[exit_idx]  # FIXED: Exit price at hold end
        print("Exit Price = ", exit_price)
    
    # Calculate return with fees on both entry and exit
    gross_ret = direction * (exit_price - entry_price) / entry_price
    print("Gross Ret = ", gross_ret)
    ret = gross_ret - 2 * trading_cost  # FIXED: Fees applied on both entry and exit
    print("Ret = ", ret)

    data.at[data.index[i], 'entry_time'] = data.index[i]
    data.at[data.index[exit_idx], 'exit_time'] = data.index[exit_idx]
    data.at[data.index[exit_idx], 'pnl'] = ret
    data.at[data.index[exit_idx], 'exit_reason'] = reason

    trades.append([data.index[i], data.index[exit_idx], direction, entry_price, exit_price, ret, reason])

    trade_counter += 1
    last_exit = exit_idx
    

print("len data = ", len(data))
trade_log = pd.DataFrame(trades, columns=['entry_time', 'exit_time', 'direction', 'entry_price', 'exit_price', 'return', 'exit_reason'])
trade_log.to_csv("trades.csv")

# Print first 500 rows with new columns
pd.set_option("display.max_rows", None)
print(data.head(1000))

# Calculate metrics from pnl only on exit rows
trade_pnls = data.loc[data['pnl'] != 0, 'pnl']
cumprod = (1 + trade_pnls).cumprod()

avg_holding_hours = (trade_log['exit_time'] - trade_log['entry_time']).dt.total_seconds().mean() / 3600
trades_per_year = (24 * 365) / avg_holding_hours
sharpe = trade_pnls.mean() / trade_pnls.std() * np.sqrt(trades_per_year)

rolling_max = cumprod.cummax()
drawdown = cumprod / rolling_max - 1
max_dd = drawdown.min()

print(Fore.YELLOW + f"Trade count: {trade_counter}" + Style.RESET_ALL)
print(Fore.CYAN + f"\nSharpe Ratio: {sharpe:.2f}")
print(f"Cumulative Return (final): {cumprod.iloc[-1]:.2f}")
print(f"Max Drawdown: {max_dd:.2%}" + Style.RESET_ALL)

# Plot cumulative returns
plt.figure(figsize=(15, 7))
cumprod.plot(title='Strategy Cumulative Returns with SL/TP')
plt.xlabel('Time')
plt.ylabel('Cumulative Return')
plt.show()
