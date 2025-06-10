import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from colorama import init, Fore, Style
init()

# Load data
data = pd.read_csv('aggTrades_aggregated_1h.csv')
data.index = pd.to_datetime(data['timestamp'])
data = data.drop('timestamp', axis=1)
data = data.loc[data.index < '2024-01-01']  # IN-SAMPLE
#data = data.loc[data.index > '2024-01-01']  # OUT-OF-SAMPLE

mean_diff = data['volume_diff'].mean()
std_diff = data['volume_diff'].std()
volume_threshold = mean_diff + 3 * std_diff

lookback = 24
holding_period = 24
stop_loss_pct = 0.01
take_profit_pct = 0.1
trailing_stop_pct = 0.01  # Example: 2%

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
data['exit_reason'] = pd.Series(dtype='object')
data['entry_time'] = pd.NaT
data['exit_time'] = pd.NaT

print(data.head(1000))
print("len data = ", len(data))

prices = data['price'].values
signals = data['signal'].values

last_exit = -1
trade_counter = 0
trades = []

for i in range(len(data) - holding_period):
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

        res = input("Continue ? ")
        if res == "":
            print("------------------------------------------------")
            continue

    exit_price = prices[exit_idx]
    gross_ret = direction * (exit_price - entry_price) / entry_price
    print("Gross Ret = ", gross_ret)
    ret = gross_ret - 2 * trading_cost
    print("Ret = ", ret)

    # FIXED: Save entry_time at entry index, not at exit index
    # Guardar los tiempos en las filas correctas
    data.at[data.index[entry_idx], 'entry_time'] = data.index[entry_idx]  # ← entrada real
    data.at[data.index[exit_idx], 'entry_time'] = data.index[entry_idx]   # ← duplicada para que aparezca en la fila de salida
    data.at[data.index[exit_idx], 'exit_time'] = data.index[exit_idx]
    data.at[data.index[exit_idx], 'pnl'] = ret
    data.at[data.index[exit_idx], 'exit_reason'] = reason


    print("Reason = ", reason)
    trades.append([data.index[entry_idx], data.index[exit_idx], direction, entry_price, exit_price, ret, reason])
    print(data.iloc[exit_idx])

    trade_counter += 1
    last_exit = exit_idx
    print("Last Exit = ", last_exit)

    res = input("continue ?")
    if res == "":
        continue

print("len data = ", len(data))
trade_log = pd.DataFrame(trades, columns=['entry_time', 'exit_time', 'direction', 'entry_price', 'exit_price', 'return', 'exit_reason'])
trade_log.to_csv("trades.csv")

pd.set_option("display.max_rows", None)
print(data.head(1000))

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

plt.figure(figsize=(15, 7))
cumprod.plot(title='Strategy Cumulative Returns with SL/TP')
plt.xlabel('Time')
plt.ylabel('Cumulative Return')
plt.show()
