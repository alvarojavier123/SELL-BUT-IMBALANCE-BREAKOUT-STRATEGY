This repo explores a trading strategy hypothesis which consists in buying when there are aggressive buyers (as takers) hitting the ask prices and when the price its at a maximum at the same 
time and sell when there are aggressive sellers (as takers) hitting the bid prices and the price its at lowest point at the same time.
The strategy logic is coded in the file STRATEGY.py, in this file I backtested the signals 1 (BUY) -1 (SELL) with hourly data (2020-07-01 until 2025-06-01). Then, I went deep and decided to backtest these signals with 1 minute data (THIS FILE IS NOT IN THE REPO BECAUSE ITS HUGE). First, I backtested this strategy with random parameters (LOOKBACK PERIOD : 180 HOURS, HOLDING PERIOD: 480 HOURS, TAKE PROFIT : 10% AND STOP LOSS : 1%). After this I went ahead and did a parameter sweep in the file MINUTE_BACKTEST_PARAMETER_SWEEP_STRATEGY TP SL IN-SAMPLE .py to see how the same strategy would have worked with different parameters...

These are the returns from the file MINUTE_BACKTEST IN-SAMPLE.py with the parameters : 180 LOOKBACK, 480 HOLDING PERIOD, 10% TAKE PROFIT, 1% STOP LOSS (IN - SAMPLE PERFORMANCE 2020-2024) 86 % COMPOUNDED RETURN, -21 DRAWDOWN, 0.65 SHARPE RATIO:
![returns 180 lkbk, 480 hrs, 1% sl, 10% tp](https://github.com/user-attachments/assets/0763fc52-d4f8-4e21-b581-03f8cd9451f5)

These are the returns from the file MINUTE_BACKTEST OUT -SAMPLE.py with the parameters : 180 LOOKBACK, 480 HOLDING PERIOD, 10% TAKE PROFIT, 1% STOP LOSS (OUT - SAMPLE PERFORMANCE 2024-2025) -1% % COMPOUNDED RETURN, -28 DRAWDOWN, 0.07 SHARPE RATIO:
![out sample](https://github.com/user-attachments/assets/4053bfbf-cd78-4727-b47d-6b6f01aef1da)


