
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("SUMMARY OF PARAMETER SWEEP(LOOKBACKS-HOLDING PERIODS) MINUTES.csv")
print(df)

pivot = df.pivot(index="lookback", columns="holding", values="sharpe")
sns.heatmap(pivot, cmap="viridis", annot=True, fmt=".1f")
plt.title("Sharpe Ratio Heatmap")
plt.show()
