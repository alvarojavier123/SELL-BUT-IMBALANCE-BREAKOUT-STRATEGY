
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
pd.set_option("display.max_rows", None)

data = pd.read_csv("SUMMARY_SWEEP_LOOKBACK_HOLD_SL_TP.csv")
data = data.drop('Unnamed: 0',axis=1)
print(data)

