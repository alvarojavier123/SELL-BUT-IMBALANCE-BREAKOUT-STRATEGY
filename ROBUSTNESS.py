
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

"""
# Filtrar por si hay valores faltantes
df = df.dropna(subset=["lookback", "holding", "sharpe"])

# Estilo bonito
sns.set(style="whitegrid")

# Crear figura 3D
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Colores por Sharpe
sc = ax.scatter(
    df["lookback"],
    df["holding"],
    df["sharpe"],
    c=df["sharpe"],
    cmap="viridis",
    s=40,
    alpha=0.8
)

# Ejes y título
ax.set_xlabel("Lookback", fontsize=12)
ax.set_ylabel("Holding Period", fontsize=12)
ax.set_zlabel("Sharpe Ratio", fontsize=12)
ax.set_title("3D Scatter: Lookback vs Holding vs Sharpe", fontsize=14)

# Barra de color
cbar = fig.colorbar(sc, ax=ax, shrink=0.5, aspect=10)
cbar.set_label("Sharpe Ratio", fontsize=12)

plt.tight_layout()
plt.show()
"""