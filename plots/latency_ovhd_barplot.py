from joblib import parallel_backend
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from color_scale import color_palette

font = {"size": 64}
matplotlib.rc("font", **font)
sns.set(palette="colorblind")

df_ovhd = pd.DataFrame(
    [
        {"#model": "2", "latency": 1},
        {"#model": "3", "latency": 1.2},
        {"#model": "4", "latency": 1.3},
        {"#model": "5", "latency": 1.4},
    ]
)

plt.figure(figsize=(25, 15))
ax = sns.barplot(x="#model", y="latency", data=df_ovhd, palette=sns.dark_palette("blue", 4))
plt.axhline(y=.8, color='r', linestyle='--', linewidth=12)

handles, labels = ax.get_legend_handles_labels()
ax.legend(handles=handles, labels=labels)

plt.xticks(fontsize=64)
plt.yticks(fontsize=64)
plt.xlabel("Number of Model", fontsize=64)
plt.ylabel("Latency", fontsize=64)
plt.legend(fontsize=64)

plt.savefig("plots/latency_ovhd_bar.png", bbox_inches="tight")