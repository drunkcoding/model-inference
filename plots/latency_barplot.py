import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from color_scale import color_palette

font = {"size": 96}
matplotlib.rc("font", **font)
sns.set(palette="colorblind")

df_tp = pd.DataFrame(
    [
        {"#GPU": "1", "type": "T", "tp": 1},
        {"#GPU": "1", "type": "MIE", "tp": 2},
        {"#GPU": "1", "type": "DS", "tp": 1},
        {"#GPU": "2", "type": "T", "tp": 3},
        {"#GPU": "2", "type": "MIE", "tp": 3.5},
        {"#GPU": "2", "type": "DS", "tp": 3},
        {"#GPU": "4", "type": "T", "tp": 4},
        {"#GPU": "4", "type": "MIE", "tp": 4.5},
        {"#GPU": "4", "type": "DS", "tp": 4},
        {"#GPU": "8", "type": "T", "tp": 4},
        {"#GPU": "8", "type": "MIE", "tp": 5.5},
        {"#GPU": "8", "type": "DS", "tp": 4},
    ]
)

plt.figure(figsize=(25, 15))
ax = sns.barplot(x="#GPU", y="tp", hue="type", data=df_tp, palette=sns.dark_palette("blue", 4))

handles, labels = ax.get_legend_handles_labels()
ax.legend(handles=handles, labels=labels)

plt.xticks(fontsize=96)
plt.yticks(fontsize=96)
plt.xlabel("Number of GPUs", fontsize=96)
plt.ylabel("Latency (ms)", fontsize=96)
plt.legend(bbox_to_anchor=(0.9, 1.2), ncol=3, fontsize=72)

plt.savefig("plots/latency_bar.png", bbox_inches="tight")