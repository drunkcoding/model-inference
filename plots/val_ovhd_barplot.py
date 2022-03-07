import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from color_scale import color_palette

font = {"size": 96}
matplotlib.rc("font", **font)
sns.set(palette="colorblind")

df_ovhd = pd.DataFrame(
    [{"model": "GPT", "process": "VAL", "runtime": 1},
    {"model": "GPT", "process": "VAL+C", "runtime": 2},
    {"model": "BERT", "process": "VAL", "runtime": 1},
    {"model": "BERT", "process": "VAL+C", "runtime": 2},
    {"model": "ViT", "process": "VAL", "runtime": 1},
    {"model": "ViT", "process": "VAL+C", "runtime": 2},
    {"model": "T5", "process": "VAL", "runtime": 1},
    {"model": "T5", "process": "VAL+C", "runtime": 2}]
)

plt.figure(figsize=(25, 15))
ax = sns.barplot(x="model", y="runtime", hue="process", data=df_ovhd, palette=sns.dark_palette("blue", 2))

handles, labels = ax.get_legend_handles_labels()
ax.legend(handles=handles, labels=labels)

plt.xticks(fontsize=96)
plt.yticks(fontsize=96)
plt.xlabel("Model", fontsize=96)
plt.ylabel("Runtime", fontsize=96)
plt.legend(bbox_to_anchor=(0.2, 1), ncol=2, fontsize=72)

plt.savefig("plots/val_ovhd_bar.png", bbox_inches="tight")