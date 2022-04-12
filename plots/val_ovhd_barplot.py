import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from color_scale import color_palette

font = {"size": 64}
matplotlib.rc("font", **font)
matplotlib.rcParams['hatch.linewidth'] = 5
sns.set(palette="colorblind")
sns.set_style("whitegrid")

df_ovhd = pd.DataFrame(
    [{"model": "GPT", "process": "VAL", "runtime": 162},
    {"model": "GPT", "process": "VAL+C", "runtime": 273},
    {"model": "GPT", "process": "TUNE", "runtime": 8218800},
    {"model": "BERT", "process": "VAL", "runtime": 307},
    {"model": "BERT", "process": "VAL+C", "runtime": 313},
    {"model": "BERT", "process": "TUNE", "runtime": 14400},
    {"model": "ViT", "process": "VAL", "runtime": 420},
    {"model": "ViT", "process": "VAL+C", "runtime": 1800},
    {"model": "ViT", "process": "TUNE", "runtime": 82400},
    {"model": "T5", "process": "VAL", "runtime": 180},
    {"model": "T5", "process": "VAL+C", "runtime": 360},
    {"model": "T5", "process": "TUNE", "runtime": 164400}]
)

hatches = ["//"] * 4 +  ["\\"] * 4 +  ["+"] * 4
# [, '\\', '-', '+', 'x', '*', 'o', 'o']

plt.figure(figsize=(20, 10))
ax = sns.barplot(x="model", y="runtime", hue="process", data=df_ovhd, palette=sns.dark_palette("blue", 3))

for i,thisbar in enumerate(ax.patches):
    thisbar.set_hatch(hatches[i])

handles, labels = ax.get_legend_handles_labels()
ax.legend(handles=handles, labels=labels)

plt.xticks(fontsize=64)
plt.yticks(fontsize=64)
# plt.ylim(100, 2e5)
plt.yscale("log")
plt.xlabel("", fontsize=64)
plt.ylabel("Runtime (seconds)", fontsize=64)
plt.legend(bbox_to_anchor=(1, 1.2), ncol=3, fontsize=48)

plt.savefig("plots/val_ovhd_bar.png", bbox_inches="tight")