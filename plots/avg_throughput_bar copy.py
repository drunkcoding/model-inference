import itertools
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from color_scale import color_palette

font = {"size": 64}
matplotlib.rc("font", **font)
matplotlib.rcParams['hatch.linewidth'] = 4
sns.set(palette="colorblind")
sns.set_style("whitegrid")

colors = ["#afd5ff", "#8bb8ff", "#3e8fda"]
palette = sns.color_palette(colors)

df = pd.DataFrame(
    [
        {"model": "GPT", "variant": "DeepSpeed", "throughput": 3},
        {"model": "ViT", "variant": "DeepSpeed", "throughput": 5},
        {"model": "T5", "variant": "DeepSpeed", "throughput": 8},
    ] + [
        {"model": "GPT", "variant": "HybridServe (AP)", "throughput": 19},
        {"model": "GPT", "variant": "HybridServe (EO)", "throughput": 40},
        {"model": "ViT", "variant": "HybridServe (AP)", "throughput": 59},
        {"model": "ViT", "variant": "HybridServe (EO)", "throughput": 64},
        {"model": "T5", "variant": "HybridServe (AP)", "throughput": 29},
        {"model": "T5", "variant": "HybridServe (EO)", "throughput": 57}, 
    ]
)

num_locations = len(df.model.unique())
hatches = itertools.cycle(['/', '//', '+', '-', 'x', '\\', '*', '.'])

plt.figure(figsize=(30, 10), dpi=300)
ax = sns.barplot(x="model", y="throughput", hue="variant", data=df, palette=palette)

# for i, bar in enumerate(ax.patches):
#     if i % num_locations == 0:
#         hatch = next(hatches)
#     # hatch = hatches[i % num_locations]
#     # print(i, hatch)
#     bar.set_hatch(hatch)
#     bar.set_edgecolor([1, 1, 1])

handles, labels = ax.get_legend_handles_labels()
ax.legend(handles=handles, labels=labels)

axbox = ax.get_position()

# plt.yscale("log")
plt.xticks(fontsize=64)
plt.yticks(fontsize=64)
# plt.ylim(100, 2e5)
plt.xlabel("", fontsize=64)
plt.ylabel("Throughput (R/s)", fontsize=64)
plt.legend(bbox_to_anchor=[0, axbox.y0+0.2,1,1], loc='upper center', ncol=4, fontsize=48)

plt.savefig("plots/avg_throughput_bar.png", bbox_inches="tight")