from audioop import reverse
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

colors = ["#afd5ff", "#006dc1", "#003671"]
palette = sns.color_palette(colors)

df = pd.DataFrame(
    [
        {"model": "GPT", "variant": "DeepSpeed", "energy": 462},
        {"model": "ViT", "variant": "DeepSpeed", "energy": 249},
        {"model": "T5", "variant": "DeepSpeed", "energy": 167},
    ] + [
        {"model": "GPT", "variant": "HybridServe (AP)", "energy": 259},
        {"model": "GPT", "variant": "HybridServe (EO)", "energy": 129},
        {"model": "ViT", "variant": "HybridServe (AP)", "energy": 34},
        {"model": "ViT", "variant": "HybridServe (EO)", "energy": 24},
        {"model": "T5", "variant": "HybridServe (AP)", "energy": 36},
        {"model": "T5", "variant": "HybridServe (EO)", "energy": 24},
    ]
)

num_locations = len(df.model.unique())
hatches = itertools.cycle(['/', '//', '+', '-', 'x', '\\', '*', '.'])

plt.figure(figsize=(30, 10), dpi=300)
ax = sns.barplot(x="model", y="energy", hue="variant", data=df, palette=palette, linewidth=3)

for i, bar in enumerate(ax.patches):
    # if i % num_locations == 0:
    #     hatch = next(hatches)
    # hatch = hatches[i % num_locations]
    # print(i, hatch)
    # bar.set_hatch(hatch)
    # bar.set_edgecolor([0, .5, .5])
    bar.set_edgecolor('k')

handles, labels = ax.get_legend_handles_labels()
ax.legend(handles=handles, labels=labels)

axbox = ax.get_position()

# plt.yscale("log")
plt.xticks(fontsize=64)
plt.yticks(fontsize=64)
# plt.ylim(100, 2e5)
plt.xlabel("", fontsize=64)
plt.ylabel("energy (J/R)", fontsize=64)
plt.legend(bbox_to_anchor=[0, axbox.y0+0.2,1,1], loc='upper center', ncol=4, fontsize=48)

plt.savefig("plots/avg_energy_bar.png", bbox_inches="tight")