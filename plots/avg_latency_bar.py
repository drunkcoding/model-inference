import itertools
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from color_scale import color_palette
from PIL import ImageColor

font = {"size": 48}
matplotlib.rc("font", **font)
matplotlib.rcParams['hatch.linewidth'] = 4
sns.set(palette="colorblind")
sns.set_style("whitegrid")

colors = ["#afd5ff", "#006dc1", "#003671"]
palette = sns.color_palette(colors)

df = pd.DataFrame(
    [
        {"model": "GPT", "variant": "DeepSpeed", "latency": 6532},
        {"model": "ViT", "variant": "DeepSpeed", "latency": 3420},
        {"model": "T5", "variant": "DeepSpeed", "latency": 2133},
    ] + [
        {"model": "GPT", "variant": "HybridServe (AP)", "latency": 4233},
        {"model": "GPT", "variant": "HybridServe (EO)", "latency": 1939},
        {"model": "ViT", "variant": "HybridServe (AP)", "latency": 662},
        {"model": "ViT", "variant": "HybridServe (EO)", "latency": 420},
        {"model": "T5", "variant": "HybridServe (AP)", "latency": 517},
        {"model": "T5", "variant": "HybridServe (EO)", "latency": 271},
    ]
)
df['latency'] = df.latency / 1000

num_locations = len(df.model.unique())
hatches = itertools.cycle(['/', '+', '-', 'x', '\\', '*', '.', '//'])

plt.figure(figsize=(10, 8), dpi=300)
ax = sns.barplot(x="model", y="latency", hue="variant", data=df, palette=palette)
ax.legend_.remove()

for i,thisbar in enumerate(ax.patches):
    # thisbar.set_hatch(hatches[i])
    thisbar.set_edgecolor('k')

# for i, bar in enumerate(ax.patches):
#     if i % num_locations == 0:
#         hatch = next(hatches)
#     # hatch = hatches[i % num_locations]
#     # print(i, hatch)
#     bar.set_hatch(hatch)
#     bar.set_edgecolor([0, .5, .5])

# handles, labels = ax.get_legend_handles_labels()
# ax.legend(handles=handles, labels=labels)

# axbox = ax.get_position()

# plt.yscale("log")
plt.xticks(fontsize=48)
plt.yticks(fontsize=48)
# plt.ylim(100, 2e5)
plt.xlabel("", fontsize=48)
plt.ylabel("Latency (s)", fontsize=48)
# legend = plt.legend(bbox_to_anchor=[0, axbox.y0+0.3,1,1], loc='upper right', ncol=1, fontsize=48)

plt.savefig("plots/avg_latency_bar.png", bbox_inches="tight")