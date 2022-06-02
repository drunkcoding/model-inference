import numpy as np
import itertools
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from color_scale import color_palette

font = {"size": 48}
matplotlib.rc("font", **font)
matplotlib.rcParams['hatch.linewidth'] = 4
sns.set(palette="colorblind")
sns.set_style("whitegrid")

colors = ["#afd5ff", "#006dc1", "#003671"]
palette = sns.color_palette(colors)

df = pd.DataFrame(
    [
        {"model": "GPT", "variant": "DeepSpeed", "latency": 11248},
        {"model": "ViT", "variant": "DeepSpeed", "latency": 4482},
        {"model": "T5", "variant": "DeepSpeed", "latency": 2304},
    ] + [
        {"model": "GPT", "variant": "HybridServe (AP)", "latency": 15057},
        {"model": "GPT", "variant": "HybridServe (EO)", "latency": 14897},
        {"model": "ViT", "variant": "HybridServe (AP)", "latency": 4651},
        {"model": "ViT", "variant": "HybridServe (EO)", "latency": 4534},
        {"model": "T5", "variant": "HybridServe (AP)", "latency": 2874},
        {"model": "T5", "variant": "HybridServe (EO)", "latency": 2617},
    ]
)
df['latency'] = df.latency / 1000

num_locations = len(df.model.unique())
hatches = itertools.cycle(['/', '//', '+', '-', 'x', '\\', '*', '.'])

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
#     bar.set_edgecolor([1, 1, 1])

# handles, labels = ax.get_legend_handles_labels()
# ax.legend(handles=handles, labels=labels)

axbox = ax.get_position()

# plt.yscale("log")
plt.xticks(fontsize=48)
plt.yticks(fontsize=48)
# plt.ylim(100, 2e5)
plt.xlabel("", fontsize=48)
plt.ylabel("Latency (s)", fontsize=48)
legend = plt.legend(bbox_to_anchor=[0, axbox.y0+0.3,1,1], loc='upper right', ncol=3, fontsize=48)

def export_legend(legend, filename="legend.png", expand=[-5,-5,5,5]):
    fig  = legend.figure
    fig.canvas.draw()
    bbox  = legend.get_window_extent()
    bbox = bbox.from_extents(*(bbox.extents + np.array(expand)))
    bbox = bbox.transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(filename, dpi="figure", bbox_inches=bbox)

export_legend(legend)

plt.savefig("plots/tail_latency_bar.png", bbox_inches="tight")
# plt.savefig("plots/bar_legend.png", bbox_inches="tight")