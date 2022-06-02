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

df = pd.DataFrame(
    [
        {"model": "GPT", "variant": "XS", "accuracy": 0.2358},
        {"model": "GPT", "variant": "S", "accuracy": 0.3113},
        {"model": "GPT", "variant": "M", "accuracy": 0.3409},
        {"model": "GPT", "variant": "L", "accuracy": 0.3542},
        {"model": "GPT", "variant": "XL", "accuracy": 0.3691},
        {"model": "GPT", "variant": "XXL", "accuracy": 0.4231},
        {"model": "ViT", "variant": "XS", "accuracy": 0.74755},
        {"model": "ViT", "variant": "S", "accuracy": 0.8084},
        {"model": "ViT", "variant": "M", "accuracy": 0.81205},
        {"model": "ViT", "variant": "L", "accuracy": 0.82535},
        {"model": "T5", "variant": "S", "accuracy": 0.7819922824067458},
        {"model": "T5", "variant": "M", "accuracy": 0.8421037587537517},
        {"model": "T5", "variant": "L", "accuracy": 0.8709732742603973},
        {"model": "T5", "variant": "XL", "accuracy": 0.9062455338002001},
    ] + [
        {"model": "GPT", "variant": "HybridServe (AP)", "accuracy": 0.4231},
        {"model": "GPT", "variant": "HybridServe (EO)", "accuracy": 0.372},
        {"model": "ViT", "variant": "HybridServe (AP)", "accuracy": 0.82535},
        {"model": "ViT", "variant": "HybridServe (EO)", "accuracy": 0.817},
        {"model": "T5", "variant": "HybridServe (AP)", "accuracy": 0.905},
        {"model": "T5", "variant": "HybridServe (EO)", "accuracy": 0.898},
    ]
)

num_locations = len(df.model.unique())
hatches = itertools.cycle(['/', '//', '+', '-', 'x', '\\', '*', '.'])
# hatches = ['/', '//', '+', '-', 'x', '\\', '*', 'o', 'O', '.']
# print(num_locations)

plt.figure(figsize=(30, 10))
ax = sns.barplot(x="model", y="accuracy", hue="variant", data=df, palette=sns.dark_palette("blue", 6))

for i, bar in enumerate(ax.patches):
    if i % num_locations == 0:
        hatch = next(hatches)
    # hatch = hatches[i % num_locations]
    # print(i, hatch)
    bar.set_hatch(hatch)
    bar.set_edgecolor([1, 1, 1])

handles, labels = ax.get_legend_handles_labels()
ax.legend(handles=handles, labels=labels)

axbox = ax.get_position()

plt.xticks(fontsize=64)
plt.yticks(fontsize=64)
# plt.ylim(100, 2e5)
plt.xlabel("", fontsize=64)
plt.ylabel("Accuracy", fontsize=64)
plt.legend(bbox_to_anchor=[0, axbox.y0+0.2,1,1], loc='upper center', ncol=4, fontsize=48)

plt.savefig("plots/accuracy_bar.png", bbox_inches="tight")