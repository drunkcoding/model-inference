from joblib import parallel_backend
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from color_scale import color_palette

font = {"size": 96}
matplotlib.rc("font", **font)
sns.set(palette="colorblind")

df_ovhd = pd.DataFrame(
    [
        {"#model": "2", "emission": 1, "label": "Estimate"},
        {"#model": "3", "emission": 1, "label": "Estimate"},
        {"#model": "4", "emission": 1, "label": "Estimate"},
        {"#model": "5", "emission": 1, "label": "Estimate"},
        {"#model": "2", "emission": 1, "label": "Measure"},
        {"#model": "3", "emission": 1, "label": "Measure"},
        {"#model": "4", "emission": 1, "label": "Measure"},
        {"#model": "5", "emission": 1, "label": "Measure"},
    ]
)

plt.figure(figsize=(25, 15))
ax = sns.barplot(x="#model", y="emission", hue="label", data=df_ovhd, palette=sns.dark_palette("blue", 2))
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles=handles, labels=labels)

plt.xticks(fontsize=96)
plt.yticks(fontsize=96)
plt.xlabel("Number of Model", fontsize=96)
plt.ylabel("Emission", fontsize=96)
plt.legend(bbox_to_anchor=(0.065, 1), ncol=2, fontsize=72)

plt.savefig("plots/cost_model_bar.png", bbox_inches="tight")
plt.close()

df_ovhd = pd.DataFrame(
    [
        {"#gpu": "1", "emission": 1, "label": "Estimate"},
        {"#gpu": "2", "emission": 1, "label": "Estimate"},
        {"#gpu": "4", "emission": 1, "label": "Estimate"},
        {"#gpu": "8", "emission": 1, "label": "Estimate"},
        {"#gpu": "1", "emission": 1, "label": "Measure"},
        {"#gpu": "2", "emission": 1, "label": "Measure"},
        {"#gpu": "4", "emission": 1, "label": "Measure"},
        {"#gpu": "8", "emission": 1, "label": "Measure"},
    ]
)

plt.figure(figsize=(25, 15))
ax = sns.barplot(x="#gpu", y="emission", hue="label", data=df_ovhd, palette=sns.dark_palette("blue", 2))
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles=handles, labels=labels)

plt.xticks(fontsize=96)
plt.yticks(fontsize=96)
plt.xlabel("Number of GPU", fontsize=96)
plt.ylabel("Emission", fontsize=96)
plt.legend(bbox_to_anchor=(0.065, 1), ncol=2, fontsize=72)

plt.savefig("plots/cost_gpu_bar.png", bbox_inches="tight")
plt.close()