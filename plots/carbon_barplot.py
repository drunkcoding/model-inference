from joblib import parallel_backend
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from color_scale import color_palette

font = {"size": 96}
matplotlib.rc("font", **font)
sns.set(palette="colorblind")

df_carbon = pd.DataFrame(
    [
        {"#GPU": "2", "carbon": 1},
        {"#GPU": "3", "carbon": 3},
        {"#GPU": "4", "carbon": 4},
        {"#GPU": "5", "carbon": 4.1},
    ]
)

plt.figure(figsize=(25, 15))
ax = sns.barplot(x="#GPU", y="carbon", data=df_carbon, palette=sns.dark_palette("blue", 4))
plt.axhline(y=10, color='r', linestyle='--', linewidth=12)
plt.axhline(y=8, color='b', linestyle='-.', linewidth=12)

handles, labels = ax.get_legend_handles_labels()
ax.legend(handles=handles, labels=labels)

plt.xticks(fontsize=96)
plt.yticks(fontsize=96)
plt.xlabel("Number of Models", fontsize=96)
plt.ylabel("kg CO2e / sample", fontsize=96)
plt.legend(bbox_to_anchor=(0.9, 1.2), ncol=3, fontsize=72)

plt.savefig("plots/carbon_bar.png", bbox_inches="tight")