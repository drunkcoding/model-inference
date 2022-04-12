from tokenize import group
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import json

type = "vit"

font = {"size": 72}
matplotlib.rc("font", **font)

with open(f"plots/thresholds/{type}.json", "r") as fp:
    best_data = json.load(fp)


df = pd.read_csv(f"plots/thresholds/{type}.csv")


group_latency = df.groupby("accuracy", as_index=False)["latency"]
latency_mean = group_latency.mean()
latency_std = group_latency.std()
latency_min = group_latency.min()
latency_max = group_latency.max()

idx = group_latency.transform(min)["latency"] == df["latency"]

print(df[idx][(df.latency <= best_data["latency"]) & (df.accuracy <= best_data["accuracy"])].tail(10))

plt.figure(figsize=(25, 15))

lower_bounds = latency_mean["latency"] - latency_std["latency"] * 3
upper_bounds = latency_mean["latency"] + latency_std["latency"] * 3
plt.fill_between(
    latency_mean["accuracy"],
    latency_min["latency"],
    latency_max["latency"],
    color="b",
    alpha=0.3,
)
plt.plot(
    latency_mean["accuracy"],
    latency_min["latency"],
    linewidth=6,
    linestyle=":",
    color="black",
)
plt.plot(
    latency_mean["accuracy"],
    latency_max["latency"],
    linewidth=6,
    linestyle="--",
    color="black",
)
plt.hlines(
    y=best_data["latency"],
    xmin=df.accuracy.min(),
    xmax=df.accuracy.max(),
    color="navy",
    linewidth=6,
)
plt.vlines(
    x=best_data["accuracy"],
    ymin=df.latency.min(),
    ymax=df.latency.max(),
    color="navy",
    linewidth=6,
)
plt.xlabel("Accuracy", fontsize=72)
plt.ylabel("Average Latency (ms)", fontsize=72)
plt.xticks(fontsize=72)
plt.yticks(fontsize=72)
# sns.lineplot(x="accuracy", y="latency", data=latency_mean)
plt.savefig(f"plots/thresholds/{type}.png", bbox_inches="tight")
