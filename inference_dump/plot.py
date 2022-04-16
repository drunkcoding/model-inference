from dataclasses import dataclass, field
import os
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import json

from transformers import HfArgumentParser

font = {"size": 64}
matplotlib.rc("font", **font)

basedir = os.path.dirname(__file__)


@dataclass
class Arguments:
    hybrid: str = field(metadata={"help": "Name of the model use as key"},)
    type: str = field(metadata={"help": "Hybrid type"},)


parser = HfArgumentParser(Arguments)
args = parser.parse_args_into_dataclasses()[0]

hybrid = args.hybrid.split(",")

with open(os.path.join(basedir, f"{args.type}.json"), "r") as fp:
    best_data = json.load(fp)

df = pd.read_csv(os.path.join(basedir, f"{args.type}.csv"))

group_latency = df.groupby("accuracy", as_index=False)["cost"]
latency_mean = group_latency.mean()
latency_std = group_latency.std()
latency_min = group_latency.min()
latency_max = group_latency.max()

data = [(row["accuracy"], row["cost"]) for _, row in latency_min.iterrows()]
data = data[::-1]

min_cost = np.inf
pivot = data[0]
reserved_data = [pivot]
for d in data:
    if d[1] <= pivot[1] and d[1] < min_cost:
        reserved_data.append(d)
        pivot = d
        min_cost = d[1]
acc, cost = zip(*reserved_data)
latency_min = pd.DataFrame({"accuracy": acc, "cost": cost})

# idx = group_latency.transform(min)["cost"] == df["cost"]

# print(df[idx][(df.cost <= best_data["cost"]) & (df.accuracy <= best_data["accuracy"])].tail(10))
# print(df[idx].tail(20))
fig = plt.figure(figsize=(20, 10))
ax = fig.add_subplot(111)

# plt.fill_between(
#     latency_mean["accuracy"],
#     latency_min["cost"],
#     latency_max["cost"],
#     color="b",
#     alpha=0.3,
# )
plt.plot(
    latency_min["accuracy"],
    latency_min["cost"],
    linewidth=3,
    linestyle="-",
    color="black",
)

# first_acc = best_data["t5-xl-lm-adapt"]["accuracy"]
# second_acc = best_data["t5-large-lm-adapt"]["accuracy"]

# first_cost = best_data["t5-xl-lm-adapt"]["cost"]
# second_cost = best_data["t5-large-lm-adapt"]["cost"]

first_acc = best_data[hybrid[-1]]["accuracy"]
second_acc = best_data[hybrid[-2]]["accuracy"]

first_cost = best_data[hybrid[-1]]["cost"]
second_cost = best_data[hybrid[-2]]["cost"]

rect = patches.Rectangle(
    (second_acc, 0),
    (first_acc - second_acc),
    (first_cost),
    linewidth=6,
    linestyle=":",
    edgecolor="r",
    facecolor="none",
)
ax.add_patch(rect)

plt.xlabel("Accuracy", fontsize=64)
plt.ylabel("Energy Cost", fontsize=64)
plt.xticks(fontsize=64)
plt.yticks(fontsize=64)
plt.savefig(
    os.path.join(os.path.dirname(__file__), f"{args.type}.png"), bbox_inches="tight"
)

latency_min = latency_min.sort_values(by=["accuracy"], ascending=True)
accuracy = latency_min["accuracy"].to_numpy()
latency = latency_min["cost"].to_numpy()
latency = latency[(accuracy > second_acc) & (accuracy < first_acc)]
accuracy = accuracy[(accuracy > second_acc) & (accuracy < first_acc)]

h = accuracy[1:-1] - accuracy[:-2]
f = latency[2:] - 2 * latency[1:-1] + latency[:-2]

dev = f / (h ** 2)
idx = np.argsort(dev)[::-1][:5]
idx = np.argmin(latency[idx])
print(dev, np.max(dev), accuracy[idx], latency[idx])

df_all = df[df.groupby(["accuracy"])["cost"].transform(min) == df["cost"]]
df_all = df_all.drop_duplicates("accuracy").reset_index()

decision = {
    "preserved": {"accuracy": 0, "cost": None, "threshold": None, "propotion": None,},
    "optimized": {
        "accuracy": None,
        "cost": None,
        "threshold": None,
        "propotion": None,
    },
}

for i, row in df_all.iterrows():
    if row["accuracy"] == accuracy[idx]:
        decision["optimized"] = {
            "accuracy": accuracy[idx],
            "cost": latency[idx],
            "threshold": json.loads(row["thresholds"]),
            "propotion": json.loads(row["propotion"]),
        }

    if (
        np.round(row["accuracy"], 3) < np.round(first_acc, 3)
        and row["accuracy"] > decision["preserved"]["accuracy"]
    ):
        decision["preserved"] = {
            "accuracy": row["accuracy"],
            "cost": row["cost"],
            "threshold": json.loads(row["thresholds"]),
            "propotion": json.loads(row["propotion"]),
        }

with open(os.path.join(basedir, f"{args.type}_decision.json"), "w") as fp:
    json.dump(decision, fp)

print(df_all)

