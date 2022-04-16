from cProfile import label
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

font = {"size": 48}
matplotlib.rc("font", **font)
sns.set(palette="colorblind", style='white')

NAS_TRAIN_CARBON_KG = 284019
BERT_BASE_TRAIN_CARBON_KG = 652.2658 / 1000
BERT_BASE_POWER = 200
DISTILBERT_BASE_POWER = 170
BERT_BASE_TIME = 9 / 1000
DISTILBERT_BASE_TIME = 5.6 / 1000
CO2_PER_KWH = 0.233

NUM_GPUS = 4

steps = np.arange(0, 1e4)+1
x = steps * BERT_BASE_POWER * BERT_BASE_TIME / (60**2) / 1000 * CO2_PER_KWH * NUM_GPUS
x_d = steps * DISTILBERT_BASE_POWER * DISTILBERT_BASE_TIME / (60**2) / 1000 * CO2_PER_KWH
x_t = x * (1 / BERT_BASE_TIME * NUM_GPUS) * (60**2)
x_dt = x_d * (1 / DISTILBERT_BASE_TIME * NUM_GPUS) * (60**2)

df = pd.DataFrame(
    {"steps": steps, "reqc": x, "timec": x_t, "name": ["Inference"]* len(steps)}
)
print(df)

plt.figure(figsize=(25, 15))
# ax = sns.lineplot(x="steps", y="timec", hue="name", data=df, palette=sns.dark_palette("blue", 1), lw=12)
plt.plot(steps, x_t / 1000, label="Inference", color='b', linewidth=12, linestyle='-')
# plt.plot(steps, x_dt / 1000, label="Optimal", color='black', linewidth=12, linestyle='--')
plt.axhline(y=BERT_BASE_TRAIN_CARBON_KG, color='r', linestyle='-.', linewidth=12, label="Train")

plt.xticks(fontsize=48)
plt.yticks(fontsize=48)
plt.xlabel("Time Elapsed (hours)", fontsize=48)
plt.ylabel(r"CO2e (ton)", fontsize=48)
plt.xscale("log")
# plt.yscale("log")
plt.grid()
plt.legend(fontsize=72)

plt.savefig("plots/inference-carbon.png", bbox_inches="tight")
