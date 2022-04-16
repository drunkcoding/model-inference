from cProfile import label
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

font = {"size": 48}
matplotlib.rc("font", **font)
# plt.rcParams['text.usetex'] = True
# sns.set(palette="colorblind", style='white')

GPT3_TRAIN_CARBON = 552 * 1000
CO2_PER_KWH = 0.475
HOURS_PER_DAY = 24
SECONDS_PER_HOUR = 3600
REQUEST_PER_SECOND = 2.8
MODEL_NUM_GPUS = 25
MODEL_RELEASE_DAYS = 461


INFERENCE_ENERGY = 510

steps = np.arange(0, 1e3)+1
 
x = steps * INFERENCE_ENERGY * (1 / REQUEST_PER_SECOND) * SECONDS_PER_HOUR * HOURS_PER_DAY

color = ["aqua", "blue", "black"]

fig = plt.figure(figsize=(20, 10))
ax = fig.add_subplot(111)
for i, rps in enumerate([100, 1000, 10000]):
    rps_label = int(rps / 1000) if int(rps / 1000) == np.ceil(rps / 1000) else rps / 1000
    plt.plot(steps, x * rps, label=f"{rps_label}k RPS", linewidth=8, color=color[i])

TRAIN_ENERGY=GPT3_TRAIN_CARBON  / CO2_PER_KWH * SECONDS_PER_HOUR * 1000
plt.axhline(y=TRAIN_ENERGY, color='r', linestyle='-.', linewidth=8)
# plt.axvline(x =MODEL_RELEASE_DAYS, color='r', linestyle='-.', linewidth=8)
ax.annotate('One-off Training', xy=(1,TRAIN_ENERGY), xytext=(1, 1e14), size=48,
            arrowprops=dict(arrowstyle="->", color='r', lw=6,
                            connectionstyle="angle3,angleA=0,angleB=-90"));
# ax.annotate('Next Generation', xy=(MODEL_RELEASE_DAYS, 1e11), xytext=(20, 1e11), size=48,
#             arrowprops=dict(arrowstyle="->", color='r', lw=6,
#                             connectionstyle="angle3,angleA=0,angleB=179"));

# x = steps * BERT_BASE_POWER * BERT_BASE_TIME / (60**2) / 1000 * CO2_PER_KWH * NUM_GPUS
# x_d = steps * DISTILBERT_BASE_POWER * DISTILBERT_BASE_TIME / (60**2) / 1000 * CO2_PER_KWH
# x_t = x * (1 / BERT_BASE_TIME * NUM_GPUS) * (60**2)
# x_dt = x_d * (1 / DISTILBERT_BASE_TIME * NUM_GPUS) * (60**2)

# df = pd.DataFrame(
#     {"steps": steps, "reqc": x, "timec": x_t, "name": ["Inference"]* len(steps)}
# )
# print(df)

# plt.figure(figsize=(25, 15))
# # ax = sns.lineplot(x="steps", y="timec", hue="name", data=df, palette=sns.dark_palette("blue", 1), lw=12)
# plt.plot(steps, x_t / 1000, label="Inference", color='b', linewidth=12, linestyle='-')
# # plt.plot(steps, x_dt / 1000, label="Optimal", color='black', linewidth=12, linestyle='--')
# plt.axhline(y=BERT_BASE_TRAIN_CARBON_KG, color='r', linestyle='-.', linewidth=12, label="Train")

plt.xticks(fontsize=48)
plt.yticks(fontsize=48)
plt.xlabel("Time Elapsed (days)", fontsize=48)
plt.ylabel("Energy (Joules)", fontsize=48)
plt.xscale("log")
plt.yscale("log")
# plt.grid()
plt.legend(bbox_to_anchor=(1, 1.15), ncol=3, fontsize=36)

plt.savefig("plots/inference-energy.png", bbox_inches="tight")
