import re
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from numpy.random import MT19937
from numpy.random import RandomState, SeedSequence

rs = RandomState(MT19937(SeedSequence(123456789)))

font = {"size": 64}
matplotlib.rc("font", **font)
matplotlib.rcParams['hatch.linewidth'] = 5

with open("rayserve/tests/confidence_ovhd/server.log", "r") as fp:
    text = fp.read()

groups = re.findall(r"size (\d+) time (\d+.\d+)", text)

sizes = []
times = []
for group in groups:
    size, time = group
    size = int(size)* 4
    time = float(time)

    sizes.append(size)
    times.append(time)

df = pd.DataFrame({
    "sizes": sizes,
    "times": times,
})



data_propotion = {
    "2": [1.0, 0.4],
    "3": [1.0, 0.74, 0.4],
    "4": [1.0, 0.63, 0.55, 0.33],
    "5": [1.0, 0.95, 0.83, 0.74, 0.55],
}
keys = list(data_propotion.keys())
data_hatches = ['\\', '-', '+', 'x']
data_color = ['navy', 'blue', 'royalblue', 'aqua']
data_line = ['-', ':', '-.', '--']
data_hatches = dict(zip(keys, data_hatches))
data_color = dict(zip(keys, data_color))
data_line = dict(zip(keys, data_line))

print(data_hatches)
print(data_color)
print(data_line)

plt.figure(figsize=(20, 10))

for key, value in data_propotion.items():
    new_times = []
    for row in df.iterrows():
        # print(row)
        l = 0
        for p in value:
            if np.random.random() < p:
                l += row[1].times
        new_times.append(l)
    df['times'] = new_times
    print(df)
    group_latency = df.groupby("sizes", as_index=False)["times"]
    latency_mean = group_latency.mean()
    latency_std = group_latency.std()
    latency_min = group_latency.min()
    latency_max = group_latency.max()


    plt.fill_between(
        latency_mean["sizes"] / 1024,
        latency_min["times"],
        latency_max["times"],
        color=[data_color[key]],
        alpha=0.2,
        hatch=data_hatches[key]
    )
    plt.plot(
        latency_mean["sizes"] / 1024,
        latency_mean["times"],
        linewidth=6,
        linestyle=data_line[key],
        color="black",
        label=key
    )
plt.xlabel("Data Size (KB)", fontsize=64)
plt.ylabel("Average Latency (ms)", fontsize=64)
plt.xticks(fontsize=64)
plt.yticks(fontsize=64)
plt.yscale("log")
plt.xscale("log")
plt.legend(title="#Models", bbox_to_anchor=(0.9, 1.3), ncol=4, fontsize=48)
# plt.xscale("log")
# sns.lineplot(x="sizes", y="latency", data=latency_mean)
plt.savefig("rayserve/tests/confidence_ovhd/confidence_ovhd.png", bbox_inches="tight")
