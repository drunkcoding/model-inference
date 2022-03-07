import os
import numpy as np
import seaborn as sns
import json
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

batches = [1,2,4,8,16,32,64]
model_names = [
    "t5-small-lm-adapt",
    "t5-base-lm-adapt",
    "t5-large-lm-adapt",
    "t5-xl-lm-adapt",
]
# batches = [1,2,4,8,16]
model_name = "t5-xl-lm-adapt"
model_keys = [
    "S",
    "M",
    "L",
    "XL"
]
id = "XL"

true_latency = []
pred_latency = []

def reject_outliers(data, m = 2.):
    data = np.array(data)
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d/mdev if mdev else 0.
    # print(s < m)
    return data[s < m]

for i, model_name in enumerate(model_names):
    id = model_keys[i]
    for bsz in batches:
        with open(os.path.join("data/single_model", f"ray-{id}-{bsz}.json"), "r") as fp:
            base = json.load(fp)

        with open(os.path.join("data/replica_model", f"ray-{id}-R2-{bsz}.json"), "r") as fp:
            data = json.load(fp)

        filename = os.path.join("data/single_profile", f"trace_{model_name}_{bsz}.npy")
        raw_data = np.load(filename, allow_pickle=False)

        source = np.load(os.path.join("data",f"{model_name}_{bsz}_{model_name}_{bsz}.npy"), allow_pickle=False)
        target = []
        for v in data.values():
            target += v

        base_t = []
        for v in base.values():
            base_t += v

        # base_t = reject_outliers(base_t)
        # source = reject_outliers(source)
        # target = reject_outliers(target)

        # source = (source - np.mean(source)) / np.std(source)
        # target = (target - np.mean(target)) / np.std(target)
        
        # source = np.random.choice(source, len(target))
        # print(bsz, np.mean(target), np.mean(source), stats.kstest(source, target))
        # print(stats.kstest(stats.norm.rvs, 'norm', N=1000))
        # exit()
        diff = np.abs(np.mean(target)- np.mean(base_t))
        if diff < 1000:
            true_latency.append(diff)
            pred_latency.append(
                # np.mean(source)
                np.mean(raw_data) * 2
                # * (np.log2(bsz) + 1) 
                * bsz
                # * bsz
                * np.log(len(raw_data) / np.mean(base_t) * 1000)
                # * np.mean(raw_data)
                # * np.mean(target)
            )
            print(np.mean(source), diff / np.mean(base_t))

true_latency = np.array(true_latency) 
pred_latency = np.array(pred_latency)

pred_latency = pred_latency / 100

print(true_latency,pred_latency)

parameters = np.polyfit(pred_latency,true_latency,1)
# parameters[-1] = 0
print(parameters)
p = np.poly1d(parameters)
# p = np.poly1d([  21.63844158, -759.41585935])
# true_latency = 0.1588068 * true_latency + 12.54144258

pred_latency = p(pred_latency)

print(true_latency,pred_latency)
print(r2_score(true_latency,pred_latency))
print(np.abs(true_latency-pred_latency)/true_latency)

sns.scatterplot(x=true_latency,y=pred_latency)
plt.xscale("log")
plt.yscale("log")
plt.xlabel("True")
plt.ylabel("Pred")
# plt.savefig(f"figures/{model_name}_cost.png")
plt.savefig(f"figures/total_cost.png")
plt.close()
