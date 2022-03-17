import functools
from multiprocessing import pool
import numpy as np
import os
import json
from tqdm import tqdm
from hfutils.plot import sns_displot, distplot
from sklearn.metrics import r2_score

batches = [1,2,4,8,16,32,64]
model_names = [
    "t5-small-lm-adapt",
    "t5-base-lm-adapt",
    "t5-large-lm-adapt",
    "t5-xl-lm-adapt",
]
model_keys = [
    "S",
    "M",
    "L",
    "XL"
]

data_trace = {}
data_kdur = {}
data_delay = {}
data_ovhd = {}
data_est = {}

np.random.seed(1234567890)

for bsz in batches:
    for i, name in enumerate(model_names):
        id = model_keys[i]
        key = f"{id}_{bsz}"
        filename = os.path.join("data", "single_profile", f"trace_{name}_{bsz}.npy")
        data_trace[key] = np.load(filename, allow_pickle=False)
        filename = os.path.join("data", "single_profile", f"dur_{name}_{bsz}.npy")
        data_kdur[key] = np.load(filename, allow_pickle=False)

        with open(os.path.join("data", "single_model", f"ray-{id}-{bsz}.json"), "r") as fp:
            base = json.load(fp)
            base_t = []
            for v in base.values():
                base_t += v 

        with open(os.path.join("data", "replica_model", f"ray-{id}-R2-{bsz}.json"), "r") as fp:
            data = json.load(fp)
            target = []
            for v in data.values():
                target += v

        sample = np.load(os.path.join("data", "combined_model",f"{id}_{bsz}_{id}_{bsz}.npy"), allow_pickle=False)
        print(sample)
        data_est[key] = np.mean(
            sample
        ) # * np.log2(len(data_trace[key]) / np.mean(base_t) * 1000)

        data_ovhd[key] = np.abs(np.mean(target)- np.mean(base_t))
        data_delay[key] = base_t

true_latency = []
kernel_latency = []

for key in data_delay:
    true_latency.append(np.mean(data_delay[key]))
    log_batch = np.log2(int(key.split("_")[-1])) + 1
    kernel_latency.append(np.sum(data_kdur[key]) / 1000 * log_batch)

parameters = np.polyfit(kernel_latency,true_latency,1)
print(parameters)
func_latency = np.poly1d(parameters)

pred_latency = func_latency(kernel_latency)

print(true_latency,pred_latency)
print(r2_score(true_latency,pred_latency))

true_ovhd = np.array(list(data_ovhd.values()))
kernel_ovhd = func_latency(np.array(list(data_est.values())))

parameters = np.polyfit(kernel_ovhd,true_ovhd,1)
print(parameters)
func_ovhd = np.poly1d(parameters)

pred_ovhd = func_ovhd(kernel_ovhd)

print(true_ovhd,pred_ovhd)
print(kernel_ovhd)
print(r2_score(true_ovhd,pred_ovhd))
print(r2_score(true_ovhd,kernel_ovhd))
print(np.abs(true_ovhd-pred_ovhd)/true_ovhd)


