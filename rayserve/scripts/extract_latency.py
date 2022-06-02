import json
import os, re
import numpy as np
from scipy.stats import describe

# Open a file
path = "/home/ubuntu/model-inference/rayserve/gpt-optim"
dirs = os.listdir(path)

# This would print all the files and directories
latency_list = []
for file in dirs:
    # print(file)
    filepath = os.path.join(path, file)
    if "e" in file or not ".log" in file: continue
    with open(filepath, "r") as fp:
        text = fp.read()

    groups = re.findall(r"request\(1\) \(.*\) (\d+.\d+)", text)
    # print(groups)
    for group in groups:
        latency_list.append(float(group))

latency_list = np.array(latency_list)
latency_list = latency_list[latency_list < 15]

print(describe(latency_list))
print(np.percentile(latency_list, 50))
print(np.percentile(latency_list, 99))

json_path = "/home/ubuntu/model-inference/rayserve/gpt-3-preserved-sync.json"
with open(json_path, "r")as fp:
    data = json.load(fp)
print(np.percentile(data["latency"], 99))
print(np.percentile(data["latency"], 50))
print(describe(data["latency"]))

hosts = [
    "172.31.35.95",
    "172.31.39.160",
    "172.31.47.240",
    "172.31.32.224",
    "172.31.44.101",
    "172.31.36.213",
    "172.31.43.33",
    "172.31.39.35",
    "172.31.43.93",
    "172.31.34.158",
    "172.31.40.86",
    "172.31.47.59",
]

ds_list = []
for host in hosts:
    json_path = f"/home/ubuntu/model-inference/tests/deepspeed/gpt-bsz1-False_{host}.json"
    with open(json_path, "r")as fp:
        data = json.load(fp)
    latency = np.array(data["latency"])
    ds_list.append(latency)
    print(host, describe(latency))
ds_records = np.sum(ds_list, axis=0)
print(np.percentile(ds_records, 99))
print(np.percentile(ds_records, 50))
print(describe(ds_records))


    