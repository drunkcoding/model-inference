import os
import re
import json
from sklearn.metrics import r2_score
import numpy as np

base_dir = os.path.dirname(__file__)
files = ['bert', 'vit', 't5', 'gpt']


memory_config = {}
memory_params = {}
latency_params = {}
mem_data = {}
latency = {}
latency_config = {}

energy_config = {}

for file in files:
    filename = os.path.join(base_dir, f"{file}.log")
    with open(filename, "r") as fp:
        lines = fp.readlines()

    idx = 0
    while idx < len(lines):
        if "Arguments" in lines[idx]:
            group = re.findall(r"model_name_or_path='(.*)', batch_size=(\d+)", lines[idx])
            basename, batch_size = group[0]
            basename = os.path.basename(basename)
            batch_size = int(batch_size)
            idx += 3

            if not basename in memory_config:
                memory_config[basename] = {}
                mem_data[basename] = []
                latency[basename] = []
                latency_config[basename] = {}
                energy_config[basename] = {}

            while not "energy" in lines[idx]:
                # print(lines[idx])
                idx += 1
                if idx >= len(lines): break
            if idx >= len(lines): break

            group = re.findall(r"energy total (\d+.\d+), request (\d+.\d+), sample (\d+.\d+)", lines[idx])
            total, request, sample = group[0]
            reserved = float(total)
            allocated = float(request)
            total = float(sample)   
            energy_config[basename][batch_size] = [float(v) for v in group[0]]           

            while not "memory" in lines[idx]:
                idx += 1
                if idx >= len(lines): break
            if idx >= len(lines): break
            group = re.findall(r"memory reserved (\d+), allocated (\d+), total (\d+)", lines[idx])
            reserved, allocated, total = group[0]
            reserved = int(reserved)
            allocated = int(allocated)
            total = int(total)

            mem_data[basename].append((batch_size, allocated / 1024**2))
            # print(basename, batch_size, data[basename])
            memory_config[basename][batch_size] = [int(v) for v in group[0]]

            
            while not "mean" in lines[idx]:
                idx += 1
                if idx >= len(lines): break
            if idx >= len(lines): break

            mean_latency = float(lines[idx].split(" ")[-1]) * 1000
            latency[basename].append((batch_size, mean_latency))
            latency_config[basename][batch_size] = mean_latency

        idx += 1
    for basename, d in mem_data.items():
        # print(memory_params[basename], basename)
        xdata, ydata = zip(*d)
        params = np.polyfit(xdata, ydata, 1)

        print(basename, r2_score(ydata, np.poly1d(params)(xdata)))
        memory_params[basename] = params.tolist()

    for basename, d in latency.items():
        # print(memory_params[basename], basename)
        xdata, ydata = zip(*d)
        params = np.polyfit(xdata, ydata, 2)
        print(basename, xdata, ydata)
        print(basename, r2_score(ydata, np.poly1d(params)(xdata)))
        latency_params[basename] = params.tolist()

with open("tests/kernel_duration/memory.json", "w") as fp:
    json.dump(memory_config, fp)

with open("tests/kernel_duration/latency.json", "w") as fp:
    json.dump(latency_config, fp)

with open("tests/kernel_duration/energy.json", "w") as fp:
    json.dump(energy_config, fp)

with open("tests/kernel_duration/memory_params.json", "w") as fp:
    json.dump(memory_params, fp)

with open("tests/kernel_duration/latency_params.json", "w") as fp:
    json.dump(latency_params, fp)