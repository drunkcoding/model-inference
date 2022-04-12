import io
import re
import subprocess
import time
import pandas as pd
import torch
import requests
import os
from functools import partial
from torch.utils.data import DataLoader
# from experiment_impact_tracker.compute_tracker import ImpactTracker

from hfutils.loader import t5_preprocess_function, load_glue_val
from hfutils.measure import get_energy_by_group
from hfutils.pipe.t5 import T5PyTorchPipe
from tqdm import tqdm
from transformers import T5Tokenizer, T5ForConditionalGeneration, DataCollatorForSeq2Seq


def get_cpu_energy():
    command = "cat /sys/class/powercap/intel-rapl:0/energy_uj"
    result = subprocess.run(command.split(), stdout=subprocess.PIPE)
    energy = int(result.stdout)
    print(energy, result.stdout)
    command = "cat /sys/class/powercap/intel-rapl:1/energy_uj"
    result = subprocess.run(command.split(), stdout=subprocess.PIPE)
    energy += int(result.stdout)
    print(energy, result.stdout)
    return energy

home_dir = "/mnt/raid0nvme1"
base_dir = os.path.join(home_dir, os.path.join("HuggingFace", "google"))

model_keys = [
    "S",
    "L",
]

device_id = 0

device_map = [
    f"cuda:{device_id}",
    f"cuda:{device_id}",
]

model_paths = [
    f"{base_dir}/t5-small-lm-adapt/",
    f"{base_dir}/t5-large-lm-adapt/",
]
model_paths = dict(zip(model_keys, model_paths))
model_device = dict(zip(model_keys, device_map))

models = dict()
for key in model_keys:
    models[key] = T5ForConditionalGeneration.from_pretrained(model_paths[key])
    models[key] = T5PyTorchPipe(models[key])
    models[key].convert(model_device[key])
    models[key].eval()

tokenizer = T5Tokenizer.from_pretrained(model_paths[model_keys[0]])

preprocess_function = partial(
    t5_preprocess_function, tokenizer=tokenizer, padding="max_length", max_length=128,
)
eval_dataset = load_glue_val(preprocess_function).shuffle()
data_collator = DataCollatorForSeq2Seq(tokenizer)

dataloader = DataLoader(
    eval_dataset,
    collate_fn=data_collator,
    batch_size=1,
    # drop_last=True,
)

@torch.no_grad()
def model_inference(model, batch, device):
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    model((input_ids, attention_mask))



def get_gpu_uuid():
    command = "nvidia-smi --query-gpu=index,uuid,gpu_bus_id --format=csv"

    result = subprocess.run(command.split(), stdout=subprocess.PIPE)
    # print(result.stdout)
    df = pd.read_csv(io.StringIO(result.stdout.decode("utf-8")), index_col="index")
    df = df.sort_index()
    df.iloc[:, 0] = df.iloc[:, 0].str.strip()
    gpu_uuid = df.iloc[device_id][" uuid"]

    return gpu_uuid

uuid = get_gpu_uuid()

# tracker = ImpactTracker("impact_tracker/synthetic_workload")
# tracker.launch_impact_monitor()

def query_power():
    last_power = None
    command = "nvidia-smi --query-gpu=uuid,power.draw --format=csv"

    energy = 0
    while True:
        result = subprocess.run(command.split(), stdout=subprocess.PIPE)
        groups = re.findall(r"(.*), (\d+.\d+) W", result.stdout.decode("utf-8"))
        data = []
        for group in groups:
            gid, power = group
            power = float(power)
            data.append((gid, power))
        data = dict(data)
        cur_power = data[uuid]
        if last_power is None:
            energy += cur_power * 0.1
        else:
            energy += (cur_power + last_power) * 0.1 / 2
        last_power = cur_power
        print("energy", energy)
        time.sleep(0.1)

import multiprocess as mp

p = mp.Process(target=query_power)
p.start()

gpu_start_energy = get_energy_by_group()[uuid]
cpu_start_energy = get_cpu_energy() / 1e6
start_energy = cpu_start_energy + gpu_start_energy
for step, batch in enumerate(tqdm(dataloader, desc="Collect Data")):
    if step > 1000: break
    for i, key in enumerate(model_keys):
        model_inference(models[key], batch, device=model_device[key])
gpu_end_energy = get_energy_by_group()[uuid]
cpu_end_energy = get_cpu_energy() / 1e6
end_energy = cpu_end_energy + gpu_end_energy

print(end_energy-start_energy)
print(gpu_end_energy-gpu_start_energy)
print(cpu_end_energy-cpu_start_energy)

p.terminate()
p.join()
