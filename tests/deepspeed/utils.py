import os
import torch
import socket
import time
import json

from hfutils.measure import get_energy_by_group
from tqdm import tqdm

def get_host_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(('8.8.8.8', 1))
    host_ip = s.getsockname()[0]
    return host_ip

@torch.no_grad()
def execute_model(dataloader, inference_func, type):
    host_ip = get_host_ip()

    start_energy = sum(list(get_energy_by_group().values()))
    inference_count = 0
    latency_list = []
    for step, batch in enumerate(tqdm(dataloader)):
        if step > 100:
            break
        start_time = time.perf_counter()
        outputs = inference_func(batch)
        if outputs != None:
            inference_count += 1
        end_time  = time.perf_counter()
        latency_list.append(end_time-start_time)
    end_energy = sum(list(get_energy_by_group().values()))

    basedir = os.path.dirname(__file__)
    data = {
        "host_ip": host_ip,
        "latency": latency_list,
        "energy": end_energy-start_energy,
        "inference_count": inference_count,
    }
    with open(os.path.join(basedir, f"{type}_{host_ip}.json"), "w") as fp:
        json.dump(data, fp)
