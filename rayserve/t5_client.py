import asyncio
from dataclasses import dataclass, field
import json
import time
import torch
import numpy as np
import ray
import ray.util
from ray import serve
import os
from functools import partial


from datasets import load_dataset, load_metric
import requests

from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from transformers import HfArgumentParser, T5Tokenizer, DataCollatorForSeq2Seq

from hfutils.measure import get_energy_by_group, get_remote_gpu_energy
from hfutils.loader import load_glue_val, t5_preprocess_function

import socket


def get_host_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(("8.8.8.8", 1))
    host_ip = s.getsockname()[0]
    return host_ip


@dataclass
class Arguments:
    namespace: str = field(metadata={"help": "test type"})
    batch_size: int = field(
        metadata={"help": "batch_size"},
    )


parser = HfArgumentParser(Arguments)
args = parser.parse_args_into_dataclasses()[0]

batch_size = args.batch_size

metric = load_metric("accuracy")


def metric_accuracy(logits, labels):
    predictions = np.argmax(logits, axis=1).flatten()
    return metric.compute(predictions=predictions, references=labels.flatten())[
        "accuracy"
    ]


home_dir = "/data"
model_paths = f"{home_dir}/HuggingFace/google/t5-small-lm-adapt"

tokenizer = T5Tokenizer.from_pretrained(model_paths)

preprocess_function = partial(
    t5_preprocess_function,
    tokenizer=tokenizer,
    padding="max_length",
    max_length=128,
)

data_collator = DataCollatorForSeq2Seq(tokenizer)
dataset = load_glue_val(preprocess_function).shuffle()

dataloader = DataLoader(
    dataset,
    shuffle=False,
    collate_fn=data_collator,
    batch_size=batch_size,
    drop_last=True,
)

from hfutils.constants import token2label

T5_TASK_LABELS = [1176, 6136, 59]

inputs_list = []
labels_list = []
for step, batch in enumerate(tqdm(dataloader)):
    if step * batch_size > 1000:
        break
    input_ids = batch["input_ids"].numpy()
    attention_mask = batch["attention_mask"].numpy()
    inputs_list.append((input_ids, attention_mask))

    label = token2label(batch["labels"][:, 0], T5_TASK_LABELS)
    labels_list.append(np.array(label))

labels_list = np.concatenate(labels_list)

host_ip = get_host_ip()
ray.init(address=f"ray://{host_ip}:10001", namespace=args.namespace)


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

handles = [
    serve.get_deployment(f"hybrid-scheduler_{host}_{r}").get_handle(sync=True)
    for host in hosts
    for r in range(8)
]

start_energy = np.array([get_remote_gpu_energy(host, 0) for host in hosts])

# start_energy = np.array(list(get_energy_by_group().values()))
exec_start_time = time.perf_counter()
async_requests = []
time_list = []
energy_list = []
async_requests = []
for step, input in enumerate(tqdm(inputs_list)):
    start_time = time.perf_counter()
    handle = handles[step % len(handles)]
    response = handle.ensemble_inference.remote(input)
    logits = ray.get(response)
    async_requests.append(logits)
    # async_requests.append(response)
    end_time = time.perf_counter()
    time_list.append(end_time - start_time)

# async_requests = ray.get(async_requests)
async_requests = np.concatenate(async_requests)
# print(metric_accuracy(async_requests, labels_list))
exec_end_time = time.perf_counter()
end_energy = np.array([get_remote_gpu_energy(host, 0) for host in hosts])

print(end_energy - start_energy)
print(np.sum(time_list))
print(exec_end_time - exec_start_time)

with open(os.path.join(os.path.dirname(__file__), f"{args.namespace}.json"), "w") as fp:
    json.dump(
        {
            "latency": time_list,
            "exec_time": exec_end_time - exec_start_time,
            "energy": (end_energy - start_energy).tolist(),
        },
        fp,
    )
