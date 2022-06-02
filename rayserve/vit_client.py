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

from datasets import load_dataset, load_metric
import requests

from tqdm import tqdm
from torch.utils.data import DataLoader, Subset
from transformers import HfArgumentParser

from hfutils.measure import get_energy_by_group, get_remote_gpu_energy, get_host_ip

home_dir = "/data"
dataset_path = os.path.join(home_dir, "ImageNet")
model_paths = f"{home_dir}/HuggingFace/WinKawaks/vit-tiny-patch16-224"

metric = load_metric("accuracy")


def metric_accuracy(logits, labels):
    predictions = np.argmax(logits, axis=1).flatten()
    # print(predictions, predictions.shape)
    # print(labels, labels.shape)
    return metric.compute(predictions=predictions, references=labels.flatten())[
        "accuracy"
    ]


@dataclass
class Arguments:
    namespace: str = field(metadata={"help": "test type"})
    batch_size: int = field(
        metadata={"help": "batch_size"},
    )


parser = HfArgumentParser(Arguments)
args = parser.parse_args_into_dataclasses()[0]

batch_size = args.batch_size

# -------------  Dataset Prepare --------------
from torchvision.datasets import ImageNet
from hfutils.preprocess import (
    split_train_test,
    vit_collate_fn,
    ViTFeatureExtractorTransforms,
)
from hfutils.measure import get_energy_by_group

print("======ImageNet==========")

dataset = ImageNet(
    dataset_path,
    split="val",
    transform=ViTFeatureExtractorTransforms(model_paths, split="val"),
)
num_labels = len(dataset)
m = torch.nn.Softmax(dim=1)

inputs_list = []
label_list = []

rnd_seed = 106033
np.random.seed(rnd_seed)

index = np.array([x for x in range(len(dataset))])
np.random.shuffle(index)
dataset = Subset(dataset, index)

eval_dataloader = DataLoader(
    dataset,
    num_workers=4,
    collate_fn=vit_collate_fn,
    batch_size=batch_size,
)

host_ip = get_host_ip()
ray.init(address=f"ray://{host_ip}:10001", namespace=args.namespace)

inputs_list = []
labels_list = []
for step, batch in enumerate(tqdm(eval_dataloader)):
    if step * batch_size > 1000:
        break
    pixel_values = batch["pixel_values"].numpy()
    inputs_list.append((pixel_values,))
    labels_list.append(batch["labels"].numpy())

labels_list = np.concatenate(labels_list)


hosts = [
    "172.31.35.95",
    # "172.31.39.160",
    # "172.31.47.240",
    # "172.31.32.224",
    # "172.31.44.101",
    # "172.31.36.213",

    # "172.31.43.33",
    # "172.31.39.35",
    # "172.31.43.93",
    # "172.31.34.158",
    # "172.31.40.86",
    # "172.31.47.59",
]

handles = [
    serve.get_deployment(f"hybrid-scheduler_{host}_{r}").get_handle(sync=True)
    for host in hosts
    for r in range(1)
]

start_energy = np.array([get_remote_gpu_energy(host, 0) for host in hosts])
exec_start_time = time.perf_counter()
async_requests = []
time_list = []
energy_list = []
async_requests = []
for step, input in enumerate(tqdm(inputs_list)):
    handle = handles[step % len(handles)]
    start_time = time.perf_counter()
    # response = handle.ensemble_inference.remote(input)
    response = handle.handle_batch.remote(input)
    # logits = ray.get(response)
    # async_requests.append(logits)
    async_requests.append(response)
    end_time = time.perf_counter()
    time_list.append(end_time - start_time)

async_requests = ray.get(async_requests)
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
