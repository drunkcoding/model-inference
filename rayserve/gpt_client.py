import asyncio
from dataclasses import dataclass, field
import functools
import sys
import json
import time
import numpy as np
import ray
import ray.util
from ray import serve
import requests
import torch

import os
from tqdm import tqdm
from transformers import GPT2Tokenizer, EvalPrediction, HfArgumentParser
from transformers.data.data_collator import (
    default_data_collator,
)
from torch.utils.data import DataLoader, Subset, Dataset
from datasets import load_metric, load_dataset
from scipy.special import softmax

from hfutils.measure import get_energy_by_group, get_host_ip, get_remote_gpu_energy

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class TextDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, iter):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data = list(iter)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = {
            "input_ids": self.data[idx][0],
            "attention_mask": torch.ones_like(self.data[idx][0]).to(torch.long),
            "labels": self.data[idx][1],
        }
        return sample


@dataclass
class Arguments:
    namespace: str = field(metadata={"help": "test namespace"})
    batch_size: int = field(
        metadata={"help": "batch_size"},
    )


parser = HfArgumentParser(Arguments)
args = parser.parse_args_into_dataclasses()[0]

batch_size = args.batch_size
home_dir = "/data"

tokenizer = GPT2Tokenizer.from_pretrained(
    f"{home_dir}/HuggingFace/gpt2",
    use_fast=True,
)

val_dataset = load_dataset("lambada", split="validation")
# val_dataset = val_dataset.select([x for x in range(80)])
# print(val_dataset)


encodings = tokenizer("\n\n".join(val_dataset["text"]), return_tensors="pt")
encodings.input_ids = encodings.input_ids.to(torch.long)

max_length = 512
stride = 128


def load_encodings(encodings):
    for i in tqdm(range(0, encodings.input_ids.size(1), stride)):
        begin_loc = max(i + stride - max_length, 0)
        end_loc = min(i + stride, encodings.input_ids.size(1))
        trg_len = end_loc - i  # may be different from stride on last loop
        input_ids = encodings.input_ids[:, begin_loc:end_loc]
        # input_ids = input_ids.to(torch.int8)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        if input_ids.size(1) != max_length:
            continue

        yield input_ids, encodings.input_ids[:, end_loc]


m = functools.partial(softmax, axis=1)

metric = load_metric("accuracy")


def metric_accuracy(logits, labels):
    predictions = np.argmax(logits, axis=1).flatten()
    # print(predictions, predictions.shape)
    # print(labels, labels.shape)
    return metric.compute(predictions=predictions, references=labels.flatten())[
        "accuracy"
    ]


dataset = TextDataset(load_encodings(encodings))

rnd_seed = 106033
np.random.seed(rnd_seed)

index = np.array([x for x in range(len(dataset))])
np.random.shuffle(index)
dataset = Subset(dataset, index)
dataloader = DataLoader(
    dataset, shuffle=False, num_workers=2, batch_size=args.batch_size
)

inputs_list = []
labels_list = []
for step, batch in enumerate(tqdm(dataloader, desc="Prepare")):
    if step * args.batch_size > 500:
        break
    input_ids = batch['input_ids'].numpy().astype(np.int64)
    attention_mask = batch['attention_mask'].numpy().astype(np.int64)
    # print(input_ids.size())
    labels_list.append(batch['labels'].numpy().astype(np.int64))
    inputs_list.append((input_ids, attention_mask))
labels_list = np.concatenate(labels_list)

# import multiprocessing as mp

# NUM_PROC = 5

# def test_body(pid):
#     async_requests = []
#     s = requests.Session()
#     for step, input in enumerate(tqdm(inputs_list, desc=str(pid))):
#         # input_ids = io.BytesIO()
#         # attention_mask = io.BytesIO()
#         # np.save(input_ids, input[0], allow_pickle=False)
#         # np.save(attention_mask, input[1], allow_pickle=False)
#         response = s.post('http://127.0.0.1:8000/hybrid-scheduler', json={
#             # "input_ids":input_ids.getvalue().decode('utf-8'),
#             # "attention_mask":attention_mask.getvalue().decode('utf-8'),
#             "input_ids": input[0].tolist(),
#             "attention_mask": input[1].tolist(),
#         })

#         # print(response.json())
#         logits = np.array(response.json()['logits'])
#         async_requests.append(logits)

#     # ray.init(address="ray://129.215.164.41:10001", namespace="gpt")

#     # handle = serve.get_deployment("hybrid-scheduler").get_handle(sync=True)

#     # start_time = time.perf_counter()
#     # start_energy = np.array(list(get_energy_by_group().values()))
#     # async_requests = []
#     # for step, input in enumerate(tqdm(inputs_list)):
#     #     response = handle.ensemble_inference.remote(input)
#     #     async_requests.append(response)

#     # async_requests = ray.get(async_requests)

#     # # for obj in async_requests:
#     # #     print(obj.shape)

#     async_requests = np.concatenate(async_requests)

#     print(metric_accuracy(async_requests, labels_list))

host_ip = get_host_ip()
ray.init(address=f"ray://{host_ip}:10001", namespace=args.namespace)

start_time = time.perf_counter()
start_energy = np.array(list())

# pool = mp.Pool(processes=NUM_PROC)
# pool.map(test_body, [i for i in range(NUM_PROC)])
# pool.close()
# pool.join()

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
    for r in range(8)
]

start_energy = np.array([get_remote_gpu_energy(host, 0) for host in hosts])
exec_start_time = time.perf_counter()
# start_energy = np.array(list(get_energy_by_group().values()))
async_requests = []
time_list = []
energy_list = []
for step, input in enumerate(tqdm(inputs_list)):
    handle = handles[step % len(handles)]
    start_time = time.perf_counter()
    response = handle.ensemble_inference.remote(input)
    logits = ray.get(response)
    # print(logits.shape)
    async_requests.append(logits)
    async_requests.append(response)
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
