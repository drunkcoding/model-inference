import asyncio
import functools
import time
import numpy as np
import ray
import ray.util
from ray import serve
import torch

import os
from tqdm import tqdm
from transformers import GPT2Tokenizer, EvalPrediction
from transformers.data.data_collator import (
    default_data_collator,
)
from torch.utils.data import DataLoader
from datasets import load_metric, load_dataset
from scipy.special import softmax

from hfutils.measure import get_energy_by_group

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

task_name = "squad_v2"
batch_size = 1
home_dir = "/mnt/raid0nvme1"

tokenizer = GPT2Tokenizer.from_pretrained(
    f"{home_dir}/HuggingFace/gpt2",
    use_fast=True,
)

val_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
val_dataset = val_dataset.select([x for x in range(80)])
print(val_dataset)


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

        yield input_ids, target_ids, trg_len, end_loc

m = functools.partial(softmax, axis=1)

inputs_list = []
labels_list = []
for step, batch in enumerate(tqdm(load_encodings(encodings), desc="Prepare")):
    input_ids, target_ids, trg_len, end_loc = batch
    labels_list.append(target_ids[-1])
    input_ids = input_ids.numpy().astype(np.int64)
    attention_mask = np.ones(input_ids.shape).astype(np.int64)
    inputs_list.append((input_ids, attention_mask))

ray.init(address="ray://129.215.164.41:10001", namespace="gpt")

handle = serve.get_deployment("hybrid-scheduler").get_handle()

start_time = time.perf_counter()
start_energy = sum(list(get_energy_by_group().values()))
async_requests = []
for step, input in enumerate(tqdm(inputs_list)):
    response = handle.ensemble_inference.remote(input)
    async_requests.append(response)

async_requests = ray.get(async_requests)
end_energy = sum(list(get_energy_by_group().values()))
end_time = time.perf_counter()
print(end_energy - start_energy)
print(end_time - start_time)
