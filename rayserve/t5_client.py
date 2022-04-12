import asyncio
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
from transformers import T5Tokenizer, DataCollatorForSeq2Seq

from hfutils.measure import get_energy_by_group
from hfutils.loader import load_glue_val, t5_preprocess_function


home_dir = "/mnt/raid0nvme1"
model_paths = f"{home_dir}/HuggingFace/google/t5-small-lm-adapt"

tokenizer = T5Tokenizer.from_pretrained(model_paths)

preprocess_function = partial(
    t5_preprocess_function, tokenizer=tokenizer, padding="max_length", max_length=128,
)

batch_size = 128

data_collator = DataCollatorForSeq2Seq(tokenizer)
dataset = load_glue_val(preprocess_function).shuffle()

dataloader = DataLoader(
    dataset, shuffle=False, collate_fn=data_collator, batch_size=batch_size, drop_last=True,
)

inputs_list = []
for step, batch in enumerate(tqdm(dataloader)):
    # if step * batch_size > 1000:
    #     break
    input_ids = batch["input_ids"].numpy()
    attention_mask = batch["attention_mask"].numpy()
    inputs_list.append((input_ids,attention_mask))
    
ray.init(address="ray://129.215.164.41:10001", namespace="t5")


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