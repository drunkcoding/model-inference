from dataclasses import dataclass, field
from functools import partial
import functools
import itertools
import json
import logging
import os
import random
import time
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import torch
from transformers import (
    AutoModelForCausalLM,
    GPT2LMHeadModel,
    GPT2Tokenizer,
)
import pandas as pd
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from hfutils.logger import Logger
from hfutils.pipe.gpt import GPTLMHeadModelPipe
from hfutils.calibration import temperature_scale
from hfutils.qa import prepare_validation_features, prepare_train_features

import sys

sys.path.append(".")
from plots.thresholds.utils import *


home_dir = "/mnt/raid0nvme1"
base_dir = os.path.join(home_dir, os.path.join("model-finetune", "outputs", "google"))

model_keys = [
    # "XS",
    # "S",
    "M",
    "L",
    "XL",
    # "XXL",
]

model_names = [
    # "distilgpt2",
    # "gpt2",
    "gpt2-medium",
    "gpt2-large",
    "gpt2-xl",
    # "gpt-j",
]

device_map = [
    # "cuda:3", 
    # "cuda:5",
    "cuda:3",
    "cuda:5",
    "cuda:7",
    # "cuda:6",
]

model_paths = [
    # f"{home_dir}/HuggingFace/distilgpt2",
    # f"{home_dir}/HuggingFace/gpt2",
    f"{home_dir}/HuggingFace/gpt2-medium",
    f"{home_dir}/HuggingFace/gpt2-large",
    f"{home_dir}/HuggingFace/gpt2-xl",
    # f"{home_dir}/HuggingFace/EleutherAI/gpt-j-6B",
]
tokenizer = GPT2Tokenizer.from_pretrained(
    f"{home_dir}/HuggingFace/gpt2", use_fast=True,
)

val_dataset = load_dataset("lambada", split="validation")
# val_dataset = val_dataset.select(range(100))

encodings = tokenizer("\n\n".join(val_dataset["text"]), return_tensors="pt")
encodings.input_ids = encodings.input_ids.to(torch.long)

def load_encodings(encodings):
    max_length = 512
    stride = 128

    for i in tqdm(range(0, encodings.input_ids.size(1) - 1, stride)):
        begin_loc = max(i + stride - max_length, 0)
        end_loc = min(i + stride, encodings.input_ids.size(1))
        trg_len = end_loc - i  # may be different from stride on last loop
        input_ids = encodings.input_ids[:, begin_loc:end_loc]

        if input_ids.size(1) != max_length:
            continue

        yield input_ids, encodings.input_ids[:, end_loc]


model_paths = dict(zip(model_keys, model_paths))
model_names = dict(zip(model_keys, model_names))
model_device = dict(zip(model_keys, device_map))


@torch.no_grad()
def model_inference(model, input_ids, temperature=None, device="cuda:0"):
    input_ids = input_ids.to(device)
    logits = model((input_ids, None))
    logits = logits.squeeze(0)[-1, :]
    # print(logits.size())
    if temperature is not None:
        # logits = temperature_scale(logits, temperature)
        logits /= temperature
    return logits.reshape((1, -1)).to("cpu")


with open("tests/kernel_duration/latency.json", "r") as fp:
    model_latency = json.load(fp)
with open("repository/repo_gpt/meta.json", "r") as fp:
    model_meta = json.load(fp)

models = load_models(
    model_keys,
    model_paths,
    model_device,
    AutoModelForCausalLM,
    GPTLMHeadModelPipe,
)

n_models = len(model_keys)
model_probs = dict(zip(model_keys, [list() for _ in range(n_models)]))
model_ans = {}
model_outputs = dict(zip(model_keys, [list() for _ in range(n_models)]))

m = torch.nn.Softmax(dim=-1)

num_labels = 0
labels = []
for batch in tqdm(load_encodings(encodings), desc="Collect Train Data"):
    num_labels += 1
    input_ids, target_id = batch
    for i, key in enumerate(model_keys):
        logits = model_inference(
            models[key],
            input_ids,
            device=model_device[key],
            temperature=model_meta[model_names[key]]["temperature"],
        )
        model_outputs[key].append(logits)
    # print((start_positions, end_positions))
    labels.append(torch.Tensor([target_id]))


def process_func(logits):
    topk, _ = torch.topk(logits, 10)
    probs = m(topk)
    return probs

model_probs, model_ans, model_outputs, labels = postprocessing_inference(
    model_keys, model_outputs, labels, process_func, "cpu", alpha=0.6
)

all_thresholds = list(
    itertools.product(np.linspace(0, 1, endpoint=True, num=100), repeat=n_models - 1)
)
max_size = 100000
if len(all_thresholds) > max_size:
    rnd_idx = np.random.randint(0, len(all_thresholds), max_size)
    all_thresholds = [all_thresholds[i] for i in rnd_idx]

profile_thresholds(
    model_keys,
    model_probs,
    model_ans,
    model_latency,
    model_names,
    all_thresholds,
    "gpt-3",
    "cpu"
)
