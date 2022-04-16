from dataclasses import dataclass, field
from functools import partial
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
    T5ForConditionalGeneration,
    HfArgumentParser,
    T5Tokenizer,
    DataCollatorForSeq2Seq,
)
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from datasets import concatenate_datasets

from hfutils.logger import Logger
from hfutils.pipe.t5 import T5PyTorchPipe
from hfutils.loader import load_glue_val, load_glue_train, t5_preprocess_function
from hfutils.measure import get_energy_by_group, get_gpu_uuid
from hfutils.constants import TASK_TO_LABELS, token2label
from hfutils.calibration import temperature_scale

import sys

sys.path.append(".")
from plots.thresholds.utils import *

home_dir = os.path.expanduser(("~"))
# home_dir = "/mnt/raid0nvme1"
base_dir = os.path.join(home_dir, os.path.join("model-finetune", "outputs", "google"))

model_keys = [
    "S",
    "M",
    # "L",
    "XL",
]

model_names = [
    "t5-small-lm-adapt",
    "t5-base-lm-adapt",
    # "t5-large-lm-adapt",
    "t5-xl-lm-adapt",
]

device_map = [
    "cuda:0",
    "cuda:1",
    # "cuda:2",
    "cuda:3",
]

model_paths = [
    f"{base_dir}/t5-small-lm-adapt/all/checkpoint-4500",
    f"{base_dir}/t5-base-lm-adapt/all/checkpoint-2000",
    # f"{base_dir}/t5-large-lm-adapt/all/checkpoint-1500",
    f"{base_dir}/t5-xl-lm-adapt/all/checkpoint-1500",
]

model_paths = dict(zip(model_keys, model_paths))
model_names = dict(zip(model_keys, model_names))
model_device = dict(zip(model_keys, device_map))


tokenizer = T5Tokenizer.from_pretrained(model_paths[model_keys[0]])

task_name = "mnli"
label_tokens = [
    tokenizer(label, max_length=2).input_ids[0]
    for label in TASK_TO_LABELS[task_name]
    if label is not None
]


def model_inference(model, batch, temperature=None, device="cuda:0"):
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    outputs = model((input_ids, attention_mask))
    logits = outputs.squeeze(1)[:, label_tokens]
    if temperature is not None:
        logits = temperature_scale(logits, temperature)
    return logits


with open("tests/kernel_duration/latency.json", "r") as fp:
    model_latency = json.load(fp)
with open("repository/repo_t5/meta.json", "r") as fp:
    model_meta = json.load(fp)


preprocess_function = partial(
    t5_preprocess_function, tokenizer=tokenizer, padding="max_length", max_length=128,
)

data_collator = DataCollatorForSeq2Seq(tokenizer)
dataset = concatenate_datasets(
    [load_glue_val(preprocess_function), load_glue_train(preprocess_function)]
).shuffle()
dataset = dataset.select(range(10000))

dataloader = DataLoader(
    dataset, shuffle=False, collate_fn=data_collator, batch_size=32, drop_last=True,
)

models = load_models(
    model_keys, model_paths, model_device, T5ForConditionalGeneration, T5PyTorchPipe,
)

n_models = len(model_keys)
model_probs = dict(zip(model_keys, [list() for _ in range(n_models)]))
model_ans = {}
model_outputs = dict(zip(model_keys, [list() for _ in range(n_models)]))

m = torch.nn.Softmax(dim=-1)

from hfutils.calibration import agg_logits

num_labels = 0
labels = []
for batch in tqdm(dataloader, desc="Collect Train Data"):
    label = torch.Tensor(token2label(batch["labels"][:, 0], label_tokens))
    num_labels += len(label)

    # hist_logits = None
    for i, key in enumerate(model_keys):
        logits = model_inference(
            models[key],
            batch,
            device=model_device[key],
            temperature=model_meta[model_names[key]]["temperature"],
        )
        # logits = agg_logits(hist_logits, logits, 0.6)
        model_outputs[key].append(logits)
    labels.append(label)

model_probs, model_ans, model_outputs, labels = postprocessing_inference(
    model_keys, model_outputs, labels, m, alpha=1.0
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
    "t5",
)
