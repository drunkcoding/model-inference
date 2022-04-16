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
    BertForQuestionAnswering,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    default_data_collator,
)
import pandas as pd
from datasets import load_dataset, concatenate_datasets
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from hfutils.logger import Logger
from hfutils.pipe.bert import BertPyTorchPipeForQuestionAnswering
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
    # "M",
    "L",
    "XL",
]

model_names = [
    # "bert-tiny-5-finetuned-squadv2",
    # "bert-mini-5-finetuned-squadv2",
    # "bert-small-2-finetuned-squadv2",
    "bert-base-uncased",
    "bert-large-uncased",
]

device_map = [
    # "cuda:6",
    # "cuda:6",
    # "cuda:6",
    "cuda:6",
    "cuda:6",
]

model_paths = [
    # f"{home_dir}/HuggingFace/mrm8488/bert-tiny-5-finetuned-squadv2",
    # f"{home_dir}/HuggingFace/mrm8488/bert-mini-5-finetuned-squadv2",
    # f"{home_dir}/HuggingFace/mrm8488/bert-small-2-finetuned-squadv2",
    f"{home_dir}/HuggingFace/twmkn9/bert-base-uncased-squad2",
    f"{home_dir}/HuggingFace/madlag/bert-large-uncased-squadv2",
]
tokenizer = AutoTokenizer.from_pretrained(
    f"{home_dir}/HuggingFace/bert-base-uncased", use_fast=True,
)

val_dataset = concatenate_datasets(
    [load_dataset("squad_v2", split="validation"),load_dataset("squad_v2", split="train")]
).shuffle()
val_dataset = val_dataset.select(range(10000))
column_names = val_dataset.column_names

dataset = val_dataset.map(
    functools.partial(
        prepare_train_features, column_names=column_names, tokenizer=tokenizer
    ),
    batched=True,
    num_proc=10,
    remove_columns=column_names,
    desc="Running tokenizer on training dataset",
)

dataloader = DataLoader(
    dataset,
    shuffle=False,
    collate_fn=default_data_collator,
    batch_size=16,
    drop_last=True,
)


model_paths = dict(zip(model_keys, model_paths))
model_names = dict(zip(model_keys, model_names))
model_device = dict(zip(model_keys, device_map))


@torch.no_grad()
def model_inference(model, batch, temperature=None, device="cuda:0"):
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    token_type_ids = batch["token_type_ids"].to(device)
    logits = model((input_ids, token_type_ids, attention_mask))
    if temperature is not None:
        logits = temperature_scale(logits, temperature)
    start_logits, end_logits = logits.split(1, dim=-1)
    start_logits = start_logits.squeeze(-1).contiguous()
    end_logits = end_logits.squeeze(-1).contiguous()
    return start_logits, end_logits


with open("tests/kernel_duration/latency.json", "r") as fp:
    model_latency = json.load(fp)
with open("repository/repo_bert/meta.json", "r") as fp:
    model_meta = json.load(fp)

models = load_models(
    model_keys,
    model_paths,
    model_device,
    BertForQuestionAnswering,
    BertPyTorchPipeForQuestionAnswering,
)

n_models = len(model_keys)
model_probs = dict(zip(model_keys, [list() for _ in range(n_models)]))
model_ans = {}
model_outputs = dict(zip(model_keys, [list() for _ in range(n_models)]))

m = torch.nn.Softmax(dim=-1)

num_labels = len(dataset)
labels = []
for batch in tqdm(dataloader, desc="Collect Train Data"):
    start_positions = batch["start_positions"].flatten()
    end_positions = batch["end_positions"].flatten()

    for i, key in enumerate(model_keys):
        start_logits, end_logits = model_inference(
            models[key],
            batch,
            device=model_device[key],
            temperature=model_meta[model_names[key]]["temperature"],
        )
        model_outputs[key].append(torch.stack((start_logits, end_logits), dim=1))
    ignored_index = start_logits.size(1)
    start_positions = start_positions.clamp(0, ignored_index)
    end_positions = end_positions.clamp(0, ignored_index)
    labels.append(torch.stack((start_positions, end_positions), dim=1))


def process_func(logits):
    probs = m(logits)
    if torch.min(probs[:, 0]) < torch.min(probs[:, 1]):
        return probs[:, 0]
    return probs[:, 1]


model_probs, model_ans, model_outputs, labels = postprocessing_inference(
    model_keys, model_outputs, labels, process_func
)

all_thresholds = list(
    itertools.product(np.linspace(0, 1, endpoint=True, num=1000), repeat=n_models - 1)
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
    "bert-2-train",
)
