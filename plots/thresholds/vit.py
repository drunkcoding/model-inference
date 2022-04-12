from dataclasses import dataclass, field
from functools import partial
import itertools
import json
import logging
import os
import time
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import torch
from transformers import AutoModelForImageClassification, ViTForImageClassification
from torchvision.datasets import ImageNet
import datasets
from hfutils.preprocess import (
    split_train_test,
    vit_collate_fn,
    ViTFeatureExtractorTransforms,
)

import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from hfutils.logger import Logger
from hfutils.pipe.vit import ViTPyTorchPipeForImageClassification
from hfutils.calibration import temperature_scale

import sys
sys.path.append(".")
from plots.thresholds.utils import *

home_dir = "/mnt/raid0nvme1"
dataset_path = os.path.join(home_dir, "ImageNet")

model_keys = [
    "XS",
    "S",
    "M",
    "L",
]

model_names = [
    "vit-tiny-patch16-224",
    "vit-small-patch16-224",
    "vit-base-patch16-224",
    "vit-large-patch16-224",
]

device_map = [
    "cuda:4",
    "cuda:4",
    "cuda:4",
    "cuda:4",
]

model_paths = [
    f"{home_dir}/HuggingFace/WinKawaks/vit-tiny-patch16-224",
    f"{home_dir}/HuggingFace/WinKawaks/vit-small-patch16-224",
    f"{home_dir}/HuggingFace/google/vit-base-patch16-224",
    f"{home_dir}/HuggingFace/google/vit-large-patch16-224",
]


model_paths = dict(zip(model_keys, model_paths))
model_names = dict(zip(model_keys, model_names))
model_device = dict(zip(model_keys, device_map))


def model_inference(model, batch, temperature=None, device="cuda:0"):
    pixel_values = batch["pixel_values"].to(device)
    logits = model((pixel_values,))
    if temperature is not None:
        logits = temperature_scale(logits, temperature)
    return logits


with open("tests/kernel_duration/latency.json", "r") as fp:
    model_latency = json.load(fp)
with open("repository/repo_vit/meta.json", "r") as fp:
    model_meta = json.load(fp)

dataset = ImageNet(
    dataset_path,
    split="train",
    transform=ViTFeatureExtractorTransforms(model_paths[model_keys[0]], split="val"),
)
dataset, _ = split_train_test(dataset, 0.98)
num_labels = len(dataset)

dataloader = DataLoader(
    dataset, shuffle=True, collate_fn=vit_collate_fn, batch_size=32, drop_last=True,
)

models = load_models(
    model_keys,
    model_paths,
    model_device,
    ViTForImageClassification,
    ViTPyTorchPipeForImageClassification,
)

n_models = len(model_keys)
model_outputs = dict(zip(model_keys, [list() for _ in range(n_models)]))

m = torch.nn.Softmax(dim=-1)


labels = []
for batch in tqdm(dataloader, desc="Collect Train Data"):
    label = batch["labels"]
    for i, key in enumerate(model_keys):
        logits = model_inference(
            models[key],
            batch,
            device=model_device[key],
            temperature=model_meta[model_names[key]]["temperature"],
        )
        model_outputs[key].append(logits)
    labels.append(label)


model_probs, model_ans, model_outputs, labels = postprocessing_inference(
    model_keys, model_outputs, labels, m
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
    "vit",
)
