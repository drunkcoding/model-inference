# Copyright 2020-2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import copy
import json
import os
from hfutils.arg_parser import HfArguments
from hfutils.constants import TASK_TO_LABELS
from hfutils.loader import DatasetLoader, ModelLoader
from numpy import random
import torch

from datasets import Dataset
from transformers.data.data_collator import DataCollatorForSeq2Seq
from tritonclient.utils import *
import tritonclient.http as httpclient

# import tritonclient.grpc as httpclient
import sys
import numpy as np
from tqdm import tqdm
import argparse
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.special import softmax
import time

from transformers import (
    AdamW,
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    PretrainedConfig,
    SchedulerType,
    default_data_collator,
    get_scheduler,
    set_seed,
)

import logging

from hfutils.calibration import temperature_scale, temperature_scaling, agg_logits
from hfutils.monte_carlo import monte_carlo_bounds
from hfutils.logger import Logger

logger = Logger(__file__, "info", 0, 0)

import datasets
from datasets import load_dataset, load_metric
from torch.utils.data import DataLoader

args = HfArguments()

tokenizer, _ = ModelLoader(args).load(load_model=False)
dataset_loader = DatasetLoader(args)
eval_dataset = dataset_loader.load(
    tokenizer, partition="validation", create_dataloader=False
)
logger.info("eval_dataset %s", eval_dataset)

data_args = args.data_args

if data_args.pad_to_max_length:
    data_collator = default_data_collator
else:
    data_collator = DataCollatorForSeq2Seq(tokenizer)

train_len = int(len(eval_dataset) * 0.4)
train = Dataset.from_dict(eval_dataset[:train_len])
test = Dataset.from_dict(eval_dataset[train_len:])

eval_dataloader = DataLoader(
    eval_dataset,
    shuffle=True,
    collate_fn=data_collator,
    batch_size=data_args.eval_bsz,
)

train_dataloader = DataLoader(
    train,
    shuffle=True,
    collate_fn=data_collator,
    batch_size=data_args.eval_bsz,
)

test_dataloader = DataLoader(
    test,
    shuffle=True,
    collate_fn=data_collator,
    batch_size=data_args.eval_bsz,
)

model_keys = [
    "S",
    "M",
    "L",
    "XL",
]

energy_discount_factor = [
    1 / 40,
    3 / 40,
    10 / 40,
    40 / 40,
]

remote_keys = [
    f"t5-small-lm-adapt_{data_args.task_name}",
    f"t5-base-lm-adapt_{data_args.task_name}",
    f"t5-large-lm-adapt_{data_args.task_name}",
    f"t5-xl-lm-adapt_{data_args.task_name}",
]

model_energy = dict(zip(model_keys, energy_discount_factor))
models = dict(zip(model_keys, remote_keys))

label_tokens = [
    tokenizer(label, max_length=2).input_ids[0]
    for label in TASK_TO_LABELS[data_args.task_name]
    if label is not None
]


def prepare_query(batch, pos):
    input_ids = batch["input_ids"].numpy()
    attention_mask = batch["attention_mask"].numpy()
    ensemble_outputs = np.ones((input_ids.shape[0], 2), dtype=np.float32) * -100
    batch_mask = np.zeros((4,input_ids.shape[0]))
    batch_mask[pos] = np.ones(input_ids.shape[0]) # WHERE TO ENTER
    batch_mask = batch_mask.astype(bool)

    inputs = [
        httpclient.InferInput(
            "input_ids", input_ids.shape, np_to_triton_dtype(input_ids.dtype)
        ),
        httpclient.InferInput(
            "attention_mask",
            attention_mask.shape,
            np_to_triton_dtype(attention_mask.dtype),
        ),
        httpclient.InferInput(
            "ensemble_outputs",
            ensemble_outputs.shape,
            np_to_triton_dtype(ensemble_outputs.dtype),
        ),
        httpclient.InferInput(
            "batch_mask",
            batch_mask.shape,
            np_to_triton_dtype(batch_mask.dtype),
        ),
    ]

    inputs[0].set_data_from_numpy(input_ids)
    inputs[1].set_data_from_numpy(attention_mask)
    inputs[2].set_data_from_numpy(ensemble_outputs)
    inputs[3].set_data_from_numpy(batch_mask)

    outputs = [
        httpclient.InferRequestedOutput("outputs"),
    ]

    return inputs, outputs


def model_inference(client, model_name, batch):
    inputs, outputs = batch
    response = client.infer(model_name, inputs, request_id=str(1), outputs=outputs)
    logits = response.as_numpy("outputs")
    return logits


# model_keys = [
#     f"t5_small_lm_adapt_glue_{args.task_name}",
#     f"t5_large_lm_adapt_glue_{args.task_name}",
# ]
# model_energy = [0.1, 1]
meta = {}
meta_path = os.path.join(
    "/sata_disk/jupyter-xue/model-inference/repository", "meta.json"
)

n_models = len(model_keys)

labels_list = []

m = torch.nn.Softmax(dim=-1)
device = "cuda:0"


# def agg_logits(hist, curr, pos, device):
#     if hist is not None:
#         hist = hist.to(device)
#         return (hist * pos + curr) / (pos+1)
#     return curr

model_probs = dict(zip(model_keys, [list() for _ in range(n_models)]))
model_outputs = dict(zip(model_keys, [list() for _ in range(n_models)]))
model_latency = dict(zip(model_keys, [list() for _ in range(n_models)]))

HOST = "localhost:8000"
PARALLEL = 1

# ============ RESET MODELS ==================

with open(meta_path, "r") as fp:
    meta = json.load(fp)

for pos, key in enumerate(model_keys):
    meta[models[key]] = {
        "threshold": 0.0,
        "temperature": 1.0,
        "ensemble_pos": pos,
    }

with open(meta_path, "w") as fp:
    json.dump(meta, fp)
    time.sleep(12)


num_train_labels = train_len
num_test_labels = len(eval_dataset) - train_len
model_metrics = {}
for key in model_keys:
    model_metrics[key] = load_metric(
        args.data_args.dataset_name, args.data_args.task_name
    )

correct_cnt = dict(zip(model_keys, [0] * n_models))
correct_prob = dict(zip(model_keys, [0] * n_models))

model_metrics = {}
for key in model_keys:
    model_metrics[key] = load_metric(
        args.data_args.dataset_name, args.data_args.task_name
    )

with httpclient.InferenceServerClient(HOST, concurrency=PARALLEL) as client:
    for step, batch in tqdm(
        enumerate(train_dataloader), desc="Get Temperature Train Logits"
    ):
        # input_ids=batch['input_ids']
        # attention_mask=batch['attention_mask']
        label = batch["labels"][:, 0] == label_tokens[-1]
        # label = label.to(torch.int64)

        labels_list.append(label)

        true_ans = label.cpu().detach().numpy().flatten()

        for i, key in enumerate(model_keys):
            triton_batch = prepare_query(batch, i)
            logits = model_inference(client, models[key], triton_batch)
            model_ans = np.argmax(logits, axis=-1)
            logits = torch.Tensor(logits).to(device)
            # print(key, logits)
            model_outputs[key].append(logits)

            model_metrics[key].add_batch(
                predictions=model_ans,
                references=true_ans,
            )
    for key in model_keys:
        logger.info("%s train metrics %s", key, model_metrics[key].compute())

    for step, batch in tqdm(
        enumerate(test_dataloader), desc="Eval without Temperature"
    ):
        label = batch["labels"][:, 0] == label_tokens[-1]
        true_ans = label.cpu().detach().numpy().flatten()

        for i, key in enumerate(model_keys):
            triton_batch = prepare_query(batch, i)
            start_time = time.perf_counter()
            logits = model_inference(client, models[key], triton_batch)
            end_time = time.perf_counter()

            model_ans = np.argmax(logits, axis=-1)

            model_metrics[key].add_batch(
                predictions=model_ans,
                references=true_ans,
            )

            model_latency[key].append((end_time - start_time) * 1000)

            probabilities = softmax(logits, -1)
            correct_prob[key] += np.sum(np.max(probabilities, axis=-1))
            correct_cnt[key] += np.count_nonzero(model_ans == true_ans)

from scipy import stats

for key in model_keys:
    logger.info(
        "%s latency summary %s \n %s",
        key,
        models[key],
        stats.describe(model_latency[key]),
    )
    sns.distplot(
        model_latency[key],
        hist=True,
        kde=True,
        bins=int(180 / 5),
        label=key,
        hist_kws={"edgecolor": "black"},
        kde_kws={"linewidth": 4},
    )
plt.legend()
plt.savefig(f"figures/model_latency_{data_args.task_name}.png", bbox_inches="tight")
plt.close()

labels = torch.cat(labels_list).to(device)
labels = labels.to(torch.int64)

model_temperature = {}

for key in model_keys:
    # model_probs[key] = np.array(model_probs[key])
    model_outputs[key] = torch.cat(model_outputs[key]).to(device)
    # labels = labels.to(device)

    temperature = (
        temperature_scaling(model_outputs[key], labels)
        .detach()
        .cpu()
        .numpy()
        .tolist()[0]
    )
    # bar = 1.5
    # temperature = bar + (temperature - bar) / 2 if temperature > bar else temperature
    meta[models[key]]["temperature"] = temperature
    model_temperature[key] = torch.nn.Parameter(
        torch.ones(1, device=device) * temperature
    )

hist_logits = None
for i, key in enumerate(model_keys):
    model_outputs[key] = temperature_scale(model_outputs[key], model_temperature[key])
    hist_logits = agg_logits(
        hist_logits if key != model_keys[-1] else None, model_outputs[key], i, device
    )
    # hist_logits = agg_logits(None, model_outputs[key], i, model_device[key])
    probabilities = torch.float_power(m(hist_logits).to(device), 2)
    model_ans = torch.argmax(probabilities, dim=-1).flatten()

    model_ans = model_ans.detach().cpu().numpy()
    probabilities = probabilities.detach().cpu().numpy()
    temp_labels = labels.detach().cpu().numpy()

    model_probs[key] = np.array(
        [
            [p[model_ans[i]], int(model_ans[i] == temp_labels[i])]
            for i, p in enumerate(probabilities)
        ]
    )
    # model_temperature[key] = torch.nn.Parameter(torch.ones(1, device=device) * 1.15)

logger.info("model_temperature %s", model_temperature)

# max_reward = 0
# min_energy = 1e10
# mc_threshold = []
# for th_s in np.arange(0.5, 1.0, 0.01):
#     for th_m in np.arange(0.5, 1.0, 0.01):
#         for th_l in np.arange(0.5, 1.0, 0.01):
#             threshold = [th_s, th_m, th_l]
#             mask = np.array([False] * num_train_labels)
#             reward = 0
#             energy = 0
#             for i, key in enumerate(model_keys):
#                 processed = (
#                     (model_probs[key][:, 0] >= threshold[i])
#                     if key in model_keys[:-1]
#                     else np.array([True] * num_train_labels)
#                 )
#                 # reward += np.sum(model_probs[key][(~mask) & processed, 1])
#                 reward += np.around(np.sum(model_probs[key][(~mask) & processed, 0]) / 8.0) * 8
#                 energy += model_energy[key] * np.count_nonzero(
#                     ~mask
#                     )
#                 mask |= processed
#             if reward > max_reward or (reward == max_reward and min_energy > energy):
#                 mc_threshold = copy.deepcopy(threshold)
#                 max_reward = reward
#                 min_energy = energy
#                 print(mc_threshold, max_reward, min_energy)


def total_reward(threshold):
    reward = 0
    energy = 0
    mask = np.array([False] * num_train_labels)
    for i, key in enumerate(model_keys):
        processed = (
            (model_probs[key][:, 0] >= threshold[i])
            if key in model_keys[:-1]
            else np.array([True] * num_train_labels)
        )
        reward += np.around(np.sum(model_probs[key][(~mask) & processed, 0]) / 8.0) * 8
        # reward += np.sum(model_probs[key][(~mask) & processed, 1])
        energy += model_energy[key] * np.count_nonzero(
            ~mask
        )  # np.count_nonzero((~mask) & processed)
        mask |= processed
    # print((reward, -energy))
    return (reward, -energy)


threshold_bounds = monte_carlo_bounds(
    total_reward,
    [(0.5, 1.0)] * (n_models - 1),
    [("reward", float), ("energy", float)],
    n=10000,
    tops=40,
    maxiter=30,
)
mc_threshold = np.mean(threshold_bounds, axis=1)
logger.info("Threshold Bounds %s", threshold_bounds)

logger.info("  Num examples = %s", num_test_labels)
logger.info("  Threshold = %s", mc_threshold)
# for key in model_keys:
#     logger.info("final temperature %s", models[key].temperature)
logger.info("***** Eval results *****")
for key in model_keys:
    logger.info(
        "%s correct count %s, percent %s, prob %s",
        key,
        correct_cnt[key],
        np.around(correct_cnt[key] / float(num_test_labels) * 100, 3),
        correct_prob[key],
    )
    logger.info("%s metrics %s", key, model_metrics[key].compute())

for i, key in enumerate(model_keys):
    meta[models[key]]["threshold"] = mc_threshold[i] if key in model_keys[:-1] else 0.0

with open(meta_path, "w") as fp:
    json.dump(meta, fp)
    time.sleep(12)

# -------------  Evaluation WITH Temperature --------------

total_metrics = load_metric(args.data_args.dataset_name, args.data_args.task_name)
total_accuracy = load_metric("accuracy")
total_time = []
with httpclient.InferenceServerClient(HOST, concurrency=PARALLEL) as client:
    for step, batch in tqdm(enumerate(test_dataloader), desc="Testing Accuracy"):
        label = (batch["labels"][:, 0] == label_tokens[-1]).to(torch.int64)
        triton_batch = prepare_query(batch, 0)
        start_time = time.perf_counter()
        logits = model_inference(client, models[model_keys[0]], triton_batch)
        end_time = time.perf_counter()
        logits = torch.Tensor(logits).to(device)
        probabilities = np.power(m(logits).cpu().detach().numpy(), 2)

        if step > 5:
            total_time.append((end_time - start_time) * 1000)

        model_ans = np.argmax(probabilities, axis=-1)
        true_ans = label.cpu().detach().numpy().flatten()

        total_metrics.add_batch(
            predictions=model_ans,
            references=true_ans,
        )
        total_accuracy.add_batch(
            predictions=model_ans,
            references=true_ans,
        )

logger.info(
    "total latency summary\n %s",
    stats.describe(total_time),
)
sns.distplot(
    total_time,
    hist=True,
    kde=True,
    bins=int(180 / 5),
    label="TOTAL",
    hist_kws={"edgecolor": "black"},
    kde_kws={"linewidth": 4},
)
plt.legend()
plt.savefig(f"figures/total_latency_{data_args.task_name}_skip.png", bbox_inches="tight")
plt.close()

logger.info("***** Collaborative Eval results *****")
logger.info("Collaborative metrics %s", total_metrics.compute())
logger.info("Collaborative accuracy %s", total_accuracy.compute())
# for key in model_keys:
#     logger.info(
#         "%s process count %s, correct count %s, percent %s, prob %s",
#         key,
#         process_cnt[key],
#         coop_cnt[key],
#         np.around(coop_cnt[key] / float(process_cnt[key]) * 100, 3)
#         if process_cnt[key] != 0
#         else 0,
#         process_prob[key],
#     )
