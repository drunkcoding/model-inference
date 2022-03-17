import copy
import functools
import gc
from hfutils.constants import TASK_TO_LABELS
from seaborn.distributions import histplot
import torch
import logging
import numpy as np
from transformers.data.data_collator import (
    DataCollatorForSeq2Seq,
    default_data_collator,
)
from transformers import T5ForConditionalGeneration, T5Tokenizer
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from scipy.optimize import minimize
from sklearn.mixture import GaussianMixture

import os
import sys

from torch.nn.modules.activation import Threshold

from datasets import Dataset, concatenate_datasets
from datasets import load_dataset, load_metric

from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, T5ForConditionalGeneration
from torch.utils.data import (
    DataLoader,
    RandomSampler,
    SequentialSampler,
    TensorDataset,
    dataloader,
)
from sklearn.model_selection import train_test_split

from hfutils.logger import Logger
from hfutils.constants import token2label
from hfutils.arg_parser import HfArguments
from hfutils.loader import ModelLoader, DatasetLoader
from hfutils.temperature_scaling import ModelWithTemperature
from hfutils.monte_carlo import monte_carlo_bounds
from hfutils.calibration import agg_logits, g_scaling_helper, temperature_scaling_helper, temperature_scale, temperature_scaling

home_dir = os.path.expanduser(("~"))
base_dir = os.path.join(home_dir, os.path.join("model-finetune", "outputs", "google"))

task_name = "mnli" # HACK the longest number of labels

tokenizer = T5Tokenizer.from_pretrained(f"{home_dir}/HuggingFace/google/t5-small-lm-adapt", use_fast=False)

label_tokens = [
    tokenizer(label, max_length=2, truncation=True).input_ids[0]
    for label in TASK_TO_LABELS[task_name]
    if label is not None
]

model_keys = [
    "S",
    "M",
    "L",
    "XL",
]

device_map = [
    "cuda:0",
    "cuda:0",
    "cuda:0",
    "cuda:1",
]

energy_discount_factor = [
    1 / 40,
    3 / 40,
    10 / 40,
    40 / 40,
]

model_paths = [
    f"{base_dir}/t5-small-lm-adapt/all/checkpoint-4500",
    f"{base_dir}/t5-base-lm-adapt/all/checkpoint-2000",
    f"{base_dir}/t5-large-lm-adapt/all/checkpoint-1500",
    f"{base_dir}/t5-xl-lm-adapt/all/checkpoint-1500",
]

model_energy = dict(zip(model_keys, energy_discount_factor))
model_paths = dict(zip(model_keys, model_paths))
model_device = dict(zip(model_keys, device_map))

logger = Logger(__file__, "info", 5000000, 5)

models = dict()
for key in model_paths:
    logger.debug("key %s, path %s, device %s", key, model_paths[key], model_device[key])
    models[key] = T5ForConditionalGeneration.from_pretrained(model_paths[key])
    models[key] = models[key].to(model_device[key])
    models[key].eval()

    torch.cuda.empty_cache()
    gc.collect()

logger.info("model loaded")

# -------------  Dataset Prepare --------------

from hfutils.loader import t5_preprocess_function, load_glue_val
from functools import partial

preprocess_function = partial(
    t5_preprocess_function, 
    tokenizer=tokenizer,
    padding="max_length",
    max_length=128,
)
eval_dataset = load_glue_val(preprocess_function).shuffle()
# eval_dataset = eval_dataset.select([x for x in range(1000)])
data_collator = DataCollatorForSeq2Seq(tokenizer)

# train_len = int(len(eval_dataset) * 0.4)

split_dataset = eval_dataset.train_test_split(train_size=0.4)
train, test = split_dataset["train"], split_dataset["test"]
# train_raw, test_raw = split_dataset["train"], split_dataset["test"]
print(train, test)

train_len = len(train)
test_len = len(test)

train_dataloader = DataLoader(
    train,
    collate_fn=data_collator,
    batch_size=16,
)

test_dataloader = DataLoader(
    test,
    collate_fn=data_collator,
    batch_size=16,
)

m = torch.nn.Softmax(dim=1)
logger.info("data loaded")

# -------------  Train Temperature --------------

print("temperature loaded")

n_models = len(model_keys)
num_labels = 0

def model_inference(model, batch, temperature=None, device="cuda:0"):
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        do_sample=False,  # disable sampling to test if batching affects output
        return_dict_in_generate=True,
        output_scores=True,
    )
    logits = outputs.scores[0][:, label_tokens]
    if temperature is not None:
        logits = temperature(logits)
    return logits

# ============= COLLECT TRAIN LOGITS =================
# import torch.multiprocessing as mp


labels_list = []
for batch in tqdm(train_dataloader, desc="Collect Labels"):
    label = token2label(batch["labels"][:, 0], label_tokens)
    labels_list += label
labels = torch.as_tensor(labels_list, dtype=torch.int64)

# def run_train(key, model_outputs):
#     all_logits = []
#     # labels_list = []
#     for batch in tqdm(train_dataloader, desc="Collect Train Data"):
#         # label = token2label(batch["labels"][:, 0], label_tokens)
#         logits = model_inference(models[key], batch, device=model_device[key])
#         all_logits.append(logits)
#         # if len(labels_list) < train_len:
#         #     labels_list += label
#     all_logits = torch.cat(all_logits)
#     model_outputs[key] = all_logits

#     # labels = torch.as_tensor(labels_list, dtype=torch.int64)
#     # return labels
# manager = mp.Manager()
# model_outputs = manager.dict()
# if __name__ == '__main__':
#     mp.set_start_method('spawn') 
#     pool = mp.Pool(len(model_keys))
#     pool.map(functools.partial(run_train, model_outputs=model_outputs), model_keys)

model_outputs = {}
for key in model_keys:
    all_logits = []
    # labels_list = []
    for batch in tqdm(train_dataloader, desc=f"Collect Train Data {key}"):
        # label = token2label(batch["labels"][:, 0], label_tokens)
        logits = model_inference(models[key], batch, device=model_device[key])
        all_logits.append(logits)
        # if len(labels_list) < train_len:
        #     labels_list += label
    all_logits = torch.cat(all_logits)
    model_outputs[key] = all_logits
    
# labels = torch.as_tensor(labels_list, dtype=torch.int64)

# =============  TRAIN TEMPERATURE =============
epoches = [
    500,
    500,
    500,
    500
]
model_epoches = dict(zip(model_keys, epoches))
model_temperature = g_scaling_helper(model_outputs, labels, model_epoches, len(label_tokens))
print("temperature", model_temperature)

for key in model_keys:
    model_outputs[key] = model_temperature[key](model_outputs[key])
    torch.save(model_temperature[key].state_dict(), os.path.join("tests", "confidence", f"t5_glue_glayer-{key}"))
# =============  TRAIN HYPERPARAMETER =============

num_models = len(model_keys)
m = torch.nn.Softmax(dim=1)

# hist_probs = []
# hist_logits = None
# for i, key in enumerate(model_keys):
#     hist_logits = agg_logits(
#         hist_logits if key != model_keys[-1] else None,
#         model_outputs[key],
#         0.6
#     )
#     probs, _ = torch.max(m(hist_logits), dim=1)
#     probs = probs.detach().cpu().numpy()
#     hist_probs.append(probs)

def total_reward(threshold):
    reward = 0
    energy = 0
    mask = np.array([False] * train_len)

    alpha = threshold[-1]
    threshold = threshold[:-1]

    hist_logits = None
    for i, key in enumerate(model_keys):
        hist_logits = agg_logits(
            hist_logits if key != model_keys[-1] else None,
            model_outputs[key],
            alpha
        )
        probs, _ = torch.max(m(hist_logits), dim=1)
        probs = probs.detach().cpu().numpy()
        # probs = hist_probs[i]
        processed = (
            (probs >= threshold[i])
            if key in model_keys[:-1]
            else np.array([True] * train_len)
        )
        # print(mask, processed)
        processed_probs = probs[(~mask) & processed]
        reward += np.around(np.sum(processed_probs) / 8.0) * 8
        energy += model_energy[key] * np.count_nonzero(~mask) 
        mask |= processed

    return (reward, -energy)

threshold_bounds = monte_carlo_bounds(
    total_reward,
    [(0.25, 1.0)] * (num_models),
    [("reward", float), ("energy", float)],
    n=1000,
    tops=40,
    maxiter=30,
)
mc_threshold = np.mean(threshold_bounds, axis=1)
alpha = mc_threshold[-1]
mc_threshold = mc_threshold[:-1]
logger.info("Threshold Bounds %s", threshold_bounds)
logger.info("Final Thresholds %s", mc_threshold)
logger.info("Alpha %s", alpha)

# -------------  Evaluation WITH Temperature --------------
model_outputs = {}

def compute_metric(logits, labels):
    metric = load_metric("accuracy")
    predictions = torch.argmax(logits, dim=-1).detach().cpu().numpy().flatten()
    metric.add_batch(
        predictions=predictions,
        references=labels
    )
    return metric.compute()

for key in model_keys:
    all_logits = []
    labels_list = []

    metric = load_metric("accuracy")
    for batch in tqdm(test_dataloader, desc=f"Individual Accuracy {key}"):
        label = token2label(batch["labels"][:, 0], label_tokens)
        logits = model_inference(models[key], batch, temperature=model_temperature[key], device=model_device[key])
        all_logits.append(logits)
        labels_list += label

    all_logits = torch.cat(all_logits)
    labels = torch.as_tensor(labels_list, dtype=torch.int64)

    model_outputs[key] = all_logits

    logger.info("indv %s %s", key, compute_metric(all_logits, labels))

mask = np.array([False] * test_len)
final_logits = torch.zeros((test_len, 3)).to("cuda")
hist_logits = None
for i, key in enumerate(model_keys):
    hist_logits = agg_logits(
        hist_logits if key != model_keys[-1] else None,
        model_outputs[key],
        alpha
    )
    print(final_logits.shape, hist_logits.shape)
    assert final_logits.shape == hist_logits.shape

    probs, _ = torch.max(m(hist_logits), dim=1)
    probs = probs.detach().cpu().numpy()
    processed = (
        (probs >= mc_threshold[i])
        if key in model_keys[:-1]
        else np.array([True] * test_len)
    )

    print(mask.shape, processed.shape, hist_logits.shape)

    delegated_logit = hist_logits[(~mask) & processed]

    logger.info(
        "%s process count (%s) %s",
        key, test_len,
        np.count_nonzero((~mask) & processed),
    )

    final_logits[(~mask) & processed] = delegated_logit.to(final_logits.device)
    mask |= processed

logger.info("***** Collaborative Eval results *****")
logger.info(
    "Collaborative metrics %s",
    compute_metric(final_logits, labels)
)