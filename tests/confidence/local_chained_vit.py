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
from transformers import AutoModelForImageClassification
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
from hfutils.arg_parser import HfArguments
from hfutils.loader import ModelLoader, DatasetLoader
from hfutils.temperature_scaling import ModelWithTemperature
from hfutils.monte_carlo import monte_carlo_bounds
from hfutils.calibration import temperature_scale, temperature_scaling


args = HfArguments()
data_args = args.data_args
task_name  = args.data_args.task_name
# print(pos_token, neg_token)
# print(tokenizer(list(TASK_TO_LABELS[task_name])).input_ids)
# exit()

home_dir = os.path.expanduser(("~"))
base_dir = os.path.join(home_dir, os.path.join("model-finetune", "outputs", "google"))

model_keys = [
    "XS",
    "S",
    "M",
    "L",
    # "XL",
]

device_map = [
    "cuda:0",
    "cuda:0",
    "cuda:0",
    "cuda:0",
    # "cuda:0",
]

energy_discount_factor = [
    0.3 / 40,
    1 / 40,
    3 / 40,
    10 / 40,
    # 40 / 40,
]

model_paths = [ 
     f"{home_dir}/HuggingFace/WinKawaks/vit-tiny-patch16-224",
     f"{home_dir}/HuggingFace/WinKawaks/vit-small-patch16-224",
    f"{home_dir}/HuggingFace/google/vit-base-patch16-224",
    f"{home_dir}/HuggingFace/google/vit-large-patch16-224",
    # f"{base_dir}/vit-huge-patch14-224-in21k/imagenet-fp16/checkpoint-4500",
]

model_energy = dict(zip(model_keys, energy_discount_factor))
model_paths = dict(zip(model_keys, model_paths))
model_device = dict(zip(model_keys, device_map))

logger = Logger(__file__, "info", 5000000, 5)

models = dict()
for key in model_paths:
    logger.debug("key %s, path %s, device %s", key, model_paths[key], model_device[key])
    models[key] = AutoModelForImageClassification.from_pretrained(
        model_paths[key]
    )  # if key != "S" else DistilBertForSequenceClassification.from_pretrained(model_paths[key])
    models[key] = models[key].to(model_device[key])
    models[key].eval()

    # if key == model_keys[-1]:
    #     models[key].half()

    torch.cuda.empty_cache()
    gc.collect()

logger.info("model loaded")

# -------------  Dataset Prepare --------------
from torchvision.datasets import ImageNet
import datasets
from hfutils.preprocess import split_train_test, vit_collate_fn, ViTFeatureExtractorTransforms
print("======ImageNet==========")

# Load the accuracy metric from the datasets package
metric = datasets.load_metric("accuracy")

home = os.path.expanduser('~')
dataset_path = os.path.join(home, "ImageNet")

model_dataset = {}
for key in model_keys:
    train, test = split_train_test(
        ImageNet(
            dataset_path, 
            download=False, 
            split="val", 
            transform=ViTFeatureExtractorTransforms(
                model_paths[key], 
                split="val"
            )
        ),
        0.4
    )

    num_labels = len(train) + len(test)

    train_dataloader = DataLoader(
        train,
        shuffle=False,
        collate_fn=vit_collate_fn,
        batch_size=data_args.train_bsz,
    )

    test_dataloader = DataLoader(
        test,
        shuffle=False,
        collate_fn=vit_collate_fn,
        batch_size=data_args.eval_bsz,
    )

    model_dataset[key] = (train_dataloader, test_dataloader)

m = torch.nn.Softmax(dim=1)
logger.info("data loaded")

# -------------  Train Temperature --------------

# for key in model_keys:
#     models[key] = ModelWithTemperature(models[key], tokenizer, model_device[key])
#     models[key].set_logger(logger)
#     models[key].set_temperature(train_dataloader)

print("temperature loaded")

n_models = len(model_keys)
num_labels = 0

@torch.no_grad()
def model_inference(model, batch, temperature=None, device="cuda:0"):
    pixel_values = batch["pixel_values"].to(device)
    outputs = model(pixel_values, return_dict=True)
    logits = outputs.logits
    if temperature is not None:
        logits = temperature_scale(logits, temperature)
    return logits

model_probs = dict(zip(model_keys, [list() for _ in range(n_models)]))
model_outputs = dict(zip(model_keys, [list() for _ in range(n_models)]))

def agg_logits(hist, curr, pos, device):
    # return curr
    alpha = 0.6
    if hist is not None:
        hist = hist.to(device)
        # return (hist * pos + curr) / (pos + 1)
        return hist * (1 - alpha) + curr * alpha
    return curr


for i, key in enumerate(model_keys):
    labels_list = []
    for batch in tqdm(model_dataset[key][0], desc=f"Collect Train Data {key}"):
        labels = batch["labels"]
        labels_list.append(labels)

        logits = model_inference(models[key], batch, device=model_device[key])
        model_outputs[key].append(logits)
    labels = torch.cat(labels_list)
    num_labels = len(labels)

model_temperature = {}

for key in model_keys:
    # model_probs[key] = np.array(model_probs[key])
    model_outputs[key] = torch.cat(model_outputs[key]).to(model_device[key])
    labels = labels.to(model_device[key])

    temperature = (
        temperature_scaling(model_outputs[key], labels)
        .detach()
        .cpu()
        .numpy()
        .tolist()[0]
    )
    # bar = 1.5
    # temperature = bar + (temperature - bar) / 2 if temperature > bar else temperature
    model_temperature[key] = torch.nn.Parameter(
        torch.ones(1, device=model_device[key]) * temperature
    )

hist_logits = None
for i, key in enumerate(model_keys):
    model_outputs[key] = temperature_scale(model_outputs[key], model_temperature[key])
    hist_logits = agg_logits(
        hist_logits if key != model_keys[-1] else None,
        model_outputs[key],
        i,
        model_device[key],
    )
    # hist_logits = agg_logits(None, model_outputs[key], i, model_device[key])
    probabilities = torch.float_power(m(hist_logits).to(model_device[key]), 2)
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

# print(model_probs[key])
logger.info("model_temperature %s", model_temperature)

def total_reward(threshold, model_keys):
    reward = 0
    energy = 0
    mask = np.array([False] * num_labels)

    # threshold = np.append(predetermined, threshold)
    # print(threshold)
    for i, key in enumerate(model_keys):
        # if i + 1 < len(predetermined): continue
        processed = (
            (model_probs[key][:, 0] >= threshold[i])
            if key in model_keys[:-1]
            else np.array([True] * num_labels)
        )
        processed_probs = model_probs[key][(~mask) & processed, 0]
        reward += np.around(np.sum(processed_probs) / 8.0) * 8
        # reward += np.sum(model_probs[key][(~mask) & processed, 1])
        energy += model_energy[key] * np.count_nonzero(
            ~mask
        )  # np.count_nonzero((~mask) & processed)
        mask |= processed
    # reward = reward - energy / np.sum(energy_discount_factor)
    return (reward, -energy)

threshold_bounds = monte_carlo_bounds(
    functools.partial(total_reward, model_keys=model_keys),
    [(0.25, 1.0)] * (n_models - 1),
    [("reward", float), ("energy", float)],
    n=10000,
    tops=40,
    maxiter=30,
)
mc_threshold = np.mean(threshold_bounds, axis=1)
logger.info("Threshold Bounds %s", threshold_bounds)

# -------------  Evaluation WITH Temperature --------------

correct_cnt = dict(zip(model_keys, [0] * n_models))
correct_prob = dict(zip(model_keys, [0] * n_models))
coop_cnt = dict(zip(model_keys, [0] * n_models))
process_prob = dict(zip(model_keys, [0] * n_models))
process_cnt = dict(zip(model_keys, [0] * n_models))

num_labels = 0


model_metrics = {}
for key in model_keys:
    # model_metrics[key] = load_metric(
    #     args.data_args.dataset_name, args.data_args.task_name
    # )
    model_metrics[key] = load_metric("accuracy")

model_labels = dict(zip(model_keys, [list() for _ in range(n_models)]))
model_prob = dict(zip(model_keys, [list() for _ in range(n_models)]))
th_stats = dict(zip(model_keys, [list() for _ in range(n_models)]))

hist_logits = None
for i, key in enumerate(model_keys):
    labels_list = []
    logits_list = []
    for batch in tqdm(model_dataset[key][1], desc=f"Collect Test Data {key}"):
        labels = batch["labels"].int().flatten()
        labels_list.append(labels)

        logits = model_inference(models[key], batch, device=model_device[key])
        logits = logits.cpu().detach()
        logits_list.append(logits)
    
    labels = torch.cat(labels_list).numpy()
    logits = torch.cat(logits_list)

    hist_logits = agg_logits(
        hist_logits if key != model_keys[-1] else None,
        logits,
        i,
        "cpu",
    )

    model_labels[key] = torch.argmax(hist_logits, -1).flatten().numpy()
    model_prob[key] = m(hist_logits).numpy()

# print(model_labels["M"])
# print(model_labels["L"])
# for key in model_keys:
#     model_labels[key] = torch.cat(model_labels[key]).numpy()
#     model_prob[key] = torch.cat(model_prob[key]).numpy()
# print(model_prob[key])

total_metrics = load_metric("accuracy")

b_size = len(labels)
mask = np.array([False] * b_size)
hist_logits = None

num_labels = b_size

for i, key in enumerate(model_keys):

    model_metrics[key].add_batch(
        predictions=model_labels[key],
        references=labels,
    )

    th_stats[key] += np.max(model_prob[key], axis=-1).tolist()

    selected_prob = np.max(model_prob[key], axis=-1)
    processed = (
        (selected_prob >= mc_threshold[i])
        if key in model_keys[:-1]
        else np.array([True] * b_size)
    )

    total_metrics.add_batch(
        predictions=model_labels[key][(~mask) & processed],
        references=labels[(~mask) & processed],
    )

    # print(model_labels[key], labels)

    correct_prob[key] += np.sum(selected_prob)
    correct_cnt[key] += np.count_nonzero(model_labels[key] == labels)

    process_prob[key] += np.sum(selected_prob[(~mask) & processed])
    coop_cnt[key] += np.count_nonzero(
        (model_labels[key] == labels) & (~mask) & processed
    )
    process_cnt[key] += np.count_nonzero((~mask) & processed)
    mask |= processed


for key in model_keys:
    logger.info("%s Mean Probability = %s", key, np.mean(th_stats[key]))
    sns.distplot(
        th_stats[key],
        hist=True,
        kde=True,
        bins=int(180 / 5),
        # color = 'darkblue',
        label=key,
        hist_kws={"edgecolor": "black"},
        kde_kws={"linewidth": 4},
    )
plt.legend()
plt.savefig(f"figures/th_stats_{task_name}.png", bbox_inches="tight")


logger.info("  Num examples = %s", num_labels)
logger.info("  Threshold = %s", mc_threshold)
# for key in model_keys:
#     logger.info("final temperature %s", models[key].temperature)
logger.info("***** Eval results *****")
for key in model_keys:
    logger.info(
        "%s correct count %s, percent %s, prob %s",
        key,
        correct_cnt[key],
        np.around(correct_cnt[key] / float(num_labels) * 100, 3),
        correct_prob[key],
    )
    logger.info("%s metrics %s", key, model_metrics[key].compute())
logger.info("***** Collaborative Eval results *****")
logger.info(
    "Collaborative metrics %s",
    total_metrics.compute()
)
for key in model_keys:
    logger.info(
        "%s process count %s, correct count %s, percent %s, prob %s",
        key,
        process_cnt[key],
        coop_cnt[key],
        np.around(coop_cnt[key] / float(process_cnt[key]) * 100, 3)
        if process_cnt[key] != 0
        else 0,
        process_prob[key],
    )