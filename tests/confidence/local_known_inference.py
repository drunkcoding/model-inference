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
from transformers.utils.dummy_pt_objects import T5ForConditionalGeneration
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
task_name  = args.data_args.task_name
tokenizer, _ = ModelLoader(args).load(load_model=False)

pos_token = tokenizer("false").input_ids[0]
neg_token = tokenizer("true").input_ids[0]

label_tokens = [
    tokenizer(label, max_length=2).input_ids[0]
    for label in TASK_TO_LABELS[task_name]
    if label is not None
]

# print(pos_token, neg_token)
# print(tokenizer(list(TASK_TO_LABELS[task_name])).input_ids)
# exit()

home_dir = os.path.expanduser(("~"))
# base_dir = "/mnt/yavin/checkpoints"
base_dir = os.path.join(home_dir, os.path.join("model-finetune", "outputs", "google"))

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
    "cuda:0",
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

# model_paths = [
#     f"{base_dir}/t5-small-lm-adapt/{task_name}/checkpoint-2420",
#     # f"{base_dir}/google/t5-small-lm-adapt/qqp",
#     f"{base_dir}/t5-base-lm-adapt/{task_name}/checkpoint-820",
#     f"{base_dir}/t5-large-lm-adapt/{task_name}/checkpoint-240",
#     # f"{base_dir}/t5-xl-lm-adapt/{task_name}/checkpoint-260",
# ]

# model_paths = [
#     f"{base_dir}/t5-small-lm-adapt/{task_name}/checkpoint-5540",
#     # f"{base_dir}/google/t5-small-lm-adapt/qqp",
#     # f"{base_dir}/t5-base-lm-adapt/{task_name}/checkpoint-1860",
#     # f"{base_dir}/t5-large-lm-adapt/{task_name}/checkpoint-1780",
#     # f"{base_dir}/t5-xl-lm-adapt/{task_name}/checkpoint-1380",
# ]

model_energy = dict(zip(model_keys, energy_discount_factor))
model_paths = dict(zip(model_keys, model_paths))
model_device = dict(zip(model_keys, device_map))

logger = Logger(__file__, "info", 5000000, 5)

models = dict()
for key in model_paths:
    logger.debug("key %s, path %s, device %s", key, model_paths[key], model_device[key])
    models[key] = T5ForConditionalGeneration.from_pretrained(
        model_paths[key]
    )  # if key != "S" else DistilBertForSequenceClassification.from_pretrained(model_paths[key])
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

# dataset_loader = DatasetLoader(args)
# # train_dataset = dataset_loader.load(tokenizer, partition="validation", create_dataloader=False)
# eval_dataset = dataset_loader.load(
#     tokenizer, partition="validation", create_dataloader=False
# )
# logger.debug("eval_dataset %s", eval_dataset)

eval_dataset = load_glue_val(preprocess_function).shuffle()

data_args = args.data_args

# if data_args.pad_to_max_length:
#     data_collator = default_data_collator
# else:
#     data_collator = DataCollatorForSeq2Seq(tokenizer)

data_collator = DataCollatorForSeq2Seq(tokenizer)

train_len = int(len(eval_dataset) * 0.4)

train = Dataset.from_dict(eval_dataset[:train_len])
test = eval_dataset
# test = Dataset.from_dict(eval_dataset[train_len:])
print(train, test)

train_dataloader = DataLoader(
    train,
    shuffle=True,
    collate_fn=data_collator,
    batch_size=data_args.train_bsz,
    # drop_last=True,
)

test_dataloader = DataLoader(
    test,
    shuffle=True,
    collate_fn=data_collator,
    batch_size=data_args.eval_bsz,
    # drop_last=True,
)

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
        logits = temperature_scale(logits, temperature)
    return logits

long_dataset = concatenate_datasets([eval_dataset] * 10)

model_probs = dict(zip(model_keys, [list() for _ in range(n_models)]))
model_outputs = dict(zip(model_keys, [list() for _ in range(n_models)]))

labels_list = []

# def agg_logits(hist, curr, pos, device):
#     if hist is not None:
#         hist = hist.to(device)
#         curr_prob, _ = torch.max(torch.float_power(m(curr), 2), dim=-1)
#         hist_prob, _ = torch.max(torch.float_power(m(hist), 2), dim=-1)

#         diff = torch.abs(hist_prob-curr_prob)
#         # print(diff)
#         for i in range(len(diff)):
#             if diff[i] > 0.2:
#                 if curr_prob[i] < hist_prob[i]:
#                     curr[i] = hist[i]
#             else:
#                 curr[i] = (hist[i] * pos + curr[i]) / (pos+1)
#     return curr


def agg_logits(hist, curr, pos, device):
    # return curr
    alpha = 0.6
    if hist is not None:
        hist = hist.to(device)
        # return (hist * pos + curr) / (pos + 1)
        return hist * (1 - alpha) + curr * alpha
    return curr

mc_threshold = [0.60289287, 0.81622027, 0.8955489]

model_temperature = dict(zip(model_keys, [1.7848,2.1796,2.1116,2.3932]))

# -------------  Evaluation WITH Temperature --------------

correct_cnt = dict(zip(model_keys, [0] * n_models))
correct_prob = dict(zip(model_keys, [0] * n_models))
coop_cnt = dict(zip(model_keys, [0] * n_models))
process_prob = dict(zip(model_keys, [0] * n_models))
process_cnt = dict(zip(model_keys, [0] * n_models))

num_labels = 0
# th_stats = []
# threshold = None

th_stats = dict(zip(model_keys, [list() for _ in range(n_models)]))

model_metrics = {}
for key in model_keys:
    # model_metrics[key] = load_metric(
    #     args.data_args.dataset_name, args.data_args.task_name
    # )
    model_metrics[key] = load_metric("accuracy")


total_metrics = load_metric("accuracy")
# f1_metrics = load_metric("f1")
corr_metrics = load_metric("matthews_correlation")
with torch.no_grad():
    for batch in tqdm(test_dataloader, desc="Testing Accuracy"):
        label = (batch["labels"][:, 0] == pos_token).to(torch.int64)
        true_ans = label.cpu().detach().numpy().flatten()

        b_size = len(label.cpu())
        mask = np.array([False] * b_size)

        hist_logits = None
        for i, key in enumerate(model_keys):
            logits = model_inference(
                models[key], batch, model_temperature[key], device=model_device[key]
            )
            direct_ans = np.argmax(m(logits).cpu().detach().numpy(), axis=-1)
            model_metrics[key].add_batch(
                predictions=direct_ans,
                references=true_ans,
            )

            hist_logits = agg_logits(
                hist_logits if key != model_keys[-1] else None,
                logits,
                i,
                model_device[key],
            )

            probabilities = np.power(m(hist_logits).cpu().detach().numpy(), 2)
            # probabilities = m(logits).cpu().detach().numpy()

            # if key in ['S']:
            #     th_stats += np.max(probabilities, axis=1).tolist()

            th_stats[key] += np.max(probabilities, axis=-1).tolist()

            model_ans = np.argmax(probabilities, axis=-1)

            # logger.debug("probabilities %s, true_ans %s", probabilities, true_ans)

            # selected_prob = np.array([p[model_ans[i]] for i, p in enumerate(probabilities)])
            selected_prob = np.max(probabilities, axis=-1)
            processed = (
                (selected_prob >= mc_threshold[i])
                if key in model_keys[:-1]
                else np.array([True] * b_size)
            )

            total_metrics.add_batch(
                predictions=model_ans[(~mask) & processed],
                references=true_ans[(~mask) & processed],
            )
            # f1_metrics.add_batch(
            #     predictions=model_ans[(~mask) & processed],
            #     references=true_ans[(~mask) & processed],
            # )
            corr_metrics.add_batch(
                predictions=model_ans[(~mask) & processed],
                references=true_ans[(~mask) & processed],
            )

            correct_prob[key] += np.sum(selected_prob)
            correct_cnt[key] += np.count_nonzero(direct_ans == true_ans)

            process_prob[key] += np.sum(selected_prob[(~mask) & processed])
            coop_cnt[key] += np.count_nonzero(
                (model_ans == true_ans) & (~mask) & processed
            )
            process_cnt[key] += np.count_nonzero((~mask) & processed)
            mask |= processed

        num_labels += b_size

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
    "Collaborative metrics %s %s",
    total_metrics.compute(),
    corr_metrics.compute(),
    # f1_metrics.compute(),
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