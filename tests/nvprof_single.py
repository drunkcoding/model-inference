import copy
import functools
import gc
import time
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
from hfutils.arg_parser import TestArguments
from hfutils.loader import ModelLoader, DatasetLoader
from hfutils.temperature_scaling import ModelWithTemperature
from hfutils.monte_carlo import monte_carlo_bounds
from hfutils.calibration import temperature_scale, temperature_scaling
from hfutils.plot import sns_displot

import scipy.stats as stats

args = TestArguments()
task_name = args.data_args.task_name
tokenizer, _ = ModelLoader(args).load(load_model=False)

model_args = args.model_args

pos_token = tokenizer("false").input_ids[0]
neg_token = tokenizer("true").input_ids[0]

label_tokens = [
    tokenizer(label, max_length=2).input_ids[0]
    for label in TASK_TO_LABELS[task_name]
    if label is not None
]

base_dir = "/mnt/yavin/checkpoints"
model_path = f"{base_dir}/t5-small-lm-adapt/{task_name}/checkpoint-2420"

model = T5ForConditionalGeneration.from_pretrained(model_args.model_name_or_path)
model = model.to("cuda:0")
model.eval()

torch.cuda.empty_cache()
gc.collect()

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

# -------------  Dataset Prepare --------------

dataset_loader = DatasetLoader(args)
# train_dataset = dataset_loader.load(tokenizer, partition="validation", create_dataloader=False)
eval_dataset = dataset_loader.load(
    tokenizer, partition="validation", create_dataloader=False
)

data_args = args.data_args

if data_args.pad_to_max_length:
    data_collator = default_data_collator
else:
    data_collator = DataCollatorForSeq2Seq(tokenizer)

m = torch.nn.Softmax(dim=1)


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

def plot_density(data, filename):
    sns.distplot(data, hist=True, kde=False, 
                bins=int(180/5), 
                hist_kws={'edgecolor':'black'},
                kde_kws={'linewidth': 2})
    plt.savefig(os.path.join("figures", filename), bbox_inches="tight")
    plt.close()

long_dataset = concatenate_datasets([eval_dataset] * 10)

# import torch.cuda.profiler as profiler
# import pyprof
from torch.profiler import profile, record_function, ProfilerActivity

# print(dir(pyprof))
# pyprof.init()
iters = 100

# profiler.start()

model_name = model_args.model_name_or_path.split("/")[-2]
logger_base_name = f"trace_{model_name}"
logger = Logger(logger_base_name, "info", 5000000, 5)

cuda_streams = [torch.cuda.Stream(device="cuda:0") for _ in range(2) ]

def model_exec(batch, stream, temperature=None, device="cuda:0"):
    with torch.cuda.stream(cuda_streams[stream]):
        logits = model_inference(model, batch, temperature, device)
    return 0

import multiprocessing as mp
pool = mp.Pool(2)

for batch_size in [1, 2, 4, 8, 16, 32, 64, 128]:
# for batch_size in [128, 256]:
    eval_dataloader = DataLoader(
        long_dataset,
        shuffle=True,
        collate_fn=data_collator,
        batch_size=batch_size,
        # drop_last=True,
    )

    latency = []
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for step, batch in tqdm(
                enumerate(eval_dataloader), desc=f"Test Acc bsz{batch_size}"
            ):
            if (step + 1) % 10 == 0:
                logits = model_inference(model, batch)
                # break
            # else:
            #     start_time = time.perf_counter()
            #     result = pool.apply_async(model_exec, (batch, 0))
            #     result = pool.apply_async(model_exec, (batch, 1))
            #     # with torch.cuda.stream(cuda_stream2):
            #     #     logits = model_inference(model, batch)
            #     # cuda_stream1.synchronize()
            #     # cuda_stream2.synchronize()
            #     torch.cuda.synchronize()
            #     result.get()
            #     end_time = time.perf_counter()
            #     latency.append((end_time - start_time) * 1000)

            if (step + 1) == 100: break

    

    trace_name_base = f"trace_{model_name}_{batch_size}"
    # latency_name_base = f"latency_{model_name}_{batch_size}"
    trace_raw = f"{trace_name_base}.json"
    trace_occ = f"{trace_name_base}.npy"
    trace_fig = f"{trace_name_base}.png"

    # latency_npy = f"{latency_name_base}.npy"
    # latency_fig = f"{latency_name_base}.png"

    prof.export_chrome_trace(os.path.join("data",trace_raw))

    import json

    with open(os.path.join("data",trace_raw), "r") as fp:
        trace = json.load(fp)
        trace_events = trace["traceEvents"]

    occupancy = []
    for event in tqdm(trace_events):
        if "args" in event:
            if "est. achieved occupancy %" in event["args"]:
                occupancy.append(event["args"]["est. achieved occupancy %"])

    np.save(os.path.join("data", trace_occ), occupancy, allow_pickle=False)
    # np.save(os.path.join("data", latency_npy), latency, allow_pickle=False)

    
    # sns_displot(trace_fig, occupancy)
    plot_density(occupancy, trace_fig)

    logger.info("batch_size %s, %s", batch_size, stats.describe(occupancy))

# profiler.stop()

# for batch_size in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
#     # for batch_size in [32, 64, 128, 256, 512]:
#     eval_dataloader = DataLoader(
#         eval_dataset,
#         shuffle=True,
#         collate_fn=data_collator,
#         batch_size=batch_size,
#     )
#     for batch in tqdm(eval_dataloader, desc=f"Test Acc bsz{batch_size}"):
#         logits = model_inference(model, batch)


# nvprof -f -o net.sql --profile-from-start off  python net.py

pool.terminate()
pool.join()
