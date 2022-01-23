import functools
import json
import os
from hfutils.arg_parser import HfArguments
from hfutils.constants import TASK_TO_LABELS
from hfutils.loader import DatasetLoader, ModelLoader
from numpy import random
from packaging.version import parse
import torch
from transformers.data.data_collator import (
    DataCollatorForSeq2Seq,
    default_data_collator,
)
import numpy as np
from tqdm import tnrange, tqdm, trange
from scipy.special import softmax
import time
import multiprocessing as mp
import requests

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import logging
from hfutils.measure import ModelMetricsWriter, ModelMetricsWriterBackend

logger = logging.getLogger(__name__)

import datasets
from datasets import load_dataset, load_metric, concatenate_datasets
from torch.utils.data import DataLoader

import torch.multiprocessing as mp

args = HfArguments()
data_args = args.data_args

tokenizer, _ = ModelLoader(args).load(load_model=False)
dataset_loader = DatasetLoader(args)
eval_dataset = dataset_loader.load(
    tokenizer, partition="validation", create_dataloader=False
)
# eval_dataset = concatenate_datasets([eval_dataset] * 10)
logger.info("eval_dataset %s", eval_dataset)
remote = "localhost"

if data_args.pad_to_max_length:
    data_collator = default_data_collator
else:
    data_collator = DataCollatorForSeq2Seq(tokenizer)


def prepare_query(batch):
    input_ids = batch["input_ids"].numpy().tolist()
    attention_mask = batch["attention_mask"].numpy().tolist()
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }


label_tokens = [
    tokenizer(label, max_length=2).input_ids[0]
    for label in TASK_TO_LABELS[data_args.task_name]
    if label is not None
]
URL = "http://127.0.0.1:8000/composed"

from transformers import T5ForConditionalGeneration
from hfutils.calibration import temperature_scale, temperature_scaling


base_dir = "/mnt/yavin/checkpoints"
task_name = data_args.task_name
model_path = f"{base_dir}/t5-small-lm-adapt/{task_name}/checkpoint-5540"
model_gold = T5ForConditionalGeneration.from_pretrained(model_path)
model_gold = model_gold.to("cuda:0")
model_gold.eval()


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


for batch_size in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
    metric = load_metric("glue", data_args.task_name)

    eval_dataloader = DataLoader(
        eval_dataset,
        shuffle=True,
        collate_fn=data_collator,
        batch_size=batch_size,
    )

    for step, batch in enumerate(eval_dataloader):
        inputs_dict = prepare_query(batch)

        resp = requests.post(URL, json=inputs_dict)
        try:
            predictions = resp.json()
        except:
            print(resp.content)
            exit()


        predictions = np.array(predictions['labels'])
        predictions_gold = np.argmax(model_inference(model_gold, batch).detach().cpu().numpy(), axis=-1)

        print("batch_size", batch_size, predictions, predictions_gold)
        assert np.all(predictions == predictions_gold)

        label = batch["labels"][:, 0] == label_tokens[-1]
        metric.add_batch(
            predictions=predictions,
            references=label
        )
    
    print(batch_size, metric.compute())
