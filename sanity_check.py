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

import json
import os
from numpy import random
import torch
from tritonclient.utils import *
import tritonclient.http as httpclient
import tritonclient.grpc as grpcclient
import sys
import numpy as np
from tqdm import tqdm
import argparse
from scipy.special import softmax

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

from deepspeed.utils.zero_to_fp32 import load_state_dict_from_zero_checkpoint

import logging

from triton_inference.calibration import temperature_scaling
from triton_inference.monte_carlo import monte_carlo_bounds

logger = logging.getLogger(__name__)

import datasets
from datasets import load_dataset, load_metric
from torch.utils.data import DataLoader

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a text classification task")
    parser.add_argument(
        "--task_name",
        type=str.lower,
        default=None,
        help="The name of the glue task to train on.",
        choices=list(task_to_keys.keys()),
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--train_file", type=str, default=None, help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--validation_file", type=str, default=None, help="A csv or a json file containing the validation data."
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_lengh` is passed."
        ),
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=1,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )
    args = parser.parse_args()
    return args

args = parse_args()
raw_datasets = load_dataset("glue", args.task_name)

is_regression = args.task_name == "stsb"
if not is_regression:
    label_list = raw_datasets["train"].features["label"].names
    num_labels = len(label_list)
else:
    num_labels = 1

label_to_id = None
# label_to_id = {str(v): i for i, v in enumerate(label_list)}
# print(label_to_id)
if args.task_name is not None:
    sentence1_key, sentence2_key = task_to_keys[args.task_name]
else:
    # Again, we try to have some nice defaults but don't hesitate to tweak to your use case.
    non_label_column_names = [name for name in raw_datasets["train"].column_names if name != "label"]
    if "sentence1" in non_label_column_names and "sentence2" in non_label_column_names:
        sentence1_key, sentence2_key = "sentence1", "sentence2"
    else:
        if len(non_label_column_names) >= 2:
            sentence1_key, sentence2_key = non_label_column_names[:2]
        else:
            sentence1_key, sentence2_key = non_label_column_names[0], None

tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.bos_token
padding = "max_length" if args.pad_to_max_length else False

def preprocess_function(examples):
    # Tokenize the texts
    texts = (
        (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
    )
    result = tokenizer(*texts, padding=padding, max_length=args.max_length, truncation=True)

    if "label" in examples:
        if label_to_id is not None:
            # print(examples["label"])
            # Map labels to IDs (not necessary for GLUE tasks)
            result["labels"] = [label_to_id[l] for l in examples["label"]]
        else:
            # In all cases, rename the column to labels because the model will expect that.
            result["labels"] = examples["label"]
    return result

processed_datasets = raw_datasets.map(
        preprocess_function,
        batched=True,
        remove_columns=raw_datasets["train"].column_names,
        desc="Running tokenizer on dataset",
    )

# DataLoaders creation:
if args.pad_to_max_length:
    # If padding was already done ot max length, we use the default data collator that will just convert everything
    # to tensors.
    data_collator = default_data_collator
else:
    # Otherwise, `DataCollatorWithPadding` will apply dynamic padding for us (by padding to the maximum length of
    # the samples passed). When using mixed precision, we add `pad_to_multiple_of=8` to pad all tensors to multiple
    # of 8s, which will enable the use of Tensor Cores on NVIDIA hardware with compute capability >= 7.5 (Volta).
    data_collator = DataCollatorWithPadding(tokenizer)


eval_dataset = processed_datasets["validation_matched" if args.task_name == "mnli" else "validation"]
eval_dataloader = DataLoader(eval_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)

def callback(user_data, result, error):
    if error:
        user_data.append(error)
    else:
        user_data.append(result)

# TEST SANITY
from partitioner import GPTModelPipe, get_attn_mask
from transformers import BatchEncoding
from helpers import test_parameters_consistency

user = os.path.expanduser("~")

checkpoint_path = "/jmain01/home/JAD003/sxr06/lxx22-sxr06/model-finetune/outputs/gpt-neo-2.7B/QQP/checkpoint-1350/"

model_gold = AutoModelForSequenceClassification.from_pretrained(checkpoint_path).cpu()
model_gold = load_state_dict_from_zero_checkpoint(model_gold, checkpoint_path)
model_gold.eval()
model_gold.to("cuda:1")

model_test = GPTModelPipe(model_gold.config, "classification", model_gold)
model_test.exec_map = (0, 34)
model_test.to("cuda:0")

# test_parameters_consistency(model_gold, model_test)

if args.task_name is not None:
    metric = load_metric("glue", args.task_name)
else:
    metric = load_metric("accuracy")

for step, batch in tqdm(enumerate(eval_dataloader), desc="Requesting"):
    if step > 1000: break
    print(batch.keys())
    batch = BatchEncoding(batch).to("cuda:1")

    outputs_gold = model_gold(**batch, output_hidden_states=True)
    logits_gold = outputs_gold.logits.detach().cpu().numpy()
    hidden_states_gold = list(outputs_gold.hidden_states)

    batch = batch.to("cuda:0")
    hidden_states_test = []
    for i in range(model_test.num_layers):
        model_test.exec_map = (i,i+1)
        if i == 0:
            output = model_test.forward_layers((batch['input_ids'], get_attn_mask(batch['attention_mask'])))
        else:
            output = model_test.forward_layers((hidden_states_test[-1], batch['input_ids'], get_attn_mask(batch['attention_mask'])))
        
        if i < model_test.num_layers-1:
            hidden_states_test.append(output)
        else:
            logits_test = output.detach().cpu().numpy()

    # for i in range(len(hidden_states_test)):
    #     hidden_states_gold[i] = hidden_states_gold[i].detach().cpu().numpy()
    #     hidden_states_test[i] = hidden_states_test[i].detach().cpu().numpy()
    #     print(i, hidden_states_gold[i]-hidden_states_test[i])
    #     assert np.all(np.isclose(
    #         hidden_states_gold[i],
    #         hidden_states_test[i]
    #     ))
    
    # logits_test = model_test.forward_layers((batch['input_ids'], batch['attention_mask'])).detach().cpu().numpy()
    # logits_test, _ = model_test((batch['input_ids'], batch['attention_mask']))
    # logits_test = logits_test.detach().cpu().numpy()
    predictions = logits_test.argmax(axis=1)

    print("logits_gold", logits_gold)
    print("logits_test", logits_test)
    print(logits_gold-logits_test)

    assert np.all(np.isclose(
        logits_gold,
        logits_test
    ))

    metric.add_batch(
        predictions=predictions,
        references=batch["labels"],
    )

eval_metric = metric.compute()
print(f"eval_metric: {eval_metric}")