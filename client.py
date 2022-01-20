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

from tritonclient.utils import *
import tritonclient.http as httpclient
import sys
import numpy as np
from tqdm import tqdm
import argparse
import torch

from transformers import (
    AdamW,
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoModel,
    AutoTokenizer,
    DataCollatorWithPadding,
    PretrainedConfig,
    SchedulerType,
    default_data_collator,
    get_scheduler,
    set_seed,
)

import logging

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

# model_name = "gpt2_ensemble"
model_name = "gpt2"
shape = [128]

def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a text classification task")
    parser.add_argument(
        "--task_name",
        type=str,
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

model = AutoModel.from_pretrained(args.model_name_or_path)
model.eval()
tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
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
eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)

if args.task_name is not None:
    metric = load_metric("glue", args.task_name)
else:
    metric = load_metric("accuracy")

with torch.no_grad():
    with httpclient.InferenceServerClient("localhost:8000") as client:
        for step, batch in tqdm(enumerate(eval_dataloader), desc="Requesting"):

            local_logits = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], return_dict=True).last_hidden_state

            input_ids = batch['input_ids'].numpy()
            attention_mask = batch['attention_mask'].numpy()

            inputs = [
                httpclient.InferInput("0", input_ids.shape,
                                np_to_triton_dtype(input_ids.dtype)),
                httpclient.InferInput("1", attention_mask.shape,
                                np_to_triton_dtype(attention_mask.dtype)),
            ]
            # input_names = ["input_ids", "attention_mask"]
            input_names = ["0", "1"]

            inputs[0].set_data_from_numpy(input_ids)
            inputs[1].set_data_from_numpy(attention_mask)

            # inputs = [dict(zip(input_names, inputs))]

            outputs = [
                httpclient.InferRequestedOutput("2122"),
            ]

            response = client.infer(model_name,
                                inputs,
                                request_id=str(1),
                                outputs=outputs)

            result = response.get_response()
            logits = response.as_numpy("2122")

            # print(np.isclose(local_logits, logits))
            # print(local_logits - logits)

            # predictions = logits.argmax(axis=1) if not is_regression else logits.reshape((-1,1))
            # print(predictions)
            # metric.add_batch(
            #     predictions=predictions,
            #     references=batch["labels"],
            # )


        # eval_metric = metric.compute()
        # print(f"eval_metric: {eval_metric}")
    
        # print('PASS: pytorch')
# sys.exit(0)
