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
# import tritonclient.grpc as httpclient
import sys
import numpy as np
from tqdm import tqdm
import argparse
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

from triton_inference.calibration import temperature_scaling
from triton_inference.monte_carlo import monte_carlo_bounds

logger = logging.getLogger(__name__)

import datasets
from datasets import load_dataset, load_metric
from torch.utils.data import DataLoader

task_to_labels = {
    # "cola": ("not_acceptable", "acceptable"),
    "cola": ("true", "false"),
    # "mnli": None,
    "mrpc": ("not_equivalent", "equivalent"),
    "qnli": ("entailment", "not_entailment"),
    "qqp": ("not_duplicate", "duplicate"),
    # "rte": ("entailment", "not_entailment"),
    "rte": ("true", "false"),
    "sst2": ("negative", "positive"),
    # "stsb": ("sentence1", "sentence2"),
    # "wnli": ("sentence1", "sentence2"),
}

def label2text(task_name, label):
    easy_labels = ("true", "false")
    return easy_labels[label]

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
        "--repository",
        type=str,
        help="Tritonserver model repository, used to store metadata.",
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
padding = "max_length" if args.pad_to_max_length else False

def preprocess_function(examples):
    # Tokenize the texts
    sentence1_examples = examples[sentence1_key]
    sentence2_examples = None if sentence2_key is None else examples[sentence2_key]
    processed_examples = []
    for i in range(len(sentence1_examples)):
        elements = [
                args.task_name, 
                sentence1_key+":",
                sentence1_examples[i],
            ]
        if sentence2_examples is not None:
            elements += [
                sentence2_key+":",
                sentence2_examples[i],
            ]
        processed_examples.append(" ".join(elements))

    texts = (
        (processed_examples,)
    )
    result = tokenizer(*texts, padding=padding, max_length=args.max_length, truncation=True, return_tensors="np")

    if "label" in examples:

        labels = examples["label"]
        labels = [label2text(args.task_name, label) for label in labels]

        # Setup the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(labels, max_length=2, padding=padding, truncation=True, return_tensors="np")

        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if padding == "max_length":
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]

        result["labels"] = labels["input_ids"]

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

requests = []


model_keys = [f"t5_small_lm_adapt_glue_{args.task_name}", f"t5_large_lm_adapt_glue_{args.task_name}"]
model_energy = [0.1, 1]
meta = {}
meta_path = os.path.join(args.repository, "meta.json")

vocab = tokenizer.get_vocab()
pos_token = tokenizer(task_to_labels[args.task_name][1]).input_ids[0]
neg_token = tokenizer(task_to_labels[args.task_name][0]).input_ids[0]


# TEST PERFORMANCE
# Overall Acc
metric = load_metric("glue", args.task_name)
acc_metric = load_metric("accuracy")
with httpclient.InferenceServerClient("dgj101:8000", concurrency=8) as client:
    for step, batch in tqdm(enumerate(eval_dataloader), desc="Requesting"):
        # if step > 1000: break
        input_ids = batch['input_ids'].numpy()
        attention_mask = batch['attention_mask'].numpy()
        inputs = [
            httpclient.InferInput("input_ids", input_ids.shape,
                            np_to_triton_dtype(input_ids.dtype)),
            httpclient.InferInput("attention_mask", attention_mask.shape,
                            np_to_triton_dtype(attention_mask.dtype)),
        ]

        inputs[0].set_data_from_numpy(input_ids)
        inputs[1].set_data_from_numpy(attention_mask)

        outputs = [
            httpclient.InferRequestedOutput("outputs"),
        ]
        response = client.infer(model_keys[0],
                            inputs,
                            request_id=str(1),
                            outputs=outputs)

        result = response.get_response()
        logits = response.as_numpy("outputs")
        # print(logits)
        if logits.shape[1] == 1:
            predictions = np.where(logits > 0.5, 1, 0).flatten()
        else:
            predictions = logits.argmax(axis=1) if not is_regression else logits.reshape((-1,1))
        # print(predictions, batch["labels"])
        labels = batch["labels"][:, 0] == pos_token
        # print(predictions, labels)
        metric.add_batch(
            predictions=predictions,
            references=labels,
        )
        acc_metric.add_batch(
            predictions=predictions,
            references=labels,
        )      
        # if (step + 1) % 1000 == 0:
eval_metric = metric.compute()
accuracy = acc_metric.compute()
print(f"Overall eval_metric: {eval_metric}, accuracy {accuracy}")
# exit()
# # TEST SANITY
# from partitioner import GPTModelPipe, get_attn_mask
# from transformers import BatchEncoding
# from helpers import test_parameters_consistency

# user = os.path.expanduser("~")

# model_gold = AutoModelForSequenceClassification.from_pretrained("/jmain01/home/JAD003/sxr06/lxx22-sxr06/model-finetune/outputs/gpt-neo-2.7B/QQP/checkpoint-1350/").cpu()
# model_gold.eval()
# model_gold.config.pad_token_id = 50256
# model_test = GPTModelPipe(model_gold.config, "classification", model_gold).cpu()
# model_test.eval()

# model_test.config.pad_token_id = 50256

# # test_parameters_consistency(model_gold, model_test)

# with torch.no_grad():
#     for step, batch in enumerate(tqdm(eval_dataloader)):
#         batch = BatchEncoding(batch).to("cpu")
#         input_ids = batch['input_ids']
#         attention_mask = batch['attention_mask']
#         labels = batch['labels']

#         output_gold = model_gold(**batch, output_hidden_states=True)
#         hidden_states_gold = output_gold.hidden_states
#         logits_gold = output_gold.logits.detach().cpu().numpy()

#         # args = (input_ids, attention_mask)
#         hidden_states_test = []
#         for i in range(34):
#             model_test.exec_map = (i,i+1)
#             if i == 0:
#                 output = model_test.forward_layers((input_ids, get_attn_mask(attention_mask)))
#             else:
#                 output = model_test.forward_layers((hidden_states_test[-1], input_ids, get_attn_mask(attention_mask)))
#             if i != 33:
#                 hidden_states_test.append(output)
#             else:
#                 logits_test = output.detach().cpu().numpy()
        
#         # output_test, hidden_states_test = model_test((input_ids, attention_mask), output_hidden_states=True)
#         # logits_test = output_test.detach().cpu().numpy()
        
#         # hidden_states_test = output_test[1]
#         print("logits_gold", logits_gold)
#         print("logits_test", logits_test)
#         print(logits_gold-logits_test)
#         print(len(hidden_states_test), len(hidden_states_gold))
#         assert len(hidden_states_test) == len(hidden_states_gold)
#         for i in range(len(hidden_states_gold)):
#             print(i, hidden_states_gold[i]-hidden_states_test[i])
#             assert np.all(np.isclose(
#                 hidden_states_gold[i].detach().cpu().numpy(),
#                 hidden_states_test[i].detach().cpu().numpy()
#             ))

#         assert np.all(np.isclose(
#             logits_gold,
#             logits_test
#         ))

#         break




for i, model_name in enumerate(model_keys):
    meta[model_name] = {
        "threshold": 0.0,
        "temperature": 1.0,
    }

with open(meta_path, "w") as fp:
    json.dump(meta, fp)
    time.sleep(10)

for i, model_name in enumerate(model_keys):
    meta[model_name] = {
        "threshold": 0.0,
        "temperature": 1.0,
        "energy": model_energy[i], # HACK
        "labels": [],
        "outputs": [],
        "metric": load_metric("glue", args.task_name),
        "acc": load_metric("accuracy"),
    }

# with open(meta_path, "w") as fp:
#     json.dump(meta, fp)

# random.seed(0)
# torch.manual_seed(0)

if args.task_name is not None:
    metric = load_metric("glue", args.task_name)
else:
    metric = load_metric("accuracy")

with httpclient.InferenceServerClient("dgj101:8000", concurrency=8) as client:
    for step, batch in tqdm(enumerate(eval_dataloader), desc="Requesting"):
        if step > 1000: break
        input_ids = batch['input_ids'].numpy()
        attention_mask = batch['attention_mask'].numpy()
        inputs = [
            httpclient.InferInput("input_ids", input_ids.shape,
                            np_to_triton_dtype(input_ids.dtype)),
            httpclient.InferInput("attention_mask", attention_mask.shape,
                            np_to_triton_dtype(attention_mask.dtype)),
        ]

        inputs[0].set_data_from_numpy(input_ids)
        inputs[1].set_data_from_numpy(attention_mask)

        outputs = [
            httpclient.InferRequestedOutput("outputs"),
        ]

        for model_name in model_keys:

            response = client.infer(model_name,
                                inputs,
                                request_id=str(1),
                                outputs=outputs)

            result = response.get_response()
            logits = response.as_numpy("outputs")
            # print(logits)
            if logits.shape[1] == 1:
                predictions = np.where(logits > 0.5, 1, 0).flatten()
            else:
                predictions = logits.argmax(axis=1) if not is_regression else logits.reshape((-1,1))
            # print(predictions, batch["labels"])
            labels = batch["labels"][:, 0] == pos_token
            meta[model_name]['metric'].add_batch(
                predictions=predictions,
                references=labels,
            )
            meta[model_name]['acc'].add_batch(
                predictions=predictions,
                references=labels,
            )

            meta[model_name]['labels'].append(labels)
            meta[model_name]['outputs'].append(logits)
        
        # labels_list = torch.Tensor(np.concatenate(labels_list)).long()
        # outputs_list = torch.Tensor(np.concatenate(outputs_list))
        

        # meta[model_name]['labels'] = labels_list
        # meta[model_name]['outputs'] = outputs_list        
for model_name in model_keys:
    eval_metric = meta[model_name]['metric'].compute()
    accuracy = meta[model_name]['acc'].compute()
    print(f"{model_name} eval_metric: {eval_metric}, accuracy: {accuracy}")

for model_name in model_keys:

    meta[model_name]['labels'] = torch.Tensor(np.concatenate(meta[model_name]['labels'])).long()
    meta[model_name]['outputs'] = torch.Tensor(np.concatenate(meta[model_name]['outputs']))

    temperature = temperature_scaling(meta[model_name]['outputs'], meta[model_name]['labels']).squeeze().item()
    meta[model_name]['temperature'] = temperature
    meta[model_name]['probs'] = softmax(meta[model_name]['outputs'].numpy() / temperature, axis=1)

data_size = len(meta[model_keys[0]]['labels'])
acc = data_size / 100.0

def total_reward(threshold):
    reward = 0
    energy = 0
    mask = np.array([False]*data_size)
    for i, key in enumerate(model_keys):
        processed = (meta[key]['probs'][~mask, 0] >= threshold[i]
                        ) if key in model_keys[:-1] else np.array([True]*data_size)
        # correct_count = np.sum(
        #     model_probs[key][(~mask) & processed, 1])
        reward += np.around(np.sum(meta[key]['probs'][(~mask) & processed, 1]) / acc) * acc
        # reward += np.around(correct_count /
        #                     (int(correct_count * 0.025) + 1)) * (int(correct_count * 0.025) + 1)
        energy += model_energy[i] * np.count_nonzero(~mask)
        mask |= processed
    return (reward, -energy)

# def total_reward(threshold):
#     reward = 0
#     energy = 0
#     mask = np.array([False]*data_size)
#     for i, key in enumerate(model_keys):
#         processed = (meta[key]['probs'][~mask, 0] >= threshold[i]
#                         ) if key in model_keys[:-1] else np.array([True]*data_size)
#         reward += np.sum(meta[key]['probs'][(~mask) & processed, 1])
#         energy += model_energy[i] * np.count_nonzero(~mask)
#         mask |= processed
#     return (reward, -energy)


# dtype = [('reward', float), ('energy', float)]
# candidate_th = np.linspace(0.5, 1.0, num=500, endpoint=True)
# rewards = np.array([total_reward([th]) for th in candidate_th], dtype=dtype)
# tops = 40
# idx = np.argpartition(rewards, -tops, order=[d[0] for d in dtype])[-tops:]
# rewards = rewards[idx] 
# candidate_th = candidate_th[idx]

# print("rewards", rewards)
# print("candidate_th", candidate_th)

# mc_threshold = [np.min(candidate_th)]

n_models = len(model_keys)        
threshold_bounds = monte_carlo_bounds(
    total_reward,
    [(0.8, 1.0)] * (n_models-1),
    [('reward', float), ('energy', float)],
    n=10000,
    tops=40,
    maxiter=15,
)
mc_threshold = np.min(
    threshold_bounds, axis=1
)

for i, key in enumerate(model_keys):
    meta[key]["threshold"] = mc_threshold[i] if key in model_keys[:-1] else 0.0
    del meta[key]["labels"]
    del meta[key]["outputs"]
    del meta[key]["probs"]
    del meta[key]["metric"]
    del meta[key]["acc"]

with open(meta_path, "w") as fp:
    json.dump(meta, fp)
    time.sleep(10)


# Overall Acc
metric = load_metric("glue", args.task_name)
acc_metric = load_metric("accuracy")
with httpclient.InferenceServerClient("127.0.0.1:8000", concurrency=8) as client:
    for step, batch in tqdm(enumerate(eval_dataloader), desc="Requesting"):
        # if step > 1000: break
        input_ids = batch['input_ids'].numpy()
        attention_mask = batch['attention_mask'].numpy()
        inputs = [
            httpclient.InferInput("input_ids", input_ids.shape,
                            np_to_triton_dtype(input_ids.dtype)),
            httpclient.InferInput("attention_mask", attention_mask.shape,
                            np_to_triton_dtype(attention_mask.dtype)),
        ]

        inputs[0].set_data_from_numpy(input_ids)
        inputs[1].set_data_from_numpy(attention_mask)

        outputs = [
            httpclient.InferRequestedOutput("outputs"),
        ]
        response = client.infer(model_keys[0],
                            inputs,
                            request_id=str(1),
                            outputs=outputs)

        result = response.get_response()
        logits = response.as_numpy("outputs")
        # print(logits)
        if logits.shape[1] == 1:
            predictions = np.where(logits > 0.5, 1, 0).flatten()
        else:
            predictions = logits.argmax(axis=1) if not is_regression else logits.reshape((-1,1))
        # print(predictions, batch["labels"])
        metric.add_batch(
            predictions=predictions,
            references=batch["labels"][:, 0] == pos_token,
        )
        acc_metric.add_batch(
            predictions=predictions,
            references=batch["labels"][:, 0] == pos_token,
        )     

        # if (step + 1) % 1000 == 0:
eval_metric = metric.compute()
accuracy = acc_metric.compute()
print(f"Overall eval_metric: {eval_metric}, accuracy: {accuracy}")