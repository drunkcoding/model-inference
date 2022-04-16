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

from functools import partial
import os
import random
import requests
import torch
from typing import List
from tqdm import tqdm
from tritonclient.utils import *
import tritonclient.http as httpclient
import time
import logging

from transformers import T5Tokenizer, DataCollatorForSeq2Seq, default_data_collator
from datasets import load_dataset, load_metric
from torch.utils.data import DataLoader

from hfutils.constants import TASK_TO_KEYS, TASK_TO_LABELS
from hfutils.measure import get_energy_by_group
from hfutils.loader import t5_preprocess_function, load_glue_val, load_glue_train

logger = logging.getLogger(__name__)

# home_dir = os.path.expanduser("~")
home_dir = "/mnt/raid0nvme1"
base_dir = os.path.join(home_dir, os.path.join("HuggingFace", "google"))

tokenizer = T5Tokenizer.from_pretrained(os.path.join(base_dir, "t5-small-lm-adapt"))
padding = "max_length"
max_seq_length = 128

remote = "localhost"

# data_collator = DataCollatorForSeq2Seq(tokenizer)
data_collator = partial(default_data_collator, return_tensors="np")
label_tokens = [1176, 6136, 59]  # HACK with GLUE labels


def token2label(tokens, label_tokens: List):
    return [label_tokens.index(t) for t in tokens]


def preprocess_function(examples, task_name):
    # Tokenize the texts
    sentence1_key = TASK_TO_KEYS[task_name][0]
    sentence2_key = (
        None if len(TASK_TO_KEYS[task_name]) == 1 else TASK_TO_KEYS[task_name][1]
    )
    sentence1_examples = examples[sentence1_key]
    sentence2_examples = None if sentence2_key is None else examples[sentence2_key]
    processed_examples = []
    for i in range(len(sentence1_examples)):
        elements = [
            task_name,
            sentence1_key + ":",
            sentence1_examples[i],
        ]
        if sentence2_examples is not None:
            elements += [
                sentence2_key + ":",
                sentence2_examples[i],
            ]
        processed_examples.append(" ".join(elements))

    texts = (processed_examples,)
    result = tokenizer(
        *texts,
        padding=padding,
        max_length=max_seq_length,
        truncation=True,
        return_tensors="np",
    )

    if "label" in examples:
        result["labels"] = examples["label"]
    return result


rnd_seed = 106033

preprocess_function = partial(
    t5_preprocess_function, tokenizer=tokenizer, padding="max_length", max_length=128,
)
dataset = load_glue_val(preprocess_function).shuffle(seed=rnd_seed)
data_collator = DataCollatorForSeq2Seq(tokenizer)

batch_size = 1


dataloader = DataLoader(dataset, collate_fn=data_collator, batch_size=batch_size)

from hfutils.constants import token2label
T5_TASK_LABELS = [1176, 6136, 59]

inputs_list = []
label_list = []
for step, batch in enumerate(tqdm(dataloader)):
    input_ids = batch["input_ids"].numpy()
    attention_mask = batch["attention_mask"].numpy()
    label = token2label(batch["labels"][:, 0], T5_TASK_LABELS)

    # batch_mask = np.ones((4,batch_size)).astype(bool)
    batch_mask = np.zeros((4,batch_size))
    batch_mask[0, :] = 1
    batch_mask = batch_mask.astype(bool)

    # print(input_ids.shape, attention_mask.shape, batch_mask.shape)

    logits = np.zeros((batch_size, 3)).astype(np.float32)

    if step * batch_size > 300: break

    inputs = [
        httpclient.InferInput(
            "encoder_input_ids", input_ids.shape, 
            np_to_triton_dtype(input_ids.dtype)
        ),
        httpclient.InferInput(
            "encoder_attention_mask",
            attention_mask.shape,
            np_to_triton_dtype(attention_mask.dtype),
        ),
        httpclient.InferInput(
            "batch_mask",
            batch_mask.shape,
            np_to_triton_dtype(batch_mask.dtype),
        ),
        httpclient.InferInput(
            "logits",
            logits.shape,
            np_to_triton_dtype(logits.dtype),
        ),
    ]
    inputs[0].set_data_from_numpy(input_ids)
    inputs[1].set_data_from_numpy(attention_mask)
    inputs[2].set_data_from_numpy(batch_mask)
    inputs[3].set_data_from_numpy(logits)
    outputs = [
        httpclient.InferRequestedOutput("logits"),
        httpclient.InferRequestedOutput("batch_mask"),
    ]
    inputs_list.append(inputs)
    label_list.append(label)


import multiprocessing as mp

NUM_PROC = 3
barrier = mp.Barrier(NUM_PROC)

def test_body(pid):
    print(pid)
    model_name = "t5_ensemble" # if pid == 0 else "t5_e0p0"
    metric = load_metric("accuracy")
    with httpclient.InferenceServerClient(f"{remote}:8000", concurrency=8) as client:
        for step, input in enumerate(tqdm(inputs_list)):
            response = client.infer(
                model_name, input, request_id=str(step), outputs=outputs,
            )

            # result = response.get_response()
            logits = response.as_numpy("logits")
            # print(logits)
            predictions = np.argmax(logits, axis=1).flatten()
            # print(predictions)
            # labels = token2label(batch["labels"][:, 0].flatten(), label_tokens)
            # print(label_list[step], predictions)
            metric.add_batch(predictions=predictions, references=label_list[step])
    print(metric.compute())

start_energy = np.array(list(get_energy_by_group().values()))
pool = mp.Pool(processes=NUM_PROC)
pool.map(test_body, [i for i in range(NUM_PROC)])
pool.close()
pool.join()
end_energy = np.array(list(get_energy_by_group().values()))
print(end_energy - start_energy)


