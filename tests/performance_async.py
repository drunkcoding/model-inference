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
from hfutils.arg_parser import HfArguments
from hfutils.constants import TASK_TO_LABELS
from hfutils.loader import DatasetLoader, ModelLoader
from numpy import random
from packaging.version import parse
import torch
from transformers.data.data_collator import DataCollatorForSeq2Seq, default_data_collator
from tritonclient.utils import *
import tritonclient.http as httpclient
import tritonclient.grpc as grpcclient
import numpy as np
from tqdm import tnrange, tqdm, trange
from scipy.special import softmax
import time
import multiprocessing as mp

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

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
eval_dataset = concatenate_datasets([eval_dataset] * 10)
logger.info("eval_dataset %s", eval_dataset)
RUN_SEC = 60

# model_name = "gpt_neo_2.7B_standalone"
# model_name = "gpt_neo_2stage"
# model_name = "distilgpt2_cola"
# model_name = "t5-xl-lm-adapt_sst2"
model_tag = "g2r1p1-async"
model_name = "t5_cola_ensemble_pp"

remote = "localhost"
tensorboard_base = "/sata_disk/jupyter-xue/model-inference/tritonserver/"
tensorboard_logdir = os.path.join(tensorboard_base, model_tag)

if data_args.pad_to_max_length:
    data_collator = default_data_collator
else:
    data_collator = DataCollatorForSeq2Seq(tokenizer)

def dummy(result, error):
    pass

def prepare_query(batch):
    input_ids = batch["input_ids"].numpy()
    attention_mask = batch["attention_mask"].numpy()
    ensemble_outputs = np.ones((input_ids.shape[0], 2), dtype=np.float32) * -100
    batch_mask = np.zeros((4,input_ids.shape[0]))
    batch_mask[-1] = np.ones(input_ids.shape[0]) # WHERE TO ENTER
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

label_tokens = [
    tokenizer(label, max_length=2).input_ids[0]
    for label in TASK_TO_LABELS[data_args.task_name]
    if label is not None
]

NUM_PROC = 4
barrier = mp.Barrier(NUM_PROC)


def test_body(pid):
    print(pid)
    with httpclient.InferenceServerClient(f"{remote}:8000", concurrency=4) as client:
        for batch_size in [1,2,4,8,16,32,64,128,256]:
        # for batch_size in [32, 64, 128, 256, 512]:
            
            # metric = load_metric("glue", args.task_name)

            eval_dataloader = DataLoader(
                eval_dataset,
                shuffle=True,
                collate_fn=data_collator,
                batch_size=batch_size,
            )
            
            inputs_list = []
            label_list = []
            # outputs_list = []
            for step, batch in enumerate(eval_dataloader):
                inputs, outputs = prepare_query(batch)
                inputs_list.append(inputs)
                label_list.append((batch["labels"][:, 0] == label_tokens[-1]).to(torch.int64).numpy().tolist())
                # outputs_list.append(outputs)

            start_time = time.perf_counter()
            ready = False

            query_times = []
            cnt = 0
            while not ready:
                

                async_requests = []
                metric = load_metric("glue", data_args.task_name)

                # for step, batch in tqdm(enumerate(eval_dataloader), mininterval=5.0, desc=f"bsz-{batch_size}"):
                # for step, batch in tqdm(enumerate(eval_dataloader), f"bsz{batch_size}-send"):
                for step in tqdm(range(len(inputs_list)), f"{pid} bsz{batch_size}-send"):
                    if step > 500: break
                    query_times.append(time.perf_counter())
                    async_requests.append(client.async_infer(model_name,
                                        inputs_list[step],
                                        request_id=str(step),
                                        outputs=outputs))
                for idx, async_request in tqdm(enumerate(async_requests), desc=f"{pid} bsz{batch_size}-async"):
                    query_times[cnt] = (time.perf_counter() - query_times[cnt]) * 1000
                    cnt += 1
                    response = async_request.get_result()
                    result = response.get_response()
                    logits = response.as_numpy("outputs")
                    predictions = logits.argmax(axis=-1)
                    # print(predictions, label_list[idx])
                #     metric.add_batch(
                #         predictions=predictions,
                #         references=label_list[idx],
                #     )      

                # eval_metric = metric.compute()
                # print(f"Overall eval_metric: {eval_metric}")
            
                curr_time = time.time()
                # print(curr_time - start_time)
                if curr_time - start_time > RUN_SEC:
                    ready = True
                    break

            np.save(f"data/query_times_{model_name}_{model_tag}", np.array(query_times), allow_pickle=False)
            barrier.wait()

    return pid


writer_backend = ModelMetricsWriterBackend(tensorboard_logdir, f"{model_name}")
writer_backend.remote = remote
# writer_backend.step = batch_size
writer_backend.start()



pool = mp.Pool(processes=NUM_PROC)

pool.map(test_body, [i for i in range(NUM_PROC)])

pool.join()
writer_backend.stop()

# with grpcclient.InferenceServerClient(f"{remote}:8001") as client:
