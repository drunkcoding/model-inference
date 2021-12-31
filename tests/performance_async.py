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
from packaging.version import parse
import torch
from tritonclient.utils import *
import tritonclient.http as httpclient
import tritonclient.grpc as grpcclient
import numpy as np
from tqdm import tnrange, tqdm, trange
from scipy.special import softmax
import time
import multiprocessing as mp


import logging
from triton_inference.arg_parser import GlueArgParser
from triton_inference.measure import ModelMetricsWriter, ModelMetricsWriterBackend
from triton_inference.srv_ctx import GlueContext

logger = logging.getLogger(__name__)

import datasets
from datasets import load_dataset, load_metric
from torch.utils.data import DataLoader

import torch.multiprocessing as mp

glue_parser = GlueArgParser()
args = glue_parser.parse()
glue_ctx = GlueContext(glue_parser)

RUN_SEC = 60

# model_name = "gpt_neo_2.7B_standalone"
# model_name = "gpt_neo_2stage"
# model_name = "distilgpt2_cola"
model_name = "gpt_neo_cola_ensemble"

remote = "dgj110"
tensorboard_base = "/jmain01/home/JAD003/sxr06/lxx22-sxr06/model-inference/tritonserver/"
tensorboard_logdir = os.path.join(tensorboard_base, "m1g2-g-async")

# grpcclient.InferenceServerClient

def dummy(result, error):
    pass

def test_body(pid):
    print(pid)
    with httpclient.InferenceServerClient(f"{remote}:8000", concurrency=4) as client:
        for batch_size in range(1, 24 + 1):
            
            # metric = load_metric("glue", args.task_name)

            eval_dataloader = glue_ctx.get_eval_dataloader(batch_size=batch_size)
            
            inputs_list = []
            label_list = []
            # outputs_list = []
            for step, batch in enumerate(eval_dataloader):
                input_ids = batch['input_ids'].numpy()
                attention_mask = batch['attention_mask'].numpy()
                label_list.append(batch['labels'].numpy())

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
                inputs_list.append(inputs)
                # outputs_list.append(outputs)

            start_time = time.time()
            ready = False
            while not ready:
                
                async_requests = []
                metric = load_metric("glue", args.task_name)

                # for step, batch in tqdm(enumerate(eval_dataloader), mininterval=5.0, desc=f"bsz-{batch_size}"):
                # for step, batch in tqdm(enumerate(eval_dataloader), f"bsz{batch_size}-send"):
                for step in tqdm(range(len(inputs_list)), f"{pid} bsz{batch_size}-send"):
                    if step > 5000: break
                    async_requests.append(client.async_infer(model_name,
                                        inputs_list[step],
                                        # request_id=str(1),
                                        outputs=outputs))
                for idx, async_request in tqdm(enumerate(async_requests), desc=f"{pid} bsz{batch_size}-async"):
                    response = async_request.get_result()
                    result = response.get_response()
                    logits = response.as_numpy("outputs")
                    predictions = logits.argmax(axis=1)
                    # print(predictions, batch["labels"])
                    # metric.add_batch(
                    #     predictions=predictions,
                    #     references=label_list[idx],
                    # )      

                # eval_metric = metric.compute()
                # print(f"Overall eval_metric: {eval_metric}")
            
                curr_time = time.time()
                # print(curr_time - start_time)
                if curr_time - start_time > RUN_SEC:
                    ready = True
                    break


writer_backend = ModelMetricsWriterBackend(tensorboard_logdir, f"{model_name}")
writer_backend.remote = remote
# writer_backend.step = batch_size
writer_backend.start()

NUM_PROC = 1

pool = mp.Pool(processes=NUM_PROC)

pool.map(test_body, [i for i in range(NUM_PROC)])

pool.join()
writer_backend.stop()

# with grpcclient.InferenceServerClient(f"{remote}:8001") as client:
