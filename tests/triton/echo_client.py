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
from scipy import stats

import logging

logger = logging.getLogger(__name__)

import datasets
from datasets import load_dataset, load_metric
from torch.utils.data import DataLoader

import torch.multiprocessing as mp

RUN_SEC = 60

model_name = "echo"

remote = "localhost"

with httpclient.InferenceServerClient(f"{remote}:8000", concurrency=1) as client:

    input_ids = np.ones((64,128)).astype(int)
    attention_mask = np.ones((64,128)).astype(int)

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


    time_list = []
    for _ in tqdm(range(1000)):
        start_time = time.perf_counter()
        response = client.infer(model_name, inputs, outputs=outputs)
        end_time = time.perf_counter()
        time_list.append((end_time-start_time)*1000)

print(stats.describe(time_list))