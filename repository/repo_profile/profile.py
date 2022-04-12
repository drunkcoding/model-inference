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
import pandas as pd
from packaging.version import parse
import torch
from tritonclient.utils import *
import tritonclient.http as httpclient
import tritonclient.grpc as grpcclient
import numpy as np
from tqdm import tnrange, tqdm, trange
from scipy.special import softmax
import time


import logging

logger = logging.getLogger(__name__)

import datasets
from datasets import load_dataset, load_metric
from torch.utils.data import DataLoader

import torch.multiprocessing as mp

RUN_SEC = 15

model_name = "node_ensemble"
# model_name = "echo"
remote = "localhost"


inputs_list = []
for size in range(1, 21):
    data_size = 2 ** size

    data = np.ones(data_size, dtype=np.float32)
    inputs = [
        httpclient.InferInput("input", data.shape, np_to_triton_dtype(data.dtype)),
    ]

    inputs[0].set_data_from_numpy(data)
    outputs = [
        httpclient.InferRequestedOutput("output"),
    ]
    inputs_list.append((data_size, inputs))


records = {
    "req_time": list(),
    "rsp_time": list(),
    "latency": list(),
    "data_size": list(),
}

# with grpcclient.InferenceServerClient(f"{remote}:8001") as client:
with httpclient.InferenceServerClient(f"{remote}:8000", concurrency=1) as client:

    for data_size, input in tqdm(inputs_list, f"send"):

        start_time = time.perf_counter()
        ready = False
        while not ready:
            req_time = time.perf_counter()
            response = client.infer(model_name, input, outputs=outputs)
            rsp_time = time.perf_counter()

            curr_time = time.perf_counter()
            if curr_time - start_time > RUN_SEC:
                ready = True
                break

            if curr_time - start_time > 1:
                records['req_time'].append(req_time)
                records['rsp_time'].append(rsp_time)
                records['latency'].append(rsp_time-req_time)
                records['data_size'].append(data_size)

        df = pd.DataFrame(records)
        df.to_csv(os.path.join(os.path.dirname(__file__), "profile_same_numa.csv"), index=False)