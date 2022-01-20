
# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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
from torch import nn
import torch
import os
import requests
import multiprocessing as mp
import requests
import subprocess
import time
import schedule
import threading
import asyncio

from triton_inference.measure import ModelMetricsWriter, ModelMetricsWriterBackend

class TritonPythonModel:
    def initialize(self, args):
        self.lock = mp.Value('i', 0)
        self.lock.value = 1

        remote = "dgj101"
        tensorboard_base = "/jmain01/home/JAD003/sxr06/lxx22-sxr06/model-inference/tritonserver/"
        tensorboard_logdir = os.path.join(tensorboard_base, "server")

        self.writer_backend = ModelMetricsWriterBackend(tensorboard_logdir)
        self.writer_backend.remote = remote
        self.writer_backend.start()
        
    # def monitor(self):
    #     time.sleep(60)
    #     print("monitor started")
    #     remote = "dgj101"
    #     tensorboard_base = "/jmain01/home/JAD003/sxr06/lxx22-sxr06/model-inference/tritonserver/"
    #     tensorboard_logdir = os.path.join(tensorboard_base, "server")

    #     writer_backend = ModelMetricsWriterBackend(tensorboard_logdir)
    #     writer_backend.remote = remote
    #     writer_backend.start()
        
    #     while self.lock.value > 0:
    #         r = subprocess.run(['curl', 'http://dgj101:8002/metrics'], capture_output=True, text=True)
    #         # self.f.write("timestamp %s\n" % time.time())
    #         # self.f.write(r.stdout)
    #         writer.text = r.stdout
    #         writer.nv_energy_consumption()
    #         writer.nv_gpu_utilization()
    #         writer.nv_gpu_memory_used_bytes()
    #         writer.nv_gpu_power_usage()
    #         time.sleep(0.1)
    #     writer.writer.close()
    #     # r = subprocess.run(['curl', 'http://172.31.124.43:8002/metrics'], capture_output=True, text=True)
    #     # self.f.write("timestamp %s" % time.time())
    #     # self.f.write(r.stdout)


    def execute(self, requests):
        pass

    def finalize(self):
        # self.lock.value = 0
        # self.job.terminate()
        self.writer_backend.stop()
        print('Cleaning up...')
