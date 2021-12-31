
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

import ctypes
import json
import threading
from torch import nn
import torch
import os
import asyncio


from transformers import AutoModelForSequenceClassification
from tritonclient.utils import np_to_triton_dtype
from watchdog.events import EVENT_TYPE_MODIFIED
from triton_inference.event import CfgHandler

# triton_python_backend_utils is available in every Triton Python model. You
# need to use this module to create inference requests and responses. It also
# contains some utility functions for extracting information from model_config
# and converting Triton input/output types to numpy types.
import triton_python_backend_utils as pb_utils
import tritonclient.http as httpclient
import tritonclient.grpc as grpcclient
from scipy.special import softmax
import numpy as np
import multiprocessing as mp
import json
import time

from deepspeed.utils.zero_to_fp32 import load_state_dict_from_zero_checkpoint

class TritonPythonModel:
    """Your Python model must use the same class name. Every Python model
    that is created must have "TritonPythonModel" as the class name.
    """
    def initialize(self, args):
        """`initialize` is called only once when the model is being loaded.
        Implementing `initialize` function is optional. This function allows
        the model to intialize any state associated with this model.
        Parameters
        ----------
        args : dict
          Both keys and values are strings. The dictionary keys and values are:
          * model_config: A JSON string containing the model configuration
          * model_instance_kind: A string containing model instance kind
          * model_instance_device_id: A string containing model instance device ID
          * model_repository: Model repository path
          * model_version: Model version
          * model_name: Model name
        """

        # You must parse model_config. JSON string is not parsed here
        self.model_config = model_config = json.loads(args['model_config'])
        # print(model_config)

        outputs_config = pb_utils.get_output_config_by_name(
            model_config, "outputs")
        input_ids_config = pb_utils.get_input_config_by_name(
            model_config, "input_ids")
        attention_mask_config = pb_utils.get_input_config_by_name(
            model_config, "attention_mask")


        # Convert Triton types to numpy types
        self.outputs_dtype = pb_utils.triton_string_to_numpy(
            outputs_config['data_type'])
        self.input_ids_dtype = pb_utils.triton_string_to_numpy(
            input_ids_config['data_type'])
        self.attention_mask_dtype = pb_utils.triton_string_to_numpy(
            attention_mask_config['data_type'])

        self.model_instance_device_id = "cuda:" + args['model_instance_device_id']

        self.model_repository = model_repository = args['model_repository']
        model_version = args['model_version']
        self.model_name = model_name = args['model_name']
        model_name_or_path = os.path.join(
            os.path.join(model_repository, model_version),
            "model"
        )
        self.cls_model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path)
        self.cls_model = load_state_dict_from_zero_checkpoint(self.cls_model, model_name_or_path)
        self.cls_model = self.cls_model.to(self.model_instance_device_id)
        # self.cls_model.parallelize()
        self.cls_model.eval()

        # shared values
        # self.load_meta_event = mp.Event()

        
        repository = os.path.join(self.model_repository, os.path.pardir)
        meta_file = os.path.join(repository, "meta.json")
        print(meta_file)
        
        # self.cfg_handler = CfgHandler()
        # self.cfg_handler.register_action(self.load_meta, EVENT_TYPE_MODIFIED)
        # self.cfg_handler.begin_watch(meta_file)

        self.cfg_timer = threading.Timer(10, self.read_cfg, (meta_file, ))
        self.cfg_timer.start()
        # self.read_meta = mp.Process(target=self.load_meta)
        # self.load_meta_event.set()
        # self.read_meta.start()
        
        # self.cls_model = deepspeed.init_inference(cls_model, mp_size=torch.cuda.device_count())

        # self.client = httpclient.InferenceServerClient("172.31.124.43:8000")

    def read_cfg(self, path):
        with open(path, "r") as fp:
            config = json.load(fp)
            print(config)
            self.threshold = mp.Value(ctypes.c_float, config[self.model_name]['threshold'])
            self.temperature = mp.Value(ctypes.c_float, config[self.model_name]['temperature'])
        self.cfg_timer.cancel()
        self.cfg_timer = threading.Timer(10, self.read_cfg, (path, ))
        self.cfg_timer.start()

    def load_meta(self, config):
        print("reload config", config)
        self.threshold.value = config[self.model_name]['threshold']
        self.temperature.value = config[self.model_name]['temperature']

    # def load_meta(self):
    #     repository = os.path.join(self.model_repository, os.path.pardir)
    #     meta_file = os.path.join(repository, "meta.json")
    #     while self.load_meta_event.is_set():
    #         try:
    #             with open(meta_file, "r") as fp:
    #                 meta = json.load(fp)
    #                 # write to shared value, in other places read only
    #                 self.threshold.value = meta[self.model_name]['threshold']
    #                 self.temperature.value = meta[self.model_name]['temperature']
    #             # print("%s: temperature %s, threshold %s" % (self.model_name, self.temperature.value, self.threshold.value))
    #         except:
    #             pass
    #         time.sleep(10)

    def mask_offload(self, logits):
        if len(logits.shape) == 1:
            probabilities = np.abs(logits - 0.5) * 2
        else:
            probabilities = softmax(logits / self.temperature.value, axis=1)
        max_prob = np.max(probabilities, axis=1)
        # print(max_prob, self.threshold.value, probabilities)
        mask = max_prob < self.threshold.value
        return mask

    async def execute(self, requests):

        responses = []

        # Every Python backend must iterate over everyone of the requests
        # and create a pb_utils.InferenceResponse for each of them.
        # infer_response_awaits = []

        infer_response_awaits = []
        infer_outputs = []
        infer_masks = []
        for request in requests:
            input_ids = pb_utils.get_input_tensor_by_name(request, "input_ids").as_numpy()
            attention_mask = pb_utils.get_input_tensor_by_name(request, "attention_mask").as_numpy()
            
            with torch.no_grad():
                local_outputs = self.cls_model(
                    input_ids=torch.Tensor(input_ids).long().to(self.model_instance_device_id), 
                    attention_mask=torch.Tensor(attention_mask).long().to(self.model_instance_device_id), 
                    return_dict=True).logits.detach().cpu().numpy()

            mask = self.mask_offload(local_outputs)
            # remote_outputs = None
            if np.any(mask):
                # print("offload %s/%s" % (np.sum(mask), local_outputs.shape[0]))
                input_ids = input_ids[mask]
                attention_mask = attention_mask[mask]

                inference_request = pb_utils.InferenceRequest(
                    model_name='gpt_neo_cola_ensemble',
                    requested_output_names=['outputs'],
                    inputs=[
                        pb_utils.Tensor("input_ids", input_ids.astype(self.input_ids_dtype)),
                        pb_utils.Tensor("attention_mask", attention_mask.astype(self.attention_mask_dtype)),
                    ])
                
                # inference_response = inference_request.async_exec()
                infer_response_awaits.append(inference_request.async_exec())
            infer_outputs.append(local_outputs)
            infer_masks.append(mask)

        # Wait for all of the inference requests to complete.
        infer_responses = await asyncio.gather(*infer_response_awaits)
        idx = 0
        for i in range(len(infer_outputs)):
            mask = infer_masks[i]
            outputs = infer_outputs[i]
            if np.any(mask):
                infer_response = infer_responses[idx]
                if infer_response.has_error():
                    raise pb_utils.TritonModelException(infer_response.error().message())
                else:
                    # Extract the output tensors from the inference response.
                    remote_outputs = pb_utils.get_output_tensor_by_name(infer_response, 'outputs').as_numpy()
                outputs[mask] = remote_outputs
                idx += 1

            outputs_tensor = pb_utils.Tensor("outputs", outputs.astype(self.outputs_dtype))
            inference_response = pb_utils.InferenceResponse(
                output_tensors=[outputs_tensor])
            responses.append(inference_response)
        
        # You should return a list of pb_utils.InferenceResponse. Length
        # of this list must match the length of `requests` list.
        return responses

    def finalize(self):
        # self.load_meta_event.clear()
        # self.read_meta.terminate()
        # self.cfg_handler.stop_watch()
        self.cfg_timer.cancel()
        print('Cleaning up...')
