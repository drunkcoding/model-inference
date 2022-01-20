
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

from transformers import AutoModelForSequenceClassification
import triton_python_backend_utils as pb_utils
from partitioner import GPTModelPipe, get_attn_mask

from deepspeed.utils.zero_to_fp32 import load_state_dict_from_zero_checkpoint

class TritonPythonModel:
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
        print(model_config)
        # exit()
        self.model_instance_device_id = "cuda:" + args['model_instance_device_id']

        model_repository = args['model_repository']
        model_version = args['model_version']
        model_name = args['model_name']
        model_name_or_path = os.path.join(
            os.path.join(model_repository, model_version),
            "model"
        )
        hf_model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path)
        hf_model = load_state_dict_from_zero_checkpoint(hf_model, model_name_or_path)
        # hf_model.eval()
        self.cls_model = GPTModelPipe(hf_model.config, "classification", hf_model)
        self.part = int(model_name[-1]) # HACK one digit partition
        uniform = self.cls_model.num_layers // 4
        self.cls_model.exec_map = (uniform * self.part, uniform * (self.part + 1) if self.part != 3 else self.cls_model.num_layers)
        print(model_name, self.part, self.cls_model.exec_map)
        self.cls_model = self.cls_model.to(self.model_instance_device_id)
        # self.cls_model.half()


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

    def execute(self, requests):
        responses = []
        for request in requests:
            input_ids = pb_utils.get_input_tensor_by_name(request, "input_ids")
            attention_mask = pb_utils.get_input_tensor_by_name(request, "attention_mask")
            hidden_states = pb_utils.get_input_tensor_by_name(request, "hidden_states")



            args = (
                torch.Tensor(input_ids.as_numpy()).long().to(self.model_instance_device_id), 
                get_attn_mask(torch.Tensor(attention_mask.as_numpy())).to(self.model_instance_device_id),
            )

            if hidden_states != None:
                hidden_states = torch.Tensor(hidden_states.as_numpy()).to(self.model_instance_device_id)
                args = (hidden_states, ) + args
            
            with torch.no_grad():
                outputs = self.cls_model.forward_layers(args).detach().cpu().numpy()

            outputs_tensor = pb_utils.Tensor("outputs",
                                           outputs.astype(self.outputs_dtype))
            inference_response = pb_utils.InferenceResponse(
                output_tensors=[outputs_tensor])
            responses.append(inference_response)

        return responses

    def finalize(self):
        print('Cleaning up...')
