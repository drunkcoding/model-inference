import ctypes
import json
import threading
from datasets.utils.filelock import logger
import torch
import os
import asyncio

import triton_python_backend_utils as pb_utils
import numpy as np
import multiprocessing as mp
import json
import time
from hfutils.constants import np_to_torch_dtype
import gc

# from deepspeed.utils.zero_to_fp32 import load_state_dict_from_zero_checkpoint

m = torch.nn.Softmax(dim=-1)

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

        self.device = "cuda:0"
        self.model_config = model_config = json.loads(args["model_config"])

    async def execute(self, requests):
        responses = []

        for request in requests:
            input_ids = self.parse_input(request, "input_ids")
            attention_mask = self.parse_input(request, "attention_mask")

            outputs = np.random.random((input_ids.shape[0], 2))

            inference_response = pb_utils.InferenceResponse(
                output_tensors=[
                    self.parse_output(outputs, "outputs")
                ]
            )
            responses.append(inference_response)
        return responses

    def parse_input(self, request, field):
        input = pb_utils.get_input_tensor_by_name(request, field).as_numpy()
        # input = torch.as_tensor(input).type(np_to_torch_dtype(input.dtype))
        # input = input.to(self.device)
        input_zero_copy = torch.as_tensor(input, dtype=np_to_torch_dtype(input.dtype), device=self.device)

        return input_zero_copy

    def parse_output(self, output, field):
        output_config = pb_utils.get_output_config_by_name(self.model_config, field)
        output_dtype = pb_utils.triton_string_to_numpy(output_config["data_type"])

        if isinstance(output, torch.Tensor):
            output = output.detach().cpu().numpy()

        output_pb = pb_utils.Tensor(field, output.astype(output_dtype))
        return output_pb

    def finalize(self):
        print("Cleaning up...")
 