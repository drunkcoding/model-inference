import ctypes
import json
import threading
from datasets.utils.filelock import logger
from torch import nn
import torch
from torch.utils.dlpack import to_dlpack
from torch.utils.dlpack import from_dlpack
import os
import asyncio

from transformers import AutoTokenizer, AutoConfig
from tritonclient.utils import np_to_triton_dtype

import triton_python_backend_utils as pb_utils
import json
import time

from hfutils.logger import Logger

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

        self.logger = Logger(__file__, "info", 5000000, 5)

        model_name = args['model_name']

        self.tokenizer = 

        name_components = model_name.split("_")
        self.model_name = name_components[-1]

        model_task = name_components[1]
        model_type = name_components[0]

        self.model_ensemble_name = "_".join([model_type, model_task, "ensemble"])

        model_repository = args["model_repository"]
        repository = os.path.join(model_repository, os.path.pardir)
        meta_file = os.path.join(repository, "meta.json")

        with open(meta_file, "r") as fp:
            config = json.load(fp)
            self.threshold = config[self.model_name]["threshold"]
            self.parallel_stages = config[self.model_name]["parallel_stages"]
            self.ensemble_pos = config[self.model_name]["ensemble_pos"]
            self.ensemble_weight = config[self.model_ensemble_name]["weights"][
                self.ensemble_pos
            ]

    async def execute(self, requests):
        responses = []

        exec_start_time = time.perf_counter()
        for request in requests:
            logits = self.parse_input(request, "logits")
            ensembled_logits = self.parse_input(request, "ensembled_logits")
            batch_mask = self.parse_input(request, "batch_mask")

            local_mask = batch_mask[self.ensemble_pos]
            self.logger.debug("local_mask %s", local_mask)

            if torch.any(local_mask):
                ensembled_logits = self.model_ensemble(
                    ensembled_logits, logits, local_mask
                ) if self.ensemble_pos != 0 else logits

                local_mask, max_prob = self.offload_mask(logits, local_mask)
                self.logger.debug(
                    "%s local_mask updated %s", self.model_name, local_mask
                )
                if torch.any(local_mask):
                    batch_mask = self.update_batch_mask(
                        max_prob, batch_mask, local_mask
                    )
                    self.logger.debug(
                        "%s batch_mask updated %s", self.model_name, batch_mask
                    )

            self.logger.debug(
                "%s outputs %s, batch_mask %s", self.model_name,
                ensembled_logits.shape,
                batch_mask
            )
            inference_response = pb_utils.InferenceResponse(
                output_tensors=[
                    self.parse_output(ensembled_logits, "ensembled_logits"),
                    self.parse_output(batch_mask, "batch_mask"),
                ]
            )
            responses.append(inference_response)

        exec_end_time = time.perf_counter()
        self.logger.info(
            "%s requests elapsed %s (ms)",
            self.model_name,
            (exec_end_time - exec_start_time) * 1000,
        )
        return responses

    def parse_input(self, request, field):
        input = pb_utils.get_input_tensor_by_name(request, field).as_numpy()
        input = from_dlpack(input.to_dlpack())
        return input.to(self.device) 

    def parse_output(self, output, field):
        return pb_utils.Tensor.from_dlpack(field, to_dlpack(output))

    def model_ensemble(self, ensemble_outputs, local_outputs, mask):
        start_time = time.perf_counter()
        outputs = ensemble_outputs
        outputs[mask] = (
            ensemble_outputs[mask] * (1 - self.ensemble_weight)
            + local_outputs * self.ensemble_weight
        )
        end_time = time.perf_counter()
        self.logger.info(
            "%s model_ensemble time elapsed %s (ms)",
            self.model_name,
            (end_time - start_time) * 1000,
        )
        self.logger.debug(
            "%s ensemble_outputs %s",
            self.model_name,
            outputs,
        )
        return outputs

    def update_batch_mask(self, max_prob, mask, local_mask):
        num_next_models = self.num_ensembles - self.ensemble_pos - 1
        base_step = (self.threshold - 0.25) / num_next_models
        for skip in range(num_next_models):
            skip_th_lower = base_step * (num_next_models - 1 - skip) + 0.25
            skip_th_upper = base_step * (num_next_models - skip) + 0.25
            skip_mask = (
                (max_prob >= skip_th_lower) & (max_prob < skip_th_upper) & local_mask
            )
            self.logger.debug(
                "%s skip_th_lower %s, skip_th_upper %s, skip_mask %s",
                self.model_name,
                skip_th_lower,
                skip_th_upper,
                skip_mask,
            )
            mask[skip + 1 + self.ensemble_pos] |= skip_mask

        mask[self.ensemble_pos] &= ~local_mask
        return mask

    def offload_mask(self, logits, mask):
        probabilities = torch.float_power(m(logits), 2)
        max_prob, _ = torch.max(probabilities, dim=-1)
        prob_mask = max_prob < self.threshold
        self.logger.debug(
            "%s (offload_mask) prob_mask %s %s",
            self.model_name,
            prob_mask,
            mask,
        )
        combined_mask = mask & prob_mask
        # combined_mask[mask] &= prob_mask[mask]
        self.logger.debug("max_prob %s, combined_mask %s", max_prob, combined_mask)
        return combined_mask, max_prob

    def finalize(self):
        print("Cleaning up...")
