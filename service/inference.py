from dataclasses import dataclass
import json
import re
from typing import Optional
import torch
from torch.utils.dlpack import to_dlpack
from torch.utils.dlpack import from_dlpack
import os

from experiment_impact_tracker.compute_tracker import ImpactTracker

from transformers import T5ForConditionalGeneration

import triton_python_backend_utils as pb_utils
import json
import time

from hfutils.logger import Logger
import gc
from hfutils.pipe.t5 import (
    T5_ENCODER_INPUTS,
    T5_ENCODER_OUTPUTS,
    T5_DECODER_INPUTS,
    T5_DECODER_OUTPUTS,
    T5PyTorchPipe,
)
from hfutils.calibration import temperature_scale

m = torch.nn.Softmax(dim=-1)

T5_TASK_LABELS = [1176, 6136, 59] # HACK with GLUE labels

@dataclass
class ModelConfig:
    # name:   Optional[str]   # model full name
    path:   Optional[str]   # model checkpoint path
    type:   Optional[str]   # model type, e.g., t5 or bert
    stages: Optional[int]   # number of parallel stages
    ppos:   Optional[int]   # current stage
    epos:   Optional[int]   # current ensemble
    ens:   Optional[int]    # number of total ensembles
    alpha:  Optional[float] # ensemble exp smooth weight
    temp:   Optional[float] # temperature scaling
    th:     Optional[float] # confidence threshold

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

        self.logger = Logger(__file__, "debug", 5000000, 5)
        self.config = ModelConfig()

        self.model_name = args["model_name"] # HACK model_name = <TYPE>_e<epos>p<ppos>
        self.config.type, deploy = tuple(self.model_name.split("_"))
        groups = re.findall(r"e(\d+)p(\d+)", deploy)[0]
        # self.config.name = model_name[:model_name.rfind("_")] # HACK model_name always end with partx, indicating parallel stages
        self.config.ppos = int(groups[1])
        self.config.epos = int(groups[0])
        # self.config.type = model_name[:model_name.find("-")] # HACK model_name always start with type

        meta_path = os.path.join(args["model_repository"], os.path.pardir, "meta.json")
        print(meta_path, flush=True)
        self._read_cfg(meta_path)

        self.device = torch.device("cuda:" + args["model_instance_device_id"])

        self._load_model()

        self.tracker = ImpactTracker(os.path.join("impact_tracker", self.model_name), (55.953251, -3.188267))
        self.tracker.launch_impact_monitor()

    def _load_model(self):
        if self.config.type == "t5":
            model = T5ForConditionalGeneration.from_pretrained(self.model_name_or_path)
            self.model = T5PyTorchPipe(model)
        
        
        self.model.eval()
        self.model.partition_by_parameter(self.config.stages, self.config.ppos)
        # total_pp_layers = len(self.model.layers)
        # pp_layers = int(total_pp_layers / self.parallel_stage)

        # self.model.exec_map = (
        #     pp_layers * self.parallel_pos, 
        #     pp_layers * (self.parallel_pos + 1) if self.parallel_pos < self.parallel_stage - 1 else total_pp_layers
        # )
        self.model.convert(self.device)

        del model
        gc.collect()
        torch.cuda.empty_cache()

    def _read_cfg(self, path):
        ensemble_name = "_".join([self.config.type, "ensemble"])
        
        with open(path, "r") as fp:
            config = json.load(fp)

        self.config.alpha = config[ensemble_name]["alpha"]
        self.config.ens = len(config[ensemble_name]["ensembles"])

        model_name = config[ensemble_name]["ensembles"][self.config.epos]
        self.config.stages = config[model_name]["parallel_stages"]
        self.config.path = config[model_name]["path"]
        self.config.temp = config[model_name]["temperature"]
        self.config.th = config[model_name]["threshold"]
        # self.config.epos = config[model_name]["ensemble_pos"]

        self.is_last_stage = self.config.ppos == self.config.stages - 1

        self.logger.info("%s",self.config)

    @torch.no_grad()
    async def model_inference(self, args):
        start_time = time.perf_counter()

        outputs = self.model(args)
        if self.is_last_stage:
            if self.config.type == "t5":
                outputs = outputs[:, T5_TASK_LABELS]
                outputs = temperature_scale(outputs, self.config.temp)

        end_time = time.perf_counter()
        self.logger.info(
            "%s model_inference time elapsed %s (ms)",
            self.model_name,
            (end_time - start_time) * 1000,
        )

        return outputs

    async def execute(self, requests):
        responses = []

        exec_start_time = time.perf_counter()
        for request in requests:

            local_mask = batch_mask[self.config.epos]
            self.logger.debug("local_mask %s", local_mask)

            if torch.any(local_mask):
                args = self.parse_request(request, local_mask)
                outputs = await self.model_inference(args)  # MOVE TO CPU, SAVE GPU MEMORY
                # local_mask = local_mask.to(outputs.device)
                # ensemble_outputs = ensemble_outputs.to(outputs.device)

                if self.is_last_stage:

                    hist_outputs = self.parse_input(request, "outputs")
                    batch_mask = self.parse_input(request, "batch_mask")

                    outputs = self.model_ensemble(
                        hist_outputs, outputs, local_mask
                    )

                    local_mask, max_prob = self.offload_mask(outputs, local_mask)
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

                    # inference_response = pb_utils.InferenceResponse(
                    #     output_tensors=[
                    #         self.parse_output(outputs, "outputs"),
                    #         self.parse_output(batch_mask, "batch_mask"),
                    #     ]
                    # )
                    # responses.append(inference_response)
                # else:

            inference_response = pb_utils.InferenceResponse(
                output_tensors=self.parse_output(outputs) + [self.parse_output(batch_mask, "batch_mask")]
            )
            responses.append(inference_response)
            
        exec_end_time = time.perf_counter()
        self.logger.info(
            "%s requests handle time elapsed %s (ms)",
            self.model_name,
            (exec_end_time - exec_start_time) * 1000,
        )
        return responses

    def parse_response(self, outputs):
        layer = self.model.layers[self.model.exec_map[-1] - 1]
        layer_name = type(layer).__name__
        output_tensors = []
        if self.config.type == "t5":
            input_names = T5_DECODER_OUTPUTS[layer_name] if layer.is_decoder else T5_ENCODER_OUTPUTS[layer_name]
            for i, name in enumerate(input_names):
                tensor = self.parse_output(outputs[i], name)
                output_tensors.append(tensor)

        return output_tensors


    def parse_request(self, request, local_mask):
        layer = self.model.layers[self.model.exec_map[0]]
        layer_name = type(layer).__name__
        args = []
        if self.config.type == "t5":
            input_names = T5_DECODER_INPUTS[layer_name] if layer.is_decoder else T5_ENCODER_INPUTS[layer_name]
            for name in input_names:
                arg = self.parse_input(request, name)
                if self.config.epos == 0 and self.config.ppos == 0:
                    arg = arg[local_mask]
                args.append(arg)
        args = tuple(args)
        return args

    def parse_input(self, request, field):
        input = pb_utils.get_input_tensor_by_name(request, field).as_numpy()
        input = from_dlpack(input.to_dlpack())
        return input.to(self.device) 

    def parse_output(self, output, field):
        return pb_utils.Tensor.from_dlpack(field, to_dlpack(output))

    def update_batch_mask(self, max_prob, mask, local_mask):
        num_next_models = len(local_mask) - self.config.epos - 1
        base_step = (self.config.th - 0.25) / num_next_models
        for skip in range(num_next_models):
            skip_th_lower = base_step * (num_next_models - 1 - skip) + 0.25
            skip_th_upper = base_step * (num_next_models - skip) + 0.25
            skip_mask = (
                (max_prob >= skip_th_lower) & (max_prob < skip_th_upper) & local_mask
            )
            self.logger.debug(
                "skip_th_lower %s, skip_th_upper %s, skip_mask %s",
                skip_th_lower,
                skip_th_upper,
                skip_mask,
            )
            mask[skip + 1 + self.config.epos] |= skip_mask

        mask[self.config.epos] &= ~local_mask
        return mask

    def offload_mask(self, logits, mask):
        probabilities = torch.float_power(m(logits), 2)
        max_prob, _ = torch.max(probabilities, dim=-1)
        prob_mask = max_prob < self.threshold
        self.logger.debug(
            "(offload_mask) prob_mask %s %s",
            prob_mask,
            mask,
        )
        combined_mask = mask & prob_mask
        # combined_mask[mask] &= prob_mask[mask]
        self.logger.debug("max_prob %s, combined_mask %s", max_prob, combined_mask)
        return combined_mask, max_prob

    def model_ensemble(self, ensemble_outputs, local_outputs, mask):
        start_time = time.perf_counter()
        outputs = ensemble_outputs
        outputs[mask] = (
            ensemble_outputs[mask] * (1 - self.config.alpha)
            + local_outputs * self.config.alpha
        )
        end_time = time.perf_counter()
        self.logger.info(
            "%s model_ensemble time elapsed %s (ms)",
            self.model_name,
            (end_time - start_time) * 1000,
        )
        return outputs

    def finalize(self):
        self.tracker.stop()
        print("Cleaning up...")
