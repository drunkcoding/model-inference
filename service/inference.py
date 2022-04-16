from dataclasses import dataclass
import json
import logging
import subprocess
import random
import re
from typing import List, Optional, Tuple
import numpy as np
import pandas as pd
import io
import torch
from torch.utils.dlpack import to_dlpack
from torch.utils.dlpack import from_dlpack
import os
import nn_pruning
import aiohttp
import concurrent
from requests import post as POST
from scipy.special import softmax

# from experiment_impact_tracker.compute_tracker import ImpactTracker

from transformers import (
    T5ForConditionalGeneration,
    AutoModelForQuestionAnswering,
    DistilBertForQuestionAnswering,
    ViTForImageClassification,
    GPT2LMHeadModel,
    AutoModelForCausalLM,
)

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
from hfutils.pipe.bert import (
    BERT_INPUTS,
    BERT_OUTPUTS,
    BertPyTorchPipeForQuestionAnswering,
)
from hfutils.pipe.vit import (
    VIT_INPUTS,
    VIT_OUTPUTS,
    ViTPyTorchPipeForImageClassification,
)
from hfutils.pipe.distilbert import (
    DISTILBERT_INPUTS,
    DISTILBERT_OUTPUTS,
    DistilBertPyTorchPipeForQuestionAnswering,
)
from hfutils.pipe.gpt import GPT_INPUTS, GPT_OUTPUTS, GPTLMHeadModelPipe
from hfutils.calibration import temperature_scale
from hfutils.constants import np_to_torch_dtype
import dill

m = torch.nn.Softmax(dim=1)

T5_TASK_LABELS = [1176, 6136, 59]  # HACK with GLUE labels

from multiprocessing import shared_memory


@dataclass
class ModelConfig:
    # name:   Optional[str]   # model full name
    path: Optional[str]  # model checkpoint path
    type: Optional[str]  # model type, e.g., t5 or bert
    stages: Optional[int]  # number of parallel stages
    ppos: Optional[int]  # current stage
    epos: Optional[int]  # current ensemble
    ens: Optional[int]  # number of total ensembles
    alpha: Optional[float]  # ensemble exp smooth weight
    temp: Optional[float]  # temperature scaling
    th: Optional[float]  # confidence threshold


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

        self.logger = Logger(__file__, logging.DEBUG, 50000000, 5)
        self.config = ModelConfig("", "", 0, 0, 0, 0, 0, 0, 0)

        self.model_config = json.loads(args["model_config"])
        self.device_id = int(args["model_instance_device_id"])
        self.device = torch.device("cuda:" + args["model_instance_device_id"])

        self._get_gpu_uuid()

        self.model_name = args["model_name"]  # HACK model_name = <TYPE>_e<epos>p<ppos>
        self.config.type, deploy = tuple(self.model_name.split("_"))
        groups = re.findall(r"e(\d+)p(\d+)", deploy)[0]
        # self.config.name = model_name[:model_name.rfind("_")] # HACK model_name always end with partx, indicating parallel stages
        self.config.ppos = int(groups[1])
        self.config.epos = int(groups[0])
        # self.config.type = model_name[:model_name.find("-")] # HACK model_name always start with type

        meta_path = os.path.join(args["model_repository"], os.path.pardir, "meta.json")
        print(meta_path, flush=True)
        self._read_cfg(meta_path)

        self._load_model()

        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=3,)

        # self.tracker = ImpactTracker(os.path.join("impact_tracker", self.model_name))
        # self.tracker.launch_impact_monitor()

    def _get_gpu_uuid(self):
        command = "nvidia-smi --query-gpu=index,uuid,gpu_bus_id --format=csv"

        result = subprocess.run(command.split(), stdout=subprocess.PIPE)
        # print(result.stdout)
        df = pd.read_csv(io.StringIO(result.stdout.decode("utf-8")), index_col="index")
        df = df.sort_index()
        df.iloc[:, 0] = df.iloc[:, 0].str.strip()
        self.gpu_uuid = df.iloc[self.device_id][" uuid"]
        # print(df)

    def _load_model(self):
        if "t5" == self.config.type:
            model = T5ForConditionalGeneration.from_pretrained(self.config.path)
            self.model = T5PyTorchPipe(model)

        if "bert" == self.config.type:
            model = AutoModelForQuestionAnswering.from_pretrained(self.config.path)
            self.model = BertPyTorchPipeForQuestionAnswering(model)

        if "distilbert" == self.config.type:
            model = DistilBertForQuestionAnswering.from_pretrained(self.config.path)
            self.model = DistilBertPyTorchPipeForQuestionAnswering(model)

        if "vit" == self.config.type:
            model = ViTForImageClassification.from_pretrained(self.config.path)
            self.model = ViTPyTorchPipeForImageClassification(model)

        if "gpt" == self.config.type:
            model = AutoModelForCausalLM.from_pretrained(self.config.path)
            self.model = GPTLMHeadModelPipe(model)

        self.model.eval()
        self.model.partition_by_parameter(self.config.ppos, self.config.stages)
        # self.model.partition_by_parameter(self.config.ppos, 4) # TEST MULTIPLEX
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
        self.config.path = os.path.join(config["base_dir"], config[model_name]["path"])
        self.config.temp = config[model_name]["temperature"]
        self.config.th = config[model_name]["threshold"]
        # self.config.epos = config[model_name]["ensemble_pos"]

        util_params = config[model_name]["util_params"]
        self.util_func = np.poly1d(util_params)

        self.is_last_stage = self.config.ppos == self.config.stages - 1
        # self.is_last_stage = False # TEST MULTIPLEX
        with open(
            f"/home/xly/model-inference/inference_dump/{model_name}_calibrator", "rb"
        ) as f:
            self.calibrator = dill.load(f)
        # HACK bert tiny model
        if "bert" == self.config.type and "distilbert" in self.config.path:
            self.config.type = "distilbert"

        self.logger.info("%s", self.config)

    @torch.no_grad()
    async def model_inference(self, args):
        start_time = time.perf_counter()
        uuid = random.randint(1e9, 2e9)
        self.logger.info(
            "%s inference[%s] start %s", self.model_name, uuid, time.time_ns(),
        )
        outputs = self.model(args)
        self.logger.info(
            "%s inference[%s] end %s", self.model_name, uuid, time.time_ns(),
        )
        if self.is_last_stage:
            # print(outputs.shape, flush=True)
            outputs = outputs.squeeze(1) / self.config.temp
            if "t5" == self.config.type:
                outputs = outputs[:, T5_TASK_LABELS]
            if "gpt" == self.config.type:
                outputs = outputs[:, -1, :50257]
            outputs = outputs.detach().cpu().numpy()
            # outputs = temperature_scale(outputs, self.config.temp)

        end_time = time.perf_counter()
        self.logger.info(
            "%s inference %s (ms)", self.model_name, (end_time - start_time) * 1000,
        )

        return outputs

    async def execute(self, requests):
        responses = []

        exec_start_time = time.perf_counter()
        for request in requests:

            # request_id = int(request.request_id())
            # correlation_id = int(request.correlation_id())
            # if self.is_last_stage:
            batch_mask = self.parse_input(request, "batch_mask").detach().cpu().numpy()
            hist_outputs = self.parse_input(request, "logits").detach().cpu().numpy()

            local_mask = batch_mask[self.config.epos]
            self.logger.debug("local_mask %s", local_mask)

            output_tensors = []
            # request_start_time = time.perf_counter()
            if np.any(local_mask):
                args = self.parse_request(request, local_mask)
                outputs = await self.model_inference(
                    args
                )  # MOVE TO CPU, SAVE GPU MEMORY

                # local_mask = local_mask.to(outputs.device)
                # ensemble_outputs = ensemble_outputs.to(outputs.device)
                start_time = time.perf_counter()
                if self.is_last_stage:

                    # self.logger.trace("%s outputs %s", self.model_name, outputs)

                    outputs = self.model_ensemble(
                        None
                        if self.config.epos == 0
                        or np.all(hist_outputs.astype(int) == 0)
                        else hist_outputs,
                        outputs,
                        local_mask,
                    )

                    # self.logger.trace(
                    #     "%s ensemble outputs %s", self.model_name, outputs
                    # )

                    local_mask, max_prob = self.offload_mask(outputs, local_mask)
                    self.logger.debug(
                        "%s local_mask updated %s", self.model_name, local_mask
                    )

                    if np.any(local_mask):
                        batch_mask = self.update_batch_mask(
                            max_prob, batch_mask, local_mask
                        )
                        # batch_mask[self.config.epos] &= ~local_mask
                        # batch_mask[self.config.epos + 1] |= local_mask
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
                end_time = time.perf_counter()
                self.logger.info(
                    "%s postprocessing time elapsed (%s, %s) %s (ms)",
                    self.model_name,
                    start_time,
                    end_time,
                    (end_time - start_time) * 1000,
                )

                if not isinstance(outputs, Tuple):
                    outputs = (outputs,)

                for output in outputs:
                    self.logger.debug(
                        "%s output %s", self.model_name, output.shape,
                    )

                output_tensors = self.parse_response(outputs)
                if not self.is_last_stage:
                    output_tensors += [
                        self.parse_output(hist_outputs, "logits"),
                        self.parse_output(batch_mask, "batch_mask"),
                    ]
                else:
                    output_tensors += [
                        self.parse_output(batch_mask, "batch_mask"),
                    ]
                # # TEST MULTIPLEX
                # inference_response = pb_utils.InferenceResponse(
                #     output_tensors=[
                #         self.parse_output(hist_outputs, "logits"),
                #         self.parse_output(batch_mask, "batch_mask"),
                #     ]
                # )
            else:
                output_tensors = [
                    self.parse_output(hist_outputs, "logits"),
                    self.parse_output(batch_mask, "batch_mask"),
                ]

                if "gpt" == self.config.type and not self.is_last_stage:
                    output_tensors += [
                        self.parse_output(np.zeros((1, 512, 4096)), "hidden_states"),
                    ]

            # request_end_time = time.perf_counter()
            # self.executor.submit(
            #     POST,
            #     url=f"http://127.0.0.1:10000/meter/{self.gpu_uuid}",
            #     json={
            #         "request_id": request_id,
            #         "correlation_id": correlation_id,
            #         "epos": self.config.epos,
            #         "ppos": self.config.ppos,
            #         "type": self.config.type,
            #         "start": request_start_time,
            #         "end": request_end_time,
            #         "util": self.util_func(np.sum(local_mask))
            #         if np.any(local_mask)
            #         else 0,
            #     },
            #     timeout=1,
            # )

            # output_tensors.append(self.parse_output(timestamp, "timestamp"))
            inference_response = pb_utils.InferenceResponse(
                output_tensors=output_tensors
            )
            responses.append(inference_response)

        exec_end_time = time.perf_counter()
        self.logger.info(
            "%s requests (%s, %s) %s (ms)",
            self.model_name,
            exec_end_time,
            exec_start_time,
            (exec_end_time - exec_start_time) * 1000,
        )
        return responses

    def parse_response(self, outputs):
        layer = self.model.layers[self.model.exec_map[-1] - 1]
        layer_name = type(layer).__name__
        output_tensors = []
        if "t5" == self.config.type:
            input_names = (
                T5_DECODER_OUTPUTS[layer_name]
                if layer.is_decoder
                else T5_ENCODER_OUTPUTS[layer_name]
            )
        if "bert" == self.config.type:
            input_names = BERT_OUTPUTS[layer_name]
        if "distilbert" == self.config.type:
            input_names = DISTILBERT_OUTPUTS[layer_name]
        if "vit" == self.config.type:
            input_names = VIT_OUTPUTS[layer_name]
        if "gpt" == self.config.type:
            input_names = GPT_OUTPUTS[layer_name]

        for i, name in enumerate(input_names):
            if "gpt" == self.config.type and name == "attention_mask":
                continue
            tensor = self.parse_output(outputs[i], name)
            output_tensors.append(tensor)

        return output_tensors

    def parse_request(self, request, local_mask):
        layer = self.model.layers[self.model.exec_map[0]]
        layer_name = type(layer).__name__
        args = []
        if "t5" == self.config.type:
            input_names = (
                T5_DECODER_INPUTS[layer_name]
                if layer.is_decoder
                else T5_ENCODER_INPUTS[layer_name]
            )
        if "bert" == self.config.type:
            input_names = BERT_INPUTS[layer_name]
        if "distilbert" == self.config.type:
            input_names = DISTILBERT_INPUTS[layer_name]
        if "vit" == self.config.type:
            input_names = VIT_INPUTS[layer_name]
        if "gpt" == self.config.type:
            input_names = GPT_INPUTS[layer_name]

        for name in input_names:
            arg = self.parse_input(request, name)
            if self.config.epos == 0 and self.config.ppos == 0:
                arg = arg[local_mask]
            args.append(arg)
        return tuple(args)

    def parse_input(self, request, field):
        # self.logger.debug("parse_input: request %s, field %s", request, field)
        input = pb_utils.get_input_tensor_by_name(request, field)
        if input is None:
            return
        # input = from_dlpack(input.to_dlpack())
        input = input.as_numpy()
        input = torch.as_tensor(
            input, dtype=np_to_torch_dtype(input.dtype), device=self.device
        )
        return input

    def parse_output(self, output, field):
        # return pb_utils.Tensor.from_dlpack(field, to_dlpack(output))
        output_config = pb_utils.get_output_config_by_name(self.model_config, field)
        output_dtype = pb_utils.triton_string_to_numpy(output_config["data_type"])
        if isinstance(output, torch.Tensor):
            output = output.detach().cpu().numpy()
        output_pb = pb_utils.Tensor(field, output.astype(output_dtype))
        return output_pb

    def update_batch_mask(self, max_prob, mask, local_mask):
        start_time = time.perf_counter()
        num_next_models = self.config.ens - self.config.epos - 1
        base_step = self.config.th / num_next_models
        for skip in range(num_next_models):
            skip_th_lower = base_step * (num_next_models - 1 - skip)
            skip_th_upper = base_step * (num_next_models - skip)
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
        end_time = time.perf_counter()
        self.logger.info(
            "%s update_batch_mask time elapsed (%s,%s) %s (ms)",
            start_time,
            end_time,
            self.model_name,
            (end_time - start_time) * 1000,
        )
        return mask

    def offload_mask(self, logits, mask):
        start_time = time.perf_counter()
        probabilities = np.power(softmax(logits, axis=1), 2)
        max_prob = self.calibrator.calibrate(probabilities)
        prob_mask = max_prob < self.config.th

        # prob_mask = np.all(probabilities < self.config.th, axis=1)
        # max_prob = np.max(probabilities, axis=1)
        if "bert" in self.config.type:
            prob_mask = prob_mask.squeeze(1)
            max_prob = max_prob.squeeze(1)
        # max_prob = np.max(probabilities, axis=1)
        # # probabilities = torch.float_power(m(logits), 2)
        # # max_prob, _ = torch.max(probabilities, dim=1)
        # if "bert" in self.config.type:
        #     # max_prob, _ = torch.min(max_prob.squeeze(1), dim=1)
        #     max_prob = np.min(max_prob.squeeze(1), axis=1)
        # prob_mask = max_prob < self.config.th
        self.logger.debug(
            "(offload_mask) prob_mask %s %s", prob_mask, mask,
        )
        combined_mask = mask & prob_mask
        # combined_mask[mask] &= prob_mask[mask]
        self.logger.debug("max_prob %s, combined_mask %s", max_prob, combined_mask)
        end_time = time.perf_counter()
        self.logger.info(
            "%s offload_mask time elapsed (%s, %s) %s (ms)",
            self.model_name,
            start_time,
            end_time,
            (end_time - start_time) * 1000,
        )
        return combined_mask, max_prob

    def model_ensemble(self, hist_outputs, outputs, mask):
        start_time = time.perf_counter()
        if hist_outputs is not None:
            outputs[mask] = (
                hist_outputs[mask] * (1 - self.config.alpha)
                + outputs[mask] * self.config.alpha
            )
        end_time = time.perf_counter()
        self.logger.info(
            "%s model_ensemble time elapsed (%s, %s) %s (ms)",
            self.model_name,
            start_time,
            end_time,
            (end_time - start_time) * 1000,
        )
        return outputs

    def finalize(self):
        # self.tracker.stop()
        print("Cleaning up...")
