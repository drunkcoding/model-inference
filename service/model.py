import ctypes
import json
import threading
from datasets.utils.filelock import logger
from torch import nn
import torch
import os
import asyncio

from transformers import AutoTokenizer, AutoConfig
from tritonclient.utils import np_to_triton_dtype
from watchdog.events import EVENT_TYPE_MODIFIED

import triton_python_backend_utils as pb_utils
from scipy.special import softmax
import numpy as np
import multiprocessing as mp
import json
import time

from hfutils.logger import Logger
import gc
from hfutils.model_pipe import T5Pipe
from hfutils.calibration import agg_logits, temperature_scale
from hfutils.constants import (
    MODEL_TASK_TO_CLASS,
    ENSEMBLE_ORDER,
    TASK_TO_LABELS,
    np_to_torch_dtype,
)

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

        self.logger = Logger(__file__, "debug", 5000000, 5)

        # You must parse model_config. JSON string is not parsed here
        self.model_config = model_config = json.loads(args["model_config"])
        model_name = args["model_name"]
        self.model_name = model_name if "part" not in model_name else model_name[:model_name.rfind("_")]
        self.model_repository = model_repository = args["model_repository"]

        model_version = args["model_version"]
        model_name_or_path = os.path.join(model_repository, model_version, "model", "")

        self.device = "cuda:" + args["model_instance_device_id"]

        self.model_parallel = "part" in model_name
        if self.model_parallel:
            self.parallel_pos = int(model_name[-1])
            self.exec_map = (0, 28) if self.parallel_pos == 0 else (28, 56)

        cls_model_config = AutoConfig.from_pretrained(model_name_or_path)
        model_task = cls_model_config.finetuning_task
        model_type = model_name[: model_name.find("-")]

        self.model_ensemble_name = "_".join([model_type, model_task, "ensemble"])
        # model_type = self.cls_model_config.model_type

        repository = os.path.join(self.model_repository, os.path.pardir)
        meta_file = os.path.join(repository, "meta.json")
        print(meta_file, flush=True)
        self.read_cfg(meta_file)

        # print(model_name_or_path)
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=False)
        self.label_tokens = [
            tokenizer(label, max_length=2).input_ids[0]
            for label in TASK_TO_LABELS[model_task]
            if label is not None
        ]
        self.cls_model = MODEL_TASK_TO_CLASS[model_task][model_type].from_pretrained(
            model_name_or_path
        )

        # self.cls_model = load_state_dict_from_zero_checkpoint(self.cls_model, model_name_or_path)

        if self.model_parallel:
            self.cls_model = T5Pipe(self.cls_model, self.exec_map).to(self.device)
        else:
            self.cls_model = self.cls_model.to(self.device)
        self.cls_model.eval()

        gc.collect()
        torch.cuda.empty_cache()

        # shared values
        # self.load_meta_event = mp.Event()


        # self.cfg_handler = CfgHandler()
        # self.cfg_handler.register_action(self.load_meta, EVENT_TYPE_MODIFIED)
        # self.cfg_handler.begin_watch(meta_file)

        # self.threshold = mp.Value(ctypes.c_float, 0.0)
        # self.temperature = mp.Value(ctypes.c_float, 1.0)

        # self.threshold = 0.0
        # self.temperature = torch.nn.Parameter(torch.ones(1, device=self.device) * 1.0)
        # self.ensemble_weight = 1.0
        # self.ensemble_pos = None
        # self.num_ensembles = 1

        

        # self.cfg_timer = threading.Timer(10, self.read_cfg, (meta_file,))
        # self.cfg_timer.start()
        # self.read_meta = mp.Process(target=self.load_meta)
        # self.load_meta_event.set()
        # self.read_meta.start()

        # # self.cls_model = deepspeed.init_inference(cls_model, mp_size=torch.cuda.device_count())
        # self.next_models = []
        # for i, model_size in enumerate(ENSEMBLE_ORDER):
        #     if model_size in model_name:
        #         self.ensemble_pos = i
        #         self.next_models = [
        #             model_name.replace(model_size, text)
        #             for text in ENSEMBLE_ORDER[i + 1 : -1]
        #         ]

        # self.logger.info(
        #     "%s extracted: next_model_name %s, model_name %s",
        #     args["model_name"],
        #     self.next_models,
        #     self.model_name,
        # )

    def read_cfg(self, path):
        torch.cuda.empty_cache()
        gc.collect()
        try:
            with open(path, "r") as fp:
                config = json.load(fp)
                self.parallel_stages = config[self.model_name]["parallel_stages"]
                self.threshold = config[self.model_name]["threshold"]
                temperature = config[self.model_name]["temperature"]
                self.temperature = torch.nn.Parameter(
                    torch.ones(1, device=self.device) * temperature
                )
                self.ensemble_pos = config[self.model_name]["ensemble_pos"]
                self.ensemble_weight = config[self.model_ensemble_name]["weights"][
                    self.ensemble_pos
                ]
                self.num_ensembles = len(config[self.model_ensemble_name]["weights"])
                self.logger.info(
                    "%s load meta from %s \n threshold %s, temperature %s, ensemble_pos %s, ensemble_weight %s, num_ensembles %s",
                    self.model_name,
                    path,
                    self.threshold,
                    self.temperature,
                    self.ensemble_pos,
                    self.ensemble_weight,
                    self.num_ensembles,
                )
        except Exception as e:
            self.logger.warn("%s", e)

        self.cfg_timer = threading.Timer(10, self.read_cfg, (path,))
        self.cfg_timer.start()

    # def load_meta(self, config):
    #     # print("reload config", config)
    #     self.threshold.value = config[self.model_name]["threshold"]
    #     self.temperature.value = config[self.model_name]["temperature"]

    # def mask_offload(self, logits):
    #     probabilities = np.power(softmax(logits, axis=-1), 2)
    #     max_prob = np.max(probabilities, axis=-1)
    #     mask = max_prob < self.threshold
    #     self.logger.debug("max_prob %s, mask %s", max_prob, mask)
    #     return mask, max_prob.flatten()

    def t5_parallel_inference(self, input_ids, attention_mask, ensemble_outputs):
        batch_size = input_ids.shape[0]
        outputs = self.cls_model(
            (
                None if self.parallel_pos == 0 else ensemble_outputs,
                None,
                None,
                None,
                input_ids,
                attention_mask,
            )
        )

        if self.parallel_pos == self.parallel_stages - 1:
            logits = outputs[1].view(batch_size, -1)[:, self.label_tokens]
            logits = temperature_scale(logits, self.temperature)
        else:
            return outputs[0]

    def t5_inference(self, input_ids, attention_mask):
        outputs = self.cls_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            do_sample=False,  # disable sampling to test if batching affects output
            return_dict_in_generate=True,
            output_scores=True,
        )
        logits = outputs.scores[0][:, self.label_tokens]
        logits = temperature_scale(logits, self.temperature)
        return logits

    def default_inference(self, input_ids, attention_mask):
        logits = self.cls_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        ).logits
        logits = temperature_scale(logits, self.temperature)
        return logits

    async def execute(self, requests):
        responses = []

        exec_start_time = time.perf_counter()
        for request in requests:
            input_ids = self.parse_input(request, "input_ids")
            attention_mask = self.parse_input(request, "attention_mask")
            ensemble_outputs = self.parse_input(request, "ensemble_outputs")
            batch_mask = self.parse_input(request, "batch_mask")

            local_mask = batch_mask[self.ensemble_pos]
            self.logger.debug("local_mask %s", local_mask)

            # if self.model_parallel and self.parallel_pos > 0:
            #     ensemble_outputs = ensemble_outputs.reshape(input_ids.shape + (self.cls_model.embed_dim, ))
            #     print("ensemble_outputs", ensemble_outputs.shape)

            if torch.any(local_mask):
                outputs = self.model_inference(
                    input_ids, attention_mask, ensemble_outputs, local_mask
                )  # MOVE TO CPU, SAVE GPU MEMORY
                # local_mask = local_mask.to(outputs.device)
                # ensemble_outputs = ensemble_outputs.to(outputs.device)

                if not self.model_parallel or (self.model_parallel and self.parallel_pos == self.parallel_stages - 1):
                    ensemble_outputs = self.model_ensemble(
                        ensemble_outputs, outputs, local_mask
                    )

                    local_mask, max_prob = self.offload_mask(ensemble_outputs, local_mask)
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
                if self.model_parallel and self.parallel_pos < self.parallel_stages - 1:
                    ensemble_outputs = outputs

            assert torch.sum(batch_mask) == input_ids.shape[0]
            self.logger.debug(
                "%s outputs %s, batch_mask %s", self.model_name,
                ensemble_outputs.shape,
                batch_mask
            )
            inference_response = pb_utils.InferenceResponse(
                output_tensors=[
                    self.parse_output(ensemble_outputs, "outputs"),
                    self.parse_output(batch_mask, "batch_mask"),
                ]
            )
            responses.append(inference_response)

        exec_end_time = time.perf_counter()
        self.logger.info(
            "%s requests handle time elapsed %s (ms)",
            self.model_name,
            (exec_end_time - exec_start_time) * 1000,
        )
        return responses

    def parse_input(self, request, field):
        input = pb_utils.get_input_tensor_by_name(request, field).as_numpy()
        # input = torch.as_tensor(input).type(np_to_torch_dtype(input.dtype))
        # input = input.to(self.device)
        input_zero_copy = torch.as_tensor(input, dtype=np_to_torch_dtype(input.dtype), device=self.device)

        self.logger.debug("%s %s %s", field, input_zero_copy.shape, input_zero_copy.device)

        return input_zero_copy

    def parse_output(self, output, field):
        output_config = pb_utils.get_output_config_by_name(self.model_config, field)
        output_dtype = pb_utils.triton_string_to_numpy(output_config["data_type"])

        if isinstance(output, torch.Tensor):
            output = output.detach().cpu().numpy()

        output_pb = pb_utils.Tensor(field, output.astype(output_dtype))
        return output_pb

    # def format_request(self, input, field):
    #     input_config = pb_utils.get_input_config_by_name(self.model_config, field)
    #     input_dtype = pb_utils.triton_string_to_numpy(input_config["data_type"])
    #     input_pb = pb_utils.Tensor(
    #         field,
    #         input.numpy().astype(input_dtype),
    #     )
    #     return input_pb

    @torch.no_grad()
    def model_inference(self, input_ids, attention_mask, ensemble_outputs, mask):
        start_time = time.perf_counter()

        masked_inputs = (input_ids[mask], attention_mask[mask])

        if "t5" in self.model_name and not self.model_parallel:
            outputs = self.t5_inference(*masked_inputs)
        elif "t5" in self.model_name and self.model_parallel:
            outputs = self.t5_parallel_inference(*masked_inputs, ensemble_outputs)
        else:
            outputs = self.default_inference(*masked_inputs)
        end_time = time.perf_counter()
        self.logger.info(
            "%s model_inference time elapsed %s (ms)",
            self.model_name,
            (end_time - start_time) * 1000,
        )
        return outputs  # .detach().cpu()

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
        self.cfg_timer.cancel()
        print("Cleaning up...")

    # def offload_request(self, input_ids, attention_mask, outputs, mask):
    #     inference_request = pb_utils.InferenceRequest(
    #         model_name=next_model,
    #         requested_output_names=["outputs"],
    #         inputs=[
    #             self.format_request(input_ids, "input_ids"),
    #             self.format_request(attention_mask, "attention_mask"),
    #             self.format_request(outputs, "ensemble_outputs"),
    #             self.format_request(mask, "mask"),
    #         ],
    #     )

    # async def execute(self, requests):
    #     responses = []

    #     # Every Python backend must iterate over everyone of the requests
    #     # and create a pb_utils.InferenceResponse for each of them.
    #     # infer_response_awaits = []

    #     exec_start_time = time.perf_counter()

    #     infer_response_awaits = []
    #     infer_outputs = []
    #     infer_masks = []
    #     for request in requests:
    #         input_ids = pb_utils.get_input_tensor_by_name(
    #             request, "input_ids"
    #         ).as_numpy()
    #         attention_mask = pb_utils.get_input_tensor_by_name(
    #             request, "attention_mask"
    #         ).as_numpy()
    #         ensemble_outputs = pb_utils.get_input_tensor_by_name(
    #             request, "ensemble_outputs"
    #         ).as_numpy()

    #         input_ids = torch.Tensor(input_ids).long()
    #         attention_mask = torch.Tensor(attention_mask).long()
    #         ensemble_outputs = torch.Tensor(ensemble_outputs).float()

    #         self.logger.debug(
    #             "input_ids %s\n attention_mask %s\n ensemble_outputs %s",
    #             input_ids.shape,
    #             attention_mask.shape,
    #             ensemble_outputs.shape,
    #         )

    #         with torch.no_grad():
    #             start_time = time.perf_counter()
    #             local_outputs = (
    #                 self.cls_model(
    #                     input_ids=input_ids.to(self.device),
    #                     attention_mask=attention_mask.to(self.device),
    #                     return_dict=True,
    #                 ).logits
    #                 if "t5" not in self.model_name
    #                 else self.t5_inference(input_ids, attention_mask)
    #             )
    #             end_time = time.perf_counter()
    #             self.logger.info(
    #                 "inference time elapsed %s (ms)", (end_time - start_time) * 1000
    #             )
    #             start_time = time.perf_counter()
    #             local_outputs = (
    #                 agg_logits(
    #                     ensemble_outputs
    #                     if self.threshold > 0.0
    #                     and ENSEMBLE_ORDER[0] not in self.model_name
    #                     and ENSEMBLE_ORDER[-2] not in self.model_name
    #                     else None,
    #                     local_outputs,
    #                     self.ensemble_pos,
    #                     self.device,
    #                 )
    #                 .detach()
    #                 .cpu()
    #                 .numpy()
    #             )
    #             end_time = time.perf_counter()
    #             self.logger.info(
    #                 "agg_logits time elapsed %s (ms)", (end_time - start_time) * 1000
    #             )
    #         mask, max_prob = self.mask_offload(local_outputs)
    #         # remote_outputs = None
    #         skip_masks = []
    #         if np.any(mask):
    #             for skip, next_model in enumerate(self.next_models):
    #                 skip_th_lower = (self.threshold - 0.25) / len(self.next_models) * (
    #                     len(self.next_models) - 1 - skip
    #                 ) + 0.25
    #                 skip_th_upper = (self.threshold - 0.25) / len(self.next_models) * (
    #                     len(self.next_models) - skip
    #                 ) + 0.25
    #                 skip_mask = (
    #                     (max_prob >= skip_th_lower) & (max_prob < skip_th_upper) & mask
    #                 )
    #                 if np.any(skip_mask):
    #                     # print("offload %s/%s" % (np.sum(mask), local_outputs.shape[0]))
    #                     offload_input_ids = input_ids[skip_mask].numpy()
    #                     offload_attention_mask = attention_mask[skip_mask].numpy()
    #                     offload_outputs = local_outputs[skip_mask]

    #                     self.logger.debug(
    #                         "next %s skip_th %s, offload_input_ids %s, offload_attention_mask %s, offload_outputs %s",
    #                         next_model,
    #                         (skip_th_lower, skip_th_upper),
    #                         offload_input_ids.shape,
    #                         offload_attention_mask.shape,
    #                         offload_outputs.shape,
    #                     )

    #                     inference_request = pb_utils.InferenceRequest(
    #                         model_name=next_model,
    #                         requested_output_names=["outputs"],
    #                         inputs=[
    #                             pb_utils.Tensor(
    #                                 "input_ids",
    #                                 offload_input_ids.astype(self.input_ids_dtype),
    #                             ),
    #                             pb_utils.Tensor(
    #                                 "attention_mask",
    #                                 offload_attention_mask.astype(
    #                                     self.attention_mask_dtype
    #                                 ),
    #                             ),
    #                             pb_utils.Tensor(
    #                                 "ensemble_outputs",
    #                                 offload_outputs.astype(self.ensemble_outputs_dtype),
    #                             ),
    #                         ],
    #                     )

    #                     # inference_response = inference_request.async_exec()
    #                     infer_response_awaits.append(inference_request.async_exec())
    #                     skip_masks.append(skip_mask)
    #         infer_outputs.append(local_outputs)
    #         infer_masks.append(skip_masks)

    #     # Wait for all of the inference requests to complete.
    #     infer_responses = await asyncio.gather(*infer_response_awaits)
    #     idx = 0
    #     for i in range(len(infer_outputs)):
    #         outputs = infer_outputs[i]
    #         for j, mask in enumerate(infer_masks[i]):
    #             # if np.any(mask):
    #             infer_response = infer_responses[idx]
    #             if infer_response.has_error():
    #                 raise pb_utils.TritonModelException(
    #                     infer_response.error().message()
    #                 )
    #             else:
    #                 # Extract the output tensors from the inference response.
    #                 remote_outputs = pb_utils.get_output_tensor_by_name(
    #                     infer_response, "outputs"
    #                 ).as_numpy()
    #             outputs[mask] = remote_outputs
    #             idx += 1
    #         self.logger.debug("combined outputs %s", outputs)
    #         outputs_tensor = pb_utils.Tensor(
    #             "outputs", outputs.astype(self.outputs_dtype)
    #         )
    #         inference_response = pb_utils.InferenceResponse(
    #             output_tensors=[outputs_tensor]
    #         )
    #         responses.append(inference_response)

    #     exec_end_time = time.perf_counter()
    #     self.logger.info(
    #         "requests handle time elapsed %s (ms)",
    #         (exec_end_time - exec_start_time) * 1000,
    #     )

    #     # You should return a list of pb_utils.InferenceResponse. Length
    #     # of this list must match the length of `requests` list.
    #     return responses
