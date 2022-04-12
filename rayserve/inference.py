import copy
from dataclasses import dataclass
import dataclasses
import functools
import io
import logging
from multiprocessing.connection import wait
import random
import subprocess
import sys
import traceback
from typing import Any, Dict, List, Optional, Tuple, Union
from attr import field
from concurrent.futures import ProcessPoolExecutor
import pandas as pd

from transformers import (
    AutoConfig,
    HfArgumentParser,
    T5ForConditionalGeneration,
    AutoModelForQuestionAnswering,
    DistilBertForQuestionAnswering,
    ViTForImageClassification,
    AutoModelForCausalLM,
)
import requests
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"

import torch
import time
import threading
import multiprocessing as mp
from multiprocessing import Process, Manager
from multiprocessing.managers import BaseManager
from fastapi import FastAPI
import json
from requests import Request
import numpy as np
from scipy.special import softmax
from transformers import AutoTokenizer
from tqdm import tqdm
import gc

# ====== ray serve
import ray
from ray import data, serve
from ray.serve import pipeline
from ray.util.metrics import Counter, Gauge, Histogram, Metric

# ====== hfutils
from hfutils.arg_parser import RayArguments
from hfutils.logger import Logger
from hfutils.calibration import agg_logits, temperature_scale
from hfutils.constants import MODEL_KEYS
from hfutils.pipe.t5 import (
    T5_ENCODER_INPUTS,
    T5_ENCODER_OUTPUTS,
    T5_DECODER_INPUTS,
    T5_DECODER_OUTPUTS,
    T5PyTorchPipe,
    T5PytorchPipeRandom
)
from hfutils.pipe.bert import (
    BERT_INPUTS,
    BERT_OUTPUTS,
    BertPyTorchPipeForQuestionAnswering,
    BertPytorchPipeRandom
)
from hfutils.pipe.vit import (
    VIT_INPUTS,
    VIT_OUTPUTS,
    ViTPyTorchPipeForImageClassification,
    ViTPytorchPipeRandom
)
from hfutils.pipe.gpt import (
    GPTPytorchPipeRandom
)
from hfutils.pipe.distilbert import (
    DISTILBERT_INPUTS,
    DISTILBERT_OUTPUTS,
    DistilBertPyTorchPipeForQuestionAnswering,
)
from hfutils.pipe.gpt import GPT_INPUTS, GPT_OUTPUTS, GPTLMHeadModelPipe
from hfutils.calibration import temperature_scale
from hfutils.constants import np_to_torch_dtype
from hfutils.options import (
    ReplicationOptions,
    SystemOptions,
    EnsembleOptions,
    ParallelOptions,
    ModelConfig,
    HostOptions,
)

# ======= DEFINE CONSTANTS =========
T5_TASK_LABELS = [1176, 6136, 59]  # HACK with GLUE labels
m = functools.partial(softmax, axis=1)
VISIBLE_GPUS = [str(i) for i in range(torch.cuda.device_count())]
m = torch.nn.Softmax(dim=1)


@dataclass
class Arguments:
    # ensemble_cfg: str = field(metadata={"help": "Path to configuration meta file for ensemble, including partition and replications"})
    model_cfg: str = field(
        metadata={
            "help": "Path to configuration meta file, including thresholds and temperatures"
        }
    )
    namespace: str = field(metadata={"help": "Namespace for ray serve"})


parser = HfArgumentParser(Arguments)
args = parser.parse_args_into_dataclasses()[0]


# ======= PARSE CONFIGURATION =========
# with open(args.ensemble_cfg, "r") as fp:
#     ensemble_config = json.load(fp)

with open(args.model_cfg, "r") as fp:
    model_config = json.load(fp)

ensembles = model_config["ensembles"]
base_dir = model_config["base_dir"]
alpha = model_config["alpha"]
type = model_config["type"]
instance = model_config["instance"]


host_options = {
    ins["host"]: HostOptions(
        host=ins["host"],
        # alpha=alpha,
        # ens=len(ensembles),
        type=type,
        placement={
            gid: [
                ModelConfig(
                    name=model["name"],
                    path=os.path.join(base_dir, model_config[model["name"]]["path"]),
                    type=model_config[model["name"]]['type'],
                    stages=model_config[model["name"]]["parallel_stages"],
                    ppos=model["stage"],
                    epos=ensembles.index(model["name"]),
                    temp=model_config[model["name"]]["temperature"],
                    util_params=model_config[model["name"]]["util_params"],
                    ray_actor_options={
                        "num_cpus": 1,
                        "num_gpus": 1 / len(models),
                        "resources": {ins["host"]: 1},
                    },
                    key = "_".join([ins["host"], model["name"], gid, str(i)])
                )
                for i, model in enumerate(models)
            ]
            for gid, models in ins["placement"].items()
        },
    )
    for ins in instance
}

# host_resource = {
#     ins["host"]: sum([len(models) for gid, models in ins["placement"].items()])
#     for ins in instance
# }

# model_replicas = {
#     name: sum(
#         [
#             1
#             for ins in instance
#             for gid, models in ins["placement"].items()
#             for model in models
#             if model["name"] == name
#         ]
#     )
#     for name in ensembles
# }

system_options = SystemOptions(
    alpha=alpha,
    ens=len(ensembles),
    type=type,
    ensemble_options=[
        EnsembleOptions(
            epos=i,
            th=model_config[name]["threshold"],
            name=name,
            parallel_options = [
                ParallelOptions(
                    stages=model_config[name]["parallel_stages"],
                    ppos=p,
                    replications=[
                        model.key
                        for host in host_options.values()
                        for models in host.placement.values()
                        for model in models
                        if model.epos == i and model.ppos == p
                    ]
                )
                for p in range(model_config[name]["parallel_stages"])
            ]
        )
        for i, name in enumerate(ensembles)
    ]
)



# for idx, name in enumerate(ensembles):
#     meta = model_config[name]

#     path = os.path.join(base_dir, meta["path"])
#     threshold = meta["threshold"]
#     temperature = meta["temperature"]
#     stages = meta["parallel_stages"]
#     util_params = meta["util_params"]

#     instance = meta["instance"]

#     parallel_options = [
#         ParallelOptions(
#             stages=stages,
#             ppos=p,
#             replication_options=[
#                 ReplicationOptions(
#                     k,
#                     "_".join([name, idx, ins["stage"], k]),
#                     torch.device(ins["device"]),
#                 )
#                 for k in range(ins["count"])
#                 for ins in instance
#                 if ins["stage"] == p
#             ],
#         )
#         for p in range(stages)
#     ]

#     for i, ins in enumerate(instance):
#         for k in range(ins["count"]):
#             key = "_".join([name, idx, ins["stage"], k])
#             replication_options = ReplicationOptions(
#                 k, key, torch.device(ins["device"])
#             )
#             config = ModelConfig(
#                 name,
#                 path,
#                 type,
#                 stages,
#                 ins["stage"],
#                 idx,
#                 len(ensembles),
#                 alpha,
#                 temperature,
#                 threshold,
#                 util_params,
#                 ins["device"],
#                 k,
#             )
#             deploy_config.append(config)

# ====== MODEL DEFINATION ==============


@serve.deployment(max_concurrent_queries=10)
class HServeModel:
    def __init__(self, options: Dict, model_id: int, key: str) -> None:
        self.logger = Logger(__file__, logging.INFO, 50000000, 5)

        # os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(VISIBLE_GPUS)

        gid = str(ray.get_gpu_ids()[0])
        print("gid", gid, ray._private.utils.get_cuda_visible_devices(), flush=True)
        host = ray._private.services.get_node_ip_address()
        print("host", host, flush=True)
        self.device = torch.cuda.current_device() # torch.device(f"cuda:{gid}")
        print("device", "cuda:" + str(gid), flush=True)

        print(gid, host, self.device, flush=True)
        print(options, model_id, key, flush=True)

        self.config = options[host].placement[gid][model_id]

        print(self.config, flush=True)

        self.key = key

        self.cuda_stream = torch.cuda.Stream(device=self.device, priority=-1)
        print("stream", self.key, self.cuda_stream, flush=True)

        # self._get_gpu_uuid()
        self._load_model()

        self.is_last_stage = self.config.ppos == self.config.stages - 1

    # def _get_gpu_uuid(self):
    #     command = "nvidia-smi --query-gpu=index,uuid,gpu_bus_id --format=csv"
    #     result = subprocess.run(command.split(), stdout=subprocess.PIPE)
    #     df = pd.read_csv(io.StringIO(result.stdout.decode("utf-8")), index_col="index")
    #     df = df.sort_index()
    #     print(df)
    #     df.iloc[:, 0] = df.iloc[:, 0].str.strip()
    #     self.gpu_uuid = df.iloc[self.device.index][" uuid"]
    #     print(self.gpu_uuid, flush=True)

    def _load_model(self):
        if "t5" == self.config.type:
            model = T5ForConditionalGeneration.from_pretrained(self.config.path)
            self.model = T5PyTorchPipe(model)

        elif "bert" == self.config.type:
            model = AutoModelForQuestionAnswering.from_pretrained(self.config.path)
            self.model = BertPyTorchPipeForQuestionAnswering(model)

        elif "distilbert" == self.config.type:
            model = DistilBertForQuestionAnswering.from_pretrained(self.config.path)
            self.model = DistilBertPyTorchPipeForQuestionAnswering(model)

        elif "vit" == self.config.type:
            model = ViTForImageClassification.from_pretrained(self.config.path)
            self.model = ViTPyTorchPipeForImageClassification(model)

        elif "gpt" == self.config.type:
            model = AutoModelForCausalLM.from_pretrained(self.config.path)
            self.model = GPTLMHeadModelPipe(model)

        elif "random" == self.config.type:
            config = AutoConfig.from_pretrained(self.config.path)
            if "bert" in self.config.name:
                self.model = BertPytorchPipeRandom(config)
            elif "vit" in self.config.name:
                self.model = ViTPytorchPipeRandom(config)
            elif "gpt" in self.config.name:
                self.model = GPTPytorchPipeRandom(config)
            elif "t5" in self.config.name:
                self.model = T5PytorchPipeRandom(config)
            else:
                raise ValueError("%s undefined random model name %s" % (self.key, self.config.name))
        else:
            raise ValueError("%s unknown model type %s" % (self.key, self.config.type))

        self.model.eval()
        self.model.partition_by_parameter(self.config.ppos, self.config.stages)
        # self.model.partition_by_parameter(self.config.ppos, 4) # TEST MULTIPLEX
        self.model.convert(self.device)

        del model
        gc.collect()
        torch.cuda.empty_cache()

    @torch.no_grad()
    def model_inference(self, args, mask):
        start_time = time.perf_counter()
        # self.logger.debug("%s args %s", self.key, args)
        if self.config.ppos == 0:
            args = tuple(torch.from_numpy(arg[mask]).to(self.device) for arg in args)
        else:
            args = tuple(arg.to(self.device) for arg in args)
        with torch.cuda.stream(self.cuda_stream):
            outputs = self.model(args)
        self.cuda_stream.synchronize()  # MUST sync otherwise outputs are zeros
        end_time = time.perf_counter()
        if self.is_last_stage:
            outputs = outputs.squeeze(1) / self.config.temp
            if "t5" == self.config.type:
                outputs = outputs[:, T5_TASK_LABELS]
            if "gpt" == self.config.type:
                outputs = outputs[:, -1, :50257]
            outputs = outputs.detach().cpu().numpy()
            # outputs = temperature_scale(outputs, self.config.temp)

        self.logger.info(
            "%s inference %s (ms)", self.key, (end_time - start_time) * 1000,
        )

        return outputs

    async def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.model_inference(*args, **kwds)


@serve.deployment(max_concurrent_queries=100)
class HybridScheduler:
    def __init__(self, options: SystemOptions):

        self.config = options

        self.logger = Logger(__file__, logging.INFO, 50000000, 5)
        self.logger.info("HybridScheduler logger initialized")

        # os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(VISIBLE_GPUS)
        # self.logger.info("CUDA_VISIBLE_DEVICES: %s", os.environ["CUDA_VISIBLE_DEVICES"])

        self.logger.info("HybridScheduler cfg %s", options)

        self.ensembles = options.ensemble_options
        self.num_ensembles = len(options.ensemble_options)

        self.logger.info("Current ensembles %s", self.ensembles)
        self.logger.info("Worker IP %s", ray._private.services.get_node_ip_address())

    def schedule_handle(self, parallel_options):
        # keys = list(parallel_options.keys())
        # values = list(parallel_options.values())
        # key = keys[values.index(min(values))]

        # idx = np.random.choice(list(range(len(keys))), 1)[0]
        # key = keys[idx]
        # parallel_options[key] += 1
        self.logger.info("parallel_options %s", parallel_options)
        num_replicas = len(parallel_options.replications)
        parallel_options.rr_counter += 1
        r = parallel_options.rr_counter % num_replicas
        key = parallel_options.replications[r]
        self.logger.info("parallel_options %s", parallel_options)
        return serve.get_deployment(key).get_handle(sync=False)

    async def __call__(self, request) -> Any:
        data = await request.json()
        self.logger.info("data %s", data, ray.ObjectRef(bytes.fromhex(data['args'])))
        args = ray.get(ray.ObjectRef(bytes.fromhex(data['args'])))
        self.logger.info("args %s", args)
        ref =  await self.ensemble_inference(args)
        self.logger.info("ref %s", ref)
        return {
            "logits": ref.hex()
        }

    async def post_processing(self, ensemble_outputs, outputs, batch_mask, idx):
        local_mask = batch_mask[idx]
        outputs = ray.get(outputs)
        ensemble_outputs = self.model_ensemble(
            ensemble_outputs, outputs, local_mask, idx
        )

        extended_mask, _ = self.offload_mask(
            ensemble_outputs, local_mask, idx
        )

        num_next_models = self.num_ensembles - idx - 1
        if np.any(extended_mask) and num_next_models > 0:
            batch_mask[idx] &= ~extended_mask
            batch_mask[idx + 1] |= extended_mask
            # batch_mask = self.update_batch_mask(
            #     max_prob, batch_mask.copy(), extended_mask, idx
            # )
            # self.logger.trace(
            #     "%s batch_mask updated %s", options.name, batch_mask
            # )

        return ensemble_outputs, np.any(extended_mask)
    

    # This method can be called concurrently!
    async def ensemble_inference(self, args):
        start_time = time.perf_counter()

        batch_size = len(args[0])
        ensemble_outputs = None
        batch_mask = np.zeros((self.num_ensembles, batch_size))
        # batch_mask = np.ones((self.num_ensembles, batch_size))
        batch_mask[0, :] = 1  # WHERE TO ENTER
        batch_mask = batch_mask.astype(bool)

        for idx, options in enumerate(self.ensembles):
            outputs = args
            local_mask = batch_mask[idx]
            self.logger.trace("%s local_mask %s", options.name, local_mask)

            if np.any(local_mask):
                for parallel_options in options.parallel_options:
                    handle = self.schedule_handle(parallel_options)
                    outputs = await handle.model_inference.remote(outputs, local_mask)

                ensemble_outputs, process_next = await self.post_processing(ensemble_outputs, outputs, batch_mask, idx)
                if not process_next: break

                # outputs = ray.get(outputs)
                # ensemble_outputs = self.model_ensemble(
                #     ensemble_outputs, outputs, local_mask, idx
                # )

                # extended_mask, _ = self.offload_mask(
                #     ensemble_outputs, local_mask, idx
                # )

                # if np.all(~extended_mask): break

                # self.logger.trace(
                #     "%s local_mask updated %s", options.name, extended_mask
                # )
                # num_next_models = self.num_ensembles - idx - 1
                # if np.any(extended_mask) and num_next_models > 0:
                #     batch_mask[idx] &= ~extended_mask
                #     batch_mask[idx + 1] |= extended_mask
                #     # batch_mask = self.update_batch_mask(
                #     #     max_prob, batch_mask.copy(), extended_mask, idx
                #     # )
                #     # self.logger.trace(
                #     #     "%s batch_mask updated %s", options.name, batch_mask
                #     # )
            assert np.sum(batch_mask) == batch_size

        end_time = time.perf_counter()
        self.logger.info(
            "request %s (ms)", (end_time - start_time) * 1000,
        )
        gc.collect()
        torch.cuda.empty_cache()
        return ensemble_outputs

    def offload_mask(self, logits, mask, idx):
        start_time = time.perf_counter()
        probabilities = np.power(softmax(logits, axis=1), 2)
        max_prob = np.max(probabilities, axis=1)
        if "bert" == self.config.type:
            if max_prob.shape[1] == 1:
                max_prob = max_prob.squeeze(1)
            max_prob = np.min(max_prob, axis=1)
        prob_mask = max_prob < self.config.ensemble_options[idx].th
        self.logger.debug(
            "(offload_mask) prob_mask %s %s", prob_mask, mask,
        )
        combined_mask = mask & prob_mask
        self.logger.debug("max_prob %s, combined_mask %s", max_prob, combined_mask)
        end_time = time.perf_counter()
        self.logger.info(
            "%s offload_mask time elapsed (%s, %s) %s (ms)",
            self.config.ensemble_options[idx].name,
            start_time,
            end_time,
            (end_time - start_time) * 1000,
        )
        return combined_mask, max_prob

    # def offload_mask(self, logits, mask, idx):
    #     probabilities = np.power(m(logits), 2)
    #     max_prob = np.max(probabilities, axis=-1)
    #     prob_mask = max_prob < self.ensembles[idx].threshold
    #     self.logger.trace(
    #         "%s (offload_mask) prob_mask %s %s",
    #         self.ensembles[idx].name,
    #         prob_mask,
    #         mask,
    #     )
    #     extended_mask = mask & prob_mask
    #     # combined_mask[mask] &= prob_mask[mask]
    #     self.logger.trace("max_prob %s, extended_mask %s", max_prob, extended_mask)
    #     return extended_mask, max_prob

    def model_ensemble(self, hist_outputs, outputs, mask, idx):
        start_time = time.perf_counter()
        if hist_outputs is not None:
            hist_outputs[mask] = (
                hist_outputs[mask] * (1 - self.config.alpha)
                + outputs * self.config.alpha
            )
        else:
            hist_outputs = outputs.copy()
        end_time = time.perf_counter()
        self.logger.info(
            "%s model_ensemble time elapsed (%s, %s) %s (ms)",
            self.config.ensemble_options[idx].name,
            start_time,
            end_time,
            (end_time - start_time) * 1000,
        )
        return hist_outputs  # MEMCOPY MUTABLE

    # def model_ensemble(self, ensemble_outputs, local_outputs, mask, idx):
    #     # start_time = time.perf_counter()
    #     if ensemble_outputs is not None:
    #         ensemble_weight = self.ensembles[idx].ensemble_weight
    #         ensemble_outputs[mask] = (
    #             ensemble_outputs[mask] * (1 - ensemble_weight)
    #             + local_outputs * ensemble_weight
    #         )
    #     self.logger.trace(
    #         "%s ensemble_outputs %s", self.ensembles[idx].name, ensemble_outputs,
    #     )
    #     return (
    #         ensemble_outputs if ensemble_outputs is not None else local_outputs.copy()
    #     )  # MEMCOPY

    # def update_batch_mask(self, max_prob, mask, local_mask, idx):
    #     num_next_models = self.num_ensembles - idx - 1

    #     if num_next_models <= 0:
    #         return mask

    #     if self.ensembles[idx].skip_connection:
    #         base_step = (self.ensembles[idx].threshold - 0.25) / num_next_models
    #         for skip in range(num_next_models):
    #             skip_th_lower = base_step * (num_next_models - 1 - skip) + 0.25
    #             skip_th_upper = base_step * (num_next_models - skip) + 0.25
    #             skip_mask = (
    #                 (max_prob >= skip_th_lower)
    #                 & (max_prob < skip_th_upper)
    #                 & local_mask
    #             )
    #             self.logger.trace(
    #                 "%s skip_th_lower %s, skip_th_upper %s, skip_mask %s",
    #                 self.ensembles[idx].name,
    #                 skip_th_lower,
    #                 skip_th_upper,
    #                 skip_mask,
    #             )
    #             mask[skip + 1 + idx] |= skip_mask
    #     else:
    #         mask[1 + idx] |= (max_prob < self.ensembles[idx].threshold) & local_mask

    #     mask[idx] &= ~local_mask
    #     return mask

    def update_batch_mask(self, max_prob, mask, local_mask, idx):
        start_time = time.perf_counter()
        num_next_models = len(mask) - self.config.ensemble_options[idx].epos - 1
        base_step = (self.config.ensemble_options[idx].th - 0.25) / num_next_models
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
            mask[skip + 1 + self.config.ensemble_options[idx].epos] |= skip_mask

        mask[self.config.ensemble_options[idx].epos] &= ~local_mask
        end_time = time.perf_counter()
        self.logger.info(
            "%s update_batch_mask time elapsed (%s,%s) %s (ms)",
            start_time,
            end_time,
            self.config.ensemble_options[idx].name,
            (end_time - start_time) * 1000,
        )
        return mask


# ray.init(address="ray://129.215.164.41:10001")

# ====== START SERVER ==============
# ray.init(namespace=args.namespace, num_cpus=80, num_gpus=torch.cuda.device_count())
ray.init(address="ray://129.215.164.41:10001", namespace=args.namespace)
serve.start(detached=True)

# print("ray initialized", args)

for host, h_op in host_options.items():
    for gid, models in h_op.placement.items():
        for i, model in enumerate(models):
            key = "_".join([host, model.name, gid, str(i)])
            HServeModel.options(
                name=key, ray_actor_options=model.ray_actor_options
            ).deploy(options=host_options, model_id=i, key=key)

# for e_op in system_options.ensemble_options:
#     for p_op in e_op.parallel_options:
#         for r_op in p_op.replication_options:
#             HServeModel.options(
#                 name=r_op.key, ray_actor_options={"num_cpus": 4, "num_gpus": 2},
#             ).deploy(
#                 options=system_options,
#                 epos=e_op.epos,
#                 ppos=p_op.ppos,
#                 replica=r_op.replica,
#             )


HybridScheduler.options(
    name="hybrid-scheduler",
    num_replicas=20,
    ray_actor_options={"num_cpus": 2},
).deploy(system_options)
