import copy
from dataclasses import dataclass
import dataclasses
import functools
import traceback
from typing import Any, Dict, List, Optional, Tuple, Union
from async_timeout import enum
from attr import field
from concurrent.futures import ProcessPoolExecutor

from transformers import AutoConfig, T5ForConditionalGeneration
import requests
import os

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
import sqlite3
import gc

# ====== ray serve
import ray
from ray import data, serve
from ray.serve import pipeline
from ray.util.metrics import Counter, Gauge, Histogram, Metric

# ====== hfutils
from hfutils.arg_parser import RayArguments
from hfutils.logger import Logger
from hfutils.options import EnsembleOptions, ParallelOptions, ReplicationOptions
from hfutils.model_pipe import T5Pipe, T5PyTorchPipe, get_num_layers
from hfutils.calibration import agg_logits, temperature_scale
from hfutils.constants import (
    MODEL_TASK_TO_CLASS,
    ENSEMBLE_ORDER,
    TASK_TO_LABELS,
    np_to_torch_dtype,
    MODEL_KEYS,
)

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# import pyprof
# import torch.cuda.profiler as profiler


class LatencyMonitor:
    def __init__(self, record_tag: str, config: Union[Dict, List[Dict]]):


        self.fp = open(f"mie_latency-{record_tag}", "w")

        # self.con = sqlite3.connect('mie_latency.db')
        # self.cur = self.con.cursor()

        # self.cur.execute('''
        #         CREATE TABLE IF NOT EXISTS latency (
        #             record_id TEXT NOT NULL,
        #             model_id TEXT NOT NULL,
        #             latency FLOAT NOT NULL
        #         )''')

        # self.cur.execute('''
        #         CREATE TABLE IF NOT EXISTS config (
        #             record_id TEXT NOT NULL,
        #             config BLOB NOT NULL
        #         )''')

        # self.lock = mp.Lock()
        # self.record_id = record_tag

        # self.cur.execute('''
        #     INSERT INTO config VALUES (
        #         "%s", "%s"
        #     )''' % (record_tag, json.dumps(config)))

    async def observe(self, value: Union[int, float], tag: str):
        if not isinstance(value, (int, float)):
            raise TypeError(f"value must be int or float, got {type(value)}.")


        with self.lock:
            # self.cur.execute('''
            #     INSERT INTO latency VALUES (
            #         %s, "%s", 
            #     )''' % (self.record_id, tag, value))
            # self.con.commit()
            self.fp.write("%s:%s;" % (tag, value))
        # self.record(value, tags, _internal=True)

    def __del__(self):
        # self.con.commit()
        # self.con.close()
        self.fp.close()

        fp.close()


args = RayArguments()
task_name = args.data_args.task_name
serve_args = args.serve_args


home = "/jmain02/home/J2AD002/jxm12/lxx22-jxm12"
tokenizer = AutoTokenizer.from_pretrained(os.path.join(home, "HuggingFace", "google", "t5-small-lm-adapt"))
label_tokens = [
    tokenizer(label, max_length=2).input_ids[0]
    for label in TASK_TO_LABELS[task_name]
    if label is not None
]

print(args)
print(serve_args)

ray.init(
    namespace=f"t5-{task_name}", num_cpus=40, num_gpus=torch.cuda.device_count()
)
serve.start(detached=False)

print("ray initialized")

# print(torch.cuda.is_available())

# m = torch.nn.Softmax(dim=-1)
m = functools.partial(softmax, axis=-1)

# task_name = "sst2"
model_ensemble_name = serve_args.deployment # "_".join(["t5", task_name, "test"])

# tokenizer = AutoTokenizer.from_pretrained("google/t5-small-lm-adapt", use_fast=False)
# label_tokens = [
#     tokenizer(label, max_length=2).input_ids[0]
#     for label in TASK_TO_LABELS[task_name]
#     if label is not None
# ]

deploy_options = []

with open(serve_args.cfg, "r") as fp:
    config = json.load(fp)
    ensemble_config = config[model_ensemble_name]
    ensemble_names = ensemble_config["ensembles"]
    ensemble_weights = ensemble_config["weights"]

    for idx, name in enumerate(ensemble_names):
        model_config = config[name]

        ckpt_path = model_config["ckpt"]

        visible_gpus = [str(i) for i in model_config["devices"]]
        num_gpus = len(visible_gpus)
        num_replicas = model_config["count"]

        hf_config = AutoConfig.from_pretrained(ckpt_path)
        total_pp_layers = (get_num_layers(hf_config) + 4) * 2 + 1

        # print(name, "num_layers", total_pp_layers)

        parallel_stages = model_config["parallel_stages"]
        pp_layers = int(total_pp_layers / parallel_stages)

        deploy_options.append(
            EnsembleOptions(
                ensemble_weight=ensemble_weights[idx],
                ensemble_pos=idx,
                ckpt_path=ckpt_path,
                threshold=model_config["threshold"],
                temperature=model_config["temperature"],
                name=name,
                scheduler=ensemble_config["scheduler"],
                parallel=parallel_stages > 1,
                skip_connection=ensemble_config["skip_connection"],
                ray_actor_options={"num_gpus": 1, "num_cpus": 5},
                parallel_options=[
                    ParallelOptions(
                        num_stages=parallel_stages,
                        parallel_stage=p,
                        first_parallel_layer=pp_layers * p,
                        last_parallel_layer=pp_layers * (p + 1)
                        if p < parallel_stages - 1
                        else total_pp_layers,
                        replication_options=[
                            ReplicationOptions(
                                replica_id=r,
                                key=f"{MODEL_KEYS[idx]}R{r}P{p}",
                                device=torch.device(
                                    "cuda:" + visible_gpus[(r + p) % num_gpus]
                                ),
                            )
                            for r in range(num_replicas)
                        ],
                    )
                    for p in range(parallel_stages)
                ],
            )
        )


visible_gpus = [str(i) for i in range(torch.cuda.device_count())]


@serve.deployment(max_concurrent_queries=10)
class T5Model:
    def __init__(self, options: EnsembleOptions, replica: int, stage: int) -> None:

        # pyprof.init()

        self.key = options.parallel_options[stage].replication_options[replica].key

        # self.logger = Logger(self.key, "debug", 5000000, 5)
        self.logger = Logger(__file__, "debug", 50000000, 5)
        self.logger.info("%s logger initialized", options.name)

        self.options = options

        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(visible_gpus)
        self.logger.info("CUDA_VISIBLE_DEVICES: %s", os.environ["CUDA_VISIBLE_DEVICES"])

        # visible_gpus = ray.get_gpu_ids()

        self.logger.info("options: %s", options)

        # self.logger.info("ray.get_gpu_ids(): {}".format(ray.get_gpu_ids()))
        # self.logger.info("torch visible devices: {}".format(torch.cuda.device_count()))

        self.device = (
            options.parallel_options[stage].replication_options[replica].device
        )
        self.logger.info("%s device %s", options.name, self.device)
        self.cuda_stream = torch.cuda.Stream(
            device=self.device, priority=-1
        )
        # self.cuda_stream = torch.cuda.Stream()
        self.logger.info("%s stream %s", options.name, self.cuda_stream)

        self.model_name = model_name = options.name
        self.logger.debug("%s model_name %s", self.key, self.model_name)
        self.temperature = torch.nn.Parameter(
            torch.ones(1, device=self.device) * options.temperature
        )
        self.logger.debug("%s temperature %s", self.key, self.temperature)
        self.ensemble_pos = options.ensemble_pos
        self.logger.debug("%s ensemble_pos %s", self.key, self.ensemble_pos)
        self.ensemble_weight = options.ensemble_weight
        self.logger.debug("%s ensemble_weight %s", self.key, self.ensemble_weight)
        self.parallel_stage = options.parallel_options[stage].parallel_stage
        self.logger.debug("%s parallel_stage %s", self.key, self.parallel_stage)
        self.num_stages = options.parallel_options[stage].num_stages
        self.logger.debug("%s num_stages %s", self.key, self.num_stages)

        # self.cuda_stream = torch.cuda.Stream()
        # from deepspeed.utils.zero_to_fp32 import load_state_dict_from_zero_checkpoint

        model = T5ForConditionalGeneration.from_pretrained(options.ckpt_path)
        # if not "small" in model_name:
        #     model = load_state_dict_from_zero_checkpoint(model, options.ckpt_path)
        self.model_parallel = options.parallel

        # exec_map = (
        #     options.parallel_options[stage].first_parallel_layer,
        #     options.parallel_options[stage].last_parallel_layer,
        # )

        # self.logger.info("%s garbage collected", model_name)
        # if self.model_parallel:
        #     self.model = T5PyTorchPipe(model, exec_map)
        #     self.model = self.model.to(self.device)
        #     # self.model.device = self.device
        #     self.logger.debug(
        #         "%s T5PyTorchPipe num_layers %s ", model_name, len(self.model.pipe)
        #     )
        # else:
        #     self.model = model.to(self.device)
        self.model = T5PyTorchPipe(model)
        self.model.partition_by_parameter(
            stage, options.parallel_options[stage].num_stages
        )

        self.logger.debug(
            "%s T5PyTorchPipe num_layers %s ", model_name, len(self.model.layers)
        )
        # self.model = self.model.to(self.device)
        self.model.convert(self.device)
        self.logger.info("%s model initialized %s", model_name, type(self.model))
        self.model.eval()

        del model
        torch.cuda.empty_cache()
        gc.collect()

        self.logger.debug("%s garbage collected", model_name)

        # profiler.start()

        self.logger.info("%s full initialization complete", model_name)

    def t5_parallel_inference(self, args):
        # batch_size = input_ids.shape[0]

        outputs = self.model.forward(args)
        # for i, t in enumerate(outputs):
        #     self.logger.trace(
        #         "%s t5_parallel_inference outputs size (%s) %s",
        #         self.model_name,
        #         i,
        #         t.size() if t is not None else None,
        #     )
        if self.parallel_stage == self.num_stages - 1:
            logits = torch.squeeze(outputs, 1)[:, label_tokens]
            # logits = outputs[1].view(batch_size, -1)
            outputs = temperature_scale(logits, self.temperature)

        return outputs

    # def t5_inference(self, input_ids, attention_mask, *args):
    #     outputs = self.model.generate(
    #         input_ids=input_ids,
    #         attention_mask=attention_mask,
    #         do_sample=False,  # disable sampling to test if batching affects output
    #         return_dict_in_generate=True,
    #         output_scores=True,
    #     )
    #     logits = outputs.scores[0][:, label_tokens]
    #     # logits = outputs.scores[0]
    #     logits = temperature_scale(logits, self.temperature)
    #     return logits

    def default_inference(self, args):
        logits = self.model(
            input_ids=args[0],
            attention_mask=args[1],
            return_dict=True,
        ).logits
        logits = temperature_scale(logits, self.temperature)
        return logits

    # @torch.no_grad()
    def model_inference(self, args, types, **kwds):
        tag = kwds.get("tag", "")
        start_time = time.perf_counter()

        # input_ids = torch.Tensor(input_ids.copy()).to(self.device).to(torch.long)
        # attention_mask = (
        #     torch.Tensor(attention_mask.copy()).to(self.device).to(torch.long)
        # )

        # input_ids = torch.as_tensor(input_ids, dtype=torch.int, device=self.device)
        # attention_mask = torch.as_tensor(attention_mask, dtype=torch.int, device=self.device)

        args = tuple(
            # torch.Tensor(arg.copy()).to(self.device).to(torch.float)
            [
                torch.as_tensor(arg, dtype=types[i], device=self.device)
                if arg is not None
                else None
                for i, arg in enumerate(args)
            ]
        )

        # masked_inputs = (
        #     input_ids,
        #     attention_mask,
        #     *tensor_args,
        # )
        end_time = time.perf_counter()
        self.logger.debug(
            "L(%s:%s[%s]:%s:%s)",
            tag, self.key, os.getpid(),
            "datacopy",
            (end_time - start_time) * 1000,
        )

        # self.cuda_stream.synchronize()
        start_time = time.perf_counter()
        with torch.cuda.stream(self.cuda_stream):
            if "t5" in self.model_name:
                outputs = self.t5_parallel_inference(args)
            # if "t5" in self.model_name and not self.model_parallel:
            #     outputs = self.t5_inference(*masked_inputs)
            # elif "t5" in self.model_name and self.model_parallel:
            #     outputs = self.t5_parallel_inference(*masked_inputs)
            else:
                outputs = self.default_inference(args)
        end_time = time.perf_counter()
        self.logger.debug(
            "L(%s:%s[%s]:%s:%s)",
            tag, self.key, os.getpid(),
            "submitstream",
            (end_time - start_time) * 1000,
        )
        
        start_time = time.perf_counter()
        self.cuda_stream.synchronize()  # MUST sync otherwise outputs are zeros
        end_time = time.perf_counter()
        # self.logger.info(
        #     "(%s) %s model_inference time elapsed %s (ms)",
        #     self.key,
        #     self.model_name,
        #     (end_time - start_time) * 1000,
        # )
        self.logger.info(
            "L(%s:%s[%s]:%s:%s)",
            tag, self.key, os.getpid(),
            "cudasync",
            (end_time - start_time) * 1000,
        )
        # print(outputs.shape, isinstance(outputs, tuple))

        start_time = time.perf_counter()
        if isinstance(outputs, tuple):
            outputs = tuple(
                [
                    output.detach().cpu().numpy() if output is not None else None
                    for output in outputs
                ]
            )
        else:
            outputs = outputs.detach().cpu().numpy()

        end_time = time.perf_counter()
        self.logger.info(
            "L(%s:%s[%s]:%s:%s)",
            tag, self.key, os.getpid(),
            "outputscopy",
            (end_time - start_time) * 1000,
        )
        
        return outputs

    async def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.model_inference(*args, **kwds)

    # async def __call__(self, request):
    #     data = await request.json()

    #     batch_size = len(data["input_ids"])
    #     input_ids = np.array(data["input_ids"])
    #     attention_mask = np.array(data["attention_mask"])

    #     step = data["step"]
    #     pid = data["pid"]



    #     outputs = self.model_inference(
    #         input_ids, attention_mask, (None, None, None, None)
    #     )
    #     # print("a", type(outputs))

    #     return {
    #         "step": step,
    #         "pid": pid,
    #         "logits": outputs.tolist(),
    #         # "labels": np.argmax(outputs, axis=-1).flatten().tolist(),
    #     }


# remote_handles = []
for d, ensemble_option in enumerate(deploy_options):
    for p, parallel_options in enumerate(ensemble_option.parallel_options):
        for r, replication_options in enumerate(parallel_options.replication_options):
            T5Model.options(
                name=replication_options.key,
                ray_actor_options=ensemble_option.ray_actor_options,
            ).deploy(options=ensemble_option, replica=r, stage=p)
            # deploy_options[d].parallel_options[p].replication_options[
            #     r
            # ].handle = serve.get_deployment(replication_options.key).get_handle()


# remote_handles.append(
#     serve.get_deployment(replication_options.key).get_handle()
# )
# latency_monitor = LatencyMonitor(serve_args.tag, config)

# key = "12345"
# key = bytes(key, encoding='utf8')

# BaseManager.register("LatencyMonitor", LatencyMonitor)
# manager = BaseManager(authkey=key)
# manager.start()
# latency_monitor = manager.LatencyMonitor(serve_args.tag, config)

# print("=============================================================================================")

@serve.deployment(max_concurrent_queries=100, route_prefix="/composed")
class T5Ensemble:
    def __init__(self, deploy_options):

        # self.deploy_options = deploy_options
        # self.latency_monitor = LatencyMonitor(serve_args.tag, config)

        self.logger = Logger(__file__, "info", 50000000, 5)

        self.logger.info("T5Ensemble logger initialized")

        self.model_name = model_name = model_ensemble_name

        self.logger.info("T5Ensemble read cfg %s", serve_args.cfg)

        self.ensembles = deploy_options
        self.num_ensembles = len(deploy_options)

        # self.thresholds = [cfg[t]["threshold"] for t in model_subtypes]

        self.logger.info("Current ensembles %s", self.ensembles)

    def schedule_handle(self, type, parallel_options):
        num_replicas = len(parallel_options.replication_options)
        if type == "rand":
            r = np.random.choice(num_replicas)
        elif type == "rr":
            parallel_options.rr_counter += 1
            r = parallel_options.rr_counter % num_replicas
        else:
            raise ValueError("scheduler type %s not defined")
        key = parallel_options.replication_options[r].key
        # print(key, parallel_options.rr_counter)
        return serve.get_deployment(key).get_handle(sync=False), key

    # async def __call__(self, request):
    #     data = await request.json()

    #     batch_size = len(data["input_ids"])
    #     input_ids = np.array(data["input_ids"])
    #     attention_mask = np.array(data["attention_mask"])
    #     ensemble_outputs = None

    #     step = data["step"]
    #     pid = data["pid"]

    #     # options = self.ensembles[-1]
    #     for idx, options in enumerate(self.ensembles):
    #         outputs = (None, None, None, None)
    #         for parallel_options in options.parallel_options:
    #             handle = self.schedule_handle(options.scheduler, parallel_options)
    #             outputs = await handle.model_inference.remote(
    #                 input_ids, attention_mask, outputs
    #             )
    #         # print(type(outputs))
    #         outputs = ray.get(outputs)
    #         # print(outputs.shape, None if ensemble_outputs is None else ensemble_outputs.shape)
    #         if ensemble_outputs is None:
    #             ensemble_outputs = copy.deepcopy(outputs)
    #         else:
    #             ensemble_outputs += copy.deepcopy(outputs)

    #     return {
    #         "step": step,
    #         "pid": pid,
    #         "logits": ensemble_outputs.tolist(),
    #         "labels": np.argmax(ensemble_outputs, axis=-1).flatten().tolist(),
    #     }

    # This method can be called concurrently!
    async def __call__(self, request):
        data = await request.json()

        batch_size = len(data["input_ids"])
        # input_ids = await torch.as_tensor(data["input_ids"], dtype=torch.long, device="cuda:0")
        # attention_mask = await torch.as_tensor(data["attention_mask"], dtype=torch.long, device="cuda:0")
        input_ids = np.array(data["input_ids"])
        input_ids.setflags(write=0)
        attention_mask = np.array(data["attention_mask"])
        attention_mask.setflags(write=0)
        ensemble_outputs = None
        # hist_outputs = []
        batch_mask = np.zeros((self.num_ensembles, batch_size))
        # batch_mask = np.ones((self.num_ensembles, batch_size))
        batch_mask[0, :] = 1  # WHERE TO ENTER
        batch_mask = batch_mask.astype(bool)

        tag = data["tag"]
        pid = data["pid"]

        for idx, options in enumerate(self.ensembles):
            local_mask = batch_mask[idx]
            self.logger.trace("%s local_mask %s", options.name, local_mask)

            # print(len(options.parallel_options))

            if np.any(local_mask):

                outputs = (input_ids[local_mask], attention_mask[local_mask]) + tuple([None] * 6)
                types = (torch.int, torch.int) + tuple([torch.float] * 6)
                for parallel_options in options.parallel_options:
                    handle, key = self.schedule_handle(options.scheduler, parallel_options)
                    # start_time = time.monotonic()
                    outputs = await handle.model_inference.remote(outputs, types, tag=tag)
                    # end_time = time.monotonic()
                    # self.logger.info(
                    #     "L(%s:%s:%s)",
                    #     tag, key, (end_time - start_time) * 1000,
                    # )
                    # await self.latency_monitor.observe((end_time-start_time)*1000, key)
                # print(outputs.shape, outputs)
                outputs = ray.get(outputs)

                start_time = time.perf_counter()
                ensemble_outputs = self.model_ensemble(
                    ensemble_outputs, outputs, local_mask, idx
                )
                end_time = time.perf_counter()
                self.logger.info(
                    "E(%s:%s:%s)",
                    tag,
                    self.ensembles[idx].name,
                    (end_time - start_time) * 1000,
                )

                extended_mask, max_prob = self.offload_mask(
                    ensemble_outputs, local_mask, idx
                )

                self.logger.trace(
                    "%s local_mask updated %s", options.name, extended_mask
                )
                num_next_models = self.num_ensembles - idx - 1
                if np.any(extended_mask) and num_next_models > 0:
                    batch_mask[idx] &= ~extended_mask
                    batch_mask[idx+1] |= extended_mask
                    # batch_mask = self.update_batch_mask(
                    #     max_prob, batch_mask.copy(), extended_mask, idx
                    # )
                    # self.logger.trace(
                    #     "%s batch_mask updated %s", options.name, batch_mask
                    # )
            # print(options.name, num_next_models, batch_mask)
            assert np.sum(batch_mask) == batch_size
        # assert ensemble_outputs.shape == (batch_size, 2)
        # return {"labels": np.argmax(ensemble_outputs, axis=-1).flatten().tolist()}
        return {
            "tag": tag,
            "pid": pid,
            "logits": ensemble_outputs.tolist(),
            # "labels": np.argmax(ensemble_outputs, axis=-1).flatten().tolist(),
        }

    def offload_mask(self, logits, mask, idx):
        probabilities = np.power(m(logits), 2)
        max_prob = np.max(probabilities, axis=-1)
        prob_mask = max_prob < self.ensembles[idx].threshold
        self.logger.trace(
            "%s (offload_mask) prob_mask %s %s",
            self.ensembles[idx].name,
            prob_mask,
            mask,
        )
        extended_mask = mask & prob_mask
        # combined_mask[mask] &= prob_mask[mask]
        self.logger.trace("max_prob %s, extended_mask %s", max_prob, extended_mask)
        return extended_mask, max_prob

    # async def model_ensemble(
    #     self, hist_outputs, mask, idx
    # ):
    #     start_time = time.perf_counter()
    #     hist_outputs = np.array(hist_outputs)
    #     hist_mask = mask[:(idx+1)]
    #     ensemble_outputs = hist_outputs
    #     end_time = time.perf_counter()
    #     self.logger.info(
    #         "%s model_ensemble time elapsed %s (ms)",
    #         model_subtypes[idx],
    #         (end_time - start_time) * 1000,
    #     )
    #     self.logger.trace(
    #         "%s ensemble_outputs %s",
    #         model_subtypes[idx],
    #         ensemble_outputs,
    #     )
    #     return ensemble_outputs if ensemble_outputs is not None else local_outputs

    def model_ensemble(self, ensemble_outputs, local_outputs, mask, idx):
        # start_time = time.perf_counter()
        if ensemble_outputs is not None:
            ensemble_weight = self.ensembles[idx].ensemble_weight
            ensemble_outputs[mask] = (
                ensemble_outputs[mask] * (1 - ensemble_weight)
                + local_outputs * ensemble_weight
            )
        self.logger.trace(
            "%s ensemble_outputs %s",
            self.ensembles[idx].name,
            ensemble_outputs,
        )
        return (
            ensemble_outputs if ensemble_outputs is not None else local_outputs.copy()
        )  # MEMCOPY

    def update_batch_mask(self, max_prob, mask, local_mask, idx):
        num_next_models = self.num_ensembles - idx - 1

        if num_next_models <= 0:
            return mask

        if self.ensembles[idx].skip_connection:
            base_step = (self.ensembles[idx].threshold - 0.25) / num_next_models
            for skip in range(num_next_models):
                skip_th_lower = base_step * (num_next_models - 1 - skip) + 0.25
                skip_th_upper = base_step * (num_next_models - skip) + 0.25
                skip_mask = (
                    (max_prob >= skip_th_lower)
                    & (max_prob < skip_th_upper)
                    & local_mask
                )
                self.logger.trace(
                    "%s skip_th_lower %s, skip_th_upper %s, skip_mask %s",
                    self.ensembles[idx].name,
                    skip_th_lower,
                    skip_th_upper,
                    skip_mask,
                )
                mask[skip + 1 + idx] |= skip_mask
        else:
            mask[1 + idx] |= (max_prob < self.ensembles[idx].threshold) & local_mask

        mask[idx] &= ~local_mask
        return mask


T5Ensemble.options(
    num_replicas=4, ray_actor_options={"num_cpus": 5}
).deploy(deploy_options)

# input_ids = [[200, 200, 200, 200, 0, 0, 0, 0]]
# attention_mask = [[1, 1, 1, 1, 0, 0, 0, 0]]
# ensemble_outputs = ray.put(np.array([[-100, -100]]))
# batch_mask = ray.put(np.array([[True], [False], [False], [False]]))
# for _ in tqdm(range(1000)):
#     resp = requests.post(
#         "http://127.0.0.1:8000/composed",
#         json={
#             "input_ids": input_ids,
#             "attention_mask": attention_mask,
#         },
#     )
# print(resp.content)
# print(np.frombuffer(resp.content, dtype=np.float64))

# import signal
# import sys

# def signal_handler(sig, frame):
#     serve.shutdown()
#     sys.exit(0)


while True:
    pass
