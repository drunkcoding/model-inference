from dataclasses import dataclass
import functools
import traceback
from typing import Any, Dict, List, Tuple
from async_timeout import enum
from attr import field
from ray import data, serve
from ray.serve import pipeline
from transformers import AutoConfig, T5ForConditionalGeneration
import requests
import os
import ray
import torch
import time
import threading
from fastapi import FastAPI
import json
from requests import Request
import numpy as np
from scipy.special import softmax
from transformers import AutoTokenizer
from tqdm import tqdm

from hfutils.arg_parser import RayArguments
from hfutils.logger import Logger
import gc
from hfutils.options import EnsembleOptions, ParallelOptions, ReplicationOptions
from hfutils.model_pipe import T5Pipe, get_num_layers
from hfutils.calibration import agg_logits, temperature_scale
from hfutils.constants import (
    MODEL_TASK_TO_CLASS,
    ENSEMBLE_ORDER,
    TASK_TO_LABELS,
    np_to_torch_dtype,
    MODEL_KEYS,
)

args = RayArguments()
serve_args = args.serve_args

print(args)
print(serve_args)

app = FastAPI()
ray.init(num_cpus=os.cpu_count(), num_gpus=torch.cuda.device_count())
serve.start()

# print(torch.cuda.is_available())

# m = torch.nn.Softmax(dim=-1)
m = functools.partial(softmax, axis=-1)

home_dir = "/sata_disk/jupyter-xue"
base_dir = os.path.join(home_dir, os.path.join("model-finetune", "outputs"))
model_repository = os.path.join(home_dir, os.path.join("model-inference", "repository"))
task_name = "sst2"
model_ensemble_name = "_".join(["t5", task_name, "test"])

tokenizer = AutoTokenizer.from_pretrained("google/t5-small-lm-adapt", use_fast=False)
label_tokens = [
    tokenizer(label, max_length=2).input_ids[0]
    for label in TASK_TO_LABELS[task_name]
    if label is not None
]


# model_subtypes = [
#     "t5-small-lm-adapt_sst2",
#     "t5-base-lm-adapt_sst2",
#     "t5-large-lm-adapt_sst2",
#     "t5-xl-lm-adapt_sst2",
# ]

# model_options = [
#     {"num_replicas": 1, "ray_actor_options": {"num_gpus": 0.01, "num_cpus": 4}},
#     {"num_replicas": 1, "ray_actor_options": {"num_gpus": 0.15, "num_cpus": 4}},
#     {"num_replicas": 1, "ray_actor_options": {"num_gpus": 0.25, "num_cpus": 4}},
#     {"num_replicas": 1, "ray_actor_options": {"num_gpus": 0.5, "num_cpus": 4}},
# ]


# model_paths = [
#     f"{base_dir}/t5-small-lm-adapt/{task_name}/checkpoint-5540",
#     f"{base_dir}/t5-base-lm-adapt/{task_name}/checkpoint-1860",
#     f"{base_dir}/t5-large-lm-adapt/{task_name}/checkpoint-1780",
#     f"{base_dir}/t5-xl-lm-adapt/{task_name}/checkpoint-1380",
# ]

# model_paths = dict(zip(MODEL_KEYS, model_paths))
# # model_names = dict(zip(model_keys, model_subtypes))
# deploy_options = dict(zip(MODEL_KEYS, model_options))

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
        num_replicas = model_config["count"] * num_gpus

        hf_config = AutoConfig.from_pretrained(ckpt_path)
        total_pp_layers = (get_num_layers(hf_config) + 4) * 2 + 1

        print(name, "num_layers", total_pp_layers)

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
                ray_actor_options={"num_gpus": 0.01, "num_cpus": 4},
                parallel_options=[
                    ParallelOptions(
                        num_stages=parallel_stages,
                        parallel_stage=p,
                        first_parallel_layer=pp_layers * p,
                        last_parallel_layer=min(total_pp_layers, pp_layers * (p + 1)),
                        replication_options=[
                            ReplicationOptions(
                                replica_id=r,
                                key=f"{MODEL_KEYS[idx]}R{r}P{p}",
                                device=torch.device(
                                    "cuda:" + visible_gpus[(r * p) % num_gpus]
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
        self.logger = Logger(__file__, "debug", 5000000, 5)
        self.logger.info("%s logger initialized", options.name)

        self.options = options

        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(visible_gpus)
        self.logger.info("CUDA_VISIBLE_DEVICES: %s", os.environ["CUDA_VISIBLE_DEVICES"])

        # visible_gpus = ray.get_gpu_ids()

        # torch.cuda.dev
        self.logger.info("options: %s", options)

        # self.logger.info("ray.get_gpu_ids(): {}".format(ray.get_gpu_ids()))
        # self.logger.info("torch visible devices: {}".format(torch.cuda.device_count()))

        self.device = (
            options.parallel_options[stage].replication_options[replica].device
        )
        self.logger.debug("%s device %s", options.name, self.device)
        self.key = options.parallel_options[stage].replication_options[replica].key
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

        model = T5ForConditionalGeneration.from_pretrained(options.ckpt_path)
        self.model_parallel = options.parallel
        exec_map = (
            options.parallel_options[stage].first_parallel_layer,
            options.parallel_options[stage].last_parallel_layer,
        )
        # self.logger.info("%s garbage collected", model_name)
        if self.model_parallel:
            self.model = T5Pipe(model, exec_map)
            self.logger.debug("%s T5Pipe num_layers %s ", model_name, len(self.model.pipe))
            # # for i in range(len(self.model.pipe)):
            # #     print((i - 1) // (len(self.model.pipe) // len(visible_gpus)))
            # device_map = [
            #     torch.device(
            #         "cuda:"
            #         + str(
            #             visible_gpus[
            #                 (i - 1) // (len(self.model.pipe) // len(visible_gpus))
            #             ]
            #         )
            #     )
            #     for i in range(len(self.model.pipe))
            # ]
            # # device_map = [torch.device("cpu")] * len(self.model.pipe)
            # self.logger.info(
            #     "%s model layers map to devices %s", model_name, device_map
            # )
            # self.model.parallize(device_map)
        else:
            self.model = model.to(self.device)
        self.logger.info("%s model initialized %s", model_name, type(self.model))

        del model
        torch.cuda.empty_cache()
        gc.collect()

        self.logger.debug("%s garbage collected", model_name)

        # meta_file = os.path.join(model_repository, "meta.json")
        # print(meta_file, flush=True)

        # self.cfg_timer = threading.Timer(10, self.read_cfg, (meta_file,))
        # self.cfg_timer.start()

        # self.read_cfg(serve_args.cfg)

        self.logger.info("%s full initialization complete", model_name)

    async def t5_parallel_inference(self, input_ids, attention_mask, *args):
        batch_size = input_ids.shape[0]
        outputs = await self.model.forward_async(
            (
                input_ids,
                attention_mask,
                *args,
            )
        )
        for i, t in enumerate(outputs):
            self.logger.debug(
                "%s t5_parallel_inference outputs size (%s) %s",
                self.model_name,
                i,
                t.size() if t is not None else None,
            )
        if self.parallel_stage == self.num_stages - 1:
            logits = outputs[1].view(batch_size, -1)[:, label_tokens]
            logits = temperature_scale(logits, self.temperature)
        else:
            return outputs

    async def t5_inference(self, input_ids, attention_mask, *args):
        outputs = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            do_sample=False,  # disable sampling to test if batching affects output
            return_dict_in_generate=True,
            output_scores=True,
        )
        logits = outputs.scores[0][:, label_tokens]
        logits = temperature_scale(logits, self.temperature)
        return logits

    async def default_inference(self, input_ids, attention_mask, *args):
        logits = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        ).logits
        logits = temperature_scale(logits, self.temperature)
        return logits

    @torch.no_grad()
    async def model_inference(self, input_ids, attention_mask, parallel_args, **kwds):
        start_time = time.perf_counter()

        input_ids = torch.as_tensor(input_ids, dtype=torch.long, device=self.device)
        attention_mask = torch.as_tensor(
            attention_mask, dtype=torch.long, device=self.device
        )

        tensor_args = (
            torch.as_tensor(arg, dtype=torch.float, device=self.device)
            if arg is not None
            else None
            for arg in parallel_args
        )

        masked_inputs = (
            input_ids,
            attention_mask,
            *tensor_args,
        )

        # with torch.cuda.stream(self.cuda_stream):
        if "t5" in self.model_name and not self.model_parallel:
            outputs = await self.t5_inference(*masked_inputs)
        elif "t5" in self.model_name and self.model_parallel:
            outputs = await self.t5_parallel_inference(*masked_inputs)
        else:
            outputs = await self.default_inference(*masked_inputs)
        end_time = time.perf_counter()
        self.logger.info(
            "%s model_inference time elapsed %s (ms)",
            self.model_name,
            (end_time - start_time) * 1000,
        )

        if isinstance(outputs, tuple):
            return tuple(
                [
                    output.detach().cpu().numpy() if output is not None else None
                    for output in outputs
                ]
            )
        else:
            return outputs.detach().cpu().numpy()

    async def __call__(self, *args: Any, **kwds: Any) -> Any:
        return await self.model_inference(*args, **kwds)

    # def read_cfg(self, path):
    #     torch.cuda.empty_cache()
    #     gc.collect()
    #     # print(torch.cuda.is_available())
    #     try:
    #         with open(path, "r") as fp:
    #             config = json.load(fp)
    #         self.parallel_stages = config[self.model_name]["parallel_stages"]
    #         self.threshold = config[self.model_name]["threshold"]
    #         temperature = config[self.model_name]["temperature"]
    #         self.temperature = torch.nn.Parameter(
    #             torch.ones(1, device=self.device) * temperature
    #         )
    #         self.ensemble_pos = config[self.model_name]["ensemble_pos"]
    #         self.ensemble_weight = config[model_ensemble_name]["weights"][
    #             self.ensemble_pos
    #         ]
    #         self.num_ensembles = len(config[model_ensemble_name]["weights"])
    #         self.logger.info(
    #             "%s load meta from %s \n threshold %s, temperature %s, ensemble_pos %s, ensemble_weight %s, num_ensembles %s",
    #             self.model_name,
    #             path,
    #             self.threshold,
    #             self.temperature,
    #             self.ensemble_pos,
    #             self.ensemble_weight,
    #             self.num_ensembles,
    #         )
    #     except Exception as e:
    #         self.logger.warn("read_cfg exception %s", traceback.print_stack())

    #     self.cfg_timer = threading.Timer(10, self.read_cfg, (path,))
    #     self.cfg_timer.start()


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


@serve.deployment(max_concurrent_queries=10, route_prefix="/composed")
class T5Ensemble:
    def __init__(self, deploy_options):

        # self.deploy_options = deploy_options

        self.logger = Logger(__file__, "info", 5000000, 5)

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
        return serve.get_deployment(key).get_handle(sync=False)

    # This method can be called concurrently!
    async def __call__(self, request):
        data = await request.json()

        batch_size = len(data["input_ids"])
        # input_ids = await torch.as_tensor(data["input_ids"], dtype=torch.long, device="cuda:0")
        # attention_mask = await torch.as_tensor(data["attention_mask"], dtype=torch.long, device="cuda:0")
        input_ids = np.array(data["input_ids"])
        attention_mask = np.array(data["attention_mask"])
        ensemble_outputs = None
        # hist_outputs = []
        batch_mask = np.zeros((self.num_ensembles, batch_size))
        batch_mask[0] = np.ones(batch_size)  # WHERE TO ENTER
        batch_mask = batch_mask.astype(bool)

        for idx, options in enumerate(self.ensembles):
            local_mask = batch_mask[idx].view()
            self.logger.debug("%s local_mask %s", options.name, local_mask)

            if np.any(local_mask):

                outputs = ray.put((None, None, None, None))
                for parallel_options in options.parallel_options:
                    handle = self.schedule_handle(options.scheduler, parallel_options)
                    outputs = await handle.remote(
                        input_ids[local_mask], attention_mask[local_mask], outputs
                    )
                outputs = ray.get(outputs)
                # hist_outputs.append(outputs)
                # print(type(outputs), "===========")
                # if not self.model_parallel or (self.model_parallel and self.parallel_stage == self.num_stages - 1):
                ensemble_outputs = await self.model_ensemble(
                    ensemble_outputs, outputs, local_mask, idx
                )

                local_mask, max_prob = await self.offload_mask(
                    ensemble_outputs, local_mask, idx
                )
                self.logger.debug("%s local_mask updated %s", options.name, local_mask)
                if np.any(local_mask):
                    batch_mask = await self.update_batch_mask(
                        max_prob, batch_mask, local_mask, idx
                    )
                    self.logger.debug(
                        "%s batch_mask updated %s", options.name, batch_mask
                    )
        # assert np.sum(batch_mask) == batch_size
        return {"labels": np.argmax(ensemble_outputs, axis=-1).flatten().tolist()}

    async def offload_mask(self, logits, mask, idx):
        probabilities = np.power(m(logits), 2)
        max_prob = np.max(probabilities, axis=-1)
        prob_mask = max_prob < self.ensembles[idx].threshold
        self.logger.debug(
            "%s (offload_mask) prob_mask %s %s",
            self.ensembles[idx].name,
            prob_mask,
            mask,
        )
        combined_mask = mask & prob_mask
        # combined_mask[mask] &= prob_mask[mask]
        self.logger.debug("max_prob %s, combined_mask %s", max_prob, combined_mask)
        return combined_mask, max_prob

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
    #     self.logger.debug(
    #         "%s ensemble_outputs %s",
    #         model_subtypes[idx],
    #         ensemble_outputs,
    #     )
    #     return ensemble_outputs if ensemble_outputs is not None else local_outputs

    async def model_ensemble(self, ensemble_outputs, local_outputs, mask, idx):
        start_time = time.perf_counter()
        if ensemble_outputs is not None:
            ensemble_outputs[mask] = (
                ensemble_outputs[mask] * (1 - self.ensembles[idx].ensemble_weight)
                + local_outputs * self.ensembles[idx].ensemble_weight
            )
        end_time = time.perf_counter()
        self.logger.info(
            "%s model_ensemble time elapsed %s (ms)",
            self.ensembles[idx].name,
            (end_time - start_time) * 1000,
        )
        self.logger.debug(
            "%s ensemble_outputs %s",
            self.ensembles[idx].name,
            ensemble_outputs,
        )
        return (
            ensemble_outputs if ensemble_outputs is not None else local_outputs
        )  # MEMCOPY

    async def update_batch_mask(self, max_prob, mask, local_mask, idx):
        num_next_models = self.num_ensembles - idx - 1

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
                self.logger.debug(
                    "%s skip_th_lower %s, skip_th_upper %s, skip_mask %s",
                    self.ensembles[idx].name,
                    skip_th_lower,
                    skip_th_upper,
                    skip_mask,
                )
                mask[skip + 1 + idx] |= skip_mask

        elif num_next_models > 0:
            mask[1 + idx] |= (max_prob < self.ensembles[idx].threshold) & local_mask

        mask[idx] &= ~local_mask
        return mask

    # @app.post("/log-level/{level}")
    # def update_log_level(self, level: str):
    #     self.logger = Logger(__file__, level, 5000000, 5)
    #     return f"log level updated to {level}"


T5Ensemble.options(num_replicas=4).deploy(deploy_options)

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

while True:
    pass
