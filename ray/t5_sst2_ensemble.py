from ray import serve
from ray.serve import pipeline
from transformers import T5ForConditionalGeneration
import requests
import os
import ray
import torch
import time
import threading
import json
from requests import Request
import numpy as np

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

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

ray.init()
serve.start()

print(torch.cuda.is_available())

home_dir = "/sata_disk/jupyter-xue"
base_dir = os.path.join(home_dir, os.path.join("model-finetune", "outputs"))
model_repository = os.path.join(home_dir, os.path.join("model-inference", "repository"))
task_name = "sst2"
model_ensemble_name = "_".join(["t5", task_name, "ensemble"])

class PythonModel:
    def __init__(self, model_name) -> None:
        self.logger = Logger(__file__, "info", 5000000, 5)

        os.environ["CUDA_VISIBLE_DEVICES"] = "1"

        self.model_name = model_name

        self.device = "cuda"  # HACK for testing
        self.model_parallel = False

        meta_file = os.path.join(model_repository, "meta.json")
        print(meta_file, flush=True)

        # self.cfg_timer = threading.Timer(10, self.read_cfg, (meta_file,))
        # self.cfg_timer.start()

        self.read_cfg(meta_file)

    def t5_parallel_inference(self, input_ids, attention_mask, ensemble_outputs):
        outputs = self.model(
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
            logits = outputs[1][:, self.label_tokens]
            logits = temperature_scale(logits, self.temperature)
        else:
            return outputs[0]

    def t5_inference(self, input_ids, attention_mask):
        outputs = self.model.generate(
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
        logits = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        ).logits
        logits = temperature_scale(logits, self.temperature)
        return logits

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

    def parse_input(self, content, field):

        input = ray.get(content[field])
        input_zero_copy = torch.as_tensor(
            input, dtype=np_to_torch_dtype(input.dtype), device=self.device
        )

        self.logger.debug(
            "%s %s %s", field, input_zero_copy.shape, input_zero_copy.device
        )

        return input_zero_copy

    def parse_output(self, output):
        object_id = ray.put(output.as_numpy())
        return object_id

    async def __call__(self, request):
        # content = request.get_json()
        content = request.json()

        input_ids = self.parse_input(content, "input_ids")
        attention_mask = self.parse_input(content, "attention_mask")
        ensemble_outputs = self.parse_input(content, "ensemble_outputs")
        batch_mask = self.parse_input(content, "batch_mask")

        local_mask = batch_mask[self.ensemble_pos]
        self.logger.debug("local_mask %s", local_mask)

        if self.model_parallel and self.parallel_pos > 0:
            ensemble_outputs = ensemble_outputs.reshape(
                input_ids.shape + (self.cls_model.embed_dim,)
            )
            print("ensemble_outputs", ensemble_outputs.shape)

        if torch.any(local_mask):
            outputs = self.model_inference(
                input_ids, attention_mask, ensemble_outputs, local_mask
            )  # MOVE TO CPU, SAVE GPU MEMORY

            if not self.model_parallel or (
                self.model_parallel and self.parallel_pos == self.parallel_stages - 1
            ):
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
                ensemble_outputs = outputs.view(-1)

        assert torch.sum(batch_mask) == input_ids.shape[0]
        self.logger.debug(
            "%s outputs %s, batch_mask %s",
            self.model_name,
            ensemble_outputs.shape,
            batch_mask,
        )

        request.json["ensemble_outputs"] = ensemble_outputs
        request.json["batch_mask"] = batch_mask
        return request

    def read_cfg(self, path):
        torch.cuda.empty_cache()
        gc.collect()
        print(torch.cuda.is_available())
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
                self.ensemble_weight = config[model_ensemble_name]["weights"][
                    self.ensemble_pos
                ]
                self.num_ensembles = len(config[model_ensemble_name]["weights"])
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
            self.logger.warn("read_cfg %s", e)

        self.cfg_timer = threading.Timer(10, self.read_cfg, (path,))
        self.cfg_timer.start()


@pipeline.step(execution_mode="actors", num_replicas=1)
class T5Small(PythonModel):
    def __init__(self) -> None:
        super().__init__("t5-small-lm-adapt_sst2")
        # print("T5Small", torch.cuda.is_available())
        self.model = T5ForConditionalGeneration.from_pretrained(
            f"{base_dir}/t5-small-lm-adapt/{task_name}/checkpoint-5540"
        ).to("cuda")

    def __call__(self, request):
        return super().__call__(request)


@pipeline.step(execution_mode="actors", num_replicas=1)
class T5Base(PythonModel):
    def __init__(self):
        super().__init__("t5-base-lm-adapt_sst2")

        self.model = T5ForConditionalGeneration.from_pretrained(
            f"{base_dir}/t5-base-lm-adapt/{task_name}/checkpoint-1860",
        ).to("cuda")

    def __call__(self, request):
        return super().__call__(request)


@pipeline.step(execution_mode="actors", num_replicas=1)
class T5Large(PythonModel):
    def __init__(self):
        super().__init__("t5-large-lm-adapt_sst2")

        self.model = T5ForConditionalGeneration.from_pretrained(
            f"{base_dir}/t5-base-lm-adapt/{task_name}/checkpoint-1860",
        ).to("cuda")

    def __call__(self, request):
        return super().__call__(request)


@pipeline.step(execution_mode="actors", num_replicas=1)
class T5XLarge(PythonModel):
    def __init__(self):
        super().__init__("t5-xl-lm-adapt_sst2")

        self.model = T5ForConditionalGeneration.from_pretrained(
            f"{base_dir}/t5-base-lm-adapt/{task_name}/checkpoint-1860",
        ).to("cuda")

    def __call__(self, request):
        return super().__call__(request)


sequential_pipeline = T5Base()(T5Small()(pipeline.INPUT)).deploy()
# result = sequential_pipeline.call(dummy_png_bytes)

input_ids = ray.put(np.array([[200,200,200,200, 0,0,0,0]]))
attention_mask = ray.put(np.array([[1,1,1,1, 0,0,0,0]]))
ensemble_outputs = ray.put(np.array([[-100, -100]]))
batch_mask = ray.put(np.array([[True], [False], [False], [False]]))

resp = requests.post("http://127.0.0.1:8000/T5Small", json={
    "input_ids": input_ids,
    "attention_mask": attention_mask,
    "ensemble_outputs": ensemble_outputs,
    "batch_mask": batch_mask,
})
print(resp.json())

while True:
    pass
