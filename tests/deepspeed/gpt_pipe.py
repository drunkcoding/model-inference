from dataclasses import dataclass, field
import logging
import os
import time
import torch
import deepspeed
from deepspeed.utils import RepeatingLoader
from tqdm import tqdm
import requests
import re

from transformers import AutoConfig, HfArgumentParser
from transformers import GPT2Tokenizer, AutoModelWithLMHead
from transformers.models.gpt2.configuration_gpt2 import GPT2Config
from transformers.utils.model_parallel_utils import get_device_map
from datasets import load_dataset, load_metric, concatenate_datasets
from hfutils.pipe.gpt import GPTDeepSpeedPipe
from hfutils.logger import Logger
from hfutils.pipe.base import get_num_layers

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TORCH_EXTENSIONS_DIR"] = "."


@dataclass
class Arguments:
    model_name_or_path: str = field(
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        }
    )
    deepspeed_config: str = field(
        default=None, metadata={"help": "DeepSpeed configuration path."},
    )
    local_rank: int = field(
        default=-1, metadata={"help": "Place holder for deepspeed launcher."},
    )


local_rank = int(os.getenv("LOCAL_RANK", "0"))
world_size = int(os.getenv("WORLD_SIZE", "1"))

logger = Logger(__file__, logging.CRITICAL, 50000000, 5)
parser = HfArgumentParser(Arguments)
args = parser.parse_args_into_dataclasses()[0]

if local_rank == 0:
    logger.critical("=================================")
    logger.critical("%s", args)

tokenizer = GPT2Tokenizer.from_pretrained(args.model_name_or_path)
config = GPT2Config.from_pretrained(args.model_name_or_path)

val_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
# val_dataset = val_dataset.select([x for x in range(500)])
print(val_dataset)
encodings = tokenizer("\n\n".join(val_dataset["text"]), return_tensors="pt")
encodings.input_ids = encodings.input_ids.to(torch.long)
print(encodings.input_ids.shape)


def load_encodings(encodings):
    max_length = 512
    stride = 128

    for i in tqdm(range(0, encodings.input_ids.size(1), stride)):
        begin_loc = max(i + stride - max_length, 0)
        end_loc = min(i + stride, encodings.input_ids.size(1))
        trg_len = end_loc - i  # may be different from stride on last loop
        input_ids = encodings.input_ids[:, begin_loc:end_loc]
        # input_ids = input_ids.to(torch.int8)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        if input_ids.size(1) != max_length:
            continue

        yield input_ids, target_ids, trg_len, end_loc


# if local_rank == 0:
#     model = AutoModelWithLMHead.from_pretrained(args.model_name_or_path)
#     config = AutoConfig.from_pretrained(args.model_name_or_path)
#     if hasattr(model, "parallelize"):
#         model.parallelize(
#             get_device_map(get_num_layers(config), range(world_size))
#         )
#     else:
#         model = model.to("cuda:0")
#     for step, batch in enumerate(tqdm(load_encodings(encodings), desc="single model")):
#         if step > 500:
#             break
#         input_ids = batch[0].to("cuda:0")
#         start_time = time.perf_counter()
#         model(input_ids=input_ids)
#         end_time = time.perf_counter()

#         logger.critical(
#             "(-1) start_time %s, end_time %s, diff %s",
#             start_time,
#             end_time,
#             end_time - start_time,
#         )

#     del model

deepspeed.init_distributed()
model = GPTDeepSpeedPipe(config, num_stages=2)

engine, _, _, _ = deepspeed.initialize(config=args.deepspeed_config, model=model)


def get_energy_by_group():
    response = requests.get("http://localhost:8002/metrics")
    text = response.text
    energy_groups = re.findall(
        r'nv_energy_consumption{gpu_uuid="(.*)"} (\d+.\d+)', text
    )
    energy_groups = dict(energy_groups)
    for k in energy_groups:
        energy_groups[k] = float(energy_groups[k])
    return energy_groups

import socket
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s.connect(('8.8.8.8', 1))
host_ip = s.getsockname()[0]


start_energy = sum(list(get_energy_by_group().values()))
inference_count = 0
for step, batch in enumerate(RepeatingLoader(load_encodings(encodings))):
    if step > 100:
        break
    shape = batch[0].shape
    inputs = ((batch[0], torch.ones(shape),), torch.zeros(shape[0]))
    start_time = time.perf_counter()
    outputs = engine.eval_batch(iter([inputs] * 1), compute_loss=False)
    if outputs != None:
        inference_count += 1
    # print(local_rank, step, outputs if outputs is None else outputs.size())
    end_time  = time.perf_counter()
    if local_rank == 0:
        logger.critical(
            "(%s) start_time %s, end_time %s, diff %s",
            world_size,
            start_time,
            end_time,
            end_time - start_time,
        )
logger.critical("%s %s ", os.getpid(), inference_count)

end_energy = sum(list(get_energy_by_group().values()))
if local_rank == 0:
    logger.critical(
        "(%s) start_energy %s, end_energy %s, diff %s",
        world_size,
        start_energy,
        end_energy,
        end_energy - start_energy,
    )
