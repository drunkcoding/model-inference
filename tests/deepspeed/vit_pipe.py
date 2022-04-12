from dataclasses import dataclass, field
import os
import time
import torch
import deepspeed
from deepspeed.utils import RepeatingLoader
from tqdm import tqdm
import requests
import re
from torchvision.datasets import ImageNet
from torch.utils.data import DataLoader
import logging

from transformers import (
    ViTFeatureExtractor,
    HfArgumentParser,
    ViTForImageClassification,
)
from transformers.models.vit.configuration_vit import ViTConfig
from transformers.utils.model_parallel_utils import get_device_map
from hfutils.pipe.vit import ViTDeepSpeedPipe
from hfutils.logger import Logger
from hfutils.preprocess import (
    vit_collate_fn,
    ViTFeatureExtractorTransforms,
)
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

home_dir = "/mnt/raid0nvme1"
dataset_path = os.path.join(home_dir, "ImageNet")

dataset = ImageNet(
    dataset_path,
    split="val",
    transform=ViTFeatureExtractorTransforms(args.model_name_or_path, split="val"),
)

batch_size = 128

def eval_generator():
    dataloader = DataLoader(
        dataset,
        shuffle=False,
        collate_fn=vit_collate_fn,
        batch_size=batch_size,
        num_workers=world_size,
        drop_last=True
    )
    pid = os.getpid()

    batches = []
    for batch in dataloader:
        batches.append(batch)
        if len(batches) > 510:
            break
    for batch in tqdm(batches):
        shape = batch["pixel_values"].shape
        yield (
            (batch["pixel_values"],),
            torch.zeros(shape[0]),
        )


config = ViTConfig.from_pretrained(args.model_name_or_path)

if local_rank == 0 and batch_size == 1:
    model = ViTForImageClassification.from_pretrained(args.model_name_or_path)
    config = ViTConfig.from_pretrained(args.model_name_or_path)
    if hasattr(model, "parallelize"):
        model.parallelize(get_device_map(get_num_layers(config), range(world_size)))
    else:
        model = model.to("cuda:0")
    records = []
    for step, batch in enumerate(tqdm(eval_generator(), desc="single model")):
        if step > 500:
            break
        pixel_values = batch[0][0].to("cuda:0")
        start_time = time.perf_counter()
        model(pixel_values=pixel_values)
        end_time = time.perf_counter()
        records.append((start_time, end_time))
    for start_time, end_time in records:
        logger.critical(
            "(-1) start_time %s, end_time %s, diff %s",
            start_time,
            end_time,
            end_time - start_time,
        )

    del model


deepspeed.init_distributed()

local_rank = int(os.getenv("LOCAL_RANK", "0"))
world_size = int(os.getenv("WORLD_SIZE", "1"))

model = ViTDeepSpeedPipe(config, num_stages=1)

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


start_energy = sum(list(get_energy_by_group().values()))
records = []
inference_count = 0
for step, batch in enumerate(RepeatingLoader(eval_generator())):
    if step > 500:
        break
    start_time = time.perf_counter()
    outputs = engine.eval_batch(iter([batch] * 1), compute_loss=False)
    end_time = time.perf_counter()
    if outputs != None:
        inference_count += 1
    records.append((start_time, end_time))
logger.critical("%s %s ", os.getpid(), inference_count)

if local_rank == 0:
    for start_time, end_time in records:
        logger.critical(
            "(%s) start_time %s, end_time %s, diff %s",
            world_size,
            start_time,
            end_time,
            end_time - start_time,
        )

end_energy = sum(list(get_energy_by_group().values()))
if local_rank == 0:
    logger.critical(
        "(%s) start_energy %s, end_energy %s, diff %s",
        world_size,
        start_energy,
        end_energy,
        end_energy - start_energy,
    )

