from dataclasses import dataclass, field
from functools import partial
import logging
import os
import time
import torch
from transformers import (
    HfArgumentParser,
    ViTForImageClassification,
)
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from torchvision.datasets import ImageNet

from hfutils.logger import Logger
from hfutils.pipe.vit import ViTPyTorchPipeForImageClassification
from hfutils.preprocess import (
    vit_collate_fn,
    ViTFeatureExtractorTransforms,
)
from hfutils.measure import get_energy_by_group, get_gpu_uuid


@dataclass
class Arguments:
    model_name_or_path: str = field(
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        }
    )
    batch_size: int = field(metadata={"help": "batch size for profiling kernel"})


logger = Logger(__file__, logging.INFO, 50000000, 5)
parser = HfArgumentParser(Arguments)
args = parser.parse_args_into_dataclasses()[0]
basename = os.path.basename(args.model_name_or_path)

logger.info("=================================")
logger.info("%s", args)

home_dir = "/mnt/raid0nvme1"
dataset_path = os.path.join(home_dir, "ImageNet")
dataset = ImageNet(
    dataset_path,
    split="val",
    transform=ViTFeatureExtractorTransforms(args.model_name_or_path, split="val"),
)

dataloader = DataLoader(
    dataset,
    shuffle=False,
    collate_fn=vit_collate_fn,
    batch_size=args.batch_size,
    num_workers=20,
    drop_last=True,
)

device_id = 1
device = f"cuda:{device_id}"
uuid = get_gpu_uuid(device_id)

model = ViTForImageClassification.from_pretrained(args.model_name_or_path)
model = ViTPyTorchPipeForImageClassification(model)
model.convert(device)

start_energy = get_energy_by_group()[uuid]
records_start = []
records_end = []
for step, batch in enumerate(tqdm(dataloader, desc=f"{args.batch_size}")):
    pixel_values = batch["pixel_values"].to(device)
    start_time = time.perf_counter()
    model((pixel_values,))
    end_time = time.perf_counter()
    if step > 10:
        records_start.append(start_time)
        records_end.append(end_time)
    if step > 100:
        break
end_energy = get_energy_by_group()[uuid]

diff = end_energy - start_energy
logger.info(
    "energy total %s, request %s, sample %s",
    diff,
    diff / step,
    diff / step / args.batch_size,
)
logger.info(
    "memory reserved %s, allocated %s, total %s",
    torch.cuda.memory_reserved(device_id),
    torch.cuda.memory_allocated(device_id),
    torch.cuda.get_device_properties(device_id).total_memory,
)


df = pd.DataFrame({
    "model": [basename]*len(records_end),
    "batch_size": [args.batch_size]*len(records_end),
    "start_time": records_start,
    "end_time": records_end,
    "latency": np.array(records_end) - np.array(records_start),
})
df.to_csv(os.path.join("profile", f"latency_{basename}_{args.batch_size}.csv"), index=False)
logger.info("%s", df.describe())
