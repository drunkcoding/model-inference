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
from hfutils.pipe.vit import ViTDeepSpeedPipe, ViTPytorchPipeRandom
from hfutils.logger import Logger
from hfutils.preprocess import (
    vit_collate_fn,
    ViTFeatureExtractorTransforms,
)
from hfutils.pipe.base import get_num_layers

from tests.deepspeed.utils import execute_model

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TORCH_EXTENSIONS_DIR"] = "."


@dataclass
class Arguments:
    model_name_or_path: str = field(
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        }
    )
    batch_size: int = field(
        metadata={"help": "Batch size."},
    )
    deepspeed_config: str = field(
        metadata={"help": "DeepSpeed configuration path."},
    )
    local_rank: int = field(
        default=-1, metadata={"help": "Place holder for deepspeed launcher."},
    )
    inference: bool = field(
        default=False,
        metadata={
            "help": "Pipeline engine or inference engine"
        }
    )


local_rank = int(os.getenv("LOCAL_RANK", "0"))
world_size = int(os.getenv("WORLD_SIZE", "1"))

logger = Logger(__file__, logging.CRITICAL, 50000000, 5)
parser = HfArgumentParser(Arguments)
args = parser.parse_args_into_dataclasses()[0]

if local_rank == 0:
    logger.critical("=================================")
    logger.critical("%s", args)

home_dir = "/data"
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
    num_workers=4,
    drop_last=True
)
config = ViTConfig.from_pretrained(args.model_name_or_path)

# if local_rank == 0 and batch_size == 1:
#     model = ViTForImageClassification.from_pretrained(args.model_name_or_path)
#     config = ViTConfig.from_pretrained(args.model_name_or_path)
#     if hasattr(model, "parallelize"):
#         model.parallelize(get_device_map(get_num_layers(config), range(world_size)))
#     else:
#         model = model.to("cuda:0")
#     records = []
#     for step, batch in enumerate(tqdm(eval_generator(), desc="single model")):
#         if step > 500:
#             break
#         pixel_values = batch[0][0].to("cuda:0")
#         start_time = time.perf_counter()
#         model(pixel_values=pixel_values)
#         end_time = time.perf_counter()
#         records.append((start_time, end_time))
#     for start_time, end_time in records:
#         logger.critical(
#             "(-1) start_time %s, end_time %s, diff %s",
#             start_time,
#             end_time,
#             end_time - start_time,
#         )

#     del model


deepspeed.init_distributed()

local_rank = int(os.getenv("LOCAL_RANK", "0"))
world_size = int(os.getenv("WORLD_SIZE", "1"))

model = ViTDeepSpeedPipe(config, num_stages=world_size if not args.inference else int( world_size / 2) )
engine, _, _, _ = deepspeed.initialize(config=args.deepspeed_config, model=model)

# if args.inference:
#     model = ViTPytorchPipeRandom(config)
#     engine = deepspeed.init_inference(model, mp_size=world_size)
# else:
#     model = ViTDeepSpeedPipe(config, num_stages=world_size)
#     engine, _, _, _ = deepspeed.initialize(config=args.deepspeed_config, model=model)

def model_inference(batch):
    shape = batch["pixel_values"].shape
    input_args = (batch["pixel_values"],)
    inputs = (input_args, torch.zeros(shape[0]))
    
    # if args.inference:
    #     outputs = engine(input_args)
    # else:
    #     outputs = engine.eval_batch(iter([inputs] * 1), compute_loss=False)
    outputs = engine.eval_batch(iter([inputs] * 1), compute_loss=False)
    return outputs

execute_model(dataloader, model_inference, f"vit-bsz{args.batch_size}-{args.inference}")