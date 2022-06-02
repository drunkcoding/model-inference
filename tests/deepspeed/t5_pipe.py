import sys
import requests
import os
from dataclasses import dataclass, field
from functools import partial
import re
import time

import torch
from datasets import concatenate_datasets, load_dataset, load_metric
from hfutils.loader import load_glue_val, t5_preprocess_function
from hfutils.pipe.t5 import T5DeepSpeedPipe, T5PyTorchPipe, T5PytorchPipeRandom
from hfutils.logger import Logger
from hfutils.pipe.base import get_num_layers
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from tqdm import tqdm
from transformers import (
    HfArgumentParser,
    T5ForConditionalGeneration,
    T5Tokenizer,
    DataCollatorForSeq2Seq,
    default_data_collator,
)
from transformers.models.t5.configuration_t5 import T5Config

import deepspeed
import logging
from deepspeed.utils import RepeatingLoader

sys.path.append(".")
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

logger = Logger(__file__, logging.INFO, 50000000, 5)
parser = HfArgumentParser(Arguments)
args = parser.parse_args_into_dataclasses()[0]

tokenizer = T5Tokenizer.from_pretrained(args.model_name_or_path)
config = T5Config.from_pretrained(args.model_name_or_path)

preprocess_function = partial(
    t5_preprocess_function, tokenizer=tokenizer, padding="max_length", max_length=128,
)

data_collator = DataCollatorForSeq2Seq(tokenizer)
dataset = load_glue_val(preprocess_function).shuffle()

dataloader = DataLoader(
    dataset, shuffle=False, collate_fn=data_collator, batch_size=args.batch_size, drop_last=True,
)

deepspeed.init_distributed()
# if args.inference:
#     model = T5PytorchPipeRandom(config)
#     engine = deepspeed.init_inference(model, mp_size=world_size)
# else:
#     model = T5DeepSpeedPipe(config, num_stages=world_size)
#     engine, _, _, _ = deepspeed.initialize(config=args.deepspeed_config, model=model)


# def model_inference(batch):
#     shape = batch["input_ids"].shape
#     input_args = (batch["input_ids"], batch["attention_mask"])
#     inputs = (input_args, torch.zeros(shape[0]))
    
#     if args.inference:
#         outputs = engine(input_args)
#     else:
#         outputs = engine.eval_batch(iter([inputs] * 1), compute_loss=False)

#     return outputs

model = T5DeepSpeedPipe(config, num_stages=world_size if not args.inference else int( world_size / 2) )
engine, _, _, _ = deepspeed.initialize(config=args.deepspeed_config, model=model)

def model_inference(batch):
    shape = batch["input_ids"].shape
    input_args = (batch["input_ids"],batch["attention_mask"])
    inputs = (input_args, torch.zeros(shape[0]))
    
    outputs = engine.eval_batch(iter([inputs] * 1), compute_loss=False)
    return outputs

execute_model(dataloader, model_inference, f"t5-bsz{args.batch_size}-{args.inference}")