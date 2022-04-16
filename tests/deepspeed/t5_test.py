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

model = T5PytorchPipeRandom(config)
model.exec_map = (0,2)
model.convert_layer_specs("cuda")

def model_inference(batch):
    shape = batch["input_ids"].shape
    input_args = (batch["input_ids"], batch["attention_mask"])
    
    outputs = model(input_args)
    return outputs

execute_model(dataloader, model_inference, f"t5-bsz{args.batch_size}-test")