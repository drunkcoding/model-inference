from dataclasses import dataclass, field
import logging
import os
import sys
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
from hfutils.pipe.gpt import GPTDeepSpeedPipe, GPTPytorchPipeRandom
from hfutils.logger import Logger
from hfutils.pipe.base import get_num_layers

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

logger = Logger(__file__, logging.CRITICAL, 50000000, 5)
parser = HfArgumentParser(Arguments)
args = parser.parse_args_into_dataclasses()[0]

tokenizer = GPT2Tokenizer.from_pretrained(args.model_name_or_path)
config = AutoConfig.from_pretrained(args.model_name_or_path)

val_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
# val_dataset = val_dataset.select([x for x in range(500)])
print(val_dataset)
encodings = tokenizer("\n\n".join(val_dataset["text"]), return_tensors="pt")
encodings.input_ids = encodings.input_ids.to(torch.long)
# print(encodings.input_ids.shape)

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

        yield input_ids.repeat(args.batch_size, 1), target_ids, trg_len, end_loc

deepspeed.init_distributed()

# if args.inference:
#     model = GPTPytorchPipeRandom(config)
#     engine = deepspeed.init_inference(model, mp_size=world_size)
# else:
#     model = GPTDeepSpeedPipe(config, num_stages=world_size)
#     engine, _, _, _ = deepspeed.initialize(config=args.deepspeed_config, model=model)

# def model_inference(batch):
#     shape = batch[0].shape
#     input_args = (batch[0], torch.ones(shape),)
#     inputs = (input_args, torch.zeros(shape[0]))
    
#     if args.inference:
#         outputs = engine(input_args)
#     else:
#         outputs = engine.eval_batch(iter([inputs] * 1), compute_loss=False)

#     return outputs

model = GPTDeepSpeedPipe(config, num_stages=world_size if not args.inference else int( world_size / 2) )
engine, _, _, _ = deepspeed.initialize(config=args.deepspeed_config, model=model)

def model_inference(batch):
    shape = batch["pixel_values"].shape
    input_args = (batch["pixel_values"],)
    inputs = (input_args, torch.zeros(shape[0]))
    
    outputs = engine.eval_batch(iter([inputs] * 1), compute_loss=False)
    return outputs

execute_model(RepeatingLoader(load_encodings(encodings)), model_inference, f"gpt-bsz{args.batch_size}-{args.inference}")
