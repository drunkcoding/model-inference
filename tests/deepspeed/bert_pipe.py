from dataclasses import dataclass, field
import functools
import os
import time
import requests
import logging
import torch
import deepspeed
from deepspeed.utils import RepeatingLoader
from tqdm import tqdm
from argparse_dataclass import ArgumentParser
from torch.utils.data import DataLoader
import nn_pruning
import re

from transformers import (
    BertTokenizer,
    DistilBertConfig,
    DistilBertTokenizer,
    HfArgumentParser,
    default_data_collator,
)
from transformers.models.bert.configuration_bert import BertConfig
from datasets import load_dataset, load_metric, concatenate_datasets
from hfutils.pipe.bert import BertDeepSpeedPipeForQuestionAnswering
from hfutils.pipe.distilbert import DistilBertDeepSpeedPipeForQuestionAnswering
from hfutils.logger import Logger
from hfutils.pipe.base import get_num_layers
from hfutils.qa import prepare_validation_features


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

tokenizer = (
    BertTokenizer.from_pretrained(args.model_name_or_path)
    if not "distil" in args.model_name_or_path
    else DistilBertTokenizer
)
config = (
    BertConfig.from_pretrained(args.model_name_or_path)
    if not "distil" in args.model_name_or_path
    else DistilBertConfig.from_pretrained(args.model_name_or_path)
)

val_dataset = load_dataset("squad_v2")[
    "validation"
].shuffle()  # .select([x for x in range(1000)])
column_names = val_dataset.column_names


batch_size = 128

data_collator = default_data_collator

dataset = val_dataset.map(
    functools.partial(
        prepare_validation_features, tokenizer=tokenizer, column_names=column_names
    ),
    batched=True,
    num_proc=10,
    remove_columns=column_names,
    desc="Running tokenizer on validation dataset",
)


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


deepspeed.init_distributed()
# model = T5DeepSpeedPipe(config, num_stages=torch.cuda.device_count())
model = (
    BertDeepSpeedPipeForQuestionAnswering(config, num_stages=1)
    if not "distil" in args.model_name_or_path
    else DistilBertDeepSpeedPipeForQuestionAnswering.from_pretrained(
        args.model_name_or_path
    )
)


def eval_generator():
    dataloader = DataLoader(
        dataset,
        shuffle=False,
        collate_fn=data_collator,
        batch_size=batch_size,
        drop_last=True,
    )
    pid = os.getpid()

    for batch in tqdm(dataloader, desc=f"{pid}-eval_generator"):
        shape = batch["input_ids"].shape
        yield (
            (batch["input_ids"], batch["attention_mask"],),
            torch.zeros(shape[0]),
        )


engine, _, _, _ = deepspeed.initialize(config=args.deepspeed_config, model=model)

start_energy = sum(list(get_energy_by_group().values()))
inference_count = 0
for step, batch in enumerate(RepeatingLoader(eval_generator())):
    if step > 500:
        break
    start_time = time.perf_counter()
    outputs = engine.eval_batch(iter([batch] * 1), compute_loss=False)
    end_time = time.perf_counter()
    if outputs != None:
        inference_count += 1
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
