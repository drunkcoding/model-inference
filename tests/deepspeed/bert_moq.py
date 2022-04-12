import functools
import logging
import os
import time
from dataclasses import dataclass, field
from datasets import concatenate_datasets
import deepspeed
import numpy as np
from scipy import stats
import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from torch.utils.data.dataset import Dataset
from transformers import BertForQuestionAnswering, pipeline
from tqdm import tqdm

from hfutils.logger import Logger
from hfutils.qa import prepare_validation_features
from transformers.models.bert.configuration_bert import BertConfig
from transformers import HfArgumentParser, BertTokenizerFast, default_data_collator
from datasets import load_dataset

import warnings

warnings.filterwarnings("ignore")


@dataclass
class Arguments:
    model_name_or_path: str = field(
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        }
    )
    local_rank: int = field(
        default=-1, metadata={"help": "Place holder for deepspeed launcher."},
    )


logger = Logger(__file__, logging.INFO, 50000000, 5)

parser = HfArgumentParser(Arguments)
args = parser.parse_args_into_dataclasses()[0]

tokenizer = BertTokenizerFast.from_pretrained(args.model_name_or_path)
config = BertConfig.from_pretrained(args.model_name_or_path)

val_dataset = load_dataset("squad_v2")[
    "validation"
].shuffle()  # .select([x for x in range(1000)])
column_names = val_dataset.column_names
# model_name = args.model_args.model_name_or_path

local_rank = int(os.getenv("LOCAL_RANK", "0"))
world_size = int(os.getenv("WORLD_SIZE", "1"))

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

# with torch.no_grad():
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

model = BertForQuestionAnswering.from_pretrained(args.model_name_or_path)
model = deepspeed.init_inference(model, mp_size=world_size, dtype=torch.int8)

