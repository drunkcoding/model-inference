from dataclasses import dataclass, field
from functools import partial
import functools
import logging
import os
import time
import torch
from transformers import (
    BertTokenizer,
    BertTokenizerFast,
    DistilBertTokenizer,
    DistilBertTokenizerFast,
    HfArgumentParser,
    AutoModelForQuestionAnswering,
    default_data_collator,
)
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from datasets import load_dataset

from hfutils.logger import Logger
from hfutils.pipe.bert import BertPyTorchPipeForQuestionAnswering
from hfutils.pipe.distilbert import DistilBertPyTorchPipeForQuestionAnswering
from hfutils.qa import prepare_validation_features
from hfutils.measure import get_energy_by_group, get_gpu_uuid
from nn_pruning.inference_model_patcher import optimize_model

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

tokenizer = (
    BertTokenizerFast.from_pretrained(args.model_name_or_path, use_fast=True)
    if not "distil" in args.model_name_or_path
    else DistilBertTokenizerFast.from_pretrained(args.model_name_or_path, use_fast=True)
)

val_dataset = load_dataset(
    "squad_v2", split="validation"
).shuffle()  # .select([x for x in range(1000)])
column_names = val_dataset.column_names
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
dataloader = DataLoader(
    dataset.remove_columns(["example_id", "offset_mapping"]),
    shuffle=False,
    collate_fn=data_collator,
    batch_size=args.batch_size,
    drop_last=True,
    num_workers=20,
)

device_id = 4
device = f"cuda:{device_id}"
uuid = get_gpu_uuid(device_id)

model = AutoModelForQuestionAnswering.from_pretrained(args.model_name_or_path)
if "hybrid" in args.model_name_or_path:
    model = optimize_model(model, "dense")
    model = model.to(device)
else:
    model = (
        DistilBertPyTorchPipeForQuestionAnswering(model)
        if "distil" in args.model_name_or_path
        else BertPyTorchPipeForQuestionAnswering(model)
    )
    model.convert(device)


start_energy = get_energy_by_group()[uuid]
records_start = []
records_end = []
for step, batch in enumerate(tqdm(dataloader, desc=f"{args.batch_size}")):
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    start_time = time.perf_counter()
    if "distil" in args.model_name_or_path:
        model((input_ids, attention_mask))
    elif "hybrid" in args.model_name_or_path:
        token_type_ids = batch["token_type_ids"].to(device)
        model(input_ids, token_type_ids, attention_mask)
    else:
        token_type_ids = batch["token_type_ids"].to(device)
        model((input_ids, token_type_ids, attention_mask))
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

df = pd.DataFrame(
    {
        "model": [basename] * len(records_end),
        "batch_size": [args.batch_size] * len(records_end),
        "start_time": records_start,
        "end_time": records_end,
        "latency": np.array(records_end) - np.array(records_start),
    }
)
df.to_csv(
    os.path.join("profile", f"latency_{basename}_{args.batch_size}.csv"), index=False
)
logger.info("%s", df.describe())
