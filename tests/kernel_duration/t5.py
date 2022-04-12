from dataclasses import dataclass, field
from functools import partial
import logging
import os
import time
import torch
from transformers import (
    T5ForConditionalGeneration,
    HfArgumentParser,
    T5Tokenizer,
    DataCollatorForSeq2Seq,
)
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from hfutils.logger import Logger
from hfutils.pipe.t5 import T5PyTorchPipe
from hfutils.loader import load_glue_val, t5_preprocess_function
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

tokenizer = T5Tokenizer.from_pretrained(args.model_name_or_path)

preprocess_function = partial(
    t5_preprocess_function, tokenizer=tokenizer, padding="max_length", max_length=128,
)

data_collator = DataCollatorForSeq2Seq(tokenizer)
dataset = load_glue_val(preprocess_function).shuffle()

dataloader = DataLoader(
    dataset,
    shuffle=False,
    collate_fn=data_collator,
    batch_size=args.batch_size,
    drop_last=True,
)

device_id = 0
device = f"cuda:{device_id}"
uuid = get_gpu_uuid(device_id)


model = T5ForConditionalGeneration.from_pretrained(args.model_name_or_path)
model = T5PyTorchPipe(model)
model.convert(device)

start_energy = get_energy_by_group()[uuid]
records_start = []
records_end = []
for step, batch in enumerate(tqdm(dataloader, desc=f"{args.batch_size}")):
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    start_time = time.perf_counter()
    model((input_ids, attention_mask))
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
