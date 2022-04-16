from dataclasses import dataclass, field
from functools import partial
import logging
import os
import time
from transformers import (
    AutoModelWithLMHead,
    GPT2Tokenizer,
    HfArgumentParser,
    AutoTokenizer
)
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm
import numpy as np
import torch

from hfutils.logger import Logger
from hfutils.pipe.gpt import GPTLMHeadModelPipe
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

# logger.info("=================================")
# logger.info("%s", args)

tokenizer = GPT2Tokenizer.from_pretrained(args.model_name_or_path)

val_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
# val_dataset = val_dataset.select([x for x in range(500)])
print(val_dataset)
encodings = tokenizer("\n\n".join(val_dataset["text"]), return_tensors="pt")
encodings.input_ids = encodings.input_ids.to(torch.long)
print(encodings.input_ids.shape)

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

        yield input_ids, target_ids, trg_len, end_loc

device_id = 6
device = f"cuda:{device_id}"
uuid = get_gpu_uuid(device_id)

model = AutoModelWithLMHead.from_pretrained(args.model_name_or_path)
model = GPTLMHeadModelPipe(model)
if "gpt-j" in args.model_name_or_path:
    model.partition_by_parameter(0, 4)
if "gpt-xl" in args.model_name_or_path:
    model.partition_by_parameter(0, 2)
model.convert(device)

start_energy = get_energy_by_group()[uuid]
records_start = []
records_end = []
for step, batch in enumerate(tqdm(load_encodings(encodings), desc=f"{args.batch_size}")):
    input_ids = batch[0].to(device)
    start_time = time.perf_counter()
    outputs = model((input_ids, None))
    print(None, outputs[1].shape)
    exit()
    torch.cuda.empty_cache()
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