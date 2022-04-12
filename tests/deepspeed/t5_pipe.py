import requests
import os
from dataclasses import dataclass, field
from functools import partial
import re
import time

import torch
from datasets import concatenate_datasets, load_dataset, load_metric
from hfutils.loader import load_glue_val, t5_preprocess_function
from hfutils.pipe.t5 import T5DeepSpeedPipe
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

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TORCH_EXTENSIONS_DIR"] = "."


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

if local_rank == 0:
    logger.critical("=================================")
    logger.critical("%s", args)

tokenizer = T5Tokenizer.from_pretrained(args.model_name_or_path)
config = T5Config.from_pretrained(args.model_name_or_path)

preprocess_function = partial(
    t5_preprocess_function, tokenizer=tokenizer, padding="max_length", max_length=128,
)

data_collator = DataCollatorForSeq2Seq(tokenizer)
dataset = load_glue_val(preprocess_function).shuffle()


# print(eval_dataset[0])
batch_size = 128

def eval_generator():
    dataloader = DataLoader(
        dataset, shuffle=False, collate_fn=data_collator, batch_size=batch_size, drop_last=True,
    )
    pid = os.getpid()

    for batch in tqdm(dataloader, desc=f"{pid}-eval_generator"):
        shape = batch["input_ids"].shape
        yield (
            (batch["input_ids"], batch["attention_mask"],),
            torch.zeros(shape[0]),
        )


if local_rank == 0 and batch_size == 1:
    model = T5ForConditionalGeneration.from_pretrained(args.model_name_or_path)
    config = T5Config.from_pretrained(args.model_name_or_path)
    # if hasattr(model, "parallelize"):
    #     model.parallelize(
    #         get_device_map(get_num_layers(config), range(world_size))
    #     )
    # else:
    model = model.to("cuda:0")
    records = []
    for step, batch in enumerate(tqdm(eval_generator(), desc="single model")):
        if step > 500:
            break
        input_ids = batch[0][0].to("cuda:0")
        attention_mask = batch[0][1].to("cuda:0")
        decoder_input_ids = torch.zeros((input_ids.shape[0], 1), dtype=torch.long, device="cuda:0")
        decoder_attention_mask = torch.ones((input_ids.shape[0], 1), dtype=torch.long, device="cuda:0")
        start_time = time.perf_counter()
        model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
        )
        end_time = time.perf_counter()
        records.append((start_time, end_time))
    for start_time, end_time in records:
        logger.critical(
            "(-1) start_time %s, end_time %s, diff %s",
            start_time,
            end_time,
            end_time - start_time,
        )

    del model


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
model = T5DeepSpeedPipe(config, num_stages=8)

engine, _, _, _ = deepspeed.initialize(config=args.deepspeed_config, model=model)

start_energy = sum(list(get_energy_by_group().values()))
inference_count = 0
outputs_list = []
for step, batch in enumerate(RepeatingLoader(eval_generator())):
    if step > 100:
        break
    start_time = time.perf_counter()
    outputs = engine.eval_batch(iter([batch] * 1), compute_loss=False)
    end_time = time.perf_counter()
    if outputs != None:
        inference_count += 1
        outputs_list.append(outputs)
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
