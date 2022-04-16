from dataclasses import dataclass, field
from hfutils.constants import TASK_TO_LABELS
import torch
import logging
from transformers.data.data_collator import (
    DataCollatorForSeq2Seq,
)
from transformers import HfArgumentParser, T5ForConditionalGeneration, T5Tokenizer

import os
from datasets import concatenate_datasets

from tqdm import tqdm
from transformers import AutoTokenizer, T5ForConditionalGeneration
from torch.utils.data import (
    DataLoader,
    dataloader,
)

from hfutils.loader import t5_preprocess_function, load_glue_val, load_glue_train
from functools import partial
from hfutils.logger import Logger
from hfutils.pipe.t5 import T5PyTorchPipe
from hfutils.constants import token2label
T5_TASK_LABELS = [1176, 6136, 59]

@dataclass
class Arguments:
    model_path: str = field(
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        }
    )
    tokenizer_path: str = field(metadata={"help": "Path to pretrained tokenizer"})
    model_name: str = field(metadata={"help": "Name of the model use as key"},)
    device: str = field(
        default="cuda:0", metadata={"help": "Model Device"},
    )


parser = HfArgumentParser(Arguments)
args = parser.parse_args_into_dataclasses()[0]

tokenizer = T5Tokenizer.from_pretrained(args.tokenizer_path)

logger = Logger(__file__, logging.INFO, 5000000, 5)

model = T5ForConditionalGeneration.from_pretrained(args.model_path)
model = T5PyTorchPipe(model)
model.convert(args.device)

logger.info("model loaded %s", args)

rnd_seed = 106033

# -------------  Dataset Prepare --------------

preprocess_function = partial(
    t5_preprocess_function, tokenizer=tokenizer, padding="max_length", max_length=128,
)
# eval_dataset = load_glue_val(preprocess_function).shuffle(seed=rnd_seed)
# train_dataset = (
#     load_glue_train(preprocess_function).shuffle(seed=rnd_seed).select(range(len(eval_dataset)))
# )
# dataset = concatenate_datasets([eval_dataset, train_dataset]).shuffle(seed=rnd_seed)
dataset = load_glue_val(preprocess_function).shuffle(seed=rnd_seed)
data_collator = DataCollatorForSeq2Seq(tokenizer)

logger.info("data loaded")

# -------------  Collect Logits --------------

dataloader = DataLoader(dataset, collate_fn=data_collator, batch_size=16)

logits_list = []
labels_list = []
for step, batch in enumerate(tqdm(dataloader)):
    input_ids = batch["input_ids"].to(args.device)
    attention_mask = batch["attention_mask"].to(args.device)
    label = token2label(batch["labels"][:, 0], T5_TASK_LABELS)

    with torch.no_grad():
        logits = model((input_ids, attention_mask))
        logits = logits.squeeze(1)[:, T5_TASK_LABELS]

    labels_list.extend(label)
    logits_list.append(logits.detach().cpu())

labels = torch.as_tensor(labels_list, dtype=torch.int64)
logits = torch.cat(logits_list)

torch.save(labels, os.path.join(os.path.dirname(__file__), f"{args.model_name}_labels"))
torch.save(logits, os.path.join(os.path.dirname(__file__), f"{args.model_name}_logits"))
