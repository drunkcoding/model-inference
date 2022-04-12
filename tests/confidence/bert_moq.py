import copy
import functools
import gc
from posixpath import split
from typing import Optional
from pyparsing import alphas
from seaborn.distributions import histplot
import torch
import logging
import numpy as np
from transformers.data.data_collator import (
    DataCollatorForSeq2Seq,
    default_data_collator,
)
from transformers import (
    AutoModelForQuestionAnswering,
    BertForQuestionAnswering,
    BertTokenizer,
    DistilBertForQuestionAnswering,
)
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

import os
import sys

from torch.nn.modules.activation import Threshold

from datasets import Dataset, concatenate_datasets
from datasets import load_dataset, load_metric
from dataclasses import dataclass, field

from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
)
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from nn_pruning.inference_model_patcher import optimize_model

from hfutils.logger import Logger
from hfutils.arg_parser import HfArguments
from hfutils.loader import ModelLoader, DatasetLoader
from hfutils.monte_carlo import monte_carlo_bounds
from hfutils.calibration import (
    temperature_scale,
    temperature_scaling_helper,
    agg_logits,
)

from utils_qa import postprocess_qa_predictions


import deepspeed

tokenizer = BertTokenizer.from_pretrained(
    "/home/xly/model-finetune/outputs/bert-large-uncased/squad_v2_moq/checkpoint-8236"
)
model = BertForQuestionAnswering.from_pretrained(
    "/home/xly/model-finetune/outputs/bert-large-uncased/squad_v2_moq/checkpoint-8236"
)
model.eval()

local_rank = int(os.getenv('LOCAL_RANK', '0'))
world_size = int(os.getenv('WORLD_SIZE', '1'))

with torch.no_grad():
    model = deepspeed.init_inference(model, mp_size=world_size, dtype=torch.int8, quantization_setting=8)

task_name = "squad_v2"
max_seq_length = 384
version_2_with_negative = True

pad_on_right = tokenizer.padding_side == "right"
max_seq_length = min(max_seq_length, tokenizer.model_max_length)
doc_stride = 128
pad_to_max_length = True
n_best_size = 20
max_answer_length = 30
output_dir = "test/."
preprocessing_num_workers = None
null_score_diff_threshold = 0

logger = Logger(__file__, "info", 5000000, 5)

logger.info("model loaded")

# -------------  Dataset Prepare --------------
metric = load_metric("squad_v2")


def compute_metrics(p: EvalPrediction):
    return metric.compute(predictions=p.predictions, references=p.label_ids)


val_dataset = load_dataset(task_name)[
    "validation"
].shuffle()  # .select([x for x in range(1000)])

column_names = val_dataset.column_names

question_column_name = "question" if "question" in column_names else column_names[0]
context_column_name = "context" if "context" in column_names else column_names[1]
answer_column_name = "answers" if "answers" in column_names else column_names[2]

# Training preprocessing
def prepare_train_features(examples):
    # Some of the questions have lots of whitespace on the left, which is not useful and will make the
    # truncation of the context fail (the tokenized question will take a lots of space). So we remove that
    # left whitespace
    examples[question_column_name] = [
        q.lstrip() for q in examples[question_column_name]
    ]

    # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
    # in one example possible giving several features when a context is long, each of those features having a
    # context that overlaps a bit the context of the previous feature.
    tokenized_examples = tokenizer(
        examples[question_column_name if pad_on_right else context_column_name],
        examples[context_column_name if pad_on_right else question_column_name],
        truncation="only_second" if pad_on_right else "only_first",
        max_length=max_seq_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length" if pad_to_max_length else False,
    )
    # print(len(tokenized_examples['input_ids']))
    # Since one example might give us several features if it has a long context, we need a map from a feature to
    # its corresponding example. This key gives us just that.
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    # The offset mappings will give us a map from token to character position in the original context. This will
    # help us compute the start_positions and end_positions.
    offset_mapping = tokenized_examples.pop("offset_mapping")

    # Let's label those examples!
    tokenized_examples["start_positions"] = []
    tokenized_examples["end_positions"] = []

    for i, offsets in enumerate(offset_mapping):
        # We will label impossible answers with the index of the CLS token.
        input_ids = tokenized_examples["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)

        # Grab the sequence corresponding to that example (to know what is the context and what is the question).
        sequence_ids = tokenized_examples.sequence_ids(i)

        # One example can give several spans, this is the index of the example containing this span of text.
        sample_index = sample_mapping[i]
        answers = examples[answer_column_name][sample_index]
        # If no answers are given, set the cls_index as answer.
        if len(answers["answer_start"]) == 0:
            tokenized_examples["start_positions"].append(cls_index)
            tokenized_examples["end_positions"].append(cls_index)
        else:
            # Start/end character index of the answer in the text.
            start_char = answers["answer_start"][0]
            end_char = start_char + len(answers["text"][0])

            # Start token index of the current span in the text.
            token_start_index = 0
            while sequence_ids[token_start_index] != (1 if pad_on_right else 0):
                token_start_index += 1

            # End token index of the current span in the text.
            token_end_index = len(input_ids) - 1
            while sequence_ids[token_end_index] != (1 if pad_on_right else 0):
                token_end_index -= 1

            # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
            if not (
                offsets[token_start_index][0] <= start_char
                and offsets[token_end_index][1] >= end_char
            ):
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
                # Note: we could go after the last offset if the answer is the last word (edge case).
                while (
                    token_start_index < len(offsets)
                    and offsets[token_start_index][0] <= start_char
                ):
                    token_start_index += 1
                tokenized_examples["start_positions"].append(token_start_index - 1)
                while offsets[token_end_index][1] >= end_char:
                    token_end_index -= 1
                tokenized_examples["end_positions"].append(token_end_index + 1)
    # for key in tokenized_examples:
    #     print(key, len(tokenized_examples[key]))
    return tokenized_examples


# Validation preprocessing
def prepare_validation_features(examples):
    # Some of the questions have lots of whitespace on the left, which is not useful and will make the
    # truncation of the context fail (the tokenized question will take a lots of space). So we remove that
    # left whitespace
    examples[question_column_name] = [
        q.lstrip() for q in examples[question_column_name]
    ]

    # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
    # in one example possible giving several features when a context is long, each of those features having a
    # context that overlaps a bit the context of the previous feature.
    tokenized_examples = tokenizer(
        examples[question_column_name if pad_on_right else context_column_name],
        examples[context_column_name if pad_on_right else question_column_name],
        truncation="only_second" if pad_on_right else "only_first",
        max_length=max_seq_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length" if pad_to_max_length else False,
    )

    # Since one example might give us several features if it has a long context, we need a map from a feature to
    # its corresponding example. This key gives us just that.
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

    # For evaluation, we will need to convert our predictions to substrings of the context, so we keep the
    # corresponding example_id and we will store the offset mappings.
    tokenized_examples["example_id"] = []

    for i in range(len(tokenized_examples["input_ids"])):
        # Grab the sequence corresponding to that example (to know what is the context and what is the question).
        sequence_ids = tokenized_examples.sequence_ids(i)
        context_index = 1 if pad_on_right else 0

        # One example can give several spans, this is the index of the example containing this span of text.
        sample_index = sample_mapping[i]
        tokenized_examples["example_id"].append(examples["id"][sample_index])

        # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
        # position is part of the context or not.
        tokenized_examples["offset_mapping"][i] = [
            (o if sequence_ids[k] == context_index else None)
            for k, o in enumerate(tokenized_examples["offset_mapping"][i])
        ]

    return tokenized_examples


# Post-processing:
def post_processing_function(examples, features, predictions, stage="eval"):
    # Post-processing: we match the start logits and end logits to answers in the original context.
    predictions = tuple(
        [
            p.detach().cpu().numpy() if isinstance(p, torch.Tensor) else p
            for p in predictions
        ]
    )

    predictions = postprocess_qa_predictions(
        examples=examples,
        features=features,
        predictions=predictions,
        version_2_with_negative=version_2_with_negative,
        n_best_size=n_best_size,
        max_answer_length=max_answer_length,
        null_score_diff_threshold=null_score_diff_threshold,
        output_dir=output_dir,
        prefix=stage,
    )
    # Format the result to the format the metric expects.
    if version_2_with_negative:
        formatted_predictions = [
            {"id": k, "prediction_text": v, "no_answer_probability": 0.0}
            for k, v in predictions.items()
        ]
    else:
        formatted_predictions = [
            {"id": k, "prediction_text": v} for k, v in predictions.items()
        ]

    references = [
        {"id": ex["id"], "answers": ex[answer_column_name]} for ex in examples
    ]
    return EvalPrediction(predictions=formatted_predictions, label_ids=references)


data_collator = (
    default_data_collator
    if pad_to_max_length
    else DataCollatorWithPadding(tokenizer, pad_to_multiple_of=None)
)

# train_len = int(len(val_dataset) * 0.4)

# train = Dataset.from_dict(val_dataset[:train_len])
# test = val_dataset

split_dataset = val_dataset.train_test_split(train_size=0.4)
train, test = split_dataset["train"], split_dataset["test"]
train_raw, test_raw = split_dataset["train"], split_dataset["test"]
train_len = len(train)

print(train)
print(test)

train = train.map(
    prepare_train_features,
    batched=True,
    batch_size=-1,
    num_proc=preprocessing_num_workers,
    remove_columns=column_names,
    desc="Running tokenizer on training dataset",
)
test = test.map(
    prepare_validation_features,
    batched=True,
    batch_size=-1,
    num_proc=preprocessing_num_workers,
    remove_columns=column_names,
    desc="Running tokenizer on validation dataset",
)

print(test.column_names)

train_dataloader = DataLoader(
    train, shuffle=False, collate_fn=data_collator, batch_size=16,
)
test_dataloader = DataLoader(
    test.remove_columns(["example_id", "offset_mapping"]),
    shuffle=False,
    collate_fn=data_collator,
    batch_size=16,
)
m = torch.nn.Softmax(dim=1)
logger.info("data loaded")

device = "cuda:0"

all_start_logits = []
all_end_logits = []
for batch in tqdm(test_dataloader, desc="test"):
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    token_type_ids = batch["token_type_ids"].to(device)

    outputs = model(
        input_ids=input_ids,
        token_type_ids=token_type_ids,
        attention_mask=attention_mask,
        return_dict=True,
    )
    start_logits = outputs.start_logits
    end_logits = outputs.end_logits
    # logits = torch.cat([start_logits, end_logits])
    all_start_logits.append(start_logits)
    all_end_logits.append(end_logits)
    
predictions = (
    all_start_logits,
    all_end_logits,
)

eval_pred = post_processing_function(test_raw, test, predictions)
logger.info("indv %s %s", compute_metrics(eval_pred))
