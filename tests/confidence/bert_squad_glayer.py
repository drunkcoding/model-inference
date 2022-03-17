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
from transformers import AutoModelForQuestionAnswering, DistilBertForQuestionAnswering
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
from transformers import AutoTokenizer, DataCollatorWithPadding, EvalPrediction, HfArgumentParser
from torch.utils.data import (
    DataLoader,
    RandomSampler,
    SequentialSampler,
    TensorDataset,
    dataloader,
)
from sklearn.model_selection import train_test_split
from nn_pruning.inference_model_patcher import optimize_model

from hfutils.logger import Logger
from hfutils.arg_parser import HfArguments
from hfutils.loader import ModelLoader, DatasetLoader
from hfutils.monte_carlo import monte_carlo_bounds
from hfutils.calibration import g_scaling, g_scaling_helper, agg_logits

from utils_qa import postprocess_qa_predictions

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    output_dir: str = field(
        metadata={"help": "The output directory where the model predictions and checkpoints will be written."},
    )

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_seq_length: int = field(
        default=384,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch (which can "
            "be faster on GPU but will be slower on TPU)."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of prediction examples to this "
            "value if set."
        },
    )
    version_2_with_negative: bool = field(
        default=True, metadata={"help": "If true, some of the examples do not have an answer."}
    )
    null_score_diff_threshold: float = field(
        default=0.0,
        metadata={
            "help": "The threshold used to select the null answer: if the best answer has a score that is less than "
            "the score of the null answer minus this threshold, the null answer is selected for this example. "
            "Only useful when `version_2_with_negative=True`."
        },
    )
    doc_stride: int = field(
        default=128,
        metadata={"help": "When splitting up a long document into chunks, how much stride to take between chunks."},
    )
    n_best_size: int = field(
        default=20,
        metadata={"help": "The total number of n-best predictions to generate when looking for an answer."},
    )
    max_answer_length: int = field(
        default=30,
        metadata={
            "help": "The maximum length of an answer that can be generated. This is needed because the start "
            "and end predictions are not conditioned on one another."
        },
    )

parser = HfArgumentParser(DataTrainingArguments)
data_args = parser.parse_args_into_dataclasses()[0]
print(data_args)
task_name = data_args.dataset_name

# assert task_name == "squad_v2"
# assert data_args.version_2_with_negative == True

home_dir = os.path.expanduser(("~"))
base_dir = os.path.join(home_dir, os.path.join("model-finetune", "outputs"))

model_keys = [
    "S",
    "M",
    "L",
    "XL",
]

device_map = [
    "cuda:0",
    "cuda:0",
    "cuda:0",
    "cuda:0",
]

energy_discount_factor = [
    1 / 40,
    3 / 40,
    10 / 40,
    40 / 40,
]

model_paths = [
    # f"{base_dir}/distilbert-base-uncased/{task_name}/checkpoint-512",
    f"{home_dir}/HuggingFace/twmkn9/distilbert-base-uncased-squad2",
    f"{home_dir}/HuggingFace/twmkn9/bert-base-uncased-squad2",
    f"{home_dir}/HuggingFace/madlag/bert-large-uncased-wwm-squadv2-x2.63-f82.6-d16-hybrid-v1",
    # f"{base_dir}/bert-large-uncased/squad_v2_moq/checkpoint-8236",
    f"{home_dir}/HuggingFace/madlag/bert-large-uncased-squadv2",
    # f"{base_dir}/bert-large-uncased/{task_name}/checkpoint-8236",
]

# model_paths = [
#     # f"{base_dir}/distilbert-base-uncased/{task_name}/checkpoint-5534",
#     # f"{home_dir}/HuggingFace/madlag/bert-base-uncased-squadv1-x1.16-f88.1-d8-unstruct-v1",
#     # f"{base_dir}/bert-base-uncased/{task_name}/checkpoint-2767",
#     f"{home_dir}/HuggingFace/csarron/bert-base-uncased-squad-v1",
#     # f"{base_dir}/bert-large-uncased/{task_name}/checkpoint-640",
# ]

tokenizer = AutoTokenizer.from_pretrained(
    f"{home_dir}/HuggingFace/bert-base-uncased",
    use_fast=True,
)
# tokenizer = AutoTokenizer.from_pretrained(
#     model_paths[1],
#     use_fast=True,
# )
pad_on_right = tokenizer.padding_side == "right"
max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

model_energy = dict(zip(model_keys, energy_discount_factor))
model_paths = dict(zip(model_keys, model_paths))
model_device = dict(zip(model_keys, device_map))

logger = Logger(__file__, "info", 5000000, 5)

models = dict()
for key in model_paths:
    logger.debug("key %s, path %s, device %s", key, model_paths[key], model_device[key])
    models[key] = AutoModelForQuestionAnswering.from_pretrained(
        model_paths[key]
    )
    if key == "L":
        models[key] = optimize_model(models[key], "dense")
    models[key] = models[key].to(model_device[key])
    models[key].eval()
    torch.cuda.empty_cache()
    gc.collect()

logger.info("model loaded")

# -------------  Dataset Prepare --------------
metric = load_metric("squad_v2" if data_args.version_2_with_negative else "squad")

def compute_metrics(p: EvalPrediction):    
    return metric.compute(predictions=p.predictions, references=p.label_ids)

val_dataset = load_dataset(task_name)["validation"].shuffle()#.select([x for x in range(1000)])

# data_files = {}
# train_file = "tests/confidence/dev-v1.1.json"
# data_files["validation"] = train_file
# extension = train_file.split(".")[-1]

# val_dataset = load_dataset(extension, data_files=data_files, field="data")['validation']
# val_dataset = Dataset.from_json(pd.DataFrame(val_dataset['paragraphs']))

# print(val_dataset)

column_names = val_dataset.column_names

question_column_name = "question" if "question" in column_names else column_names[0]
context_column_name = "context" if "context" in column_names else column_names[1]
answer_column_name = "answers" if "answers" in column_names else column_names[2]

# Training preprocessing
def prepare_train_features(examples):
    # Some of the questions have lots of whitespace on the left, which is not useful and will make the
    # truncation of the context fail (the tokenized question will take a lots of space). So we remove that
    # left whitespace
    examples[question_column_name] = [q.lstrip() for q in examples[question_column_name]]
    
    # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
    # in one example possible giving several features when a context is long, each of those features having a
    # context that overlaps a bit the context of the previous feature.
    tokenized_examples = tokenizer(
        examples[question_column_name if pad_on_right else context_column_name],
        examples[context_column_name if pad_on_right else question_column_name],
        truncation="only_second" if pad_on_right else "only_first",
        max_length=max_seq_length,
        stride=data_args.doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length" if data_args.pad_to_max_length else False,
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
            if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
                # Note: we could go after the last offset if the answer is the last word (edge case).
                while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
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
    examples[question_column_name] = [q.lstrip() for q in examples[question_column_name]]

    # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
    # in one example possible giving several features when a context is long, each of those features having a
    # context that overlaps a bit the context of the previous feature.
    tokenized_examples = tokenizer(
        examples[question_column_name if pad_on_right else context_column_name],
        examples[context_column_name if pad_on_right else question_column_name],
        truncation="only_second" if pad_on_right else "only_first",
        max_length=max_seq_length,
        stride=data_args.doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length" if data_args.pad_to_max_length else False,
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
    predictions = tuple([
        p.detach().cpu().numpy() if isinstance(p, torch.Tensor) else p
        for p in predictions
    ])
    
    predictions = postprocess_qa_predictions(
        examples=examples,
        features=features,
        predictions=predictions,
        version_2_with_negative=data_args.version_2_with_negative,
        n_best_size=data_args.n_best_size,
        max_answer_length=data_args.max_answer_length,
        null_score_diff_threshold=data_args.null_score_diff_threshold,
        output_dir=data_args.output_dir,
        prefix=stage,
    )
    # Format the result to the format the metric expects.
    if data_args.version_2_with_negative:
        formatted_predictions = [
            {"id": k, "prediction_text": v, "no_answer_probability": 0.0} for k, v in predictions.items()
        ]
    else:
        formatted_predictions = [{"id": k, "prediction_text": v} for k, v in predictions.items()]

    references = [{"id": ex["id"], "answers": ex[answer_column_name]} for ex in examples]
    return EvalPrediction(predictions=formatted_predictions, label_ids=references)

data_collator = (
    default_data_collator
    if data_args.pad_to_max_length
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
    num_proc=data_args.preprocessing_num_workers,
    remove_columns=column_names,
    desc="Running tokenizer on training dataset",
)
test = test.map(
    prepare_validation_features,
    batched=True,
    batch_size=-1,
    num_proc=data_args.preprocessing_num_workers,
    remove_columns=column_names,
    desc="Running tokenizer on validation dataset",
)

print(test.column_names)

train_dataloader = DataLoader(
    train,
    shuffle=False,
    collate_fn=data_collator,
    batch_size=16,
)
# train_raw_dataloader = DataLoader(
#     Dataset.from_dict(val_dataset[:train_len]),
#     shuffle=False,
#     collate_fn=data_collator,
#     batch_size=16,
# )
test_dataloader = DataLoader(
    test.remove_columns(['example_id', 'offset_mapping']),
    shuffle=False,
    collate_fn=data_collator,
    batch_size=16,
)
# test_raw_dataloader = DataLoader(
#     val_dataset,
#     shuffle=False,
#     collate_fn=data_collator,
#     batch_size=16,
# )

m = torch.nn.Softmax(dim=1)
logger.info("data loaded")

# ============= MODEL INFERENCE FRUNCTION =================
@torch.no_grad()
def model_inference(model, batch, temperature=None, device="cuda:0"):
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    token_type_ids = batch["token_type_ids"].to(device)

    if isinstance(model, DistilBertForQuestionAnswering):
        outputs = model(input_ids=input_ids,attention_mask=attention_mask, return_dict=True)
    else:
        outputs = model(input_ids=input_ids,token_type_ids=token_type_ids,attention_mask=attention_mask, return_dict=True)
    start_logits = outputs.start_logits
    end_logits = outputs.end_logits
    
    # print(start_logits.shape, end_logits.shape)
    logits = torch.cat([start_logits, end_logits])
    if temperature is not None:
        start_logits = temperature(start_logits)
        end_logits = temperature(end_logits)
        logits = torch.cat([start_logits, end_logits])
    return logits

# ============= COLLECT TRAIN LOGITS =================

epoches = [
    2500,
    2500,
    2500,
    2500
]
model_epoches = dict(zip(model_keys, epoches))

model_outputs = {}
for key in model_keys:
    all_logits = []
    labels_list = []
    for batch in tqdm(train_dataloader, desc=f"Individual Accuracy {key}"):
        logits = model_inference(models[key], batch, device=model_device[key])
        start_logits, end_logits = logits.chunk(2)
        all_logits.append(start_logits)
        all_logits.append(end_logits)

        if len(labels_list) < len(train):
            start_positions = batch['start_positions'].to(model_device[key]).flatten()
            end_positions = batch['end_positions'].to(model_device[key]).flatten()
            labels_list.append(start_positions)
            labels_list.append(end_positions)

    all_logits = torch.cat(all_logits)
    model_outputs[key] = all_logits

    labels = torch.cat(labels_list).flatten()
    g_scaling(all_logits, labels, 5000, 384)
    # print(all_start_logits[0].shape, predictions[0].shape)
    # eval_pred = post_processing_function(val_dataset, test, predictions)
    # logger.info("indv %s %s", key, compute_metrics(eval_pred))


print(len(train))
print(labels.shape)
print(model_outputs[model_keys[0]].shape)

# =============  TRAIN TEMPERATURE =============

model_temperature = g_scaling_helper(model_outputs, labels, model_epoches, 384)
print("temperature", model_temperature)

for key in model_keys:
    model_outputs[key] = model_temperature[key](model_outputs[key])
    torch.save(model_temperature[key].state_dict(), os.path.join("tests", "confidence", f"bert_squad_glayer-{key}"))
# =============  TRAIN HYPERPARAMETER =============

num_train_labels = len(train)
num_models = len(model_keys)

def total_reward(threshold):
    reward = 0
    energy = 0
    mask = np.array([False] * num_train_labels)

    alpha = threshold[-1]
    threshold = threshold[:-1]
     
    hist_start_logits = None
    hist_end_logits = None
    for i, key in enumerate(model_keys):
        hist_start_logits = agg_logits(
            hist_start_logits if key != model_keys[-1] else None,
            model_outputs[key][::2],
            alpha
        )
        hist_end_logits = agg_logits(
            hist_start_logits if key != model_keys[-1] else None,
            model_outputs[key][1::2],
            alpha
        )

        start_probs, _ = torch.max(m(hist_start_logits), dim=1)
        end_probs, _ = torch.max(m(hist_end_logits), dim=1)
        
        min_probs = torch.where(start_probs > end_probs, end_probs, start_probs).detach().cpu().numpy()
        processed = (
            (min_probs >= threshold[i])
            if key in model_keys[:-1]
            else np.array([True] * num_train_labels)
        )
        processed_probs = min_probs[(~mask) & processed]
        reward += np.around(np.sum(processed_probs) / 8.0) * 8
    
        energy += model_energy[key] * np.count_nonzero(~mask) 
        mask |= processed
    return (reward, -energy)

threshold_bounds = monte_carlo_bounds(
    # functools.partial(total_reward, model_keys=model_keys),
    total_reward,
    [(0.25, 1.0)] * (num_models),
    [("reward", float), ("energy", float)],
    n=1000,
    tops=40,
    maxiter=30,
)
mc_threshold = np.mean(threshold_bounds, axis=1)
alpha = mc_threshold[-1]
mc_threshold = mc_threshold[:-1]
# logger.info("Threshold Bounds %s", threshold_bounds)
logger.info("Final Thresholds %s", mc_threshold)
logger.info("Alpha %s", alpha)

# ============= MODEL INFERENCE WITH HYPERPARAMETER =================
model_outputs = {}
# model_probs = {}

num_test_labels = len(test)

model_metrics = {}
for key in model_keys:
    model_metrics[key] = load_metric(task_name)

# mask = np.array([False] * num_train_labels)
for key in model_keys:
    all_start_logits = []
    all_end_logits = []
    for batch in tqdm(test_dataloader, desc=f"Individual Accuracy {key}"):
        logits = model_inference(models[key], batch, temperature=model_temperature[key], device=model_device[key])
        start_logits, end_logits = logits.chunk(2)
        all_start_logits.append(start_logits)
        all_end_logits.append(end_logits)
    
    all_end_logits = torch.cat(all_end_logits)
    all_start_logits = torch.cat(all_start_logits)

    predictions = (
        all_start_logits,
        all_end_logits,
    )

    assert num_test_labels == all_end_logits.shape[0]
    assert num_test_labels == all_start_logits.shape[0]

    # probs = (
    #     m(all_start_logits).detach().cpu().numpy(),
    #     m(all_end_logits).detach().cpu().numpy(),
    # )

    # model_probs[key] = probs
    model_outputs[key] = predictions

    # print(all_start_logits[0].shape, predictions[0].shape)
    
    eval_pred = post_processing_function(test_raw, test, predictions)
    logger.info("indv %s %s", key, compute_metrics(eval_pred))

mask = np.array([False] * num_test_labels)
final_start_logits = torch.zeros((num_test_labels, 384)).to("cuda")
final_end_logits = torch.zeros((num_test_labels, 384)).to("cuda")
hist_start_logits = None
hist_end_logits = None
for i, key in enumerate(model_keys):
    hist_start_logits = agg_logits(
        hist_start_logits if key != model_keys[-1] else None,
        model_outputs[key][0],
        alpha
    )
    hist_end_logits = agg_logits(
        hist_start_logits if key != model_keys[-1] else None,
        model_outputs[key][1],
        alpha
    )

    assert final_start_logits.shape == hist_start_logits.shape
    assert final_end_logits.shape == hist_end_logits.shape

    start_probs, _ = torch.max(m(hist_start_logits), dim=1)
    end_probs, _ = torch.max(m(hist_end_logits), dim=1)
    min_probs = torch.where(start_probs > end_probs, end_probs, start_probs).detach().cpu().numpy()
    processed = (
        (min_probs >= mc_threshold[i])
        if key in model_keys[:-1]
        else np.array([True] * num_test_labels)
    )

    print(mask.shape, processed.shape, hist_start_logits.shape)

    delegated_start_logit = hist_start_logits[(~mask) & processed]
    delegated_end_logit = hist_end_logits[(~mask) & processed]

    true_indices = np.argwhere((~mask) & processed).flatten()
    logger.info(
        "%s process count (%s) %s",
        key, num_test_labels,
        np.count_nonzero((~mask) & processed),
    )

    final_start_logits[(~mask) & processed] = delegated_start_logit
    final_end_logits[(~mask) & processed] = delegated_end_logit

    mask |= processed

logger.info("***** Collaborative Eval results *****")
eval_pred = post_processing_function(
    test_raw, test, (final_start_logits, final_end_logits))
logger.info(
    "Collaborative metrics %s",
    compute_metrics(eval_pred)
)