import torch
import logging
import numpy as np
from transformers.data.data_collator import (
    DataCollatorForSeq2Seq,
    default_data_collator,
)
from tritonclient.utils import *
import tritonclient.http as httpclient

import requests
import os

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

from torch.nn.modules.activation import Threshold

from datasets import Dataset, concatenate_datasets
from datasets import load_dataset, load_metric

from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
)
from torch.utils.data import (
    DataLoader,
    RandomSampler,
    SequentialSampler,
    TensorDataset,
    dataloader,
)
from sklearn.model_selection import train_test_split
from nn_pruning.inference_model_patcher import optimize_model

from utils_qa import postprocess_qa_predictions
from hfutils.measure import get_energy_by_group

task_name = "squad_v2"

home_dir = "/mnt/raid0nvme1"

tokenizer = AutoTokenizer.from_pretrained(
    f"{home_dir}/HuggingFace/bert-base-uncased", use_fast=True,
)

pad_on_right = True
max_seq_length = 384
doc_stride = 128
version_2_with_negative = True
n_best_size = 20
null_score_diff_threshold = 0.0
max_answer_length = 30
output_dir = "."

# -------------  Dataset Prepare --------------
metric = load_metric("squad_v2")


def compute_metrics(p: EvalPrediction):
    return metric.compute(predictions=p.predictions, references=p.label_ids)


val_dataset = load_dataset(task_name)["validation"].select([x for x in range(5000)])


column_names = val_dataset.column_names

question_column_name = "question" if "question" in column_names else column_names[0]
context_column_name = "context" if "context" in column_names else column_names[1]
answer_column_name = "answers" if "answers" in column_names else column_names[2]

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
        padding="max_length",
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
    formatted_predictions = [
        {"id": k, "prediction_text": v, "no_answer_probability": 0.0}
        for k, v in predictions.items()
    ]

    references = [
        {"id": ex["id"], "answers": ex[answer_column_name]} for ex in examples
    ]
    return EvalPrediction(predictions=formatted_predictions, label_ids=references)


data_collator = default_data_collator

eval_dataset = val_dataset.map(
    prepare_validation_features,
    batched=True,
    remove_columns=column_names,
    desc="Running tokenizer on training dataset",
)

m = torch.nn.Softmax(dim=1)

batch_size = 128


eval_dataloader = DataLoader(
    eval_dataset.remove_columns(["example_id", "offset_mapping"]),
    shuffle=False,
    collate_fn=data_collator,
    batch_size=batch_size,
    drop_last=True
)


inputs_list = []
for step, batch in enumerate(tqdm(eval_dataloader)):
    input_ids = batch["input_ids"].numpy()
    attention_mask = batch["attention_mask"].numpy()
    token_type_ids = batch["token_type_ids"].numpy()
   
    logits = np.zeros((batch_size, 384, 2)).astype(np.float32)
    
    # batch_mask = np.ones((3, batch_size)).astype(bool)

    batch_mask = np.zeros((3, batch_size))
    batch_mask[0, :] = 1
    batch_mask = batch_mask.astype(bool)

    inputs = [
        httpclient.InferInput(
            "input_ids", input_ids.shape, np_to_triton_dtype(input_ids.dtype)
        ),
        httpclient.InferInput(
            "token_type_ids",
            token_type_ids.shape,
            np_to_triton_dtype(token_type_ids.dtype),
        ),
        httpclient.InferInput(
            "attention_mask",
            attention_mask.shape,
            np_to_triton_dtype(attention_mask.dtype),
        ),
        httpclient.InferInput(
            "batch_mask", batch_mask.shape, np_to_triton_dtype(batch_mask.dtype),
        ),
        httpclient.InferInput(
            "logits", logits.shape, np_to_triton_dtype(logits.dtype),
        ),
    ]
    inputs[0].set_data_from_numpy(input_ids)
    inputs[1].set_data_from_numpy(token_type_ids)
    inputs[2].set_data_from_numpy(attention_mask)
    inputs[3].set_data_from_numpy(batch_mask)
    inputs[4].set_data_from_numpy(logits)
    outputs = [
        httpclient.InferRequestedOutput("logits"),
        httpclient.InferRequestedOutput("batch_mask"),
    ]
    inputs_list.append(inputs)

import multiprocessing as mp

NUM_PROC = 16
barrier = mp.Barrier(NUM_PROC)

remote = "localhost"

def test_body(pid):
    print(pid)
    random_num = np.random.randint(0, 2**16)
    model_name = "bert_ensemble"
    all_start_logits = []
    all_end_logits = []
    async_requests = []
    with httpclient.InferenceServerClient(f"{remote}:8000", concurrency=10) as client:
        for step, input in enumerate(tqdm(inputs_list)):
            request_id = str(int((random_num << 16) + step))
            response = client.async_infer(
                model_name, input, request_id=request_id, outputs=outputs,
            )
            async_requests.append(response)
            # print(request_id, step, random_num << 16, (random_num << 16) + step, random_num)
        for idx, async_request in tqdm(enumerate(async_requests), desc=f"{pid} bsz{batch_size}-async"):
            response = async_request.get_result()
            logits = response.as_numpy("logits")
            logits = torch.as_tensor(logits)
            start_logits, end_logits = logits.split(1, dim=-1)
            start_logits = start_logits.squeeze(-1)
            end_logits = end_logits.squeeze(-1)
            # print(start_logits.shape, end_logits.shape)
            all_start_logits.append(start_logits)
            all_end_logits.append(end_logits)

    all_end_logits = torch.cat(all_end_logits)
    all_start_logits = torch.cat(all_start_logits)
    print(all_end_logits.shape, all_end_logits.shape)
    predictions = (
        all_start_logits,
        all_end_logits,
    )
    # eval_pred = 
    # post_processing_function(val_dataset, eval_dataset, predictions)
    # print(compute_metrics(eval_pred))

start_energy = sum(list(get_energy_by_group().values()))
pool = mp.Pool(processes=NUM_PROC)
pool.map(test_body, [i for i in range(NUM_PROC)])
pool.close()
pool.join()
end_energy = sum(list(get_energy_by_group().values()))
print(end_energy - start_energy)

