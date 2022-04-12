import asyncio
import functools
import time
import ray
import ray.util
from ray import serve

import os
from tqdm import tqdm
from transformers import AutoTokenizer, EvalPrediction
from transformers.data.data_collator import (
    default_data_collator,
)
from torch.utils.data import DataLoader
from datasets import load_metric, load_dataset
from scipy.special import softmax

from utils_qa import postprocess_qa_predictions
from hfutils.measure import get_energy_by_group
from hfutils.qa import prepare_train_features, prepare_validation_features

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

task_name = "squad_v2"
batch_size = 1
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

metric = load_metric("squad_v2")


def compute_metrics(p: EvalPrediction):
    return metric.compute(predictions=p.predictions, references=p.label_ids)


val_dataset = load_dataset(task_name)["validation"].select([x for x in range(5000)])
data_collator = default_data_collator
column_names = val_dataset.column_names

eval_dataset = val_dataset.map(
    functools.partial(
        prepare_validation_features, tokenizer=tokenizer, column_names=column_names
    ),
    batched=True,
    remove_columns=column_names,
    desc="Running tokenizer on training dataset",
)

m = functools.partial(softmax, axis=1)

eval_dataloader = DataLoader(
    eval_dataset.remove_columns(["example_id", "offset_mapping"]),
    shuffle=False,
    collate_fn=data_collator,
    batch_size=batch_size,
    drop_last=True
)

inputs_list = []
for step, batch in enumerate(tqdm(eval_dataloader, desc="Prepare")):
    input_ids = batch["input_ids"].numpy()
    attention_mask = batch["attention_mask"].numpy()
    token_type_ids = batch["token_type_ids"].numpy()

    inputs_list.append((input_ids, token_type_ids, attention_mask))


ray.init(address="ray://129.215.164.41:10001", namespace="bert")

# asyncio.run(asyncio.gather(*async_requests))

# async def main():
#     handle = serve.get_deployment("hybrid-scheduler").get_handle(sync=False)

#     async_requests = []
#     for step, input in enumerate(tqdm(inputs_list)):
#         response = handle.ensemble_inference.remote(input)
#         async_requests.append(response)

#     for obj in tqdm(async_requests):
#         ray.get(await obj)
#     # responses = await asyncio.gather(*async_requests)

#     # for obj in responses:
#     #     print(ray.get(obj))

# asyncio.run(main())

handle = serve.get_deployment("hybrid-scheduler").get_handle()

start_time = time.perf_counter()
start_energy = sum(list(get_energy_by_group().values()))
async_requests = []
for step, input in enumerate(tqdm(inputs_list)):
    response = handle.ensemble_inference.remote(input)
    async_requests.append(response)

async_requests = ray.get(async_requests)
end_energy = sum(list(get_energy_by_group().values()))
end_time = time.perf_counter()
print(end_energy - start_energy)
print(end_time - start_time)
# for idx, async_request in tqdm(enumerate(async_requests), desc=f"bsz{batch_size}-async"):
#     response = async_request.get_result()