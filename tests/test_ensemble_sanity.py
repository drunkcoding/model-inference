import functools
import json
import os
from hfutils.arg_parser import HfArguments
from hfutils.constants import TASK_TO_LABELS
from hfutils.loader import DatasetLoader, ModelLoader
from numpy import random
from packaging.version import parse
import ray
import torch
from transformers.data.data_collator import (
    DataCollatorForSeq2Seq,
    default_data_collator,
)
import numpy as np
from tqdm import tnrange, tqdm, trange
from scipy.special import softmax
import time
import multiprocessing as mp
import requests
from ray import serve

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import logging
from hfutils.measure import ModelMetricsWriter, ModelMetricsWriterBackend

logger = logging.getLogger(__name__)

import datasets
from datasets import load_dataset, load_metric, concatenate_datasets
from torch.utils.data import DataLoader

import torch.multiprocessing as mp

args = HfArguments()
data_args = args.data_args

tokenizer, _ = ModelLoader(args).load(load_model=False)
dataset_loader = DatasetLoader(args)
eval_dataset = dataset_loader.load(
    tokenizer, partition="validation", create_dataloader=False
)
# eval_dataset = concatenate_datasets([eval_dataset] * 2)
logger.info("eval_dataset %s", eval_dataset)
RUN_SEC = 60

# model_name = "gpt_neo_2.7B_standalone"
# model_name = "gpt_neo_2stage"
# model_name = "distilgpt2_cola"
model_name = "t5-xl-lm-adapt_sst2"
model_tag = "ray-g1r1p0-async"
# model_name = "t5_cola_ensemble"

remote = "localhost"
tensorboard_base = "/home/oai/share/model-inference/tritonserver/"
tensorboard_logdir = os.path.join(tensorboard_base, model_tag)

if data_args.pad_to_max_length:
    data_collator = default_data_collator
else:
    data_collator = DataCollatorForSeq2Seq(tokenizer)


def dummy(result, error):
    pass


def prepare_query(batch):
    input_ids = batch["input_ids"].numpy().tolist()
    attention_mask = batch["attention_mask"].numpy().tolist()
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }


label_tokens = [
    tokenizer(label, max_length=2).input_ids[0]
    for label in TASK_TO_LABELS[data_args.task_name]
    if label is not None
]

# ray.init("localhost:8000", namespace="t5-sst2")

NUM_PROC = 4
barrier = mp.Barrier(NUM_PROC)
URL = "http://127.0.0.1:8000/composed"
s_URL = "http://127.0.0.1:8000/SR0P0"
m_URL = "http://127.0.0.1:8000/MR0P0"
l_URL = "http://127.0.0.1:8000/LR0P0"
# np.random.seed(42)

def query_server(url, data):
    resp = requests.post(url, json=data)
    try:
        outputs = resp.json()
    except:
        print(resp.content)
        exit()

    return outputs


def test_body(pid, inputs_list, label_list):
    print(pid)
    start_time = time.perf_counter()
    ready = False

    query_times = []
    cnt = 0
    while not ready:
        async_requests = []
        metric = load_metric("glue", data_args.task_name)

        for step, inputs in tqdm(enumerate(inputs_list), f"{pid} bsz{batch_size}-send"):
            # if step > 200:
            #     break
            json_data = {**inputs, **{"step": step, "pid": pid}}

            e_outputs = query_server(URL, json_data)
            s_outputs = query_server(s_URL, json_data)
            m_outputs = query_server(m_URL, json_data)
            l_outputs = query_server(l_URL, json_data)

            e_logits = np.array(e_outputs["logits"])
            s_logits = np.array(s_outputs["logits"])
            m_logits = np.array(m_outputs["logits"])
            l_logits = np.array(l_outputs["logits"])


            if not np.all(np.argmax(m_logits + s_logits + l_logits, axis=-1) == np.argmax(e_logits, axis=-1)):
                print(s_logits, m_logits, l_logits, s_logits + m_logits + l_logits,e_logits)
            # else:
            #     print(s_logits, m_logits, l_logits)
            
            # assert np.all(np.isclose(m_logits + s_logits, e_logits))
            # assert np.all(np.argmax(m_logits + s_logits, axis=-1) == np.argmax(e_logits, axis=-1))
            assert e_outputs["step"] == step
            assert e_outputs["pid"] == pid
            # query_times[cnt] = (time.perf_counter() - query_times[cnt]) * 1000
            # cnt += 1
            # response = async_request.get_result()
            # result = response.get_response()
            # logits = response.as_numpy("outputs")
            # predictions = logits.argmax(axis=-1)
            # print(predictions, label_list[idx])

            metric.add_batch(
                predictions=e_outputs["labels"],
                references=label_list[step],
            )

        eval_metric = metric.compute()
        print(f"Overall eval_metric: {eval_metric}")

        curr_time = time.time()
        # print(curr_time - start_time)
        if curr_time - start_time > RUN_SEC:
            ready = True
            break

    np.save(
        f"data/query_times_{model_name}_{model_tag}",
        np.array(query_times),
        allow_pickle=False,
    )
    barrier.wait()

    return pid


writer_backend = ModelMetricsWriterBackend(tensorboard_logdir, f"{model_name}")
writer_backend.remote = remote
# writer_backend.step = batch_size
writer_backend.start()

for batch_size in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
# for batch_size in [4, 8, 16, 32, 64, 128, 256]:
    # for batch_size in [32, 64, 128, 256, 512]:

    # metric = load_metric("glue", args.task_name)

    # c_dataset = concatenate_datasets([eval_dataset] * int(np.log2(batch_size) + 1))
    c_dataset = concatenate_datasets([eval_dataset] * 100)

    eval_dataloader = DataLoader(
        c_dataset,
        # shuffle=True,
        collate_fn=data_collator,
        batch_size=batch_size,
    )

    inputs_list = []
    label_list = []
    # outputs_list = []
    for step, batch in enumerate(eval_dataloader):
        if step > 500: break
        inputs_dict = prepare_query(batch)
        inputs_list.append(inputs_dict)
        label_list.append(
            (batch["labels"][:, 0] == label_tokens[-1]).to(torch.int64).numpy().tolist()
        )

    pool = mp.Pool(processes=NUM_PROC)

    pool.map(
        functools.partial(test_body, inputs_list=inputs_list, label_list=label_list),
        [i for i in range(NUM_PROC)],
    )
    pool.close()
    pool.join()

writer_backend.stop()

# with grpcclient.InferenceServerClient(f"{remote}:8001") as client:
