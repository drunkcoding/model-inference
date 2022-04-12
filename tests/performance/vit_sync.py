import time
import torch
import numpy as np
from transformers.data.data_collator import (
    DataCollatorForSeq2Seq,
    default_data_collator,
)


import os
import requests

from torch.nn.modules.activation import Threshold

from datasets import Dataset, concatenate_datasets
from datasets import load_dataset, load_metric

from tqdm import tqdm
from torch.utils.data import DataLoader
from tritonclient.utils import *
import tritonclient.http as httpclient
from sklearn.model_selection import train_test_split

from hfutils.logger import Logger
from hfutils.measure import get_energy_by_group

# home_dir = os.path.expanduser(("~"))
home_dir = "/mnt/raid0nvme1"
dataset_path = os.path.join(home_dir, "ImageNet")
model_paths = f"{home_dir}/HuggingFace/WinKawaks/vit-tiny-patch16-224"

# -------------  Dataset Prepare --------------
from torchvision.datasets import ImageNet
import datasets
from hfutils.preprocess import (
    split_train_test,
    vit_collate_fn,
    ViTFeatureExtractorTransforms,
)

print("======ImageNet==========")

eval_dataset = ImageNet(
    dataset_path,
    split="val",
    transform=ViTFeatureExtractorTransforms(model_paths, split="val"),
)
num_labels = len(eval_dataset)


m = torch.nn.Softmax(dim=1)

remote = "localhost"

inputs_list = []
label_list = []

batch_size = 32

eval_dataloader = DataLoader(
    eval_dataset, shuffle=True, collate_fn=vit_collate_fn, batch_size=batch_size,
)

for step, batch in enumerate(tqdm(eval_dataloader)):
    pixel_values = batch["pixel_values"].numpy()
    # print(pixel_values.shape)
    logits = np.zeros((batch_size, 1000)).astype(np.float32)

    # batch_mask = np.ones((4, batch_size)).astype(bool)

    batch_mask = np.zeros((4, batch_size))
    batch_mask[0, :] = 1
    batch_mask = batch_mask.astype(bool)

    if step * batch_size > 100: break

    inputs = [
        httpclient.InferInput(
            "pixel_values", pixel_values.shape, np_to_triton_dtype(pixel_values.dtype)
        ),
        httpclient.InferInput(
            "batch_mask", batch_mask.shape, np_to_triton_dtype(batch_mask.dtype),
        ),
        httpclient.InferInput(
            "logits", logits.shape, np_to_triton_dtype(logits.dtype),
        ),
    ]
    inputs[0].set_data_from_numpy(pixel_values)
    inputs[1].set_data_from_numpy(batch_mask)
    inputs[2].set_data_from_numpy(logits)
    outputs = [
        httpclient.InferRequestedOutput("logits"),
        httpclient.InferRequestedOutput("batch_mask"),
    ]
    inputs_list.append(inputs)
    label_list.append(batch["labels"])

import multiprocessing as mp

NUM_PROC = 8
barrier = mp.Barrier(NUM_PROC)

def test_body(pid):
    print(pid)
    model_name = "vit_ensemble"
    metric = load_metric("accuracy")
    with httpclient.InferenceServerClient(f"{remote}:8000", concurrency=1) as client:
        for step, input in enumerate(tqdm(inputs_list)):
            response = client.infer(
                model_name, input, request_id=str(step), outputs=outputs,
            )

            result = response.get_response()
            logits = response.as_numpy("logits")
            # print(logits.shape)
            predictions = np.argmax(logits, axis=-1).flatten()
            # labels = token2label(batch["labels"][:, 0].flatten(), label_tokens)
            # print(label_list[step], predictions)
            metric.add_batch(predictions=predictions, references=label_list[step])
            # time.sleep(1)
    print(metric.compute())

start_energy = sum(list(get_energy_by_group().values()))
pool = mp.Pool(processes=NUM_PROC)
pool.map(test_body, [i for i in range(NUM_PROC)])
pool.close()
pool.join()
end_energy = sum(list(get_energy_by_group().values()))
print(end_energy - start_energy)