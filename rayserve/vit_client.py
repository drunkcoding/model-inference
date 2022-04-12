import asyncio
import time
import torch
import numpy as np
import ray
import ray.util
from ray import serve
import os

from datasets import load_dataset, load_metric
import requests

from tqdm import tqdm
from torch.utils.data import DataLoader
from tritonclient.utils import *
import tritonclient.http as httpclient
from sklearn.model_selection import train_test_split

from hfutils.measure import get_energy_by_group

home_dir = "/mnt/raid0nvme1"
dataset_path = os.path.join(home_dir, "ImageNet")
model_paths = f"{home_dir}/HuggingFace/WinKawaks/vit-tiny-patch16-224"

# -------------  Dataset Prepare --------------
from torchvision.datasets import ImageNet
from hfutils.preprocess import (
    split_train_test,
    vit_collate_fn,
    ViTFeatureExtractorTransforms,
)
from hfutils.measure import get_energy_by_group

print("======ImageNet==========")

eval_dataset = ImageNet(
    dataset_path,
    split="val",
    transform=ViTFeatureExtractorTransforms(model_paths, split="val"),
)
num_labels = len(eval_dataset)
m = torch.nn.Softmax(dim=1)

inputs_list = []
label_list = []

batch_size = 1

eval_dataloader = DataLoader(
    eval_dataset,
    shuffle=True,
    num_workers=20,
    collate_fn=vit_collate_fn,
    batch_size=batch_size,
)

ray.init(address="ray://129.215.164.41:10001", namespace="vit")

obj_list = []
inputs_list = []
for step, batch in enumerate(tqdm(eval_dataloader)):
    if step * batch_size > 1000:
        break
    pixel_values = batch["pixel_values"].numpy()
    inputs_list.append((pixel_values,))
    obj_list.append(ray.put((pixel_values,)).hex())
    # logits = np.zeros((batch_size, 1000)).astype(np.float32)
# obj_list = ray.put(inputs_list)
# obj_list = [obj.hex() for obj in obj_list ]


import multiprocessing as mp

NUM_PROC = 8


def test_body(pid):
    for step, obj in enumerate(tqdm(obj_list, desc=f"{pid}")):
        resp = requests.post(
            "http://127.0.0.1:8000/hybrid-scheduler",
            json={"args": obj},
        )
        print(resp.json())


# async def test_body(pid):
#     print(pid)
#     ray.init(address="ray://129.215.164.41:10001", namespace="vit")
#     handle = serve.get_deployment("hybrid-scheduler").get_handle(sync=False)
#     async_requests = []
#     for step, input in enumerate(tqdm(inputs_list, desc=f"{pid}")):
#         response = handle.ensemble_inference.remote(input)
#         async_requests.append(response)

#     start_time = time.perf_counter()
#     async_requests = await asyncio.gather(*async_requests)
#     async_requests = ray.get(async_requests)
#     end_time = time.perf_counter()
#     print("asyncio", end_time - start_time)

start_time = time.perf_counter()
start_energy = sum(list(get_energy_by_group().values()))

# asyncio.run(test_body(0))

pool = mp.Pool(processes=NUM_PROC)
pool.map(test_body, [i for i in range(NUM_PROC)])
pool.close()
pool.join()


end_energy = sum(list(get_energy_by_group().values()))
end_time = time.perf_counter()
print(end_energy - start_energy)
print(end_time - start_time)
