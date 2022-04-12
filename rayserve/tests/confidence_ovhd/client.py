import ray
from ray import serve
import time
import numpy as np
from tqdm import tqdm

ray.init(address="ray://129.215.164.41:10001", namespace="confidence_ovhd")
handle = serve.get_deployment("hybrid-scheduler").get_handle()

batches = [1,2,4,6,8,16,32,48,64,128]
dimensions = [256, 384, 512, 1024, 2048, 4096, 10000, 20000, 50000]

inputs_list = []
for bsz in [1,2,4,6,8,16,32,48,64,128]:
    for dim in [256, 384, 512, 1024, 2048, 4096, 10000, 20000, 50000]:
        for _ in tqdm(range(100)):
            ensemble_outputs = np.random.random((bsz, dim))
            outputs = np.random.random((bsz, dim))
            batch_mask = np.ones((2, bsz)).astype(bool)
            inputs_list.append((ensemble_outputs, outputs, batch_mask, 0))

start_time = time.perf_counter()
async_requests = []
for step, input in enumerate(tqdm(inputs_list)):
    response = handle.post_processing.remote(*input)
    # ray.get(response)
    async_requests.append(response)
    if step % 500 == 0:
        ray.get(async_requests)
        async_requests = []
async_requests = ray.get(async_requests)
end_time = time.perf_counter()
print(end_time - start_time)