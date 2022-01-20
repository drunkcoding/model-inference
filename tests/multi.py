import torch
import torch.multiprocessing as mp
import subprocess
import time
import numpy as np
from tqdm import tqdm, trange

from triton_inference.measure import ModelMetricsWriter

def monitor():
    writer = ModelMetricsWriter("/jmain01/home/JAD003/sxr06/lxx22-sxr06/model-inference/tritonserver", "stress")
    while lock.value > 0:
        r = subprocess.run(['curl', 'http://dgj101:8002/metrics'], capture_output=True, text=True)
        writer.text = r.stdout
        writer.record_gpu_metric("nv_energy_consumption")
        writer.record_gpu_metric("nv_gpu_utilization")
        writer.record_gpu_metric("nv_gpu_memory_used_bytes")
        writer.record_gpu_metric("nv_gpu_power_usage")
        writer.record_gpu_metric("nv_energy_consumption")
        time.sleep(0.1)
    writer.writer.close()

lock = mp.Value('i', 0)
lock.value = 1

job = mp.Process(target=monitor)
job.start()

RUN_SEC = 30


for size in trange(1,21):
    size_m = 2 ** size

    start_time = time.process_time()
    while True:
        A = torch.rand(size_m, size_m, device="cuda:0")
        B = torch.rand(size_m, size_m, device="cuda:0")

        C = torch.multiply(A, B)

        curr_time = time.process_time()
        if curr_time - start_time > RUN_SEC:
            break
    time.sleep(RUN_SEC)

lock.value = 0
job.join()
