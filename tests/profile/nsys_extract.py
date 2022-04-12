from dataclasses import dataclass, field
import json
import re
import os
import numpy as np
from scipy.stats import describe
import logging
from transformers import HfArgumentParser

from hfutils.cuda import CudaOccupancyCalculator
from hfutils.logger import Logger

calculator = CudaOccupancyCalculator("8.0")

@dataclass
class Arguments:
    model_name_or_path: str = field(
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        }
    )
    batch_size: int = field(metadata={"help": "batch size for profiling kernel"})


logger = Logger(__file__, logging.INFO, 50000000, 5)
parser = HfArgumentParser(Arguments)
args = parser.parse_args_into_dataclasses()[0]
basename = os.path.basename(args.model_name_or_path)
with open(f"profile/{basename}-{args.batch_size}.json") as fp:
    text = fp.read()

logger.info("=================================")
logger.info("%s", args)

# cuda_groups = re.findall(r"CudaEvent.*startNs\":\"(\d+)\",\"endNs\":\"(\d+)\"", text)
cuda_groups = re.findall(r'CudaEvent.*startNs":"(\d+)","endNs":"(\d+)".*"(memcpy|sync|kernel)"', text)
trace_groups = re.findall(r"TraceProcessEvent.*startNs\":\"(\d+)\",\"endNs\":\"(\d+)\"", text)
kernel_groups = re.findall(r'CudaEvent.*startNs":"(\d+)","endNs":"(\d+)".*"kernel".*blockX":(\d+),"blockY":(\d+),"blockZ":(\d+).*staticSharedMemory":(\d+),"dynamicSharedMemory":(\d+).*registersPerThread":(\d+)', text)
# kernel_groups = re.findall(r'kernel.*blockX":(\d+),"blockY":(\d+),"blockZ":(\d+).*registersPerThread":(\d+),"sharedMemoryExecuted":(\d+)', text)

kernel_occupancy = []
total_time = 0
ts = []

for group in kernel_groups:
    start_ns, end_ns, blockX, blockY, blockZ, staticSharedMemory, dynamicSharedMemory, registersPerThread = group
    # blockX, blockY, blockZ, registersPerThread, sharedMemoryExecuted = group
    start_ns = int(start_ns)
    end_ns = int(end_ns)
    blockX = int(blockX)
    blockY = int(blockY)
    blockZ = int(blockZ)
    staticSharedMemory = int(staticSharedMemory)
    dynamicSharedMemory = int(dynamicSharedMemory)
    registersPerThread = int(registersPerThread)
    # sharedMemoryExecuted = int(sharedMemoryExecuted)

    ts.append([start_ns, end_ns])

    calculator.set_inputs(blockX*blockY*blockZ, registersPerThread, "8.0", dynamicSharedMemory+staticSharedMemory)
    occupancy = calculator.occupancyOfMultiprocessor()
    # occupancy = occupancy * (end_ns - start_ns)
    total_time += end_ns - start_ns
    kernel_occupancy.append(occupancy)
ts = np.array(ts)
kernel_occupancy = np.array(kernel_occupancy)
# kernel_occupancy = kernel_occupancy[-10000:] 
print(describe(kernel_occupancy))
# print("kernel_occupancy", np.sum(kernel_occupancy) / (ts[-1, -1] - ts[0, 0]))
# exit()

print(len(cuda_groups), cuda_groups[0])

def convert_groups(groups, type):
    ts = []
    op  = None
    for group in groups:
        if type == "cuda":
            start_ns, end_ns, op = group
        else:
            start_ns, end_ns = group
        start_ns = int(start_ns)
        end_ns = int(end_ns)
        if op == "memcpy" or op == "sync": continue
        ts.append([start_ns, end_ns])
    ts = np.array(ts)

    # idx = ts[:, 0] >= 5e10
    # ts = ts[idx]

    ts = ts[-10000:]

    return ts

def print_ts(ts):
    logger.info("%s bsz %s: total %s, propotion %s",
        basename,
        args.batch_size,
        (ts[-1, -1] - ts[0, 0]) / 1e6,
        np.sum(ts[:, 1] - ts[:, 0]) / (ts[-1, -1] - ts[0, 0]),
    )


cuda_ts = convert_groups(cuda_groups, "cuda")
# trace_ts = convert_groups(trace_groups, "trace")
# print(len(cuda_ts), len(trace_ts))
print_ts(cuda_ts)
# print_ts(trace_ts)

# print(
#     np.sum(cuda_ts[:, 1] - cuda_ts[:, 0]) / ((cuda_ts[-1, -1] - cuda_ts[0, 0]) - np.sum(trace_ts[:, 1] - trace_ts[:, 0])),
# )
