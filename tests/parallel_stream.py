import torch
import torch.multiprocessing as mp

model_keys = [
    "S",
    "M",
    "L",
    # "XL",
]

model_names = [
    "google/t5-small-lm-adapt",
    "google/t5-base-lm-adapt",
    "google/t5-large-lm-adapt",
    "google/t5-xl-lm-adapt",
]

streams = [
    torch.cuda.Stream(),
    torch.cuda.Stream(),
    torch.cuda.Stream(),
    torch.cuda.Stream(),
]

processes = []

p = mp.Process(target=model_inference, args=)