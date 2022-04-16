import requests
import numpy as np
from datasets import load_metric, load_dataset
import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, Subset, Dataset
from tqdm import tqdm
import os

from transformers import GPT2Tokenizer

from hfutils.logger import Logger
from hfutils.measure import get_energy_by_group
from tritonclient.utils import *
import tritonclient.http as httpclient

loss_fct = CrossEntropyLoss()

val_dataset = load_dataset("lambada")['validation']
# val_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
val_dataset = val_dataset.select([x for x in range(500)])
print(val_dataset)

# home_dir = os.path.expanduser("~")
home_dir = os.path.expanduser("~")
home_dir = "/mnt/raid0nvme1"


tokenizer = GPT2Tokenizer.from_pretrained(
    f"{home_dir}/HuggingFace/gpt2",
    use_fast=True,
)

encodings = tokenizer("\n\n".join(val_dataset["text"]), return_tensors="pt")
encodings.input_ids = encodings.input_ids.to(torch.long)

max_length = 512
stride = 128

def load_encodings(encodings):
    for i in tqdm(range(0, encodings.input_ids.size(1), stride)):
        begin_loc = max(i + stride - max_length, 0)
        end_loc = min(i + stride, encodings.input_ids.size(1))
        trg_len = end_loc - i  # may be different from stride on last loop
        input_ids = encodings.input_ids[:, begin_loc:end_loc]
        # input_ids = input_ids.to(torch.int8)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        if input_ids.size(1) != max_length:
            continue

        # yield input_ids, target_ids, trg_len, end_loc
        yield input_ids.squeeze(0), encodings.input_ids[:, end_loc].squeeze(0)

class TextDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, iter):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data = list(iter)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = {
            "input_ids": self.data[idx][0].to(torch.long),
            "attention_mask": torch.ones_like(self.data[idx][0]).to(torch.long),
            "labels": self.data[idx][1],
        }
        return sample


batch_size = 2

dataset = TextDataset(load_encodings(encodings))
dataloader = DataLoader(
    dataset, shuffle=False, num_workers=20, batch_size=batch_size
)
    
m = torch.nn.Softmax(dim=1)
remote = "localhost"

inputs_list = []

all_trg_len = []
labels_list = []
for batch in tqdm(dataloader):
    input_ids = batch["input_ids"].numpy()
    attention_mask = batch["attention_mask"].numpy()
    label = batch["labels"]

    # batch_mask = np.ones((6,batch_size)).astype(bool)
    batch_mask = np.zeros((6,batch_size))
    batch_mask[0, :] = 1
    batch_mask = batch_mask.astype(bool)
    logits = np.zeros((batch_size, 50257)).astype(np.float32)

    labels_list.append(label)

    # print(input_ids.shape, attention_mask.shape)

    inputs = [
        httpclient.InferInput(
            "input_ids", input_ids.shape, np_to_triton_dtype(input_ids.dtype)
        ),
         httpclient.InferInput(
            "attention_mask", attention_mask.shape, np_to_triton_dtype(attention_mask.dtype)
        ),
        httpclient.InferInput(
            "batch_mask", batch_mask.shape, np_to_triton_dtype(batch_mask.dtype),
        ),
        httpclient.InferInput(
            "logits", logits.shape, np_to_triton_dtype(logits.dtype),
        ),
    ]
    inputs[0].set_data_from_numpy(input_ids)
    inputs[1].set_data_from_numpy(attention_mask)
    inputs[2].set_data_from_numpy(batch_mask)
    inputs[3].set_data_from_numpy(logits)
    outputs = [
        httpclient.InferRequestedOutput("logits"),
        httpclient.InferRequestedOutput("batch_mask"),
    ]
    inputs_list.append(inputs)

labels_list = torch.cat(labels_list) 

import multiprocessing as mp

NUM_PROC = 2
barrier = mp.Barrier(NUM_PROC)

def metric_accuracy(logits: torch.Tensor, labels: torch.Tensor):
    metric = load_metric("accuracy")
    predictions = torch.argmax(logits, axis=1).flatten()
    return metric.compute(predictions=predictions, references=labels.flatten())[
        "accuracy"
    ]

def test_body(pid):
    print(pid)
    model_name = "gpt_ensemble"
    logits_list = []
    with httpclient.InferenceServerClient(f"{remote}:8000", concurrency=1) as client:
        for step, input in enumerate(tqdm(inputs_list)):
            response = client.infer(
                model_name, input, request_id=str(step), outputs=outputs,
            )

            result = response.get_response()
            logits = response.as_numpy("logits")
            logits = torch.Tensor(logits)
            logits_list.append(logits)        
    logits_list = torch.cat(logits_list)
    print(metric_accuracy(logits_list, labels_list))

start_energy = np.array(list(get_energy_by_group().values()))
pool = mp.Pool(processes=NUM_PROC)
pool.map(test_body, [i for i in range(NUM_PROC)])
pool.close()
pool.join()
end_energy = np.array(list(get_energy_by_group().values()))
print(end_energy - start_energy)