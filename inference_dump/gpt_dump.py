from dataclasses import dataclass, field
import logging
import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, Dataset
from tqdm import tqdm
from datasets import concatenate_datasets, load_dataset, load_metric
from transformers import HfArgumentParser, AutoModelForCausalLM, GPT2Tokenizer

from hfutils.preprocess import (
    split_train_test,
    vit_collate_fn,
    ViTFeatureExtractorTransforms,
)
from hfutils.pipe.gpt import GPTLMHeadModelPipe
from hfutils.logger import Logger


@dataclass
class Arguments:
    model_path: str = field(
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        }
    )
    tokenizer_path: str = field(metadata={"help": "Path to pretrained tokenizer"})
    model_name: str = field(metadata={"help": "Name of the model use as key"},)
    device: str = field(
        default="cuda:0", metadata={"help": "Model Device"},
    )


max_length = 512
stride = 128

def load_encodings(encodings):
    

    for i in tqdm(range(0, encodings.input_ids.size(1) - 1, stride)):
        begin_loc = max(i + stride - max_length, 0)
        end_loc = min(i + stride, encodings.input_ids.size(1))
        trg_len = end_loc - i  # may be different from stride on last loop
        input_ids = encodings.input_ids[:, begin_loc:end_loc]

        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        # print(trg_len)

        if input_ids.size(1) != max_length:
            continue
        # print(target_ids[:, -1], encodings.input_ids[:, end_loc])
        # assert torch.all(target_ids[:, -1] == encodings.input_ids[:, end_loc])

        yield input_ids, encodings.input_ids[:, end_loc]
        # assert torch.all(target_ids[:, -1] != -100)
        # yield input_ids, target_ids[:, -1]

        # yield input_ids, target_ids

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
            "input_ids": self.data[idx][0],
            "attention_mask": torch.ones_like(self.data[idx][0]).to(torch.long),
            "labels": self.data[idx][1],
        }
        return sample


logger = Logger(__file__, logging.INFO, 5000000, 5)

parser = HfArgumentParser(Arguments)
args = parser.parse_args_into_dataclasses()[0]

# eval_dataset = load_dataset("lambada", split="validation")
# train_dataset = load_dataset("lambada", split="train").shuffle(106033)
# dataset = concatenate_datasets([train_dataset.select(range(10)), eval_dataset])
dataset = load_dataset("lambada", split="validation")
# dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
tokenizer = GPT2Tokenizer.from_pretrained(args.tokenizer_path, use_fast=True,)

print(dataset)

encodings = tokenizer("\n\n".join(dataset["text"]), return_tensors="pt")
encodings.input_ids = encodings.input_ids.to(torch.long)

dataset = TextDataset(load_encodings(encodings))

rnd_seed = 106033
np.random.seed(rnd_seed)

index = np.array([x for x in range(len(dataset))])
np.random.shuffle(index)
dataset = Subset(dataset, index)
dataloader = DataLoader(
    dataset, shuffle=False, num_workers=20, batch_size=16
)

model = AutoModelForCausalLM.from_pretrained(args.model_path)
if "gpt-j" in args.model_name or "xl" in args.model_name:
    model.parallelize()
else:
    model = model.to(args.device)
# model = GPTLMHeadModelPipe(model)
# model.convert(args.device)

logger.info("model loaded %s", args)


logits_list = []
labels_list = []
for step, batch in enumerate(tqdm(dataloader)):
    input_ids = batch["input_ids"].to(args.device)
    attention_mask = batch["attention_mask"].to(args.device)
    label = batch["labels"]
    with torch.no_grad():
        # logits = model((input_ids,attention_mask))
        logits = model(input_ids=input_ids, return_dict=True).logits
        logits = logits.squeeze(1)[:, -1, :50257]
        # print(logits.shape)
        # logits = logits.squeeze(1)

    labels_list.append(label)
    logits_list.append(logits.detach().cpu())

labels = torch.cat(labels_list)
logits = torch.cat(logits_list)

metric = load_metric("accuracy")
NUM_TOP = 10
def metric_accuracy(logits: torch.Tensor, labels: torch.Tensor):

    # print(logits.shape)
    # print(labels.shape)

    # logits = logits[:, stride:-1, :]
    # labels = labels[:, :, (stride+1):].squeeze(1)
    # # idx = labels != -100
    # # outputs = outputs[idx, :]
    # # labels = labels[idx]

    # print(logits.shape)
    # print(labels.shape)

    # idx = labels != -100
    # logits = logits[idx, :]
    # labels = labels[idx]

    # _, top_idx = torch.topk(logits, NUM_TOP)
    # top_idx = top_idx.reshape((-1,NUM_TOP))
    # labels = labels.flatten().repeat((NUM_TOP, 1)).transpose(0,1)

    # print(labels.shape)
    # print(top_idx.shape)

    # return torch.sum(labels == top_idx) / labels.shape[0]
    
    predictions = torch.argmax(logits, axis=1).flatten()
    # print(predictions, predictions.shape)
    # print(labels, labels.shape)
    return metric.compute(predictions=predictions, references=labels.flatten())[
        "accuracy"
    ]

print(metric_accuracy(logits, labels))

torch.save(labels, os.path.join(os.path.dirname(__file__), f"{args.model_name}_labels"))
torch.save(logits, os.path.join(os.path.dirname(__file__), f"{args.model_name}_logits"))

