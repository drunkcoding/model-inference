from dataclasses import dataclass, field
import logging
import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from transformers import HfArgumentParser, ViTForImageClassification
from torchvision.datasets import ImageNet

from hfutils.preprocess import (
    split_train_test,
    vit_collate_fn,
    ViTFeatureExtractorTransforms,
)
from hfutils.pipe.vit import ViTPyTorchPipeForImageClassification
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


logger = Logger(__file__, logging.INFO, 5000000, 5)

parser = HfArgumentParser(Arguments)
args = parser.parse_args_into_dataclasses()[0]

home_dir = "/mnt/raid0nvme1"
dataset_path = os.path.join(home_dir, "ImageNet")

rnd_seed = 106033
np.random.seed(rnd_seed)
# torch.seed(rnd_seed)

dataset = ImageNet(
    dataset_path,
    split="val",
    transform=ViTFeatureExtractorTransforms(args.tokenizer_path, split="val"),
)
index = np.array([x for x in range(len(dataset))])
np.random.shuffle(index)
dataset = Subset(dataset, index)
dataloader = DataLoader(
    dataset, shuffle=False, collate_fn=vit_collate_fn, num_workers=20, batch_size=4
)

model = ViTForImageClassification.from_pretrained(args.model_path)
model = ViTPyTorchPipeForImageClassification(model)
model.convert(args.device)

logger.info("model loaded %s", args)


logits_list = []
labels_list = []
for step, batch in enumerate(tqdm(dataloader)):
    pixel_values = batch["pixel_values"].to(args.device)
    label = batch["labels"]
    with torch.no_grad():
        logits = model((pixel_values,))
        # logits = logits.squeeze(1)[:, T5_TASK_LABELS]

    labels_list.extend(label)
    logits_list.append(logits.detach().cpu())

labels = torch.as_tensor(labels_list, dtype=torch.int64)
logits = torch.cat(logits_list)

torch.save(labels, os.path.join(os.path.dirname(__file__), f"{args.model_name}_labels"))
torch.save(logits, os.path.join(os.path.dirname(__file__), f"{args.model_name}_logits"))

