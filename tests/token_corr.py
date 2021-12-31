from seaborn.distributions import histplot
import torch
import logging
import numpy as np
from transformers.data.data_collator import DataCollatorForSeq2Seq, default_data_collator
from transformers.utils.dummy_pt_objects import T5ForConditionalGeneration
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from scipy.optimize import minimize
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans

import os
import sys

from torch.nn.modules.activation import Threshold

from datasets import Dataset
from datasets import load_dataset, load_metric

from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, T5ForConditionalGeneration
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset, dataloader
from sklearn.model_selection import train_test_split

from hfutils.logger import Logger
from hfutils.arg_parser import HfArguments
from hfutils.loader import ModelLoader, DatasetLoader
from hfutils.temperature_scaling import ModelWithTemperature
from hfutils.monte_carlo import monte_carlo_bounds

from triton_inference.calibration import temperature_scale, temperature_scaling

logger = Logger(__file__, "debug", 5000000, 5)


args = HfArguments()
tokenizer, _ = ModelLoader(args).load(load_model=False)

home_dir = os.path.expanduser("~")
base_dir = os.path.join(
    home_dir,
    os.path.join(
        "model-finetune",
        "outputs"
    )
)

model_keys = [
    "S", 
    "M", 
    "L",
    "XL",
]

device_map = [
    "cuda:0",
    "cuda:1",
    "cuda:2",
    "cuda:3",
]

energy_discount_factor = [
    1, 
    3, 
    10,
    40,
]

task_name = args.data_args.task_name

model_paths = [
    f"{base_dir}/t5-small-lm-adapt/{task_name}/best",
    f"{base_dir}/t5-base-lm-adapt/{task_name}/best",
    f"{base_dir}/t5-large-lm-adapt/{task_name}/best",
    f"{base_dir}/t5-xl-lm-adapt/{task_name}/best",
]

model_energy = dict(zip(model_keys, energy_discount_factor))
model_paths = dict(zip(model_keys, model_paths))
model_device = dict(zip(model_keys, device_map))

models = dict()
for key in model_keys:
    logger.debug("key %s, path %s, device %s", key, model_paths[key], model_device[key])
    models[key] = T5ForConditionalGeneration.from_pretrained(model_paths[key]) # if key != "S" else DistilBertForSequenceClassification.from_pretrained(model_paths[key])
    models[key] = models[key].to(model_device[key])
    models[key].eval()

logger.info("model loaded")

m = torch.nn.Softmax(dim=1)

# -------------  Dataset Prepare --------------

dataset_loader = DatasetLoader(args)
train_dataloader = dataset_loader.load(tokenizer, partition="validation", create_dataloader=True)
logger.debug("train_dataloader %s", train_dataloader)

n_models = len(model_keys)
num_labels = 0

pos_token = tokenizer("false").input_ids[0]
neg_token = tokenizer("true").input_ids[0]

def model_inference(model, batch, temperature=None, device="cuda:0"):
    input_ids=batch['input_ids']
    attention_mask=batch['attention_mask']
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        do_sample=False, # disable sampling to test if batching affects output
        return_dict_in_generate=True,
        output_scores=True,
    )
    logits = outputs.scores[0][:, [neg_token, pos_token]]
    if temperature is not None:
        logits = temperature_scale(logits, temperature)
    return logits

text_samples= dict(zip(model_keys, [list() for _ in range(n_models)]))
model_correct = dict(zip(model_keys, [list() for _ in range(n_models)]))
text_samples = list()

with torch.no_grad():
    for step, batch in tqdm(enumerate(train_dataloader), desc="Text Samples Correlation"):
        text_samples.append(batch['input_ids'])

        label = batch["labels"][:, 0] == pos_token
        label = label.to(torch.int64)
        
        # label = label.cpu().detach().flatten()
        
        for key in model_keys:
            logits = model_inference(models[key], batch, device=model_device[key])
            model_ans = torch.argmax(logits, dim=-1).flatten().to(model_device[key])
            label = label.to(model_device[key])

            model_correct[key].append(model_ans == label)

n_clusters = 16

text_samples = torch.concat(text_samples).detach().cpu().numpy()
text_kmeans = KMeans(n_clusters=n_clusters).fit(text_samples)
text_labels = text_kmeans.labels_
model_clusters = {}
for key in model_keys:
    model_correct[key] = torch.concat(model_correct[key]).detach().cpu().numpy()
    
    indicators = []
    for label in range(n_clusters):
        cluster_idx = np.argwhere(text_labels == label)
        n_members = len(cluster_idx)
        correct = model_correct[key][cluster_idx]
        n_correct = np.sum(correct)
        logger.debug("%s label %s n_members %s n_correct %s", key, label, n_members, n_correct)
        indicators.append(np.round(n_correct / n_members, 3))

    model_clusters[key] = indicators

logger.info("model_clusters %s", model_clusters)

