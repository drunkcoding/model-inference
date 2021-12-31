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

logger = Logger(f"{task_name}_" + __file__, "debug", 5000000, 5)

models = dict()
for key in model_keys:
    logger.debug("key %s, path %s, device %s", key, model_paths[key], model_device[key])
    models[key] = T5ForConditionalGeneration.from_pretrained(model_paths[key]) # if key != "S" else DistilBertForSequenceClassification.from_pretrained(model_paths[key])
    models[key] = models[key].to(model_device[key])
    models[key].eval()

logger.info("model loaded")

# -------------  Dataset Prepare --------------

dataset_loader = DatasetLoader(args)
train_dataset = dataset_loader.load(tokenizer, partition="validation", create_dataloader=False)
eval_dataset = dataset_loader.load(tokenizer, partition="validation", create_dataloader=False)
logger.debug("eval_dataset %s", eval_dataset)

data_args = args.data_args

if data_args.pad_to_max_length:
    data_collator = default_data_collator
else:
    data_collator = DataCollatorForSeq2Seq(tokenizer)
# else:
#     data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)

train_len = int(len(eval_dataset) * 0.4)

train, test = Dataset.from_dict(eval_dataset[:train_len]), Dataset.from_dict(eval_dataset[train_len:])
print(train, test)

train_dataloader = DataLoader(
    train, 
    shuffle=True, 
    collate_fn=data_collator, 
    batch_size=8,
    # drop_last=True,
)

test_dataloader = DataLoader(
    test, 
    shuffle=True, 
    collate_fn=data_collator, 
    batch_size=8,
    # drop_last=True,
)

m = torch.nn.Softmax(dim=1)
logger.info("data loaded")

# -------------  Train Temperature --------------

# for key in model_keys:
#     models[key] = ModelWithTemperature(models[key], tokenizer, model_device[key])
#     models[key].set_logger(logger)
#     models[key].set_temperature(train_dataloader)

print("temperature loaded")

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

model_probs = dict(zip(model_keys, [list() for _ in range(n_models)]))
model_outputs = dict(zip(model_keys, [list() for _ in range(n_models)]))

labels_list = []

def agg_logits(hist, curr, pos, device):
    if hist is not None:
        hist = hist.to(device)
        curr_prob, _ = torch.max(torch.float_power(m(curr), 2), dim=-1)
        hist_prob, _ = torch.max(torch.float_power(m(hist), 2), dim=-1)

        diff = torch.abs(hist_prob-curr_prob)
        # print(diff)
        for i in range(len(diff)):
            if diff[i] > 0.2:
                if curr_prob[i] < hist_prob[i]:
                    curr[i] = hist[i]
            else:
                curr[i] = (hist[i] * pos + curr[i]) / (pos+1)
    return curr

# def agg_logits(hist, curr, pos, device):
#     if hist is not None:
#         hist = hist.to(device)
#         return (hist * pos + curr) / (pos+1)
#     return curr

with torch.no_grad():
    for batch in tqdm(train_dataloader, desc="Eval with Temperature"):
        # input_ids=batch['input_ids']
        # attention_mask=batch['attention_mask']
        label = batch["labels"][:, 0] == pos_token
        label = label.to(torch.int64)
        # label = label.to(model_device[key])
        
        num_labels += len(label)
        label = label.cpu().detach().flatten()

        labels_list.append(label)
        
        hist_logits = None
        for i, key in enumerate(model_keys):
            logits = model_inference(models[key], batch, device=model_device[key])
            hist_logits = agg_logits(hist_logits, logits, i, model_device[key])
            
            model_outputs[key].append(hist_logits)
            probabilities = torch.float_power(m(hist_logits).cpu().detach(), 2)
            model_ans = torch.argmax(probabilities, dim=-1).flatten()
            model_probs[key] += [[p[model_ans[i]], int(model_ans[i] == label[i])] for i, p in enumerate(probabilities)]

labels = torch.cat(labels_list)

model_temperature = {}

for key in model_keys:
    # model_probs[key] = np.array(model_probs[key])
    model_outputs[key] = torch.cat(model_outputs[key]).to(model_device[key])
    # labels = labels.to(model_device[key])

    temperature = temperature_scaling(model_outputs[key], labels).detach().cpu().numpy()[0]
    if temperature > 2:
        model_temperature[key] = torch.nn.Parameter(torch.ones(1, device=model_device[key]) * (1 + (temperature-1)/2))
    else:
        model_temperature[key] = torch.nn.Parameter(torch.ones(1, device=model_device[key]) * temperature)
    # model_temperature[key] = temperature_scaling(model_outputs[key], labels).to(model_device[key])
    
    probabilities = torch.float_power(m(temperature_scale(model_outputs[key], model_temperature[key])).to(model_device[key]), 2)
    # probabilities = m(temperature_scale(model_outputs[key], model_temperature[key])).to(model_device[key])
    model_ans = torch.argmax(probabilities, dim=-1).flatten()

    model_ans = model_ans.detach().cpu().numpy()
    probabilities = probabilities.detach().cpu().numpy()
    temp_labels = labels.detach().cpu().numpy()

    model_probs[key] = np.array([[p[model_ans[i]], int(model_ans[i] == temp_labels[i])] for i, p in enumerate(probabilities)])
    # model_temperature[key] = torch.nn.Parameter(torch.ones(1, device=model_device[key]) * 1.15)

    

logger.debug("model_temperature %s", model_temperature)

# mc_threshold = []
# for key in model_keys[:-1]:
#     gm = GaussianMixture(n_components=2).fit(model_probs[key][:, 0].reshape((-1, 1)))
#     idx = np.argsort(gm.means_.flatten())[-1]
#     mean = gm.means_.flatten()[idx]
#     var = gm.covariances_.flatten()[idx]
#     mc_threshold.append(
#        mean
#     )
#     logger.info("%s means_ %s covariances_ %s mean %s var %s", key, gm.means_, gm.covariances_, mean, var)
#     # mc_threshold.append(
#     #     np.mean(model_probs[key][:, 0]) # - np.std(model_probs[key][:, 0])
#     # )
# logger.info("Threshold %s", mc_threshold)


def total_reward(threshold):
    reward = 0
    energy = 0
    mask = np.array([False]*num_labels)
    for i, key in enumerate(model_keys):
        processed = (model_probs[key][:, 0] >= threshold[i]) if key in model_keys[:-1] else np.array([True]*num_labels)
        # reward += np.around(np.sum(model_probs[key][(~mask) & processed, 1]) / 10.0) * 10
        reward += np.sum(model_probs[key][(~mask) & processed, 1])
        energy += model_energy[key]* np.count_nonzero(~mask) # np.count_nonzero((~mask) & processed)
        mask |= processed
    return (reward, -energy)

threshold_bounds = monte_carlo_bounds(
        total_reward, 
        [(0.6, 1.0)] * (n_models-1), 
        [('reward', float), ('energy', float)],
        n=10000,
        tops=40,
        maxiter=30,
    )
mc_threshold = np.mean(
    threshold_bounds, axis=1
)
logger.info("Threshold Bounds %s", threshold_bounds)

# mc_threshold.sort(reverse=True)
# mc_threshold[1] = mc_threshold[0]

# exit()

# -------------  Evaluation WITH Temperature --------------

correct_cnt = dict(zip(model_keys, [0]*n_models))
correct_prob = dict(zip(model_keys, [0]*n_models))
coop_cnt = dict(zip(model_keys, [0]*n_models))
process_prob = dict(zip(model_keys, [0]*n_models))
process_cnt = dict(zip(model_keys, [0]*n_models))

num_labels = 0
# th_stats = []
# threshold = None

th_stats = dict(zip(model_keys, [list() for _ in range(n_models)]))  

model_metrics = {}
for key in model_keys:
    model_metrics[key] = load_metric(args.data_args.dataset_name, args.data_args.task_name)

total_metrics = load_metric(args.data_args.dataset_name, args.data_args.task_name)
with torch.no_grad():
    for batch in tqdm(test_dataloader, desc="Testing Accuracy"):
        label = (batch["labels"][:, 0] == pos_token).to(torch.int64)

        b_size = len(label.cpu())
        mask = np.array([False]*b_size)

        hist_logits = None
        for i, key in enumerate(model_keys):
            logits = model_inference(models[key], batch, model_temperature[key], device=model_device[key])
            hist_logits = agg_logits(hist_logits, logits, i, model_device[key])

            probabilities = np.power(m(hist_logits).cpu().detach().numpy(), 2)
            # probabilities = m(logits).cpu().detach().numpy()

            # if key in ['S']:
            #     th_stats += np.max(probabilities, axis=1).tolist()
            
            th_stats[key] += np.max(probabilities, axis=-1).tolist()

            model_ans = np.argmax(probabilities, axis=-1)
            true_ans = label.cpu().detach().numpy().flatten()

            # logger.debug("probabilities %s, true_ans %s", probabilities, true_ans)

            selected_prob = np.array([p[model_ans[i]] for i, p in enumerate(probabilities)])
            processed = (selected_prob >= mc_threshold[i]) if key in model_keys[:-1] else np.array([True]*b_size)

            total_metrics.add_batch(
                predictions=model_ans[(~mask) & processed],
                references=true_ans[(~mask) & processed],
            )

            model_metrics[key].add_batch(
                predictions=model_ans,
                references=true_ans,
            )
            
            correct_prob[key] += np.sum(selected_prob)
            process_prob[key] += np.sum(selected_prob[(~mask) & processed])

            correct_cnt[key] += np.count_nonzero(model_ans == true_ans)
            coop_cnt[key] += np.count_nonzero((model_ans == true_ans) & (~mask) & processed)
            process_cnt[key] += np.count_nonzero((~mask) & processed)
            mask |= processed
        
        num_labels += b_size

for key in model_keys:
    logger.info("%s Mean Probability = %s", key, np.mean(th_stats[key]))
    sns.distplot(th_stats[key], hist=True, kde=True, 
                bins=int(180/5), 
                # color = 'darkblue', 
                label=key,
                hist_kws={'edgecolor':'black'},
                kde_kws={'linewidth': 4})
plt.legend()
plt.savefig(f'figures/th_stats_{task_name}.png', bbox_inches="tight")


logger.info("  Num examples = %s", num_labels)
logger.info("  Threshold = %s", mc_threshold)
# for key in model_keys:
#     logger.info("final temperature %s", models[key].temperature)
logger.info("***** Eval results *****")
for key in model_keys:
    logger.info("%s correct count %s, percent %s, prob %s", key, correct_cnt[key], np.around(correct_cnt[key]/float(num_labels) * 100, 3), correct_prob[key])
    logger.info("%s metrics %s", key, model_metrics[key].compute())
logger.info("***** Collaborative Eval results *****")
logger.info("Collaborative metrics %s", total_metrics.compute())
for key in model_keys:
    logger.info("%s process count %s, correct count %s, percent %s, prob %s", key, process_cnt[key], coop_cnt[key], np.around(coop_cnt[key]/float(process_cnt[key]) * 100, 3) if process_cnt[key] != 0 else 0, process_prob[key])

# # -------------  Evaluation WITHOUT Temperature --------------

# model_labels = []
# true_labels = []

# for data_batch in tqdm(dev_dataloader, desc="Evaluating"):
#     input_ids = data_batch['input_ids'].to(device)
#     # token_type_ids = data_batch['token_type_ids'].to(device)
#     attention_mask = data_batch['attention_mask'].to(device)
#     labels = data_batch['labels'].to(device)

#     true_labels += data_batch['labels'].numpy().flatten().tolist()

#     with torch.no_grad():
#         features = {}
#         preds = model(input_ids=input_ids, attention_mask=attention_mask)
#         logits = preds.logits
#         # logger.debug("softmax output %s", m(logits).cpu())
#         model_ans = np.argmax(m(logits).cpu(),axis=1)
#         model_labels += model_ans.flatten().tolist()
        
# true_labels = np.array(true_labels)
# model_labels = np.array(model_labels)

# logger.info("**** Label Stats (Train) ****")
# logger.info("num_labels %s, pred ones %s", len(true_labels), np.count_nonzero(model_labels)) 

# logger.info("**** Model Accuracy Before Temerature ****")
# logger.info("corrcoef %s, num_labels %s, num_correct %s (%s)", 
#     np.corrcoef(true_labels, model_labels)[0,1],
#     len(true_labels),
#     np.count_nonzero(model_labels == true_labels),
#     np.count_nonzero(model_labels == true_labels) / len(true_labels)
# )


