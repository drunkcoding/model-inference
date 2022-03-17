import copy
import gc
import numpy as np
from datasets import load_metric, load_dataset
import torch
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
import os

from transformers import AutoModelWithLMHead, GPT2Tokenizer

from hfutils.logger import Logger
from hfutils.monte_carlo import monte_carlo_bounds
from hfutils.calibration import temperature_scale, temperature_scaling_helper, agg_logits

loss_fct = CrossEntropyLoss()

# val_dataset = load_dataset("lambada")['validation']
val_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
val_dataset = val_dataset.select([x for x in range(500)])
print(val_dataset)

home_dir = os.path.expanduser("~")
base_dir = os.path.join(home_dir, os.path.join("model-finetune", "outputs"))

model_keys = [
    "XS",
    "S",
    "M",
    "L",
    "XL",
    # "XXL",
]

device_map = [
    "cuda:0",
    "cuda:0",
    "cuda:1",
    "cuda:0",
    "cuda:1",
]

energy_discount_factor = [
    1 / 40,
    3 / 40,
    10 / 40,
    40 / 40,
    80 / 40,
]

model_paths = [
    f"{home_dir}/HuggingFace/distilgpt2",
    f"{home_dir}/HuggingFace/gpt2",
    f"{home_dir}/HuggingFace/gpt2-medium",
    f"{home_dir}/HuggingFace/gpt2-large",
    f"{home_dir}/HuggingFace/gpt2-xl",
]

tokenizer = GPT2Tokenizer.from_pretrained(
    f"{home_dir}/HuggingFace/gpt2",
    use_fast=True,
)

model_energy = dict(zip(model_keys, energy_discount_factor))
model_paths = dict(zip(model_keys, model_paths))
model_device = dict(zip(model_keys, device_map))

logger = Logger(__file__, "info", 5000000, 5)


# device = "cuda"
# # model_id = "EleutherAI/gpt-j-6B"
# # model_id = "hivemind/gpt-j-6B-8bit"
# model_id = "gpt2"
# model_path = f"{home_dir}/HuggingFace/{model_id}"
# print(model_path)
# model = AutoModelWithLMHead.from_pretrained(model_path).to(device)
# model.to(torch.int8)
encodings = tokenizer("\n\n".join(val_dataset["text"]), return_tensors="pt")
encodings.input_ids = encodings.input_ids.to(torch.long)
print(encodings.input_ids.shape)

total_len = encodings.input_ids.size(1)
train_split = int(total_len * 0.4)
train = copy.deepcopy(encodings)
train.input_ids = train.input_ids[:, :train_split]
test = copy.deepcopy(encodings)
test.input_ids = test.input_ids[:, train_split:]

def load_encodings(encodings):
    max_length = 1024
    stride = 128

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

        yield input_ids, target_ids, trg_len, end_loc

@torch.no_grad()
def model_inference(model, input_ids, temperature=None, device="cuda:0"):
    input_ids = input_ids.to(device)
    logits = model(input_ids, return_dict=True).logits
    # logits = logits.squeeze()[-2].unsqueeze(0)
    if temperature is not None:
        logits = temperature_scale(logits, temperature)
    return logits

def compute_metrics(outputs, labels):
    cnt = 0
    # _, top_idx = torch.topk(logits, 10)
    for i, logits in enumerate(outputs):
        topk, top_idx = torch.topk(logits, 10)
        if torch.any(top_idx == labels[i]):
            cnt += 1

    return {
        "top10EM": cnt / len(labels)
    }

def compute_preplexity(outputs, labels, trg_lens):
    nlls = []
    for i in tqdm(range(len(trg_lens))):

        shift_logits = outputs[i][..., :-1, :].contiguous()
        shift_labels = labels[i][..., 1:].contiguous()
        # Flatten the tokens
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        nlls.append(loss * trg_lens[i][0])

    ppl = torch.exp(torch.stack(nlls).sum() / trg_lens[-1][1])
    return ppl
    
# def preplexity(encodings, model, device):

#     max_length = model.config.n_positions
#     stride = 128

#     nlls = []
#     for i in tqdm(range(0, encodings.input_ids.size(1), stride)):
#         begin_loc = max(i + stride - max_length, 0)
#         end_loc = min(i + stride, encodings.input_ids.size(1))
#         trg_len = end_loc - i  # may be different from stride on last loop
#         input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
#         # input_ids = input_ids.to(torch.int8)
#         target_ids = input_ids.clone()
#         target_ids[:, :-trg_len] = -100

#         with torch.no_grad():
#             outputs = model(input_ids, labels=target_ids)
#             neg_log_likelihood = outputs[0] * trg_len

#         nlls.append(neg_log_likelihood)

#     ppl = torch.exp(torch.stack(nlls).sum() / end_loc)

#     return ppl

# ppl = preplexity(encodings, model, device)
# print(ppl)

# if 1 == 1:
#     exit()

# ============= LOAD MODELS =================

models = dict()
for key in model_keys:
    logger.debug("key %s, path %s, device %s", key, model_paths[key], model_device[key])
    models[key] = AutoModelWithLMHead.from_pretrained(
        model_paths[key]
    )
    models[key] = models[key].to(model_device[key])
    models[key].eval()
    torch.cuda.empty_cache()
    gc.collect()

logger.info("model loaded")

# ============= COLLECT TRAIN LOGITS =================

model_outputs = {}
model_trg_len = {}
for key in model_keys:
    all_logits = []
    all_trg_len = []
    labels_list = []
    for batch in tqdm(load_encodings(train), desc=f"Train Data {key}"):
        input_ids, target_ids, trg_len, end_loc = batch
        # input_ids = input_ids.to(model_device[key])
        logits = model_inference(models[key], input_ids, device=model_device[key])
        # print(logits.shape, input_ids.shape, target_ids.shape)
        # logits = logits.detach().cpu()
        # topk, top_idx = torch.topk(logits, 5)
        # # print(topk, top_idx)
        # top_keep = logits[:, :, top_idx]
        # logits = torch.zeros_like(logits, device=model_device[key]) - 1e10
        # logits[:, :, top_idx] = top_keep

        # shift_logits = logits[..., :-1, :]
        # shift_labels = target_ids[..., 1:]
        
        all_logits.append(logits.detach().cpu())
        all_trg_len.append((trg_len, end_loc))
        # all_trg_len.append((1, end_loc))
        labels_list.append(target_ids)
        # labels_list.append(target_ids.squeeze()[-1])

    all_logits = torch.cat(all_logits)#.to(model_device[key])
    labels = torch.cat(labels_list).to(torch.long)
    model_outputs[key] = all_logits
    model_trg_len[key] = all_trg_len
train_len = len(labels)

print(labels.shape)
print(model_outputs[model_keys[0]].shape)

# =============  TRAIN TEMPERATURE =============
# model_outputs = {
#     k: v[..., :-1, :] for k,v in model_outputs.items()
# }
# labels = labels[..., 1:]

# model_temperature = temperature_scaling_helper(model_outputs, labels, model_device)
# print("temperature", model_temperature)

# for key in model_keys:
#     model_outputs[key] = temperature_scale(model_outputs[key], model_temperature[key])
model_temperature = {
    k: None for k in model_keys
}
# =============  TRAIN HYPERPARAMETER =============

num_models = len(model_keys)
m = torch.nn.Softmax(dim=-1)

def get_hist(model_outputs):
    hist_probs = []
    hist_logits = []
    logits = None
    for i, key in enumerate(model_keys):
        logits = agg_logits(
            logits if key != model_keys[-1] else None,
            model_outputs[key],
            0.9
        )
        hist_logits.append(logits)
        probs, _ = torch.max(m(logits), dim=-1)
        probs = probs.detach().cpu().numpy()
        hist_probs.append(probs)

    return hist_probs, hist_logits

hist_probs, _ = get_hist(model_outputs)

def total_reward(threshold):
    reward = 0
    energy = 0
    mask = np.zeros(model_outputs[model_keys[0]].shape[:-1]).astype(bool)

    alpha = threshold[-1]
    threshold = threshold[:-1]
     
    hist_logits = None
    for i, key in enumerate(model_keys):
        # hist_logits = agg_logits(
        #     hist_logits if key != model_keys[-1] else None,
        #     model_outputs[key],
        #     alpha
        # )

        # probs, _ = torch.max(m(hist_logits), dim=-1)
        # probs = probs.detach().cpu().numpy()
        probs = hist_probs[i]
        # print("probs", probs.shape)
        processed = (
            (probs >= threshold[i])
            if key in model_keys[:-1]
            else np.ones(probs.shape).astype(bool)
        )
        processed_probs = probs[(~mask) & processed]
        reward += np.around(np.sum(processed_probs) / 8.0) * 8
    
        energy += model_energy[key] * np.count_nonzero(~mask) 
        mask |= processed
    return (reward, -energy)


threshold_bounds = monte_carlo_bounds(
    total_reward,
    [(0.25, 1.0)] * (num_models),
    [("reward", float), ("energy", float)],
    n=1000,
    tops=40,
    maxiter=30,
)
mc_threshold = np.mean(threshold_bounds, axis=1)
alpha = mc_threshold[-1]
mc_threshold = mc_threshold[:-1]
logger.info("Threshold Bounds %s", threshold_bounds)
logger.info("Final Thresholds %s", mc_threshold)
logger.info("Alpha %s", alpha)


# ============= MODEL INFERENCE WITH HYPERPARAMETER =================
model_outputs = {}
model_trg_len = {}

for key in model_keys:
    all_logits = []
    all_trg_len = []
    labels_list = []
    for batch in tqdm(load_encodings(test), desc=f"Individual Accuracy {key}"):
        input_ids, target_ids, trg_len, end_loc = batch
        input_ids = input_ids.to(model_device[key])
        logits = model_inference(models[key], input_ids, temperature=model_temperature[key], device=model_device[key])
        # all_logits.append(logits.detach().cpu())
        # all_trg_len.append((trg_len, end_loc))
        all_logits.append(logits.detach().cpu())
        all_trg_len.append((trg_len, end_loc))
        labels_list.append(target_ids)

    all_logits = torch.cat(all_logits)
    # print(labels_list)
    labels = torch.cat(labels_list).to(torch.long)

    model_outputs[key] = all_logits
    model_trg_len[key] = all_trg_len

    logger.info("indv %s %s", key, compute_preplexity(all_logits, labels, all_trg_len))
    # logger.info("indv %s %s", key, compute_metrics(all_logits, labels))

test_len = len(labels)

hist_probs, hist_logits = get_hist(model_outputs)

mask = np.zeros(model_outputs[model_keys[0]].shape[:-1]).astype(bool)
final_logits = torch.zeros_like(model_outputs[model_keys[0]])
# hist_logits = None
for i, key in enumerate(model_keys):
    # hist_logits = agg_logits(
    #     hist_logits if key != model_keys[-1] else None,
    #     model_outputs[key],
    #     alpha
    # )
    # assert final_logits.shape == hist_logits.shape

    # probs, _ = torch.max(m(hist_logits), dim=-1)
    # probs = probs.detach().cpu().numpy()
    probs = hist_probs[i]
    processed = (
        (probs >= mc_threshold[i])
        if key in model_keys[:-1]
        else np.ones(probs.shape).astype(bool)
    )

    print(mask.shape, probs.shape, processed.shape, hist_logits[i].shape)
    print(((~mask) & processed).shape)
    delegated_logit = hist_logits[i][(~mask) & processed, :]

    logger.info(
        "%s process count (%s) %s",
        key, test_len,
        np.count_nonzero((~mask) & processed),
    )

    final_logits[(~mask) & processed, :] = delegated_logit.to(final_logits.device)
    mask |= processed

logger.info("***** Collaborative Eval results *****")
logger.info(
    "Collaborative metrics %s",
    compute_preplexity(final_logits, labels, model_trg_len[model_keys[0]])
)
# logger.info(
#     "Collaborative metrics %s",
#     compute_metrics(final_logits, labels)
# )