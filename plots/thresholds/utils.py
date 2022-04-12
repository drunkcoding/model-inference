from typing import Dict, List
import torch
import itertools
import numpy as np
import pandas as pd
import json
from tqdm import tqdm

from hfutils.calibration import agg_logits

def load_models(model_keys, model_paths, model_device, hf_cls, pipe_cls):
    models = dict()
    for key in model_keys:
        models[key] = hf_cls.from_pretrained(model_paths[key])
        models[key] = pipe_cls(models[key])
        models[key].convert(model_device[key])
        models[key].eval()

        torch.cuda.empty_cache()

    return models


def postprocessing_inference(
    model_keys: List[str],
    model_outputs: Dict[str, torch.Tensor],
    labels: List[torch.Tensor],
    func,
    device="cuda",
    alpha=1.0
):

    # print(labels[0].size())
    labels = torch.cat(labels).to(torch.long)
    labels = labels.to(device)
    num_labels = len(labels)

    model_ans = {}
    model_probs = {}
    hist_logits = None
    for key in model_keys:
        # print(model_outputs[key][0].size())
        model_outputs[key] = agg_logits(hist_logits, torch.cat(model_outputs[key]).to(device), alpha)

        # print(model_outputs[key].size(), labels.size())
        answer = torch.argmax(model_outputs[key], dim=-1).to(torch.long)
        probabilities = torch.float_power(func(model_outputs[key]), 2)
        probabilities, _ = torch.max(probabilities, dim=-1)
        model_probs[key] = probabilities
        model_ans[key] = answer == labels
        # print(model_ans[key], answer, labels)
        # print(model_ans[key].size())
        if len(model_ans[key].size()) > 1:
            model_ans[key] = torch.logical_and(
                model_ans[key][:, 0], model_ans[key][:, 1]
            )

        print(key, torch.count_nonzero(model_ans[key]) / num_labels)

    return model_probs, model_ans, model_outputs, labels


def profile_thresholds(
    model_keys,
    model_probs,
    model_ans,
    model_latency,
    model_names,
    all_thresholds,
    type: str,
    device="cuda"
):

    num_labels = len(model_ans[model_keys[0]])

    accuracy_data = []
    latency_data = []
    th_data = []
    prop_data = []
    for thresholds in tqdm(all_thresholds):
        mask = torch.Tensor([False] * num_labels).to(torch.bool).to(device)
        correct_cnt = 0
        weighted_latency = 0
        propotion = []
        for i, key in enumerate(model_keys):
            valid = (
                (model_probs[key] >= thresholds[i])
                if key in model_keys[:-1]
                else torch.Tensor([True] * num_labels).to(torch.bool).to(device)
            )
            # print(key, mask.size(), valid.size(), model_probs[key].size(), num_labels, thresholds)
            processed = (~mask) & valid

            correct_cnt += (
                torch.count_nonzero(model_ans[key][processed]).detach().cpu().item()
            )
            weighted_latency += (
                (torch.count_nonzero(~mask) * model_latency[model_names[key]]["1"])
                .detach()
                .cpu()
                .item()
            )
            propotion.append(torch.count_nonzero(~mask).item() / num_labels)
            mask |= valid

        accuracy = correct_cnt / num_labels
        latency = weighted_latency / num_labels
        accuracy_data.append(accuracy)
        latency_data.append(latency)
        th_data.append(json.dumps(thresholds))
        prop_data.append(json.dumps(propotion))

    df = pd.DataFrame(
        {
            "accuracy": accuracy_data,
            "latency": latency_data,
            "thresholds": th_data,
            "propotion": prop_data,
        }
    )
    df = df.sort_values("accuracy")
    df.to_csv(f"plots/thresholds/{type}.csv", index=False)

    best_key = model_keys[-1]
    best_data = {
        "latency": model_latency[model_names[best_key]]["1"],
        "accuracy": torch.count_nonzero(model_ans[best_key]).item() / num_labels,
    }
    with open(f"plots/thresholds/{type}.json", "w") as fp:
        json.dump(best_data, fp)

    print("biggest latency", model_latency[model_names[best_key]]["1"])
    print(
        "biggest accuracy", torch.count_nonzero(model_ans[best_key]) / num_labels,
    )


def get_all_thresholds(n_models):
    return list(
        itertools.product(
            np.linspace(0, 1, endpoint=True, num=100), repeat=n_models - 1
        )
    )
