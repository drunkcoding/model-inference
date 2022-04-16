from dataclasses import dataclass, field
import itertools
import json
import dill
import logging
import os
import pickle
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import HfArgumentParser
from datasets import load_metric

from hfutils.calibration import CalibrationLayer
from hfutils.logger import Logger
from scipy.stats import describe

import calibration as cal

basedir = os.path.dirname(__file__)
metric = load_metric("accuracy")

T5_TASK_LABELS = [1176, 6136, 59]  # HACK with GLUE labels

def metric_accuracy(logits: torch.Tensor, labels: torch.Tensor):
    predictions = torch.argmax(logits, axis=1).flatten()
    # print(predictions, predictions.shape)
    # print(labels, labels.shape)
    return metric.compute(predictions=predictions, references=labels.flatten())[
        "accuracy"
    ]


@dataclass
class Arguments:
    hybrid: str = field(
        metadata={"help": "Name of the models in hybrid, split by comma"},
    )
    type: str = field(
        metadata={"help": "Name of the models in hybrid, split by comma"},
    )
    num_labels: int = field(metadata={"help": "Number of labels"},)
    device: str = field(
        default="cuda:0", metadata={"help": "Model device list split by comma"},
    )


logger = Logger(__file__, logging.INFO, 5000000, 5)

m = torch.nn.Softmax(dim=1)

parser = HfArgumentParser(Arguments)
args = parser.parse_args_into_dataclasses()[0]

model_names = args.hybrid.split(",")
model_devices = (
    args.device.split(",") if "," in args.device else [args.device] * len(model_names)
)

model_logits = {}
model_probs = {}
model_ans = {}
model_devices = dict(zip(model_names, model_devices))

with open("tests/kernel_duration/energy.json", "r") as fp:
    model_energy = json.load(fp)

labels = (
    torch.load(os.path.join(os.path.dirname(__file__), f"{model_names[0]}_labels"))
    .detach()
    .numpy().flatten()
)

def process_func(logits):
    topk, _ = torch.topk(logits, 10)
    probs = m(topk)
    return probs

total_len = len(labels)
# total_len = 10000
test_len = int(total_len * 0.4)
train_labels = labels[:test_len]
test_labels = labels[test_len:total_len]

indv_data = {}
# calibrators = {}
for name in tqdm(model_names):
    # weight = torch.load(os.path.join(os.path.dirname(__file__), f"{name}_glayer"))
    # glayer = CalibrationLayer(args.num_labels)
    # glayer.load_state_dict(weight)
    # glayer = glayer.to("cuda")

    logits = torch.load(os.path.join(os.path.dirname(__file__), f"{name}_logits"))
    
    # print(logits.shape)total_len
    train_logits = logits[:test_len]
    test_logits = logits[test_len:total_len]
    # logits = logits[test_len:total_len]
    indv_data[name] = {
        "accuracy": metric_accuracy(test_logits, test_labels),
        "cost": model_energy[name]["64"][-1],
    }
    logger.info("%s %s", name, indv_data[name])

    probs = torch.float_power(m(train_logits), 2).detach().cpu().numpy()

    # calibrator = cal.PlattBinnerMarginalCalibrator(len(train_labels), num_bins=10)
    # print(name, "before", cal.get_calibration_error(probs, labels))
    
    # print(probs.shape, train_labels.shape)

    calibrator = cal.PlattBinnerTopCalibrator(len(train_labels), num_bins=100)
    calibrator.train_calibration(probs, train_labels)

    # print(logits.shape)
    # logits = glayer(logits)

    model_logits[name] = test_logits.detach().cpu().numpy()
    probs = m(test_logits).detach().cpu().numpy() ** 2
    print(name, "before", describe(np.max(probs, axis=1)))
    # probs = np.max(calibrator.calibrate(probs), axis=1)
    probs = calibrator.calibrate(probs)


    print(name, "after", describe(probs))
    model_probs[name] = probs

    with open(os.path.join(basedir, f"{name}_calibrator"), "wb") as f:
        dill.dump(calibrator, f)

    # with open('data', 'rb') as f:
    #     y = pickle.load(f)

    # print("probs", probs)
    # print("m(test_logits)", m(test_logits))
    # print("m(test_logits)", calibrator.calibrate(m(test_logits).cpu()))

# exit()

with open(os.path.join(os.path.dirname(__file__), f"{args.type}.json"), "w") as fp:
    json.dump(indv_data, fp)



all_thresholds = list(
    itertools.product(
        np.linspace(0, 1, endpoint=True, num=100), repeat=len(model_names) - 1
    )
)

num_labels = len(test_labels)

rnd_seed = 106033
np.random.seed(rnd_seed)

max_size = 100000
if len(all_thresholds) > max_size:
    rnd_idx = np.random.randint(0, len(all_thresholds), max_size)
    all_thresholds = [all_thresholds[i] for i in rnd_idx]

th_data = []
prop_data = []
acc_data = []
cost_data = []

final_logits = np.zeros((num_labels, args.num_labels))
for k, thresholds in enumerate(tqdm(all_thresholds)):
    mask = np.array([False] * num_labels).astype(bool)

    propotion = []
    cost = 0
    for i, name in enumerate(model_names):
        valid = (
            (model_probs[name] >= thresholds[i])
            if name in model_names[:-1]
            else np.array([True] * num_labels).astype(bool)
        )
        # print(np.count_nonzero(valid), num_labels)
        processed = (~mask) & valid

        # cost += np.count_nonzero(~mask) * (
        #     model_energy[name]["64"][-1]
        #     if not "gpt-j" in name
        #     else model_energy[name]["64"][-1] * 4
        # )

        cost += np.count_nonzero(~mask) * model_energy[name]["64"][-1]

        delegated_logit = model_logits[name][processed]
        final_logits[processed] = delegated_logit

        propotion.append(np.count_nonzero(~mask) / num_labels)

        mask |= valid
    accuracy = (
        np.count_nonzero(np.argmax(final_logits, axis=1) == test_labels) / num_labels
    )
    # accuracy = metric_accuracy(final_logits, labels)
    cost = cost / num_labels

    # print(accuracy, cost, final_logits)

    cost_data.append(cost)
    acc_data.append(accuracy)
    th_data.append(json.dumps(thresholds))
    prop_data.append(json.dumps(propotion))

    if (k + 1) % 1000 == 0:
        df = pd.DataFrame(
            {
                "accuracy": acc_data,
                "cost": cost_data,
                "thresholds": th_data,
                "propotion": prop_data,
            }
        )
        df = df.sort_values("accuracy")
        df.to_csv(
            os.path.join(os.path.dirname(__file__), f"{args.type}.csv"), index=False
        )

