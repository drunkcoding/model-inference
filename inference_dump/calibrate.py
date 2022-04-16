from dataclasses import dataclass, field
import json
import os
import torch
from datasets import load_metric
from transformers import HfArgumentParser

# from hfutils.calibration import g_scaling

basedir = os.path.dirname(__file__)

metric = load_metric("accuracy")


def metric_accuracy(logits: torch.Tensor, labels: torch.Tensor):
    predictions = torch.argmax(logits, axis=1).flatten()
    return metric.compute(predictions=predictions, references=labels.flatten())[
        "accuracy"
    ]


@dataclass
class Arguments:
    model_name: str = field(metadata={"help": "Name of the model use as key"},)
    num_labels: int = field(metadata={"help": "Number of labels"},)
    device: str = field(
        default="cuda:0", metadata={"help": "Model Device"},
    )


parser = HfArgumentParser(Arguments)
args = parser.parse_args_into_dataclasses()[0]

labels = torch.load(os.path.join(basedir, f"{args.model_name}_labels")).to(args.device)
logits = torch.load(os.path.join(basedir, f"{args.model_name}_logits")).to(args.device)

test_len = int(len(labels) * 0.4)
labels = labels[:test_len]
logits = logits[:test_len]

print(metric_accuracy(logits, labels))

import calibration as cal

m = torch.nn.Softmax(dim=1)
# probs, _ = torch.max(m(logits), dim=1)
probs = torch.float_power(m(logits), 2)

probs = probs.detach().cpu().numpy()
labels = labels.detach().cpu().numpy()

calibration_error = cal.get_calibration_error(probs, labels)

# print(probs)
print("calibration_error", calibration_error)

calibrator = cal.PlattBinnerMarginalCalibrator(len(probs) / 10, num_bins=10)
calibrator.train_calibration(probs, labels)

calibrated_probs = calibrator.calibrate(probs)

# print(calibrated_probs)

# with open("tests/kernel_duration/energy.json", "r") as fp:
#     model_energy = json.load(fp)

# with open(os.path.join(basedir, f"{args.type}.json"), "r") as fp:
#     best_data = json.load(fp)
# best_data[args.model_name] = {
#     "accuracy": metric_accuracy(logits, labels),
#     "cost": model_energy[args.model_name]["64"][-1]
# }

# with open(os.path.join(basedir, f"{args.type}.json"), "w") as fp:
#     json.dump(best_data, fp)

# glayer = g_scaling(logits, labels, 500, args.num_labels)
# print(metric_accuracy(glayer(logits), labels))
# torch.save(glayer.state_dict(), os.path.join(basedir, f"{args.model_name}_glayer"))
torch.save(calibrator, os.path.join(basedir, f"{args.model_name}_calibrator"))
