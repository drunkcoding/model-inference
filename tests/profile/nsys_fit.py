import json
import re
from statistics import mean
from timeit import repeat
import numpy as np
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score

pattern = r"-   (.*) bsz (\d+): total (\d+.\d+), propotion (\d+.\d+)"


with open("tests/profile/nsys_extract.log", "r") as fp:
    text = fp.read()

groups = re.findall(pattern, text)

kernel_config = {}

model_data = {}
for group in groups:
    type, bsz, _, prop = group
    bsz = int(bsz)
    prop = float(prop)

    if type not in model_data:
        model_data[type] = []
    model_data[type].append((bsz, prop))

    if type not in kernel_config:
        kernel_config[type] = {}

    kernel_config[type][bsz] = prop

def func(x, b, c):
    return np.power(x, 2) / (np.power(x, 2) + b * x + c)

params = {}
for type, data in model_data.items():
    xdata, ydata = zip(*data)
    xdata = np.array(xdata)
    ydata = np.array(ydata)

    params[type] = dict(zip(xdata, ydata))

    popt, pcov = curve_fit(func, xdata, ydata)
    prediction = func(xdata, *popt)
    # prediction = func(xdata, *popt) if "gpt" not in type else np.mean(ydata).repeat(len(ydata))
    score = r2_score(ydata, prediction)
    # if type != "vit-tiny-patch16-224":
    #     score = r2_score(ydata, func(xdata, *popt))
    # else:
    #     score = r2_score(ydata, np.mean(xdata).repeat(len(ydata)))
    print(type, popt, score, np.linalg.norm(prediction - ydata))
    # if score < 0.9:
    #     params[type] = np.mean(ydata)
    # else:
    #     params[type] = popt.tolist()
    params[type] = popt.tolist()
    print(xdata, ydata, prediction)

with open("tests/profile/nsys_extract.json", "w") as fp:
    json.dump(params, fp)

with open("tests/profile/kernel.json", "w") as fp:
    json.dump(kernel_config, fp)
    

    