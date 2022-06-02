import os

from hfutils.pipe.gpt import GPTPytorchPipeRandom
import numpy as np
from tqdm import tqdm
from transformers import AutoConfig

basedir = os.path.dirname(__file__)

config = AutoConfig.from_pretrained(basedir)
model = GPTPytorchPipeRandom(config)
layers = [spec.build() for spec in tqdm(model.layer_specs)]
layer_params = [
    sum([np.prod(p.size()) for p in layer.parameters()]) for layer in layers
]

print(layer_params)
