from datasets import concatenate_datasets
import deepspeed
import numpy as np
from scipy import stats
import torch
import matplotlib.pyplot as plt
from torch.utils.data.dataset import Dataset
from transformers import pipeline
from tqdm import tqdm
import deepspeed

from hfutils.logger import Logger
from transformers.models.t5.modeling_t5 import T5Block
from transformers import T5Tokenizer, T5ForConditionalGeneration

import warnings

warnings.filterwarnings("ignore")

logger = Logger(__file__, "info", 0, 0)

model_path = "/mnt/raid0nvme1/HuggingFace/google/t5-small-lm-adapt"

tokenizer = T5Tokenizer.from_pretrained(model_path)
model = T5ForConditionalGeneration.from_pretrained(model_path)

model.eval()
with torch.no_grad():
    model = deepspeed.init_inference(
        model,
        mp_size=1,
        dtype=torch.int8,
        # replace_with_kernel_inject=True,
        injection_policy={
            T5Block: ("SelfAttention.o", "EncDecAttention.o", "DenseReluDense.wo")
        },
    )
