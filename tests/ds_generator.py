import os
import time
import deepspeed
import numpy as np
from scipy import stats
import torch
import seaborn as sns
import matplotlib.pyplot as plt
from torch.utils.data.dataset import Dataset
from transformers import pipeline
from tqdm import tqdm

from hfutils.logger import Logger
from hfutils.arg_parser import HfArguments
from hfutils.loader import ModelLoader, DatasetLoader

import warnings
warnings.filterwarnings("ignore")

logger = Logger(__file__, "info", 0,0)

args = HfArguments()
dataset_loader = DatasetLoader(args)
eval_raw_dataset = dataset_loader.load(tokenizer=None, partition="validation", create_dataloader=False)

model_name = args.model_args.model_name_or_path

local_rank = int(os.getenv('LOCAL_RANK', '0'))
world_size = int(os.getenv('WORLD_SIZE', '1'))
# generator = pipeline('text2text-generation', model=model_name, device=local_rank, use_fast=False)
generator = pipeline('text-generation', model=model_name, device=local_rank, use_fast=True)

class PipeDataset(Dataset):
    def __init__(self, dataset: Dataset, key: str):
        self.dataset = dataset
        self.key = key

    def __len__(self):
        return len(self.dataset[self.key])

    def __getitem__(self, i):
        return self.dataset[self.key][i]

eval_dataset = PipeDataset(eval_raw_dataset, "input_text")

plain_time = []
start_time = time.perf_counter()
cnt = 0
for _ in tqdm(generator(eval_dataset, do_sample=False, min_length=2, batch_size=args.data_args.eval_bsz), "generation Plain"):
    cnt += 1
    end_time = time.perf_counter()
    elapsed = (end_time-start_time) * 1000
    if cnt > 10:
        plain_time.append(elapsed)
    # logger.info("Model plain %s: %s (ms)", model_name, elapsed)
    start_time = time.perf_counter()
plain_time = np.array(plain_time)

generator.model = deepspeed.init_inference(generator.model,
                                           mp_size=world_size,
                                           dtype=torch.float,
                                        #    replace_with_kernel_inject=True,
                                           replace_method='auto')

ds_time = []
start_time = time.perf_counter()
cnt = 0
for _ in tqdm(generator(eval_dataset, do_sample=False, min_length=2, batch_size=args.data_args.eval_bsz), "generation Deepspeed"):
    cnt += 1
    end_time = time.perf_counter()
    elapsed = (end_time-start_time) * 1000
    if cnt > 10:
        ds_time.append(elapsed)
    # logger.info("Model DS %s: %s (ms)", model_name, elapsed)
    start_time = time.perf_counter()
ds_time = np.array(ds_time)

logger.info(
    "plain_time latency summary\n %s",
    stats.describe(plain_time),
)
logger.info(
    "ds_time latency summary\n %s",
    stats.describe(ds_time),
)

sns.distplot(
    plain_time,
    hist=True,
    kde=True,
    bins=int(180 / 5),
    label="w/o DS",
    hist_kws={"edgecolor": "black"},
    kde_kws={"linewidth": 4},
)
sns.distplot(
    ds_time,
    hist=True,
    kde=True,
    bins=int(180 / 5),
    label="w/ DS",
    hist_kws={"edgecolor": "black"},
    kde_kws={"linewidth": 4},
)
plt.legend()
plt.savefig(f"figures/generator_latency_{world_size}.png", bbox_inches="tight")
plt.close()

# string = generator("DeepSpeed is", do_sample=False, min_length=2)
# if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
#     print(string)

# # Filename: gpt-neo-2.7b-generation.py
# import os
# import deepspeed
# import torch
# from transformers import pipeline

# local_rank = int(os.getenv('LOCAL_RANK', '0'))
# world_size = int(os.getenv('WORLD_SIZE', '1'))
# generator = pipeline('text-generation', model='EleutherAI/gpt-neo-125M', device=local_rank)


# generator.model = deepspeed.init_inference(generator.model,
#                                            mp_size=world_size,
#                                            dtype=torch.float,
#                                            replace_with_kernel_inject=True,
#                                            replace_method='auto')

# string = generator("DeepSpeed is", do_sample=True, min_length=50)
# if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
#     print(string)