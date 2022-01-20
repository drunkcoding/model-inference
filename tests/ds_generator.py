import os
import time
import deepspeed
import numpy as np
from scipy import stats
import torch
import pandas as pd
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

class PipeDataset(Dataset):
    def __init__(self, dataset: Dataset, key: str):
        self.dataset = dataset
        self.key = key

    def __len__(self):
        return len(self.dataset[self.key])

    def __getitem__(self, i):
        return self.dataset[self.key][i]

eval_dataset = PipeDataset(eval_raw_dataset, "input_text")

model_names = [
    #'distilbert-base-uncased',
    #'bert-base-uncased',
    # 'bert-large-uncased',
    # 'distilgpt2',
    #'gpt2',
    #'EleutherAI/gpt-neo-2.7B',
    'EleutherAI/gpt-neo-1.3B',
    'EleutherAI/gpt-neo-125M',
]

test_cases = [
    "plain",
    "half",
    "deepspeed",
    # "deepspeed-moq",
]

# model_name = args.model_args.model_name_or_path

local_rank = int(os.getenv('LOCAL_RANK', '0'))
world_size = int(os.getenv('WORLD_SIZE', '1'))

def run_generator(model_name, type):
    torch.cuda.empty_cache()
    generator = pipeline('text-classification', model=model_name, device=local_rank, use_fast=True)
    if type == "plain":
        pass
    elif type == "half":
        generator.model = deepspeed.init_inference(generator.model,
                                            mp_size=world_size,
                                            dtype=torch.half,
                                            replace_with_kernel_inject=True,
                                            replace_method='auto')
    elif type == "deepspeed":
        generator.model = deepspeed.init_inference(generator.model,
                                            mp_size=world_size,
                                            dtype=torch.float,
                                            replace_with_kernel_inject=True,
                                            replace_method='auto')
    elif type == "deepspeed-moq":
        generator.model = deepspeed.init_inference(generator.model,
                                            mp_size=world_size,
                                            dtype=torch.int8,
                                            replace_with_kernel_inject=True,
                                            replace_method='auto')
    else:
        raise ValueError()

    time_records = []
    start_time = time.perf_counter()
    cnt = 0
    for _ in tqdm(generator(eval_dataset), f"generation {type}"):
        cnt += 1
        end_time = time.perf_counter()
        elapsed = (end_time-start_time) * 1000
        if cnt > 10:
            time_records.append(elapsed)
        start_time = time.perf_counter()
    time_records = np.array(time_records)

    model_name = args.model_args.model_name_or_path.replace("/", "_")
    np.save(f"data/generator_latency_{model_name}_{type}_w{world_size}.npy", time_records, allow_pickle=False)

    logger.info(
        f"{model_name} {type} latency summary\n %s",
        stats.describe(time_records),
    )

    stat_des = stats.describe(time_records)

    return time_records, stat_des

all_stat_des = []

df_all = []

for model_name in model_names:
    # generator = pipeline('text2text-generation', model=model_name, device=local_rank, use_fast=False)

    model_stats = []
    for task in test_cases:
        records, stat_des = run_generator(model_name, task)
        sns.distplot(
            records,
            hist=True,
            kde=True,
            bins=int(180 / 5),
            label=task,
            hist_kws={"edgecolor": "black"},
            kde_kws={"linewidth": 4},
        )

        model_stats.append(stat_des)

    model_name = args.model_args.model_name_or_path.replace("/", "_")

    df = pd.DataFrame(model_stats, columns=stat_des._fields, index=test_cases)
    df["model"] = model_name
    df = df.set_index('model', append=True)
    df.to_csv(f"data/generator_latency_{model_name}_w{world_size}.csv", header=True, index=True)
    
    df_all.append(df)

    all_stat_des.append({model_name: df.to_dict()})

    plt.legend()
    plt.savefig(f"figures/generator_latency_{model_name}_w{world_size}.png", bbox_inches="tight")
    plt.close()

df = pd.concat(df_all)
df.to_csv(f"figures/generator_latency_all_w{world_size}.csv", header=True, index=True)

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
