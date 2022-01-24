import gc
import os
import time
from datasets import concatenate_datasets
import deepspeed
import numpy as np
from ray import data
from scipy import stats
import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from torch.utils.data.dataset import Dataset
from transformers import DataCollatorForSeq2Seq, default_data_collator, pipeline
from tqdm import tqdm

from hfutils.logger import Logger
from hfutils.arg_parser import HfArguments
from hfutils.loader import ModelLoader, DatasetLoader
from transformers.models.t5.modeling_t5 import T5Block
from transformers import T5ForConditionalGeneration
from hfutils.model_pipe import prepare_decoder_input_ids_for_generation

from torch.utils.data import DataLoader

import warnings

warnings.filterwarnings("ignore")

logger = Logger(__file__, "info", 0, 0)

args = HfArguments()
data_args = args.data_args
dataset_loader = DatasetLoader(args)
tokenizer, _ = ModelLoader(args).load(load_model=False)
eval_raw_dataset = dataset_loader.load(
    tokenizer=None, partition="validation", create_dataloader=False
)
eval_dataset = dataset_loader.load(
    tokenizer, partition="validation", create_dataloader=False
)
eval_dataset = concatenate_datasets([eval_dataset] * 100)
batch_size = args.data_args.eval_bsz
model_path = "/sata_disk/jupyter-xue/model-finetune/outputs/t5-xl-lm-adapt/sst2/checkpoint-1380/"


if data_args.pad_to_max_length:
    data_collator = default_data_collator
else:
    data_collator = DataCollatorForSeq2Seq(tokenizer)

local_rank = int(os.getenv("LOCAL_RANK", "0"))
world_size = int(os.getenv("WORLD_SIZE", "1"))

gc.collect()

time_records = []
start_time = time.perf_counter()


model_names = [
    # 'bert-large-uncased',
    #'EleutherAI/gpt-neo-2.7B',
    # "google/t5-small-lm-adapt"
    "/sata_disk/jupyter-xue/model-finetune/outputs/t5-xl-lm-adapt/sst2/checkpoint-1380/",
]
# model_name = args.model_args.model_name_or_path


eval_dataloader = DataLoader(
    eval_dataset,
    shuffle=True,
    collate_fn=data_collator,
    batch_size=batch_size,
)

@torch.no_grad()
def run_generator(model_name):
    torch.cuda.empty_cache()
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    config = model.config
    model = deepspeed.init_inference(
            model,
            mp_size=world_size,
            dtype=torch.float,
            # replace_with_kernel_inject=True,
            injection_policy={
                T5Block: ("SelfAttention.o", "EncDecAttention.o", "DenseReluDense.wo")
            }
        )
    model.eval()

    time_records = []
    start_time = time.perf_counter()
    for step, batch in tqdm(enumerate(eval_dataloader), f"generation ds {batch_size}"):
        del batch["idx"]
        del batch["labels"]
        if step > 500: break
        decoder_input_ids = prepare_decoder_input_ids_for_generation(batch['input_ids'], config.decoder_start_token_id, config.eos_token_id)
        decoder_attention_mask = decoder_input_ids.new_ones(decoder_input_ids.shape, dtype=torch.long)
        model(**batch, decoder_input_ids=decoder_input_ids, decoder_attention_mask=decoder_attention_mask)
        end_time = time.perf_counter()
        elapsed = (end_time - start_time) * 1000
        if step > 10:
            time_records.append(elapsed)
        start_time = time.perf_counter()
    time_records = np.array(time_records)

    model_name = args.model_args.model_name_or_path.replace("/", "_")
    np.save(
        f"data/ds_latency_{model_name}_{batch_size}_w{world_size}.npy",
        time_records,
        allow_pickle=False,
    )

    logger.info(
        f"{model_name} ds latency summary\n %s",
        stats.describe(time_records),
    )

    stat_des = stats.describe(time_records)

    return time_records, stat_des


all_stat_des = []

df_all = []

for model_name in model_names:
    # generator = pipeline('text2text-generation', model=model_name, device=local_rank, use_fast=False)

    model_stats = []
    records, stat_des = run_generator(model_name)
    sns.distplot(
        records,
        hist=True,
        kde=True,
        bins=int(180 / 5),
        label="ds",
        hist_kws={"edgecolor": "black"},
        kde_kws={"linewidth": 4},
    )

    model_stats.append(stat_des)

    model_name = args.model_args.model_name_or_path.replace("/", "_")

    df = pd.DataFrame(model_stats, columns=stat_des._fields, index=["ds"])
    df["model"] = model_name
    df = df.set_index("model", append=True)
    df.to_csv(
        f"data/ds_latency_{model_name}_{batch_size}_w{world_size}.csv",
        header=True,
        index=True,
    )

    df_all.append(df)

    all_stat_des.append({model_name: df.to_dict()})

    plt.legend()
    plt.savefig(
        f"figures/ds_latency_{model_name}_{batch_size}_w{world_size}.png", bbox_inches="tight"
    )
    plt.close()

gc.collect()

df = pd.concat(df_all)
df.to_csv(f"figures/ds_latency_all_w{world_size}.csv", header=True, index=True)