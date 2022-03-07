import os

import torch.cuda
from hfutils.model_pipe import T5DeepSpeedPipe
import deepspeed
from tqdm import tqdm
from transformers.models.t5.configuration_t5 import T5Config
from transformers import DataCollatorForSeq2Seq, default_data_collator
import argparse
from deepspeed.utils import RepeatingLoader
from hfutils.arg_parser import HfArguments
from hfutils.loader import ModelLoader, DatasetLoader
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from datasets import load_dataset, load_metric, concatenate_datasets

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

args = HfArguments()
data_args = args.data_args
dataset_loader = DatasetLoader(args)
tokenizer, _ = ModelLoader(args).load(load_model=False)
eval_dataset = dataset_loader.load(
    tokenizer, partition="validation", create_dataloader=False
)

batch_size = data_args.eval_bsz
user_path = os.path.expanduser("~")
# model_path = os.path.join(user_path, "HuggingFace", "google", "t5-xl-lm-adapt")
# model_path = "/mnt/yavin/checkpoints/t5-xl-lm-adapt/sst2/checkpoint-1380/"
# model_path = "google/t5-small-lm-adapt"
model_path = args.model_args.model_name_or_path

if data_args.pad_to_max_length:
    data_collator = default_data_collator
else:
    data_collator = DataCollatorForSeq2Seq(tokenizer)


class PipeDataset(Dataset):
    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        return {
            "encoder_input_ids": self.dataset[i]['input_ids'],
            "encoder_attention_mask": self.dataset[i]['attention_mask'],
        }

eval_dataset = concatenate_datasets([eval_dataset]*70)
eval_dataset = PipeDataset(eval_dataset)

# print(eval_dataset[0])

def eval_generator():
    eval_dataloader = DataLoader(
        eval_dataset,
        shuffle=True,
        collate_fn=data_collator,
        batch_size=batch_size,
    )

    for batch in tqdm(eval_dataloader, desc="eval_generator"):
        shape = batch["encoder_input_ids"].shape
        yield ((
                   batch["encoder_input_ids"],
                   batch["encoder_attention_mask"],
               ), torch.zeros(shape[0]))
        # print(shape)
        # yield (
        #         batch["encoder_input_ids"],
        #         batch["encoder_attention_mask"],
        # )


# exit()
config = T5Config.from_pretrained(
    model_path
)

deepspeed.init_distributed()
model = T5DeepSpeedPipe(config, num_stages=torch.cuda.device_count())

engine, _, _, _ = deepspeed.initialize(args.ds_args, model=model)

for step, batch in enumerate(RepeatingLoader(eval_generator())):
    if step > 500: break
    engine.eval_batch(iter([batch]*1), compute_loss=False)

# engine.eval_batch(RepeatingLoader(eval_generator()), compute_loss=False)
