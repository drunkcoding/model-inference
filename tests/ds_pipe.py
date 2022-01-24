from hfutils.model_pipe import T5DeepSpeedPipe
import deepspeed
from transformers.models.t5.configuration_t5 import T5Config
import argparse

deepspeed.init_distributed()

parser = argparse.ArgumentParser(description="Finetune a transformers model on a text classification task")
parser.add_argument(
    "--local_rank",
    type=int,
    default=-1,
    help="Deepspeed auto local rank.",
)
parser = deepspeed.add_config_arguments(parser)
parser = deepspeed.add_tuning_arguments(parser)
args = parser.parse_args()


config = T5Config.from_pretrained(
    "/sata_disk/jupyter-xue/model-finetune/outputs/t5-xl-lm-adapt/sst2/checkpoint-1380/"
)

model = T5DeepSpeedPipe(config, num_stages=1)

engine, _, _, _ = deepspeed.initialize(args, model=model)
engine.eval()

# engine.eval_batch(, compute_loss=False)
