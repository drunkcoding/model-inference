from transformers import AutoConfig

from hfutils.pipe.gpt import GPTPytorchPipeRandom

config = AutoConfig.from_pretrained("/mnt/raid0nvme1/HuggingFace/gpt2-large")

model = GPTPytorchPipeRandom(config)
model.partition_by_parameter(0, 2, True)
model.convert_layer_specs("cuda")
model.eval()