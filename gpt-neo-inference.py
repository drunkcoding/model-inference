from transformers import GPTNeoModel, GPT2Tokenizer, GPT2Model, AutoModelForCausalLM, AutoTokenizer, GPTNeoForSequenceClassification
import deepspeed
from deepspeed.pipe import PipelineModule
import torch
from transformers import pipeline

model = GPTNeoForSequenceClassification.from_pretrained("/jmain01/home/JAD003/sxr06/lxx22-sxr06/HuggingFace/gpt-neo-2.7B")
tokenizer = GPT2Tokenizer.from_pretrained("/jmain01/home/JAD003/sxr06/lxx22-sxr06/HuggingFace/gpt-neo-2.7B")
# model.parallelize()

print("model loaded")

promt = "In this tutorial we will be adding DeepSpeed to Megatron-LM GPT2 model, which is a large, powerful transformer. Megatron-LM supports model-parallel and multi-node training. Please see the corresponding paper for more details: Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism."
encoded_promt = tokenizer(promt, return_tensors="pt").to("cuda:0")
# print(model.is_parallelizable)

# print(model.__dict__)
# print(encoded_promt)

# print(model(**encoded_promt.data))

deepspeed.init_distributed()

# net = PipelineModule(layers=model.to_layers(), num_stages=2)
engine = deepspeed.init_inference(model, mp_size=8)

print("result", engine.forward(**encoded_promt))

# generator = pipeline('text-generation', model=engine, tokenizer=tokenizer)
# print(generator("EleutherAI has", do_sample=True, min_length=50))

# engine = deepspeed.init_inference(model.base_model)

# prompt = "In a shocking finding, scientists discovered a herd of unicorns living in a remote, " \
#         "previously unexplored valley, in the Andes Mountains. Even more surprising to the " \
#         "researchers was the fact that the unicorns spoke perfect English."

# input_ids = tokenizer(prompt, return_tensors="pt").input_ids

# gen_tokens = model.generate(input_ids, do_sample=True, temperature=0.9, max_length=100,)
# gen_text = tokenizer.batch_decode(gen_tokens)[0]