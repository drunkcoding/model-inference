import os
from deepspeed.runtime.pipe.module import PipelineModule
from numpy import float16
from onnx.external_data_helper import convert_model_to_external_data
import transformers
from transformers import AutoModel, AutoTokenizer
from transformers.models.gpt2 import GPT2OnnxConfig
from transformers.models.gpt_neo import GPTNeoOnnxConfig
import torch
import numpy as np
import onnx
import onnx_graphsurgeon as gs
from onnx import load_external_data_for_model
from pathlib import Path

# model = PipelineModule()

def return_submodels(graph: gs.Graph):
    tensors = graph.tensors()
    print(tensors)
    # print(tensors["1058"], type(tensors["1058"]))
    return [
        {
            "inputs": graph.inputs,
            "outputs": [tensors["1058"]],
            # "outputs": [tensors["1058"].to_variable(dtype=np.float32)],
        },
        {
            "inputs": [tensors["1058"], tensors["1"]],
            "outputs": [tensors["2122"]],
            # "inputs": [tensors["1058"].to_variable(dtype=np.float32)],
            # "outputs": [tensors["2122"].to_variable(dtype=np.float32)],
        }
    ]



base_dir = '/mnt/yavin/HuggingFace/'

tokenizer = AutoTokenizer.from_pretrained(os.path.join(base_dir, 'gpt2'))
model  = AutoModel.from_pretrained(os.path.join(base_dir, 'gpt2'))

# print(model.config.hidden_size)
# exit()

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = 50256

# text = ["I am "]
# encoded_text = tokenizer(text, padding=True, truncation=True, max_length=128)

onnx_config = GPT2OnnxConfig(model.config, "default", )
dummy_inputs = onnx_config.generate_dummy_inputs(tokenizer, 1, 128, framework='pt')
# del dummy_inputs['past_key_values']
print(onnx_config.use_past)
torch.onnx.export(
    model=model.base_model,
    args=(dummy_inputs, ),
    f="./onnx/model.onnx",
    verbose=False,
    opset_version=9,
    export_params=True,
    use_external_data_format=False,
    do_constant_folding=False,
)

# transformers.onnx.export(
#     tokenizer=tokenizer,
#     model=model, 
#     config=onnx_config,
#     opset=11,
#     output=Path(os.path.join(base_dir, 'gpt2/model.onnx')),
# )

model = onnx.load("./onnx/model.onnx", load_external_data=False)
# load_external_data_for_model(model, os.path.join(base_dir, 'gpt2'))
# model = onnx.load(os.path.join(base_dir, 'bert_large_v1_1_fake_quant.onnx'), load_external_data=True)

for i in range(2):
    graph = gs.import_onnx(model)
    submodels = return_submodels(graph)

    print("preparing submodels %s" % i)

    graph.inputs = submodels[i]["inputs"]
    graph.outputs = submodels[i]["outputs"]

    # if i > 0:
    #     graph.inputs.pop(0)
    #     for first_input in submodels[0]["inputs"]:
    #         # print(first_input.name)
    #         if first_input.name == "1":
    #             graph.inputs.append(first_input)

    graph.cleanup(True, True, True)

    # graph.cleanup()

    path_to_save_model = "./repository/gpt2_part%s/0/model.onnx" % i

    new_model = gs.export_onnx(graph)
    # convert_model_to_external_data(new_model, all_tensors_to_one_file=True, location=Path(path_to_save_model).name + ".data")
    onnx.save(new_model, path_to_save_model)

    print("Model saved to:", path_to_save_model)
    print("Inputs:", graph.inputs)
    print("Outputs:", graph.outputs)

    print(graph.tensors().keys())
    


# print(submodels)
# print(graph)