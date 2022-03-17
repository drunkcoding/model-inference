from torch2trt import torch2trt
import torch
import sys
from transformers import T5ForConditionalGeneration

model = T5ForConditionalGeneration.from_pretrained(
    sys.argv[1]
)
model.eval()
model = model.to("cuda")

input_ids = torch.ones((1,128)).int().cuda()
attention_mask = torch.ones((1, 128)).int().cuda() 
decoder_input_ids = torch.ones((1,1)).int().cuda()
decoder_attention_mask = torch.ones((1,1)).int().cuda()

model_trt = torch2trt(
    model, 
    (input_ids,attention_mask,decoder_input_ids,decoder_attention_mask), 
    input_names=["input_ids", "attention_mask","decoder_input_ids","decoder_attention_mask"], 
    use_onnx=False, max_workspace_size=1<<30)
torch.save(model_trt.state_dict(), 'tiny.pth', _use_new_zipfile_serialization=False)