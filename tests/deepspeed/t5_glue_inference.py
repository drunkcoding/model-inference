import deepspeed
import os
from functools import partial
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import T5ForConditionalGeneration, T5Tokenizer, AutoConfig
from transformers.models.t5.modeling_t5 import T5Block
from transformers.data.data_collator import (
    DataCollatorForSeq2Seq,
    default_data_collator,
)

from hfutils.loader import t5_preprocess_function, load_glue_val, load_glue_task_val

user_path = os.path.expanduser("~")
model_path = os.path.join(user_path, "HuggingFace", "google", "t5-xl-lm-adapt")
# model_path = "/jmain02/home/J2AD002/jxm12/lxx22-jxm12/model-finetune/outputs/google/t5-small-lm-adapt/all/checkpoint-4500"

config = AutoConfig.from_pretrained(model_path)
model = T5ForConditionalGeneration.from_pretrained(model_path,config=config)
tokenizer = T5Tokenizer.from_pretrained(model_path)

local_rank = int(os.getenv("LOCAL_RANK", "0"))
world_size = int(os.getenv("WORLD_SIZE", "1"))

padding = "max_length"
max_length = 128

preprocess_function = partial(
    t5_preprocess_function, 
    tokenizer=tokenizer,
    padding=padding,
    max_length=max_length,
)

batch_size = 2

# eval_dataset = load_glue_val(preprocess_function)
eval_dataset = load_glue_task_val(preprocess_function, "cola")
data_collator = DataCollatorForSeq2Seq(tokenizer)
eval_dataloader = DataLoader(
    eval_dataset,
    collate_fn=data_collator,
    batch_size=batch_size,
)

print("dataset loaded")

# deepspeed.init_distributed()

engine = deepspeed.init_inference(
    model,
    mp_size=world_size,
    # replace_with_kernel_inject=True,
    # injection_policy={
    #     T5Block: ("SelfAttention.o", "EncDecAttention.o", "DenseReluDense.wo")
    # }
)


print("dataset loaded")

device = torch.cuda.current_device()

decoder_input_ids = torch.zeros((batch_size,1)).int().to(device)
decoder_attention_mask = torch.ones((batch_size,1)).int().to(device)

# model = model.to(device)

for step, batch in enumerate(tqdm(eval_dataloader)):
    if step > 500: break
    # print(batch)
    
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    engine(input_ids=input_ids, 
        attention_mask=attention_mask, 
        decoder_input_ids=decoder_input_ids,
        decoder_attention_mask=decoder_attention_mask)




