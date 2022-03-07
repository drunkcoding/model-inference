import time
from transformers import DataCollatorForSeq2Seq, T5ForConditionalGeneration, T5Tokenizer
import os, sys
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from hfutils.model_pipe import T5Pipe, T5PyTorchPipe
from hfutils.loader import load_glue_task_val, t5_preprocess_function
from functools import partial
from datasets import concatenate_datasets



user_path = os.path.expanduser('~')
model_path = os.path.join(user_path, "HuggingFace", "google", "t5-xl-lm-adapt")

tokenizer = T5Tokenizer.from_pretrained(model_path)
model = T5ForConditionalGeneration.from_pretrained(model_path)

preprocess_function = partial(
    t5_preprocess_function, 
    tokenizer=tokenizer,
    padding="max_length",
    max_length=128,
)

eval_dataset = load_glue_task_val(preprocess_function, 'sst2').shuffle()
eval_dataset = concatenate_datasets([eval_dataset] * 10)
data_collator = DataCollatorForSeq2Seq(tokenizer)

# batch_size = int(sys.argv[1])
device_ds = torch.device("cuda:0")
device_pipe = torch.device("cuda:0")

# model_pipe_p0 = T5PyTorchPipe(model)
# model_pipe_p1 = T5PyTorchPipe(model)
# model_pipe_p0.partition_by_parameter(0, 2)
# model_pipe_p1.partition_by_parameter(1, 2)
# print(model_pipe_p0.layers)
# print(model_pipe_p0.exec_map,model_pipe_p1.exec_map, len(model_pipe_p0.layers))
# # model_pipe_p0 = model_pipe_p0.convert(device_pipe)
# # model_pipe_p1 = model_pipe_p1.convert(device_pipe)
# model_pipe_p0.convert(device_pipe)
# model_pipe_p1.convert(device_pipe)
 
model_pipe_ds = T5PyTorchPipe(model)
model_pipe_ds.convert(device_ds)

# model = model.to(device)

# decoder_input_ids = torch.zeros((batch_size,1)).int().to(device)
# decoder_attention_mask = torch.ones((batch_size,1)).int().to(device)
with torch.no_grad():
    for batch_size in [1,2,4,8,16,32,64,128]:
        eval_dataloader = DataLoader(
            eval_dataset,
            shuffle=True,
            collate_fn=data_collator,
            batch_size=batch_size,
            # drop_last=True,
        )

        # for step, batch in enumerate(tqdm(eval_dataloader, desc=f"{batch_size}-gen")):
        #     if step > 500: break
        #     input_ids = batch['input_ids'].to(device)
        #     attention_mask = batch['attention_mask'].to(device)
        #     model.generate(
        #         input_ids=input_ids,
        #         attention_mask=attention_mask,
        #         do_sample=False,  # disable sampling to test if batching affects output
        #         return_dict_in_generate=True,
        #         output_scores=True,
        #     )

        # for step, batch in enumerate(tqdm(eval_dataloader, desc=f"{batch_size}-ds")):
        #     if step > 500: break
        #     input_ids = batch['input_ids'].to(device_pipe)
        #     attention_mask = batch['attention_mask'].to(device_pipe)
        #     outputs = model_pipe_p0((input_ids, attention_mask))
        #     # print(len(outputs))
        #     outputs = model_pipe_p1(outputs)
        #     print(outputs.shape)

        for step, batch in enumerate(tqdm(eval_dataloader, desc=f"{batch_size}-ds")):
            if step > 500: break
            input_ids = batch['input_ids'].to(device_ds)
            attention_mask = batch['attention_mask'].to(device_ds)
            start_time = time.perf_counter()
            outputs = model_pipe_ds((input_ids, attention_mask))
            end_time = time.perf_counter()
            torch.cuda.synchronize()
            print(batch_size, (end_time-start_time) * 1000)
