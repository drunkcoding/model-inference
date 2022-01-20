import deepspeed
import torch

from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import T5ForConditionalGeneration, DataCollatorForSeq2Seq, default_data_collator

from hfutils.logger import Logger
from hfutils.arg_parser import HfArguments
from hfutils.loader import ModelLoader, DatasetLoader

args = HfArguments()
model_loader = ModelLoader(args)
dataset_loader = DatasetLoader(args)

tokenizer, _ = model_loader.load(load_model=False, deepspeed=False)

pos_token = tokenizer("false").input_ids[0]
neg_token = tokenizer("true").input_ids[0]

class T5DSInferenceWrapper(torch.nn.Module):

    def __init__(self, model) -> None:
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        # input_ids = input_ids.cuda()
        # attention_mask = attention_mask.cuda()
        # print(input_ids)
        # print(attention_mask)
        outputs = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            do_sample=False, # disable sampling to test if batching affects output
            return_dict_in_generate=True,
            output_scores=True,
        )
        # logits = outputs.scores[0][:, [neg_token, pos_token]]
        return outputs

model_loader.set_custom_model_class(T5ForConditionalGeneration)
_, model = model_loader.load(load_tokenizer=False, deepspeed=False)

for name, child in model.named_children():
    print(name, child)
exit()

ds_wrapper = T5DSInferenceWrapper(model)

engine = deepspeed.init_inference(ds_wrapper, replace_with_kernel_inject=True)

eval_dataset = dataset_loader.load(tokenizer, partition="validation", create_dataloader=False)
data_args = args.data_args
if data_args.pad_to_max_length:
    data_collator = default_data_collator
else:
    data_collator = DataCollatorForSeq2Seq(tokenizer)

eval_dataloader = DataLoader(
    eval_dataset, 
    shuffle=True, 
    collate_fn=data_collator, 
    batch_size=data_args.eval_bsz,
)

with torch.no_grad():
    for batch in tqdm(eval_dataloader, desc="Dry Run DeepSpeed"):
        # print(batch)
        # batch = batch.to("cuda:0")
        input_ids=batch['input_ids'].to("cuda:0")
        attention_mask=batch['attention_mask'].to("cuda:0")
        engine(input_ids=input_ids, attention_mask=attention_mask)
        # , labels=torch.rand((input_ids.shape[0], 2), device="cuda:0")
        # engine.module.generate(
        #     input_ids=input_ids,
        #     attention_mask=attention_mask,
        #     do_sample=False, # disable sampling to test if batching affects output
        #     return_dict_in_generate=True,
        #     output_scores=True,
        # )

    for batch in tqdm(eval_dataloader, desc="Dry Run Plain"):
        # batch = batch.to("cuda:0")
        input_ids=batch['input_ids'].to("cuda:0")
        attention_mask=batch['attention_mask'].to("cuda:0")
        model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            do_sample=False, # disable sampling to test if batching affects output
            return_dict_in_generate=True,
            output_scores=True,
        )

