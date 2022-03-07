from datasets import load_metric, load_dataset
import torch
from tqdm import tqdm
import os

from transformers import GPT2LMHeadModel, GPT2Tokenizer

val_dataset = load_dataset("lambada")['validation']

print(val_dataset)

home_dir = os.path.expanduser(("~"))
model_path = f"{home_dir}/HuggingFace/gpt2",

device = "cuda"
model_id = "gpt2"
model = GPT2LMHeadModel.from_pretrained(model_path).to(device)
tokenizer = GPT2Tokenizer.from_pretrained(model_path)

encodings = tokenizer("\n\n".join(val_dataset["text"]), return_tensors="pt")

def preplexity(encodings, model, device):

    max_length = model.config.n_positions
    stride = 512

    nlls = []
    for i in tqdm(range(0, encodings.input_ids.size(1), stride)):
        begin_loc = max(i + stride - max_length, 0)
        end_loc = min(i + stride, encodings.input_ids.size(1))
        trg_len = end_loc - i  # may be different from stride on last loop
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            neg_log_likelihood = outputs[0] * trg_len

        nlls.append(neg_log_likelihood)

    ppl = torch.exp(torch.stack(nlls).sum() / end_loc)

    return ppl

ppl = preplexity(encodings, model, device)
print(ppl)