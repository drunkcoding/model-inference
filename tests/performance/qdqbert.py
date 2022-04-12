import functools
import re
import time
import torch
from tqdm import tqdm
from transformers import (
    QDQBertForQuestionAnswering,
    AutoTokenizer,
    EvalPrediction,
    default_data_collator,
)
from datasets import load_metric, load_dataset
from torch.utils.data import DataLoader

from hfutils.qa import prepare_validation_features
from hfutils.measure import get_energy_by_group

from utils_qa import postprocess_qa_predictions
import pytorch_quantization.nn as quant_nn
from pytorch_quantization.tensor_quant import QuantDescriptor

def compute_metrics(p: EvalPrediction):
    return metric.compute(predictions=p.predictions, references=p.label_ids)


task_name = "squad_v2"
device = "cuda:7"

path = "/home/xly/transformers/examples/research_projects/quantization-qdqbert/finetuned_int8/bert-large-uncased"

model = QDQBertForQuestionAnswering.from_pretrained(path)
tokenizer = AutoTokenizer.from_pretrained(path)

model = model.to(device)
model.eval()

metric = load_metric(task_name)
val_dataset = load_dataset(task_name)["validation"] #.select([x for x in range(1000)])
column_names = val_dataset.column_names

question_column_name = "question" if "question" in column_names else column_names[0]
context_column_name = "context" if "context" in column_names else column_names[1]
answer_column_name = "answers" if "answers" in column_names else column_names[2]


# Post-processing:
def post_processing_function(examples, features, predictions, stage="eval"):
    # Post-processing: we match the start logits and end logits to answers in the original context.
    predictions = tuple(
        [
            p.detach().cpu().numpy() if isinstance(p, torch.Tensor) else p
            for p in predictions
        ]
    )

    predictions = postprocess_qa_predictions(
        examples=examples,
        features=features,
        predictions=predictions,
        version_2_with_negative=True,
        n_best_size=20,
        max_answer_length=30,
        null_score_diff_threshold=0.0,
        output_dir=".",
        prefix=stage,
    )
    # Format the result to the format the metric expects.
    formatted_predictions = [
        {"id": k, "prediction_text": v, "no_answer_probability": 0.0}
        for k, v in predictions.items()
    ]

    references = [
        {"id": ex["id"], "answers": ex[answer_column_name]} for ex in examples
    ]
    return EvalPrediction(predictions=formatted_predictions, label_ids=references)


batch_size = 128

data_collator = default_data_collator

dataset = val_dataset.map(
    functools.partial(
        prepare_validation_features, tokenizer=tokenizer, column_names=column_names
    ),
    batched=True,
    num_proc=10,
    remove_columns=column_names,
    desc="Running tokenizer on validation dataset",
)

dataloader = DataLoader(
    dataset.remove_columns(["example_id", "offset_mapping"]),
    shuffle=False,
    collate_fn=data_collator,
    batch_size=batch_size,
)

time.sleep(1)

start_time = time.perf_counter()
start_energy = sum(list(get_energy_by_group().values()))

all_starts = []
all_ends = []
with torch.no_grad():
    for batch in tqdm(dataloader, desc=f"test"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        # token_type_ids = batch["token_type_ids"].to(device)
        
        # t1 = time.perf_counter()
        outputs = model(input_ids, attention_mask, return_dict=True)
        # t2 = time.perf_counter()
        # print(t2-t1)
        all_starts.append(outputs.start_logits)
        all_ends.append(outputs.end_logits)

        torch.cuda.empty_cache()

end_energy = sum(list(get_energy_by_group().values()))
end_time = time.perf_counter()
print(end_energy - start_energy)
print(end_time - start_time)

predictions = (torch.cat(all_starts), torch.cat(all_ends))
eval_pred = post_processing_function(val_dataset, dataset, predictions)
print(compute_metrics(eval_pred))