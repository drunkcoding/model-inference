

from typing import Tuple

from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding, default_data_collator

from triton_inference.arg_parser import HfArguments
from triton_inference.constants import DATASET_TO_TASK_KEYS, DATASET_TO_TASK_LABELS

def format_texts(examples, data_args):
    texts = []
    fields = DATASET_TO_TASK_KEYS[data_args.dataset_name][data_args.task_name]
    for key in fields:
        if key is not None:
            texts.append(examples[key])
    return (tuple(texts))

def do_padding(data_args):
    if data_args.pad_to_max_length:
        padding = "max_length"
    else:
        # We will pad later, dynamically at batch creation, to the max sequence length in each batch
        padding = False

    return padding

def label2seq(data_args, label):
    # easy_labels = ("true", "false")
    # return easy_labels[label]
    if DATASET_TO_TASK_LABELS[data_args.dataset_name][data_args.task_name] is None:
        return label
    else:
        return DATASET_TO_TASK_LABELS[data_args.dataset_name][data_args.task_name][label]
        # return easy_labels[label]

def default_preprocess(
    tokenizer,
    examples,
    data_args,
    label_to_id=None,
):
    texts = format_texts(examples, data_args)
    result = tokenizer(*texts, padding=do_padding(data_args), max_length=data_args.max_length, truncation=True)

    if "label" in examples:
        if label_to_id is not None:
            # Map labels to IDs (not necessary for GLUE tasks)
            result["labels"] = [label_to_id[l] for l in examples["label"]]
        else:
            # In all cases, rename the column to labels because the model will expect that.
            result["labels"] = examples["label"]
    return result

def seq2seq_preprocess(
    tokenizer,
    examples,
    data_args,
    label_to_id=None,
):
    texts = format_texts(examples, data_args)[0]
    fields = DATASET_TO_TASK_KEYS[data_args.dataset_name][data_args.task_name]

    processed_examples = []
    for i in range(len(texts[0])):
        elements = [data_args.task_name]
        for j in range(len(texts)):
            elements.append(fields[j]+":")
            elements.append(texts[j][i])
        processed_examples.append(" ".join(elements))

    texts = (
        (processed_examples,)
    )
    padding = do_padding(data_args)
    result = tokenizer(*texts, padding=padding, max_length=data_args.max_length, truncation=True)

    if "label" in examples:

        labels = examples["label"]
        labels = [label2seq(data_args, label) for label in labels]

        # Setup the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(labels, max_length=2, padding=padding, truncation=truncation, return_tensors="np")

        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if padding == "max_length":
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]

        result["labels"] = labels["input_ids"]

    return result