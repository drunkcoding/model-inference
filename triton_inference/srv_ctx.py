from datasets.load import load_dataset
from transformers.data.data_collator import DataCollatorWithPadding, default_data_collator
from transformers import AutoTokenizer

from torch.utils.data import DataLoader

from triton_inference.arg_parser import GlueArgParser

class GlueContext():
    def __init__(self, parser: GlueArgParser) -> None:

        self.args = args = parser.parse()

        raw_datasets = load_dataset("glue", args.task_name)

        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.bos_token
        padding = "max_length" if args.pad_to_max_length else False

        self.is_regression = is_regression = args.task_name == "stsb"
        if not is_regression:
            label_list = raw_datasets["train"].features["label"].names
            num_labels = len(label_list)
        else:
            num_labels = 1

        label_to_id = None
        # label_to_id = {str(v): i for i, v in enumerate(label_list)}
        # print(label_to_id)
        if args.task_name is not None:
            sentence1_key, sentence2_key = parser.task_to_keys[args.task_name]
        else:
            # Again, we try to have some nice defaults but don't hesitate to tweak to your use case.
            non_label_column_names = [name for name in raw_datasets["train"].column_names if name != "label"]
            if "sentence1" in non_label_column_names and "sentence2" in non_label_column_names:
                sentence1_key, sentence2_key = "sentence1", "sentence2"
            else:
                if len(non_label_column_names) >= 2:
                    sentence1_key, sentence2_key = non_label_column_names[:2]
                else:
                    sentence1_key, sentence2_key = non_label_column_names[0], None

        def preprocess_function(examples):
            # Tokenize the texts
            texts = (
                (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
            )
            result = tokenizer(*texts, padding=padding, max_length=args.max_length, truncation=True)

            if "label" in examples:
                if label_to_id is not None:
                    # print(examples["label"])
                    # Map labels to IDs (not necessary for GLUE tasks)
                    result["labels"] = [label_to_id[l] for l in examples["label"]]
                else:
                    # In all cases, rename the column to labels because the model will expect that.
                    result["labels"] = examples["label"]
            return result

        processed_datasets = raw_datasets.map(
                preprocess_function,
                batched=True,
                remove_columns=raw_datasets["train"].column_names,
                desc="Running tokenizer on dataset",
            )

        # DataLoaders creation:
        if args.pad_to_max_length:
            # If padding was already done ot max length, we use the default data collator that will just convert everything
            # to tensors.
            self.data_collator = default_data_collator
        else:
            # Otherwise, `DataCollatorWithPadding` will apply dynamic padding for us (by padding to the maximum length of
            # the samples passed). When using mixed precision, we add `pad_to_multiple_of=8` to pad all tensors to multiple
            # of 8s, which will enable the use of Tensor Cores on NVIDIA hardware with compute capability >= 7.5 (Volta).
            self.data_collator = DataCollatorWithPadding(tokenizer)

        self.eval_dataset = processed_datasets["validation_matched" if args.task_name == "mnli" else "validation"]

    def get_eval_dataloader(self, shuffle=False, batch_size=None):
        eval_dataloader = DataLoader(
            self.eval_dataset, 
            shuffle=shuffle, 
            collate_fn=self.data_collator, 
            batch_size=self.args.per_device_eval_batch_size if batch_size is None else batch_size
        )
        return eval_dataloader