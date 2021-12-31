from torch.utils.data.dataloader import DataLoader
from functools import partial
from transformers import AutoConfig, AutoTokenizer, AutoModel
from datasets import load_dataset, load_metric
from transformers.data.data_collator import DataCollatorForSeq2Seq, DataCollatorWithPadding, default_data_collator

from arg_parser import HfArguments
from triton_inference.constants import DATASET_TO_TASK_KEYS, DATASET_TO_TASK_LABELS
from triton_inference.preprocess import DataPreprocessor, default_preprocess, seq2seq_preprocess

class ModelLoader:
    def __init__(self, args: HfArguments) -> None:
        self.model_args = args.model_args
        self.data_args = args.data_args

    def load(self, deepspeed=True):
        model_args = self.model_args
        data_args = self.data_args
        labels = DATASET_TO_TASK_LABELS[data_args.dataset_name][data_args.task_name]
        config = AutoConfig.from_pretrained(
            model_args.config_name if model_args.config_name else model_args.model_name_or_path,
            num_labels=len(labels) if labels is not None else None,
            finetuning_task=data_args.task_name,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
        )
        model = AutoModel.from_pretrained(
            model_args.model_name_or_path,
            config=config,
        )

        if deepspeed:
            from deepspeed.utils.zero_to_fp32 import load_state_dict_from_zero_checkpoint
            model = load_state_dict_from_zero_checkpoint(model, model_args.model_name_or_path)
        
        return tokenizer, model

class DatasetLoader:
    def __init__(self, args: HfArguments) -> None:
        self.data_args = args.data_args
        self.model_args = args.model_args

        #HARD CODE seq2seq model
        self.is_seq2seq = "t5" in self.model_args.model_name_or_path
    
    def load(self, tokenizer, tokenize:bool=True, partition:str="train", create_dataloader:bool=False):
        data_args = self.data_args
        raw_datasets = load_dataset(data_args.dataset_name, data_args.task_name)       

        preprocessor = partial(seq2seq_preprocess if self.is_seq2seq else default_preprocess, 
            tokenizer,
            data_args=data_args)

        raw_datasets = raw_datasets.map(
            preprocessor,
            batched=True,
            desc="Running tokenizer on dataset",
        )

        dataset = raw_datasets[partition]

        if not tokenize:
            return dataset

        if data_args.pad_to_max_length:
            # If padding was already done ot max length, we use the default data collator that will just convert everything
            # to tensors.
            data_collator = default_data_collator
        else:
            # Otherwise, `DataCollatorWithPadding` will apply dynamic padding for us (by padding to the maximum length of
            # the samples passed). When using mixed precision, we add `pad_to_multiple_of=8` to pad all tensors to multiple
            # of 8s, which will enable the use of Tensor Cores on NVIDIA hardware with compute capability >= 7.5 (Volta).
            if self.is_seq2seq:
                data_collator = DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8)
            else:
                data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)

        if create_dataloader:
            dataset = DataLoader(
                self.eval_dataset, 
                shuffle=True, 
                collate_fn=data_collator, 
                batch_size=self.data_args.train_bsz if partition == "train" else self.data_args.eval_bsz
            )

        return dataset