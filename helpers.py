
import numpy as np
from transformers import AdamW, get_scheduler
from transformers.tokenization_utils_base import BatchEncoding

def forward_wrapper_tuple(model, batch):
    input_ids = batch['input_ids'].to(model.first_device)
    attention_mask = batch['attention_mask'].to(model.first_device)
    labels = batch['labels'].to(model.last_device)
    outputs = model((input_ids, attention_mask))
    logits = outputs[0]
    
    return logits, labels

def forward_wrapper_dict(model, batch):
    batch = BatchEncoding(batch).to(0)
    logits = model(**batch).logits

    return logits, batch['labels'].to(0)

def train_step(model, batch, forward_wrapper, loss_fn):
    model.train()
    outputs, labels = forward_wrapper(model, batch)
    # print(outputs)
    loss = loss_fn(outputs.to(labels.device), labels)

    return loss

def eval_step(model, batch, forward_wrapper, loss_fn):
    model.train()
    outputs, labels = forward_wrapper(model, batch)
    loss = loss_fn(outputs, labels)

    return outputs, loss

def get_optimizer_grouped_parameters(args, model):
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    return optimizer_grouped_parameters


def create_adamw_optimizer(args, model):
    optimizer_grouped_parameters = get_optimizer_grouped_parameters(args, model)
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    return optimizer

def create_lr_cheduler(args, optimizer):
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.max_train_steps,
    )
    return lr_scheduler

def prepare_model(args, model):
    optimizer = create_adamw_optimizer(args, model)
    lr_cheduler = create_lr_cheduler(args, optimizer)

    return model, optimizer, lr_cheduler

def test_parameters_consistency(model_gold, model_test, abort=True):
    model_test_param = model_test.named_parameters()
    model_gold_param = model_gold.named_parameters()

    for test, gold in zip(model_test_param, model_gold_param):
        name_test, param_test = test
        name_gold, param_gold = gold

        param_test = param_test.detach().cpu().numpy()
        param_gold = param_gold.detach().cpu().numpy()

        if abort:
            print(name_gold, name_test, param_gold.shape, param_test.shape)
            assert np.all(np.isclose(
                param_test,
                param_gold
            ))
        else:
            print(name_test, np.linalg.norm(param_gold-param_test))
