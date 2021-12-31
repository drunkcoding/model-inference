
DATASET_TO_TASK_KEYS = {
    "glue": {
        "cola": ("sentence", None),
        "mnli": ("premise", "hypothesis"),
        "mrpc": ("sentence1", "sentence2"),
        "qnli": ("question", "sentence"),
        "qqp": ("question1", "question2"),
        "rte": ("sentence1", "sentence2"),
        "sst2": ("sentence", None),
        "stsb": ("sentence1", "sentence2"),
        "wnli": ("sentence1", "sentence2"),
    },
    "super_glue": {
        "axb": ("sentence1", "sentence2"),
        "axg": ("premise", "hypothesis"),
        "boolq": ("question", "passage"),
        "cb": ("premise", "hypothesis"),
    }
}

DATASET_TO_TASK_LABELS = {
    "glue": {
        "cola": ("true", "false"),
        "mnli": ("true", "false", "not"),
        "mrpc": ("true", "false"),
        "qnli": ("true", "false"),
        "qqp": ("true", "false"),
        "rte": ("true", "false"),
        "sst2": ("sentence", None),
        "stsb": None,
        "wnli": ("true", "false"),
    },
    "super_glue": {
        "axb": ("true", "false"),
        "axg": ("true", "false"),
        "boolq": ("yes", "no"),
        "cb": ("true", "false", "not"),
    }
}