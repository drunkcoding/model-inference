#!/bin/bash

source /home/jupyter-xue/.bashrc
source activate torch

cd /sata_disk/jupyter-xue/model-inference

deepspeed --include localhost:1 tests/ds_generator.py \
--model_name_or_path EleutherAI/gpt-neo-2.7B \
--eval_bsz 1 \
--pad_to_max_length \
--dataset_name glue \
--task_name cola

deepspeed --num_gpus 2 tests/ds_generator.py \
--model_name_or_path EleutherAI/gpt-neo-2.7B \
--eval_bsz 1 \
--pad_to_max_length \
--dataset_name glue \
--task_name cola