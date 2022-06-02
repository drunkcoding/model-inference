python inference_dump/t5_dump.py \
--model_path ~/model-finetune/outputs/google/t5-xl-lm-adapt/all/checkpoint-1500 \
--tokenizer_path /mnt/raid0nvme1/HuggingFace/google/t5-small-lm-adapt \
--model_name t5-xl-lm-adapt --device "cuda:5"

python inference_dump/t5_dump.py \
--model_path ~/model-finetune/outputs/google/t5-large-lm-adapt/all/checkpoint-1500 \
--tokenizer_path /mnt/raid0nvme1/HuggingFace/google/t5-small-lm-adapt \
--model_name t5-large-lm-adapt --device "cuda:5"

python inference_dump/t5_dump.py \
--model_path ~/model-finetune/outputs/google/t5-base-lm-adapt/all/checkpoint-2000 \
--tokenizer_path /mnt/raid0nvme1/HuggingFace/google/t5-small-lm-adapt \
--model_name t5-base-lm-adapt --device "cuda:3"

python inference_dump/t5_dump.py \
--model_path ~/model-finetune/outputs/google/t5-small-lm-adapt/all/checkpoint-4500 \
--tokenizer_path /mnt/raid0nvme1/HuggingFace/google/t5-small-lm-adapt \
--model_name t5-small-lm-adapt --device "cuda:2"