
BATCH=( 1 2 4 8 16 32 64 128 )

for bsz in ${BATCH[@]}; do
    CUDA_VISIBLE_DEVICES=5,6 deepspeed tests/deepspeed/ds_pipe.py \
        --dataset_name glue \
        --task_name cola \
        --output_dir . \
        --dataset_name glue \
        --deepspeed_config ~/model-finetune/deepspeed_cfg_pipe.json \
        --model_name_or_path ~/HuggingFace/google/t5-xl-lm-adapt/ \
        --eval_bsz ${bsz} &> tests/deepspeed/ds_pipe-XL-v100g2-${bsz}.trace
done