DEPLOY=$1

# nvprof --profile-all-processes --replay-mode disabled -fo ray-%p.nvprof &

~/.conda/envs/torch/bin/python ray/t5_sst2_composed.py \
    --cfg ray/meta.json \
    --dataset_name glue \
    --task_name mnli \
    --deployment ${DEPLOY} &