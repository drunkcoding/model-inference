# WORKERS=("172.31.13.128" "172.31.8.248")

WORKERS=( "172.31.35.95" \
"172.31.39.160" \
"172.31.47.240" \
"172.31.32.224" \
"172.31.44.101" \
"172.31.36.213" \
"172.31.43.33" \
"172.31.39.35" \
"172.31.43.93" \
"172.31.34.158" \
"172.31.40.86" \
"172.31.47.59" \
)

for WORKER in "${WORKERS[@]}"; do
    ssh ${WORKER} "/home/ubuntu/miniconda3/envs/torch/bin/ray stop -f"
done

ray start --head

for WORKER in "${WORKERS[@]}"; do
    ssh ${WORKER} "/home/ubuntu/miniconda3/envs/torch/bin/ray start --address='172.31.35.95:6379' --redis-password='5241590000000000' --num-cpus=4 --num-gpus 1 --resources '{\"${WORKER}\": 20}'"
done