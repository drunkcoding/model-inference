
BATCH=( 1 2 4 8 16 32 64 128 )

for bsz in ${BATCH[@]}; do
    ~/.conda/envs/torch/bin/python tests/model/single_model_latency.py ${bsz} &> tests/model/single-XL-v100g1-${bsz}.trace
done