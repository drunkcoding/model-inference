#!/bin/bash

BASE="/sata_disk/jupyter-xue"

tritonserver --model-repository ${BASE}/model-inference/repository/ \
--backend-directory ${HOME}/.local/opt/tritonserver/backends/ \
--exit-timeout-secs 60 \
--log-info True \
--log-warning True \
--log-error True \
--metrics-interval-ms 100 \
--buffer-manager-thread-count 20 \
--pinned-memory-pool-byte-size 2000000000 \
--response-cache-byte-size 2000000000 \
--allow-http True \
--model-control-mode explicit \
--load-model t5_cola_ensemble \
--grpc-infer-allocation-pool-size 20 &> trace.log &

# --model-control-mode poll \
# --repository-poll-secs 60
# --model-control-mode explicit \
# --load-model distilgpt2_cola \
# --load-model gpt_neo_2.7B_standalone \
# --load-model gpt_neo_cola_ensemble \
# --load-model t5_sst2_ensemble \
# --http-thread-count 10 \
# --load-model t5-xl-lm-adapt_sst2 \