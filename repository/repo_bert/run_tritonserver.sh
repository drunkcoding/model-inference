#!/bin/bash

type="bert"

tritonserver --model-repository ${HOME}/model-inference/repository/repo_${type}/ \
--backend-directory ${HOME}/.local/opt/tritonserver/backends/ \
--exit-timeout-secs 60 \
--log-info True \
--log-warning True \
--log-error True \
--metrics-interval-ms 10 \
--http-thread-count 50 \
--buffer-manager-thread-count 8 \
--allow-http True \
--model-control-mode explicit \
--load-model ${type}_e0p0 \
--load-model ${type}_e1p0 \
--load-model ${type}_ensemble \
--grpc-infer-allocation-pool-size 200 &> trace.log &

# --model-control-mode explicit \
# --load-model ${type}_e0p0 \

# --load-model ${type}_e1p0 \
# --load-model ${type}_e0p0 \

# --model-control-mode poll \
# --repository-poll-secs 60
# --model-control-mode explicit \
# --load-model distilgpt2_cola \
# --load-model gpt_neo_2.7B_standalone \
# --load-model gpt_neo_cola_ensemble \
# --load-model t5_sst2_ensemble \
# --http-thread-count 10 \
# --load-model t5-xl-lm-adapt_sst2 \
# --pinned-memory-pool-byte-size 2000000000 \
# --response-cache-byte-size 2000000000 \