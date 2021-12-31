#!/bin/bash

tritonserver --model-repository ${HOME}/model-inference/repository/ \
--backend-directory ${HOME}/server/build/opt/tritonserver/backends/ \
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
--load-model t5_large_lm_adapt_glue_cola \
--load-model t5_small_lm_adapt_glue_cola \
--grpc-infer-allocation-pool-size 20 &> trace.log &

# --model-control-mode poll \
# --repository-poll-secs 60
# --model-control-mode explicit \
# --load-model distilgpt2_cola \
# --load-model gpt_neo_2.7B_standalone \
# --load-model gpt_neo_cola_ensemble \
# --http-thread-count 10 \