#!/bin/bash

BASE="/home/xly/model-inference/repository/repo_profile"

tritonserver --model-repository ${BASE} \
--backend-directory ${HOME}/.local/opt/tritonserver/backends/ \
--exit-timeout-secs 60 \
--log-info True \
--log-warning True \
--log-error True \
--metrics-interval-ms 10 \
--http-thread-count 10 \
--buffer-manager-thread-count 8 \
--allow-http True \
--grpc-infer-allocation-pool-size 20 &> ${BASE}/trace.log &