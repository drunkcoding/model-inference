#!/bin/bash

nohup tritonserver --model-repository ${HOME}/model-inference/repository/ \
--backend-directory ${HOME}/.local/opt/tritonserver/backends/ \
--model-control-mode explicit \
--load-model empty &> ${HOME}/model-inference/repository/empty/trace.log &