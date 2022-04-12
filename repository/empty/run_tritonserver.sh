#!/bin/bash

tritonserver --model-repository ${HOME}/model-inference/repository/ \
--backend-directory ${HOME}/.local/opt/tritonserver/backends/ \
--model-control-mode explicit \
--load-model empty &> trace.log &