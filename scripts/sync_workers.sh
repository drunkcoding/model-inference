# WORKERS=( "172.31.13.128" "172.31.8.248" )

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
    echo ${WORKER}
    rsync -ahrv ${HOME}/model-inference ubuntu@${WORKER}:~/.
    rsync -ahrv ubuntu@${WORKER}:~/model-inference ${HOME}/. 
    # rsync -ahrv ${HOME}/model-finetune ubuntu@${WORKER}:~/.
    # rsync -ahrv ${HOME}/huggingface-utils ubuntu@${WORKER}:~/.
    rsync -ahr /data/HuggingFace ubuntu@${WORKER}:/data/.
    rsync -ahr /data/ImageNet ubuntu@${WORKER}:/data/.
    rsync -ahr ${HOME}/.cache ubuntu@${WORKER}:~/.
    # rsync -ahr ${HOME}/model-inference/scripts ubuntu@${WORKER}:~/model-inference/.
done

