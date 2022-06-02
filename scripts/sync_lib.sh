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


PYPATH="/home/ubuntu/miniconda3/envs/torch/bin/python"

for WORKER in "${WORKERS[@]}"; do
    rsync -ahr ~/huggingface-utils ubuntu@${WORKER}:~/.
    # rsync -ahr ~/huggingface-utils ubuntu@${WORKER}:~/.
    # ssh ${WORKER} "${PYPATH} -m pip install tensorboard"
    # ssh ${WORKER} "cd /home/ubuntu/huggingface-utils/ && ${PYPATH} -m build && ${PYPATH} -m pip install dist/*.tar.gz"
done

HOSTS=$(IFS=, ; echo "${WORKERS[*]}")

PDSH_RCMD_TYPE=ssh pdsh -w ${HOSTS} "cd /home/ubuntu/huggingface-utils/ && ${PYPATH} -m build && ${PYPATH} -m pip install dist/*.tar.gz"

# pdsh -w ${HOSTS} "cd /home/ubuntu/huggingface-utils/ && git pull && bash setup.sh"


# PDSH_RCMD_TYPE=ssh pdsh -w ${HOSTS} "sudo apt update && sudo apt --fix-broken install -y"
# PDSH_RCMD_TYPE=ssh pdsh -w ${HOSTS} "${PYPATH} -m pip install pynvml"
# PDSH_RCMD_TYPE=ssh pdsh -w ${HOSTS} "${PYPATH} -m pip install uncertainty-calibration"
