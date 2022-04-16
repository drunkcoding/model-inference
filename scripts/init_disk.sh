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
    # ssh -t ${WORKER} "sudo mkfs -t ext4 /dev/nvme1n1 && sudo mount /dev/nvme1n1 /data"
    # ssh -t ${WORKER} $'sudo echo "/dev/nvme1n1  /data  ext4  defaults,nofail  0  2" >> /etc/fstab'
    ssh -t ${WORKER} "sudo chown ubuntu /data"
done