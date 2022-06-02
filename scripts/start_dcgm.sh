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

HOSTS=$(IFS=, ; echo "${WORKERS[*]}")

PDSH_RCMD_TYPE=ssh pdsh -w ${HOSTS} "sudo nv-hostengine -t"
PDSH_RCMD_TYPE=ssh pdsh -w ${HOSTS} "sudo nv-hostengine -b ALL"