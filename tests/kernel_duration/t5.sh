BASE_PATH="/mnt/raid0nvme1/HuggingFace"
MODELS=( "google/t5-xl-lm-adapt" "google/t5-small-lm-adapt" "google/t5-base-lm-adapt" "google/t5-large-lm-adapt" )
# BATCH_SIZES=( 1 2 4 6 8 16 32 48 64 128)
BATCH_SIZES=( 64 128)

for model in "${MODELS[@]}"; do 
   for bsz in "${BATCH_SIZES[@]}"; do 
        basename=$(echo "${model}" | cut -d '/' -f 2)
        nsys profile \
            -o profile/${basename}-${bsz}.json \
            --export=json \
            --force-overwrite true \
            /home/xly/.conda/envs/xly/bin/python \
            /home/xly/model-inference/tests/kernel_duration/t5.py \
                --model_name_or_path ${BASE_PATH}/${model} \
                --batch_size ${bsz}
    done
done

