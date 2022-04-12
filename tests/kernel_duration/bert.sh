BASE_PATH="/mnt/raid0nvme1/HuggingFace"
# MODELS=( "distilbert-base-uncased" "bert-base-uncased" "bert-large-uncased" "mrm8488/bert-tiny-finetuned-squadv2" "mrm8488/bert-mini-finetuned-squadv2" "mrm8488/bert-small-finetuned-squadv2" "mrm8488/bert-tiny-5-finetuned-squadv2" "mrm8488/bert-mini-5-finetuned-squadv2" "mrm8488/bert-small-2-finetuned-squadv2" )
MODELS=( "madlag/bert-large-uncased-wwm-squadv2-x2.63-f82.6-d16-hybrid-v1" )
BATCH_SIZES=( 1 2 4 6 8 16 32 48 64 128 )

for model in "${MODELS[@]}"; do 
   for bsz in "${BATCH_SIZES[@]}"; do 
        basename=$(echo "${model}" | cut -d '/' -f 2)
        nsys profile \
            -o profile/${basename}-${bsz}.json \
            --export=json \
            --force-overwrite true \
            /home/xly/.conda/envs/xly/bin/python \
            /home/xly/model-inference/tests/kernel_duration/bert.py \
                --model_name_or_path ${BASE_PATH}/${model} \
                --batch_size ${bsz}
    done
done

