# MODELS=( "t5-xl-lm-adapt" \
#     "t5-small-lm-adapt" \
#     "t5-base-lm-adapt" \
#     "t5-large-lm-adapt" \
#     "vit-tiny-patch16-224" \
#     "vit-small-patch16-224" \
#     "vit-base-patch16-224" \
#     "vit-large-patch16-224" \
#     "distilbert-base-uncased" \
#     "bert-base-uncased" \
#     "bert-large-uncased" \
#     "bert-tiny-finetuned-squadv2" \
#     "bert-small-finetuned-squadv2" \
#     "bert-tiny-5-finetuned-squadv2" \
#     "bert-mini-5-finetuned-squadv2" \
#     "bert-small-2-finetuned-squadv2" \
#     "distilgpt2" \
#     "gpt2" \
#     "gpt2-medium" \
#     "gpt2-large" \
#     "gpt2-xl" \
#     "gpt-j-6B" \
# )

MODELS=( "vit-small-patch16-224" \
    "vit-base-patch16-224" \
    "vit-large-patch16-224" \
    "bert-tiny-finetuned-squadv2" \
)

   
BATCH_SIZES=( 1 2 4 6 8 16 32 48 64 128 )

for model in "${MODELS[@]}"; do 
   for bsz in "${BATCH_SIZES[@]}"; do 
        basename=$(echo "${model}" | cut -d '/' -f 2)
        python tests/profile/nsys_extract.py \
                --model_name_or_path ${model} \
                --batch_size ${bsz}
    done
done

