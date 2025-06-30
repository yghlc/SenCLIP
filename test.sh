#!/bin/bash
set -eE -o functrace

# run zeroshot prediction on EuroSAT dataset
# python src/zeroshot.py \
#         --ckpt_path=/home/hlc/Data/models_deeplearning/SenCLIP/SenCLIP_AvgPool_ViTB32.ckpt \
#         --root_dir=/home/hlc/Data/public_data_AI/EuroSAT/EuroSAT_RGB \
#         --template_path=./src/prompts/prompt_eurosat_aerial_mixed.json



python src/zeroshot_bigearthnet.py \
        --ckpt_path=/home/hlc/Data/models_deeplearning/SenCLIP/SenCLIP_AvgPool_ViTB32.ckpt \
        --template_path=./src/prompts/prompt_ben_ground_mixed.json \
        --download

