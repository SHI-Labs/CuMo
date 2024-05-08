#!/bin/bash

CKPT=$1
BASE=$2
TEMP='mistral_instruct_system'

python -m cumo.eval.model_vqa_mathvista \
    --model-path $CKPT \
    --model-base $BASE \
    --answers-file $CuMo_DIR/data/eval/mathvista/answers/cumo_mistral_7b.json \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode $TEMP

# extract answer
python cumo/eval/extract_answer.py \
    --output_file $CuMo_DIR/data/eval/mathvista/answers/cumo_mistral_7b.json \
    --response_label response \
    --output_label extraction 

# calculate score
python cumo/eval/calculate_score.py \
    --gt_file $CuMo_DIR/data/eval/mathvista/data/testmini.json \
    --output_dir $CuMo_DIR/data/eval/mathvista/answers \
    --output_file cumo_mistral_7b_extraction.json \
    --score_file scores_cumo_mistral_7b_extraction.json