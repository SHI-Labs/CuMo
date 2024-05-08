#!/bin/bash

CKPT=$1
BASE=$2
TEMP='mistral_instruct_system'

python -m cumo.eval.model_vqa_mmmu \
    --model-path $CKPT \
    --model-base $BASE \
    --answers-file $CuMo_DIR/data/eval/MMMU/answers/cumo_mistral_7b.json \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode $TEMP

python cumo/eval/main_eval_only.py --output_path $CuMo_DIR/data/eval/MMMU/answers/cumo_mistral_7b.json --answer_path $CuMo_DIR/data/eval/MMMU/eval/answer_dict_val.json
