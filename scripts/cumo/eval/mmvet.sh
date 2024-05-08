#!/bin/bash

CKPT=$1
BASE=$2
TEMP='mistral_instruct_system'

python -m cumo.eval.model_vqa \
    --model-path $CKPT \
    --model-base $BASE \
    --question-file $CuMo_DIR/data/eval/mm-vet/llava-mm-vet.jsonl \
    --image-folder $CuMo_DIR/data/eval/mm-vet/mm-vet/images \
    --answers-file $CuMo_DIR/data/eval/mm-vet/answers/cumo_mistral_7b.jsonl \
    --temperature 0 \
    --conv-mode $TEMP

mkdir -p $CuMo_DIR/data/eval/mm-vet/results

python scripts/convert_mmvet_for_eval.py \
    --src $CuMo_DIR/data/eval/mm-vet/answers/cumo_mistral_7b.jsonl \
    --dst $CuMo_DIR/data/eval/mm-vet/results/cumo_mistral_7b.json