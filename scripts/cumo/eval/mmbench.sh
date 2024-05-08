#!/bin/bash

CKPT=$1
BASE=$2
TEMP='mistral_instruct_system'

SPLIT="mmbench_dev_20230712"

python -m cumo.eval.model_vqa_mmbench \
    --model-path $CKPT \
    --model-base $BASE \
    --question-file $CuMo_DIR/data/eval/mmbench/$SPLIT.tsv \
    --answers-file $CuMo_DIR/data/eval/mmbench/answers/$SPLIT/cumo_mistral_7b.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode $TEMP

mkdir -p $CuMo_DIR/data/eval/mmbench/answers_upload/$SPLIT

python scripts/convert_mmbench_for_submission.py \
    --annotation-file $CuMo_DIR/data/eval/mmbench/$SPLIT.tsv \
    --result-dir $CuMo_DIR/data/eval/mmbench/answers/$SPLIT \
    --upload-dir $CuMo_DIR/data/eval/mmbench/answers_upload/$SPLIT \
    --experiment cumo_mistral_7b
