#!/bin/bash

CKPT=$1
BASE=$2
TEMP='mistral_instruct_system'

python -m cumo.eval.model_vqa \
    --model-path $CKPT \
    --model-base $BASE \
    --question-file $CuMo_DIR/data/eval/llava-bench-in-the-wild/questions.jsonl \
    --image-folder $CuMo_DIR/data/eval/llava-bench-in-the-wild/images \
    --answers-file $CuMo_DIR/data/eval/llava-bench-in-the-wild/answers/cumo_mistral_7b.jsonl \
    --temperature 0 \
    --conv-mode $TEMP

mkdir -p $CuMo_DIR/data/eval/llava-bench-in-the-wild/reviews

python cumo/eval/eval_gpt_review_bench.py \
    --question $CuMo_DIR/data/eval/llava-bench-in-the-wild/questions.jsonl \
    --context $CuMo_DIR/data/eval/llava-bench-in-the-wild/context.jsonl \
    --rule cumo/eval/table/rule.json \
    --answer-list \
        $CuMo_DIR/data/eval/llava-bench-in-the-wild/answers_gpt4.jsonl \
        $CuMo_DIR/data/eval/llava-bench-in-the-wild/answers/cumo_mistral_7b.jsonl \
    --output \
        $CuMo_DIR/data/eval/llava-bench-in-the-wild/reviews/cumo_mistral_7b.jsonl

python cumo/eval/summarize_gpt_review.py -f $CuMo_DIR/data/eval/llava-bench-in-the-wild/reviews/cumo_mistral_7b.jsonl
