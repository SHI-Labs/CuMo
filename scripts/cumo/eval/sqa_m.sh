#!/bin/bash

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

CKPT=$1
BASE=$2
TEMP='mistral_instruct_system'

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m cumo.eval.model_vqa_science \
        --model-path $CKPT \
        --model-base $BASE \
        --question-file $CuMo_DIR/data/eval/scienceqa/llava_test_CQM-A.json \
        --image-folder $CuMo_DIR/data/eval/scienceqa/images/test \
        --answers-file $CuMo_DIR/data/eval/scienceqa/answers/cumo_mistral_7b/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --single-pred-prompt \
        --temperature 0 \
        --conv-mode $TEMP &
done

wait

output_file=$CuMo_DIR/data/eval/scienceqa/answers/cumo_mistral_7b.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat $CuMo_DIR/data/eval/scienceqa/answers/cumo_mistral_7b/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

python cumo/eval/eval_science_qa.py \
    --base-dir $CuMo_DIR/data/eval/scienceqa \
    --result-file $output_file \
    --output-file $CuMo_DIR/data/eval/scienceqa/answers/cumo_mistral_7b_output.jsonl \
    --output-result $CuMo_DIR/data/eval/scienceqa/answers/cumo_mistral_7b_result.json
