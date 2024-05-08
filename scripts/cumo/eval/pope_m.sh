#!/bin/bash

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

CKPT=$1
BASE=$2
TEMP='mistral_instruct_system'

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m cumo.eval.model_vqa_loader \
        --model-path $CKPT \
        --model-base $BASE \
        --question-file $CuMo_DIR/data/eval/pope/llava_pope_test.jsonl \
        --image-folder $CuMo_DIR/data/eval/pope/val2014 \
        --answers-file $CuMo_DIR/data/eval/pope/answers/cumo_mistral_7b/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode $TEMP &
done

wait

output_file=$CuMo_DIR/data/eval/pope/answers/cumo_mistral_7b.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat $CuMo_DIR/data/eval/pope/answers/cumo_mistral_7b/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

python cumo/eval/eval_pope.py \
    --annotation-dir $CuMo_DIR/data/eval/pope/coco \
    --question-file $CuMo_DIR/data/eval/pope/llava_pope_test.jsonl \
    --result-file $output_file
