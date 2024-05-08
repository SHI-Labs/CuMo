#!/bin/bash

CKPT=$1
BASE=$2
TEMP='mistral_instruct_system'

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m cumo.eval.model_vqa_loader \
        --model-path $CKPT \
        --model-base $BASE \
        --question-file $CuMo_DIR/data/eval/seed_bench/llava-seed-bench-new.jsonl \
        --image-folder $CuMo_DIR/data/eval/seed_bench \
        --answers-file $CuMo_DIR/data/eval/seed_bench/answers/cumo_mistral_7b/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode $TEMP &
done

wait

output_file=$CuMo_DIR/data/eval/seed_bench/answers/cumo_mistral_7b/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat $CuMo_DIR/data/eval/seed_bench/answers/cumo_mistral_7b/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

# Evaluate
python scripts/convert_seed_for_submission.py \
    --annotation-file $CuMo_DIR/data/eval/seed_bench/SEED-Bench.json \
    --result-file $output_file \
    --result-upload-file $CuMo_DIR/data/eval/seed_bench/answers_upload/cumo_mistral_7b.json

