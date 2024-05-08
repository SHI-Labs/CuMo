#!/bin/bash

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

CKPT=$1
BASE=$2
TEMP='mistral_instruct_system'
SPLIT="llava_vqav2_mscoco_test-dev2015"

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m cumo.eval.model_vqa_loader \
        --model-path $CKPT \
        --model-base $BASE \
        --question-file $CuMo_DIR/data/eval/vqav2/$SPLIT.jsonl \
        --image-folder $CuMo_DIR/data/eval/vqav2/test2015 \
        --answers-file $CuMo_DIR/data/eval/vqav2/answers/$SPLIT/cumo_mistral_7b/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode $TEMP &
done

wait

output_file=$CuMo_DIR/data/eval/vqav2/answers/$SPLIT/cumo_mistral_7b/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat $CuMo_DIR/data/eval/vqav2/answers/$SPLIT/cumo_mistral_7b/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

python scripts/convert_vqav2_for_submission.py --dir $CuMo_DIR/data/eval/vqav2 --split $SPLIT --ckpt cumo_mistral_7b

