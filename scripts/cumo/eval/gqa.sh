#!/bin/bash

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

CKPT=$1
BASE=$2
TEMP='mistral_instruct_system'
SPLIT="llava_gqa_testdev_balanced"
GQADIR="$CuMo_DIR/data/eval/gqa/data"

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m cumo.eval.model_vqa_loader \
        --model-path $CKPT \
        --model-base $BASE \
        --question-file $CuMo_DIR/data/eval/gqa/$SPLIT.jsonl \
        --image-folder $CuMo_DIR/data/eval/gqa/data/images \
        --answers-file $CuMo_DIR/data/eval/gqa/answers/$SPLIT/cumo_mistral_7b/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode $TEMP &
done

wait

output_file=$CuMo_DIR/data/eval/gqa/answers/$SPLIT/cumo_mistral_7b/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat $CuMo_DIR/data/eval/gqa/answers/$SPLIT/cumo_mistral_7b/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

python scripts/convert_gqa_for_eval.py --src $output_file --dst $GQADIR/testdev_balanced_predictions.json

cd $GQADIR
python eval/eval.py --tier testdev_balanced
