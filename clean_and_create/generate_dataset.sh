#!/bin/bash

BASE_DIR="/fsx/m4/datasets/docvqa_instruct"
STEP_SIZE=9
submitted_jobs=0

for i in $(seq 0 9 1791); do
    SHARD_DIR="$BASE_DIR/shard_$((i / STEP_SIZE))"
    
    # Check if the directory does not exist
    if [ ! -d "$SHARD_DIR" ]; then
        sbatch --job-name=data-gen-$i --export=ALL,START_INDEX=$i,STEP_SIZE=$STEP_SIZE examples/question_answer_pairs/single_job.sh
        ((submitted_jobs++))
    fi
done

echo "Total number of jobs submitted: $submitted_jobs"