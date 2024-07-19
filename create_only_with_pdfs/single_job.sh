#!/bin/bash

#SBATCH --partition=hopper-cpu
#SBATCH --cpus-per-task=96
#SBATCH -o slurm/logs/%x_%j.out
#SBATCH --time=0:40:00
#SBATCH --qos=high

echo "START_INDEX: $START_INDEX"
echo "STEP_SIZE: $STEP_SIZE"

source /admin/home/andres_marafioti/.bashrc
source activate llmswarm

# Load any necessary modules
cd /fsx/andi/docmatix

# Run the Python script with the appropriate start index
python examples/question_answer_pairs/load_data.py --start_index $START_INDEX --step_size $STEP_SIZE