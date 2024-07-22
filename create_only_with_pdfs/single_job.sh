#!/bin/bash

#SBATCH --partition=hopper-cpu
#SBATCH -o slurm/logs/%x_%j_a.out
#SBATCH --time=0:40:00
#SBATCH --qos=high

echo "START_INDEX: $START_INDEX"
echo "STEP_SIZE: $STEP_SIZE"

source /admin/home/andres_marafioti/.bashrc
source /fsx/andi/docmatix/.venv/bin/activate

# Load any necessary modules
cd /fsx/andi/docmatix

# Run the Python script with the appropriate start index
python create_only_with_pdfs/load_data.py --start_index $START_INDEX --step_size $STEP_SIZE