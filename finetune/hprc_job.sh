#!/bin/bash
##############################################################
# TAMU HPRC job script — fine-tune TAMU ECE embedding model
# Submit with: sbatch hprc_job.sh
##############################################################

#SBATCH --job-name=tamu-ece-embedder
#SBATCH --output=logs/finetune_%j.out
#SBATCH --error=logs/finetune_%j.err
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:a100:1          # request 1 A100; change to v100 if unavailable
#SBATCH --partition=gpu            # check available partitions: sinfo

# Load modules
module purge
module load Python/3.11.3-GCCcore-12.3.0
module load CUDA/12.1.1

# Activate your virtual environment (create once: python -m venv ~/envs/ece-embed)
source ~/envs/ece-embed/bin/activate

# Install deps if not already installed
pip install -q sentence-transformers torch --extra-index-url https://download.pytorch.org/whl/cu121

# Create log dir
mkdir -p logs

# Run fine-tuning
python finetune_embedder.py \
    --data ../crawler/training_pairs.jsonl \
    --output ./tamu-ece-embedder \
    --epochs 3 \
    --batch 128

echo "Done. Model saved to ./tamu-ece-embedder"
