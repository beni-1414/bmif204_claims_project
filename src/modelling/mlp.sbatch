#!/bin/bash
#SBATCH -p short
#SBATCH -t 02:00:00
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH --job-name=mlp_training
#SBATCH --output=mlp_training_%j.log
#SBATCH --error=mlp_training_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=benet_fite@hms.harvard.edu

# Load required modules
module load gcc/14.2.0 python/3.13.1

# Activate Python environment (if applicable)
source ~/bmif204/bmif204_claims_project/.venv/bin/activate

python -m src.modelling.mlp_pooled_embeddings --title "MLP AE30d model with MedTok embeddings 5M pooled" --db 5 --suffix _FINAL