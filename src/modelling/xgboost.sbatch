#!/bin/bash
#SBATCH -p short
#SBATCH -t 04:00:00
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH --job-name=xgboost_training
#SBATCH --output=xgboost_training_%j.log
#SBATCH --error=xgboost_training_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=benet_fite@hms.harvard.edu

# Load required modules
module load gcc/14.2.0 python/3.13.1

# Activate Python environment (if applicable)
source ~/bmif204/bmif204_claims_project/.venv/bin/activate

python src/modelling/xgboost_trainer.py --title "Grid v1 5M: depth3-4 child5-10 subsample/colsamp 0.8-0.9 n_estimators=400" --db 5