#!/bin/bash
#SBATCH -p short
#SBATCH -t 01:00:00
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH --job-name=logreg_training
#SBATCH --output=logreg_training_%j.log
#SBATCH --error=logreg_training_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=benet_fite@hms.harvard.edu

# Load required modules
module load gcc/14.2.0 python/3.13.1

# Activate Python environment (if applicable)
source ~/bmif204/bmif204_claims_project/.venv/bin/activate

python src/modelling/logreg.py --title "LogReg AE30d model 5M" --db 5