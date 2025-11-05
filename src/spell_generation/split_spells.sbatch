#!/bin/bash
#SBATCH -p priority
#SBATCH -t 1:00:00
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH --job-name=claims_processing
#SBATCH --output=claims_extraction_%j.log
#SBATCH --error=claims_extraction_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=benet_fite@hms.harvard.edu

set -e

# Load required modules
module load gcc/14.2.0 python/3.13.1

# Activate Python environment (if applicable)
source ~/bmif204/bmif204_claims_project/.venv/bin/activate

python src/spell_generation/split_spells_from_changes.py \
    --suffix "_opioid_sample1M_grace15_minspell7_ae_censoring"