#!/bin/bash
#SBATCH -p short
#SBATCH -t 8:00:00
#SBATCH -c 4
#SBATCH --mem=64G
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

# echo "[INFO] Running ndc to ATC mapping..."
# python src/spell_generation/ndc_to_atc_mapping.py \

echo "[INFO] Starting final run..."
python src/spell_generation/spell_generation.py \
    --input_suffix "_sample5M" \
    --output_suffix "_sample5M_grace15_minspell7_ae_censoring" \
    --min_concurrent 3 \
    --grace_period 15 \
    --min_spell_len 7

python src/spell_generation/split_spells_from_changes.py \
    --suffix "_sample5M_grace15_minspell7_ae_censoring"

python src/spell_generation/drug_combo_labeling.py \
    --suffix "_sample5M_grace15_minspell7_ae_censoring"