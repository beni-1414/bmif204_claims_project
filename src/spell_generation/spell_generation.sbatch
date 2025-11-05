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

# Load required modules
module load gcc/14.2.0 python/3.13.1

# Activate Python environment (if applicable)
source ~/bmif204/bmif204_claims_project/.venv/bin/activate

# Run the Python script
# echo "[INFO] Starting 21-day extension run..."
# python spell_generation.py \
#     --output_suffix "_sample1M_30day" \
#     --opioid_flag True \
#     --min_concurrent 3 \
#     --grace_period 21

echo "[INFO] Starting final run..."
python spell_generation.py \
    --input_suffix "_sample1M" \
    --output_suffix "_sample1M_grace15_minspell7" \
    --opioid_flag True \
    --min_concurrent 3 \
    --grace_period 15 \
    --min_spell_len 7

# echo "[INFO] Starting 7-day min spell length run..."
# python spell_generation.py \
#     --output_suffix "_sample1M_7day" \
#     --opioid_flag True \
#     --min_concurrent 3 \
#     --grace_period 21 \
#     --min_spell_len 28