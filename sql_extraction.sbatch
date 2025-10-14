#!/bin/bash
#SBATCH -p priority
#SBATCH -t 2:00:00
#SBATCH -c 4
#SBATCH --mem=8G
#SBATCH --job-name=claims_extraction
#SBATCH --output=claims_extraction_%j.log
#SBATCH --error=claims_extraction_%j.err
#SBATCH --time=2:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=benet_fite@hms.harvard.edu

# Load required modules
module load gcc/14.2.0 python/3.13.1

# Activate Python environment (if applicable)
# source ~/path_to_your_env/bin/activate

# Run the Python script
python extract_raw_sql.py