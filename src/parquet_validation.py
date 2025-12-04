from pathlib import Path
import pandas as pd

"""
Simple validation script to check existence and readability of parquet files. Useful for verifying that expected output files were generated correctly.
"""

# Paths to the parquet files you expect to generate
SUFFIX = "_opioid_sample5M_grace15_minspell7_ae_censoring"
BASE = Path("/n/scratch/users/b/bef299/polypharmacy_project_fhd8SDd3U50/")
OUT_FILES = [
    # BASE / "rx_fills_opioid_sample5M.parquet",
    # BASE / "rx_fills_opioid_sample1M.parquet",
    # BASE / "adverse_events_opioid_sample1M.parquet",
    # BASE / "demographics_sample1M.parquet",
    # BASE / "enrollment.parquet",
    # BASE / f"spells_with_labels{SUFFIX}.parquet",
    # BASE / f"drug_changes{SUFFIX}.parquet",
    # BASE / f"split_spells{SUFFIX}.parquet",
    # BASE / f"split_spells{SUFFIX}_with_drugcombo.parquet",
    # BASE / f"icd10_codes_from_spells{SUFFIX}.parquet",
    # BASE / f"icd10_codes_from_spells{SUFFIX}_clustered.parquet",
]

def test_parquet_file(path: Path):
    print(f"\n🔍 Testing {path.name} ...")
    if not path.exists():
        print(f"❌ File does not exist: {path}")
        return False

    try:
        df = pd.read_parquet(path)
        print(f"✅ Successfully read {len(df):,} rows, {len(df.columns)} columns")
        print(df.head(20))  # preview a few rows
        # Print list of columns
        print(f"Columns: {df.columns.tolist()}")
        return True
    except Exception as e:
        print(f"❌ Failed to read {path}: {e}")
        return False

# Run tests for each file
all_ok = True
for f in OUT_FILES:
    if not test_parquet_file(f):
        all_ok = False

if all_ok:
    print("\n🎉 All parquet files validated successfully!")
else:
    print("\n⚠️ One or more parquet files failed validation.")