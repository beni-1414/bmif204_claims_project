from pathlib import Path
import pandas as pd

# Paths to the parquet files you expect to generate
OUT_FILES = [
    # Path("/n/scratch/users/b/bef299/polypharmacy_project_fhd8SDd3U50/rx_fills_sample1M.parquet"),
    # Path("/n/scratch/users/b/bef299/polypharmacy_project_fhd8SDd3U50/adverse_events.parquet"),
    # Path("/n/scratch/users/b/bef299/polypharmacy_project_fhd8SDd3U50/demographics.parquet"),
    # Path("/n/scratch/users/b/bef299/polypharmacy_project_fhd8SDd3U50/enrollment.parquet"),
    Path("/n/scratch/users/b/bef299/polypharmacy_project_fhd8SDd3U50/spells_with_labels_opioid_sample1M_grace15_minspell7.parquet"),
    Path("/n/scratch/users/b/bef299/polypharmacy_project_fhd8SDd3U50/drug_changes_opioid_sample1M_grace15_minspell7.parquet"),
    Path("/n/scratch/users/b/bef299/polypharmacy_project_fhd8SDd3U50/split_spells_opioid_sample1M_grace15_minspell7.parquet")
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