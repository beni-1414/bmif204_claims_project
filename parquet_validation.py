from pathlib import Path
import pandas as pd

# Paths to the parquet files you expect to generate
OUT_FILES = [
    Path("/n/scratch/users/b/bef299/polypharmacy_project/rx_fills.parquet"),
    Path("/n/scratch/users/b/bef299/polypharmacy_project/adverse_events.parquet"),
    Path("/n/scratch/users/b/bef299/polypharmacy_project/demographics.parquet"),
    Path("/n/scratch/users/b/bef299/polypharmacy_project/enrollment.parquet"),
    Path("/n/scratch/users/b/bef299/polypharmacy_project/spells_with_labels.parquet"),
]

def test_parquet_file(path: Path):
    print(f"\n🔍 Testing {path.name} ...")
    if not path.exists():
        print(f"❌ File does not exist: {path}")
        return False

    try:
        df = pd.read_parquet(path)
        print(f"✅ Successfully read {len(df):,} rows, {len(df.columns)} columns")
        print(df.head(3))  # preview a few rows
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