import pyodbc
import pandas as pd
from pathlib import Path
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Extract raw SQL data for opioid study")

    parser.add_argument("--suffix", type=str, default="_sample5M",
                        help="Suffix used in filenames (e.g., '_sample1M' or '')")
    parser.add_argument("--db", type=int, default=5, help="Database size to connect to (1 for 1M, 5 for 5M)")

    args = parser.parse_args()
    return args

# ------------------------------
# DB CONNECTION
# ------------------------------
args = parse_args()
suffix = args.suffix
database = f"InovalonSample{args.db}M"
conn_str = f'DRIVER=ODBC Driver 17 for SQL Server;Server=CCBWSQLP01.med.harvard.edu;Trusted_Connection=Yes;Database={database};TDS_Version=8.0;Encryption=require;Port=1433;REALM=MED.HARVARD.EDU'

conn = pyodbc.connect(conn_str)

# ------------------------------
# PATH CONFIG
# ------------------------------
SCRATCH_PATH = Path("/n/scratch/users/b/bef299/polypharmacy_project_fhd8SDd3U50/")
PROJECT_PATH = Path("~/bmif204/bmif204_claims_project/").expanduser()
(PROJECT_PATH / "queries").mkdir(parents=True, exist_ok=True)

OUT_RX = SCRATCH_PATH / f"rx_fills_opioid{suffix}.parquet"
OUT_AE = SCRATCH_PATH / f"adverse_events_opioid{suffix}.parquet"
OUT_DEMO = SCRATCH_PATH / f"demographics_opioid{suffix}.parquet"
OUT_ENROLL = SCRATCH_PATH / f"enrollment_opioid{suffix}.parquet"

# ------------------------------
# LOAD ICD10 CSV (for AE filter)
# ------------------------------
ICD_CSV_PATH = PROJECT_PATH / "data/icd10_codes.csv"
icd_df = pd.read_csv(ICD_CSV_PATH, dtype=str)

code_conditions = []
for code in icd_df["code"]:
    if pd.isna(code):
        continue
    code = code.strip().replace(".", "")
    if "%" not in code:
        code = code + "%"
    code_conditions.append(f"cc.CodeValue LIKE '{code}'")
icd_where_clause = " OR ".join(code_conditions)


# ------------------------------
# HELPERS
# ------------------------------
def run_and_save(query: str, out_path: Path):
    print(f"Running query for {out_path.name} ...")
    df = pd.read_sql(query, conn)
    print(f"  Retrieved {len(df):,} rows")
    df.to_parquet(out_path)
    print(f"  Saved to {out_path}")

def write_sql(query: str, out_path: Path):
    out_path.write_text(query)
    print(f"Wrote SQL to {out_path}")

def run_rx_batched(out_path: Path, batch_size: int = 200_000):
    """
    Run the Rx query in batches over elegible_members, save temporary parquet parts,
    then merge into a single parquet at `out_path`.
    """
    rx_base_sql = f"""
    WITH em AS (
        SELECT 
            MemberUID,
            ROW_NUMBER() OVER (ORDER BY MemberUID) AS rn
        FROM bef299.dbo.elegible_members_{args.db}M
    )
    SELECT 
        r.MemberUID, 
        r.filldate, 
        r.ndc11code, 
        r.supplydayscount
    FROM [RxClaim] r
    JOIN em ON r.MemberUID = em.MemberUID
    WHERE em.rn BETWEEN ? AND ?;
    """

    # Save SQL for record-keeping
    write_sql(rx_base_sql, PROJECT_PATH / "queries/rx_opioid_combo_query_batched.sql")

    part_paths = []
    start = 1
    batch_idx = 0

    while True:
        end = start + batch_size - 1
        print(f"Running RX batch {batch_idx} (rn {start}–{end}) ...")

        df = pd.read_sql(rx_base_sql, conn, params=(start, end))

        if df.empty:
            print("No more rows for RX; stopping batching.")
            break

        part_path = out_path.with_name(
            out_path.name.replace(".parquet", f"_part{batch_idx}.parquet")
        )
        print(f"  Retrieved {len(df):,} rows, saving to {part_path.name}")
        df.to_parquet(part_path)
        part_paths.append(part_path)

        batch_idx += 1
        start = end + 1

    if not part_paths:
        print("No RX data found; no parquet created.")
        return

    # Merge parts into a single parquet
    print(f"Merging {len(part_paths)} RX parquet parts into {out_path.name} ...")
    dfs = [pd.read_parquet(p) for p in part_paths]
    full_df = pd.concat(dfs, ignore_index=True)
    full_df.to_parquet(out_path)
    print(f"  Final RX file saved to {out_path} with {len(full_df):,} rows")

    # Clean up part files
    for p in part_paths:
        p.unlink()
    print("  Temporary RX parquet parts removed.")

# Now run the simpler downstream queries
queries = {
    "rx": (f"SELECT r.MemberUID, r.filldate, r.ndc11code, r.supplydayscount "
           f"FROM [RxClaim] r JOIN bef299.dbo.elegible_members_{args.db}M em ON r.MemberUID = em.MemberUID", OUT_RX),
    "ae": (f"SELECT c.MemberUID, cc.ServiceDate AS event_date, cc.CodeType, cc.CodeValue "
           f"FROM [Claim] c JOIN [ClaimCode] cc ON c.ClaimUID = cc.ClaimUID "
           f"JOIN bef299.dbo.elegible_members_{args.db}M em ON c.MemberUID = em.MemberUID "
           f"WHERE cc.CodeType IN (17,18,22,23,24) AND ({icd_where_clause})", OUT_AE),
    "demo": (f"SELECT m.MemberUID, m.birthyear, m.gendercode, m.raceethnicitytypecode, m.zip3value, m.statecode "
             f"FROM [Member] m JOIN bef299.dbo.elegible_members_{args.db}M em ON m.MemberUID = em.MemberUID", OUT_DEMO),
    "enroll": (f"SELECT e.MemberUID, e.effectivedate, e.terminationdate "
               f"FROM [MemberEnrollment] e JOIN bef299.dbo.elegible_members_{args.db}M em ON e.MemberUID = em.MemberUID", OUT_ENROLL)
}

for name, (sql, out_path) in queries.items():
    if name == "rx":
        # Special batched handling for RX, with merge to a single parquet
        run_rx_batched(out_path, batch_size=200_000)
    else:
        # One-shot query for all other tables
        write_sql(sql, PROJECT_PATH / f"queries/{name}_opioid_combo_query.sql")
        run_and_save(sql, out_path)
