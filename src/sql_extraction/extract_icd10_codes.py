import pandas as pd
from pathlib import Path
from sqlalchemy import create_engine, text
import pyarrow as pa
import pyarrow.parquet as pq
import argparse

def bulk_insert_member_windows(conn, df, batch=50_000):
    """
    Bulk-insert into #member_windows using pyodbc executemany on the SAME connection.
    Works reliably with temp tables; avoids SQLAlchemy 'multi' param packing.
    """
    # Ensure plain Python types (int + date)
    tuples = [
        (int(row.MemberUID), int(row.spell_id), int(row.split_seq), row.window_start, row.window_end)
        for row in df.itertuples(index=False)
    ]
    cur = conn.connection.cursor()
    cur.fast_executemany = True
    sql = "INSERT INTO #member_windows (MemberUID, spell_id, split_seq, window_start, window_end) VALUES (?, ?, ?, ?, ?)"
    for i in range(0, len(tuples), batch):
        chunk = tuples[i:i+batch]
        cur.executemany(sql, chunk)
    cur.close()

# ------------------------------
# CONFIG
# ------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Polypharmacy spell detection + AE labeling")

    parser.add_argument("--suffix", type=str, default="_opioid_sample1M_grace15_minspell7_ae_censoring",
                        help="Suffix used in filenames (e.g., '_sample1M' or '')")
    parser.add_argument("--db", type=int, default=5,
                        help="Database to use: 1=InnovalonSample1M, 5=InovalonSample5M")


    args = parser.parse_args()
    return args

args = parse_args()
SUFFIX = args.suffix
db = f"InovalonSample{args.db}M"

N_BUCKETS = 100               # adjust up/down for batch size (more buckets = smaller batches)
CHUNKSIZE = 200_000          # rows per fetch; tune for your RAM
COMPRESSION = "snappy"

SCRATCH_PATH = Path("/n/scratch/users/b/bef299/polypharmacy_project_fhd8SDd3U50/")
PROJECT_PATH = Path("~/bmif204/bmif204_claims_project/").expanduser()
(PROJECT_PATH / "queries").mkdir(parents=True, exist_ok=True)

SPELLS_PATH = SCRATCH_PATH / f"split_spells{SUFFIX}.parquet"
OUT_PARQUET = SCRATCH_PATH / f"icd10_codes_from_spells{SUFFIX}.parquet"

# ------------------------------
# SQL CONNECT (SQLAlchemy, single connection)
# ------------------------------
odbc = (
    "DRIVER=ODBC Driver 17 for SQL Server;"
    "SERVER=CCBWSQLP01.med.harvard.edu;"
    "Trusted_Connection=Yes;"
    f"DATABASE={db};"
    "TDS_Version=8.0;Encryption=require;Port=1433;REALM=MED.HARVARD.EDU"
)

# Safe ODBC string quoting for SQLAlchemy URL
from urllib.parse import quote_plus
engine = create_engine(f"mssql+pyodbc:///?odbc_connect={quote_plus(odbc)}", pool_pre_ping=True, fast_executemany=True)

# ------------------------------
# LOAD SPELLS & BUILD WINDOWS
# ------------------------------
print(f"Loading spells from {SPELLS_PATH} ...")
spells = pd.read_parquet(SPELLS_PATH)

req = {"MemberUID", "entry_date", "extended_exit_date"}
missing = req - set(spells.columns)
if missing:
    raise ValueError(f"Spells file missing required columns: {missing}")

spells["entry_date"] = pd.to_datetime(spells["entry_date"]).dt.date

# per-spell window: [entry_date - 180d, entry_date)  (end is exclusive)
windows = spells[["MemberUID", "spell_id", "split_seq", "entry_date"]].copy()
windows["window_start"] = (pd.to_datetime(windows["entry_date"]) - pd.Timedelta(days=180)).dt.date
windows["window_end"]  = (pd.to_datetime(windows["entry_date"]) - pd.Timedelta(days=1)).dt.date # Avoid catching the events on the entry_date itself

# clean & types
windows = windows.dropna(subset=["MemberUID", "spell_id", "split_seq", "window_start", "window_end"])
windows["MemberUID"] = windows["MemberUID"].astype("int64")
windows["spell_id"]  = windows["spell_id"].astype("int64")
windows["split_seq"] = windows["split_seq"].astype("int64")

# bucket by MemberUID so spells for a heavy member spread out
windows["bucket"] = windows["MemberUID"] % N_BUCKETS
print(f"Built windows for {len(windows):,} spells across {N_BUCKETS} buckets")

# ------------------------------
# MAIN QUERY (index-friendly)
# ------------------------------
icd10_sql = """
;WITH claims_scoped AS (
    SELECT c.ClaimUID, c.MemberUID, mw.spell_id, mw.split_seq
    FROM dbo.[Claim] AS c
    INNER JOIN #member_windows AS mw
      ON c.MemberUID = mw.MemberUID
     AND c.ServiceDate >= mw.window_start
     AND c.ServiceDate <  DATEADD(day, 1, mw.window_end)
),
codes AS (
    SELECT DISTINCT
        cs.MemberUID,
        cs.spell_id,
        cs.split_seq,
        cc.CodeValue AS icd10_code
    FROM claims_scoped AS cs
    INNER JOIN dbo.[ClaimCode] AS cc
      ON cs.ClaimUID = cc.ClaimUID
    WHERE cc.CodeType IN (17,18,22,23,24)
)
SELECT MemberUID, spell_id, split_seq, icd10_code
FROM codes
OPTION (RECOMPILE);
"""

# ------------------------------
# EXECUTE BY BUCKET -> STREAM TO SINGLE PARQUET
# ------------------------------
OUT_PARQUET.unlink(missing_ok=True)
writer = None
total_rows = 0

with engine.connect() as conn:
    # Loop buckets for reasonably even sizes
    for b in range(N_BUCKETS):
        w = windows.loc[windows["bucket"] == b, ["MemberUID", "spell_id", "split_seq", "window_start", "window_end"]]
        if w.empty:
            continue

        print(f"\n=== Bucket {b+1}/{N_BUCKETS}: {len(w):,} members ===")

        # Fresh temp table per bucket (lives for this connection)
        conn.exec_driver_sql("""
            DROP TABLE IF EXISTS #member_windows;
            CREATE TABLE #member_windows (
                MemberUID BIGINT NOT NULL,
                spell_id BIGINT NOT NULL,
                split_seq BIGINT NOT NULL,
                window_start DATE NOT NULL,
                window_end   DATE NOT NULL,
                PRIMARY KEY (MemberUID, spell_id, split_seq)
            );
            CREATE NONCLUSTERED INDEX ix_mw_dates ON #member_windows(window_start, window_end);
        """)

        # Fast append into temp table using this SAME connection
        bulk_insert_member_windows(conn, w, batch=5_000)
        # Stream out results in chunks and append to a single parquet
        chunks = pd.read_sql_query(text(icd10_sql), conn, chunksize=CHUNKSIZE)

        batch_rows = 0
        for i, chunk in enumerate(chunks, 1):
            # light cleanup
            chunk["icd10_code"] = chunk["icd10_code"].astype(str).str.strip()

            table = pa.Table.from_pandas(chunk, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(OUT_PARQUET, table.schema, compression=COMPRESSION)
            writer.write_table(table)

            n = len(chunk)
            batch_rows += n
            total_rows += n
            print(f"  wrote chunk {i:,} — bucket rows so far: {batch_rows:,} (total {total_rows:,})")

        # optional: clear temp table for this bucket (not strictly needed since next DROP re-creates)
        conn.exec_driver_sql("DROP TABLE IF EXISTS #member_windows;")

# close writer
if writer is not None:
    writer.close()

print(f"\nDone. Wrote {total_rows:,} rows to {OUT_PARQUET}")
