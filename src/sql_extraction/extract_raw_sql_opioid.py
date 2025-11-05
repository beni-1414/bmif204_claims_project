import pyodbc
import pandas as pd
from pathlib import Path
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Extract raw SQL data for opioid study")

    parser.add_argument("--suffix", type=str, default="_sample1M",
                        help="Suffix used in filenames (e.g., '_sample1M' or '')")
    parser.add_argument("--db", type=str, default="InovalonSample1M",
                        help="Database name to connect to (e.g., 'InovalonSample5M' or 'InovalonSample1M')")


    args = parser.parse_args()
    return args

# ------------------------------
# DB CONNECTION
# ------------------------------
args = parse_args()
suffix = args.suffix
database = args.db
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
# LOAD OPIOID NDC LIST
# ------------------------------
opioid_ndcs_path = Path("data/opioid_ndc11_list.csv")
opioid_df = pd.read_csv(opioid_ndcs_path, dtype=str)
opioid_ndcs = opioid_df["ndc11"].dropna().unique().tolist()

# Build SQL-safe IN clause (chunked for SQL Server)
def chunked(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]

opioid_in_clauses = []
for chunk in chunked(opioid_ndcs, 900):
    vals = ",".join(f"'{x}'" for x in chunk)
    opioid_in_clauses.append(f"m.ndc11code IN ({vals})")
opioid_where = " OR ".join(opioid_in_clauses)

# ------------------------------
# CTE BLOCK
# ------------------------------
base_cte_create = f"""
WITH member_drugs AS (
    SELECT 
        r.MemberUID,
        r.ndc11code,
        MIN(r.filldate) AS first_fill_date
    FROM [RxClaim] r
    WHERE r.supplydayscount IS NOT NULL
    GROUP BY r.MemberUID, r.ndc11code
),
opioid_flags AS (
    SELECT 
        m.MemberUID,
        MAX(CASE WHEN {opioid_where} THEN 1 ELSE 0 END) AS has_opioid,
        COUNT(DISTINCT m.ndc11code) AS total_drugs
    FROM member_drugs m
    GROUP BY m.MemberUID
)
SELECT 
    o.MemberUID
INTO #eligible_members
FROM opioid_flags o
WHERE o.has_opioid = 1
  AND o.total_drugs >= 3;
"""


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

# Build cohort once
print("Building eligible_members temporary table...")
conn.execute(base_cte_create)

# Now run the simpler downstream queries
queries = {
    "rx": ("SELECT r.MemberUID, r.filldate, r.ndc11code, r.supplydayscount "
           "FROM [RxClaim] r JOIN #eligible_members em ON r.MemberUID = em.MemberUID", OUT_RX),
    "ae": ("SELECT c.MemberUID, cc.ServiceDate AS event_date, cc.CodeType, cc.CodeValue "
           "FROM [Claim] c JOIN [ClaimCode] cc ON c.ClaimUID = cc.ClaimUID "
           "JOIN #eligible_members em ON c.MemberUID = em.MemberUID "
           f"WHERE cc.CodeType IN (17,18,22,23,24) AND ({icd_where_clause})", OUT_AE),
    "demo": ("SELECT m.MemberUID, m.birthyear, m.gendercode, m.raceethnicitytypecode, m.zip3value, m.statecode "
             "FROM [Member] m JOIN #eligible_members em ON m.MemberUID = em.MemberUID", OUT_DEMO),
    "enroll": ("SELECT e.MemberUID, e.effectivedate, e.terminationdate "
               "FROM [MemberEnrollment] e JOIN #eligible_members em ON e.MemberUID = em.MemberUID", OUT_ENROLL)
}

for name, (sql, out_path) in queries.items():
    write_sql(sql, PROJECT_PATH / f"queries/{name}_opioid_combo_query.sql")
    run_and_save(sql, out_path)

