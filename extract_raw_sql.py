import pyodbc
import pandas as pd
from pathlib import Path

# ------------------------------
# CONFIG
# ------------------------------
SCRATCH_PATH = Path(" /n/scratch/users/b/bef299/polypharmacy_project/")
PROJECT_PATH = Path("~/bmif204/bmif204_claims_project/")
ICD_CSV_PATH = PROJECT_PATH / "icd10_codes.csv"
OUT_RX = SCRATCH_PATH / "rx_fills.parquet"
OUT_AE = SCRATCH_PATH / "adverse_events.parquet"
OUT_DEMO = SCRATCH_PATH / "demographics.parquet"
OUT_ENROLL = SCRATCH_PATH / "enrollment.parquet"

# ------------------------------
# DB CONNECTION
# ------------------------------
conn_str = (
    "DRIVER={ODBC Driver 17 for SQL Server};"
    "SERVER=<your_server_name>;"
    "DATABASE=InovalonSample1M;"
    "Trusted_Connection=yes;"
)
conn = pyodbc.connect(conn_str)

# ------------------------------
# LOAD ICD10 CSV & BUILD CONDITIONS
# ------------------------------
icd_df = pd.read_csv(ICD_CSV_PATH)

# Build OR conditions like: (cc.CodeValue LIKE 'T36%' OR cc.CodeValue LIKE 'R296' ...)
code_conditions = []
for code in icd_df['code']:
    code = code.strip().replace('.', '')
    if '%' not in code:
        code = code + '%'   # ensure we match children too
    code_conditions.append(f"cc.CodeValue LIKE '{code}'")

icd_where_clause = " OR ".join(code_conditions)

# ------------------------------
# BASE CTE BLOCK
# ------------------------------
base_cte = f"""
WITH first_fills AS (
    SELECT 
        r.MemberUID,
        r.ndc11code,
        MIN(r.filldate) AS first_fill_date
    FROM [InovalonSample1M].[dbo].[RxClaim] r
    WHERE r.supplydayscount IS NOT NULL
    GROUP BY r.MemberUID, r.ndc11code
),
ranked_drugs AS (
    SELECT 
        MemberUID,
        ndc11code,
        first_fill_date,
        ROW_NUMBER() OVER (
            PARTITION BY MemberUID ORDER BY first_fill_date
        ) AS drug_rank
    FROM first_fills
),
polypharmacy_index AS (
    SELECT 
        MemberUID,
        first_fill_date AS index_date
    FROM ranked_drugs
    WHERE drug_rank = 5
),
eligible_members AS (
    SELECT 
        p.MemberUID,
        p.index_date,
        m.birthyear,
        (YEAR(p.index_date) - m.birthyear) AS age_at_index
    FROM polypharmacy_index p
    JOIN [InovalonSample1M].[dbo].[Member] m
      ON p.MemberUID = m.MemberUID
    WHERE (YEAR(p.index_date) - m.birthyear) >= 65
),
rx_filtered AS (
    SELECT 
        r.MemberUID,
        r.filldate,
        r.ndc11code,
        r.supplydayscount
    FROM [InovalonSample1M].[dbo].[RxClaim] r
    JOIN eligible_members em
      ON r.MemberUID = em.MemberUID
    WHERE r.supplydayscount IS NOT NULL
      AND r.filldate BETWEEN DATEADD(DAY, -180, em.index_date) AND DATEADD(YEAR, 1, em.index_date)
),
adverse_events AS (
    SELECT 
        c.MemberUID,
        cc.ServiceDate AS event_date,
        cc.CodeType,
        cc.CodeValue
    FROM [InovalonSample1M].[dbo].[Claim] c
    JOIN [InovalonSample1M].[dbo].[ClaimCode] cc
       ON c.ClaimUID = cc.ClaimUID
    JOIN eligible_members em
       ON c.MemberUID = em.MemberUID
    WHERE cc.CodeType IN (17,18,22,23,24)  -- ICD10 only
      AND ({icd_where_clause})
)
"""

# ------------------------------
# EXECUTE QUERIES AND SAVE TO PARQUET
# ------------------------------

def run_and_save(query: str, out_path: Path):
    print(f"Running query for {out_path.name} ...")
    df = pd.read_sql(query, conn)
    print(f"  Retrieved {len(df):,} rows")
    df.to_parquet(out_path)
    print(f"  Saved to {out_path}")

# RX fills
run_and_save(base_cte + " SELECT * FROM rx_filtered", OUT_RX)

# Adverse events (ICD10 filtered dynamically)
run_and_save(base_cte + " SELECT * FROM adverse_events", OUT_AE)

# Demographics
demo_query = base_cte + """
SELECT m.MemberUID, m.birthyear, m.gender, m.race, m.zip, m.state
FROM [InovalonSample1M].[dbo].[Member] m
JOIN eligible_members em ON m.MemberUID = em.MemberUID
"""
run_and_save(demo_query, OUT_DEMO)

# Enrollment
enroll_query = base_cte + """
SELECT e.MemberUID, e.effectivedate, e.terminationdate
FROM [InovalonSample1M].[dbo].[MemberEnrollment] e
JOIN eligible_members em ON e.MemberUID = em.MemberUID
"""
run_and_save(enroll_query, OUT_ENROLL)

# Optional: save a small CSV subset for debugging
subset = pd.read_parquet(OUT_RX).sample(frac=0.001, random_state=42)
subset.to_csv(SCRATCH_PATH / "rx_fills_subset.csv", index=False)
print("Saved small CSV subset for development.")
