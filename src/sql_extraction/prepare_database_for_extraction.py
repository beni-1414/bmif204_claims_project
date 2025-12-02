import pandas as pd
from pathlib import Path
import argparse
import pyodbc

def parse_args():
    parser = argparse.ArgumentParser(description="Extract raw SQL data for opioid study")

    parser.add_argument("--db", type=int, default=1, help="Database size to connect to (1 for 1M, 5 for 5M)")

    args = parser.parse_args()
    return args

# ------------------------------
# DB CONNECTION
# ------------------------------
args = parse_args()
database = f"InovalonSample{args.db}M"
conn_str = f'DRIVER=ODBC Driver 17 for SQL Server;Server=CCBWSQLP01.med.harvard.edu;Trusted_Connection=Yes;Database={database};TDS_Version=8.0;Encryption=require;Port=1433;REALM=MED.HARVARD.EDU'


def chunked(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]

query_ndcs_path = Path("data/opioid_ndc11_list.csv")
query_ndcs_df = pd.read_csv(query_ndcs_path, dtype=str)
query_ndcs = query_ndcs_df["ndc11"].dropna().unique().tolist()

# Build insert queries for the ndc11 codes (a table QueryDrugs with a single column ndc11code). Max insert size is 1000 rows, chunk accordingly.
ndc_insert_chunks = []
for chunk in chunked(query_ndcs, 900):
    vals = ",".join(f"('{x}')" for x in chunk)
    ndc_insert_chunks.append(f"INSERT INTO bef299.dbo.QueryDrugs (ndc11code) VALUES {vals};")

conn = pyodbc.connect(conn_str)
with conn.cursor() as cursor:
    try:
        cursor.execute("""
            CREATE TABLE bef299.dbo.QueryDrugs (
                ndc11code VARCHAR(11) NOT NULL PRIMARY KEY
            );
        """)
    except Exception as e:
            pass  # Table probably already exists
    conn.commit()
    # Clear existing table
    cursor.execute("TRUNCATE TABLE bef299.dbo.QueryDrugs;")
    conn.commit()
    # Insert new values
    for insert_query in ndc_insert_chunks:
        cursor.execute(insert_query)
        conn.commit()
conn.close()
print(f"Inserted {len(query_ndcs)} NDC11 codes into bef299.dbo.QueryDrugs in database {database}.")