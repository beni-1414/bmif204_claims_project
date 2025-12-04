import argparse
from pathlib import Path

import pandas as pd
import pyodbc


def parse_args():
    parser = argparse.ArgumentParser(
        description="Load NDC11 codes into bef299.dbo.QueryDrugs for opioid study"
    )

    parser.add_argument(
        "--db-user",
        type=str,
        required=True,
        help="Database name to connect to (e.g., 'InovalonSample1M').",
    )

    parser.add_argument(
        "--query-drugs-path",
        type=Path,
        default=Path("data/opioid_ndc11_list.csv"),
        help="Path to CSV file containing an 'ndc11' column.",
    )

    return parser.parse_args()


def chunked(lst, n):
    """Yield successive n-sized chunks from a list."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def load_query_drugs(db_user: str, query_drugs_path: Path):
    """
    Load NDC11 codes from a CSV into {db_user}.dbo.QueryDrugs in the specified database.
    """
    # Connection string: connect directly to the provided database
    conn_str = (
        "DRIVER=ODBC Driver 17 for SQL Server;"
        "Server=CCBWSQLP01.med.harvard.edu;"
        "Trusted_Connection=Yes;"
        f"Database={db_user};"
        "TDS_Version=8.0;Encryption=require;Port=1433;REALM=MED.HARVARD.EDU"
    )

    # Read NDC list
    query_ndcs_df = pd.read_csv(query_drugs_path, dtype=str)
    query_ndcs = query_ndcs_df["ndc11"].dropna().unique().tolist()

    # Build insert queries (max ~1000 rows per INSERT; use 900 for safety)
    ndc_insert_chunks = []
    for chunk in chunked(query_ndcs, 900):
        values = ",".join(f"('{x}')" for x in chunk)
        ndc_insert_chunks.append(
            f"INSERT INTO {db_user}.dbo.QueryDrugs (ndc11code) VALUES {values};"
        )

    conn = pyodbc.connect(conn_str)
    with conn.cursor() as cursor:
        # Create table if needed
        try:
            cursor.execute(
                f"""
                CREATE TABLE {db_user}.dbo.QueryDrugs (
                    ndc11code VARCHAR(11) NOT NULL PRIMARY KEY
                );
                """
            )
        except Exception:
            # Table probably already exists
            pass

        conn.commit()

        # Clear existing table
        cursor.execute(f"TRUNCATE TABLE {db_user}.dbo.QueryDrugs;")
        conn.commit()

        # Insert new values
        for insert_query in ndc_insert_chunks:
            cursor.execute(insert_query)
            conn.commit()

    conn.close()
    print(
        f"Inserted {len(query_ndcs)} NDC11 codes into {db_user}.dbo.QueryDrugs "
        f"in database {db_user}."
    )


def main():
    args = parse_args()
    load_query_drugs(db_user=args.db_user, query_drugs_path=args.query_drugs_path)


if __name__ == "__main__":
    main()
