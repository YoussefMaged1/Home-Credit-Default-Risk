import os

import duckdb
from dotenv import load_dotenv

load_dotenv()

MOTHERDUCK_TOKEN = os.getenv("MOTHERDUCK_TOKEN")

con = duckdb.connect(f"md:?motherduck_token={MOTHERDUCK_TOKEN}")
con.execute("CREATE DATABASE IF NOT EXISTS home_credit_db")
con.close()

con = duckdb.connect(f"md:home_credit_db?motherduck_token={MOTHERDUCK_TOKEN}")

tables = {
    "test_data": "data/raw/application_test.csv",
    "bureau": "data/raw/bureau.csv",
    "previous_application": "data/raw/previous_application.csv",
    "installments_payments": "data/raw/installments_payments.csv",
}

for table_name, csv_path in tables.items():
    print(f"Uploading {table_name}...")
    con.execute(f"""
        CREATE OR REPLACE TABLE {table_name} AS
        SELECT * FROM read_csv_auto('{csv_path}')
    """)
    count = con.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
    print(f"✅ {table_name}: {count} rows")

con.close()
