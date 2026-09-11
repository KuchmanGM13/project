import os

import psycopg
from dotenv import load_dotenv


load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

with psycopg.connect(DATABASE_URL) as conn:
    with conn.cursor() as cur:

        cur.execute("""
            SELECT extname
            FROM pg_extension
            ORDER BY extname;
        """)

        print("Extensions:")
        for extension in cur.fetchall():
            print("-", extension[0])

        cur.execute("""
            SELECT
                column_name,
                data_type
            FROM information_schema.columns
            WHERE table_schema = 'public'
              AND table_name = 'resumes_staging'
            ORDER BY ordinal_position;
        """)

        print("Columns:")
        for column in cur.fetchall():
            print("-", column[0], "|", column[1])