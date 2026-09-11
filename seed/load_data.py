import os

import pandas as pd
import psycopg
from dotenv import load_dotenv
from pgvector.psycopg import register_vector
from common.embeddings import embed_text

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(BASE_DIR, "data", "Resume.csv")

ALLOWED_CATEGORIES = {
    "INFORMATION-TECHNOLOGY",
    "ENGINEERING",
    "FINANCE",
}


df = pd.read_csv(CSV_PATH)

df = df[df["Category"].isin(ALLOWED_CATEGORIES)]
df = df.dropna(subset=["Resume_str"])
df = df[df["Resume_str"].str.strip() != ""]
df = df.drop_duplicates(subset=["Resume_str"])

sample_text = df.iloc[0]["Resume_str"]

embedding = embed_text(sample_text)

print("Embedding shape:", embedding.shape)
print("Embedding norm:", (embedding ** 2).sum() ** 0.5)

print(f"Resumes to load: {len(df)}")
print(df["Category"].value_counts())


with psycopg.connect(DATABASE_URL) as conn:
    register_vector(conn)

    with conn.cursor() as cur:
        for index, row in df.iterrows():
            embedding = embed_text(row["Resume_str"])

            cur.execute(
                """
                INSERT INTO resumes (source_id, category, raw_text, embedding)
                VALUES (%s, %s, %s, %s)
                """,
                (
                    str(row["ID"]),
                    row["Category"],
                    row["Resume_str"],
                    embedding,
                ),
            )

            print(f"Inserted {index + 1}/{len(df)}")

    conn.commit()

    print("All resumes inserted.")