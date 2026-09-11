import json
import os

import psycopg

from pgvector.psycopg import register_vector
from dotenv import load_dotenv
from common.embeddings import embed_text

load_dotenv()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VACANCIES_PATH = os.path.join(BASE_DIR, "data", "vacancies.json")


with open(VACANCIES_PATH, "r", encoding="utf-8") as f:
    vacancies = json.load(f)


print(f"Vacancies to load: {len(vacancies)}")

for vacancy in vacancies:
    print(vacancy["title"])

with psycopg.connect(os.getenv("DATABASE_URL")) as conn:
    register_vector(conn)

    with conn.cursor() as cur:
        for index, vacancy in enumerate(vacancies, start=1):
            embedding = embed_text(vacancy["raw_text"])

            cur.execute(
                """
                INSERT INTO vacancies
                    (title, category, raw_text, required_skills, embedding)
                VALUES
                    (%s, %s, %s, %s, %s)
                """,
                (
                    vacancy["title"],
                    vacancy["category"],
                    vacancy["raw_text"],
                    vacancy["required_skills"],
                    embedding,
                ),
            )

            print(f"Inserted {index}/{len(vacancies)}")

    conn.commit()

    print("All vacancies inserted.")