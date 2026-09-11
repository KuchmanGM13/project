import os
import re

import psycopg
from dotenv import load_dotenv
from pgvector.psycopg import register_vector


load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

VACANCY_ID = 2
TOP_K = 10


def skill_in_text(skill, text):
    pattern = r"\b" + re.escape(skill.lower()) + r"\b"
    return re.search(pattern, text.lower()) is not None


with psycopg.connect(DATABASE_URL) as conn:
    register_vector(conn)

    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT title, required_skills, embedding
            FROM vacancies
            WHERE id = %s
            """,
            (VACANCY_ID,),
        )

        vacancy = cur.fetchone()

        title, required_skills, vacancy_embedding = vacancy

        cur.execute(
            """
            SELECT
                id,
                source_id,
                category,
                raw_text,
                1 - (embedding <=> %s) AS similarity
            FROM resumes
            ORDER BY embedding <=> %s
            LIMIT %s
            """,
            (vacancy_embedding, vacancy_embedding, TOP_K),
        )

        candidates = cur.fetchall()


print(f"Vacancy: {title}")
print(f"Required skills: {required_skills}")
print()

for rank, candidate in enumerate(candidates, start=1):
    resume_id, source_id, category, raw_text, similarity = candidate

    matching_skills = [
        skill
        for skill in required_skills
        if skill_in_text(skill, raw_text)
    ]

    print(
        f"{rank}. CV {resume_id} | "
        f"{category} | "
        f"similarity={similarity:.3f}"
    )
    print(f"   Matching skills: {matching_skills}")
    print(
        f"   Skill coverage: "
        f"{len(matching_skills)}/{len(required_skills)}"
    )
    print()