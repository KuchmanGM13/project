import os

import psycopg
from dotenv import load_dotenv

from common.llm import evaluate_candidate


load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

VACANCY_ID = 2
RESUME_ID = 76


with psycopg.connect(DATABASE_URL) as conn:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT title, required_skills, raw_text
            FROM vacancies
            WHERE id = %s
            """,
            (VACANCY_ID,),
        )

        title, required_skills, vacancy_text = cur.fetchone()

        cur.execute(
            """
            SELECT raw_text
            FROM resumes
            WHERE id = %s
            """,
            (RESUME_ID,),
        )

        (resume_text,) = cur.fetchone()


result = evaluate_candidate(
    vacancy_title=title,
    required_skills=required_skills,
    vacancy_text=vacancy_text,
    resume_text=resume_text,
)

print(result.model_dump_json(indent=2))