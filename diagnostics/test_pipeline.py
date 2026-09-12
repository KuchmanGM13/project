import os

import psycopg
from dotenv import load_dotenv
from pgvector.psycopg import register_vector

from common.llm import evaluate_candidate


load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

VACANCY_ID = 2
TOP_K = 10


with psycopg.connect(DATABASE_URL) as conn:
    register_vector(conn)

    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT title, required_skills, raw_text, embedding
            FROM vacancies
            WHERE id = %s
            """,
            (VACANCY_ID,),
        )

        vacancy = cur.fetchone()

        title, required_skills, vacancy_text, vacancy_embedding = vacancy

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
print(f"Retrieved candidates: {len(candidates)}")
print()

results = []

for candidate in candidates:
    resume_id, source_id, category, resume_text, similarity = candidate

    result = evaluate_candidate(
        vacancy_title=title,
        required_skills=required_skills,
        vacancy_text=vacancy_text,
        resume_text=resume_text,
    )

    results.append(
        {
            "resume_id": resume_id,
            "similarity": similarity,
            "result": result,
        }
    )

results.sort(
    key=lambda item: item["result"].match_score,
    reverse=True,
)


for item in results:
    result = item["result"]

    print(
        f"CV {item['resume_id']} | "
        f"similarity={item['similarity']:.3f} | "
        f"score={result.match_score}"
    )
    print(f"  Matching: {result.matching_skills}")
    print(f"  Missing: {result.missing_skills}")
    print(f"  Verdict: {result.verdict}")
    print()