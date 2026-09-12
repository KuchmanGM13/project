import os

import psycopg
from dotenv import load_dotenv
from pgvector.psycopg import register_vector
from pydantic import BaseModel
from common.llm import evaluate_candidate


load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

class CandidateMatch(BaseModel):
    resume_id: int
    similarity: float
    match_score: int
    matching_skills: list[str]
    missing_skills: list[str]
    verdict: str

def match_candidates(vacancy_id: int, k: int = 10) -> list[CandidateMatch]:
    with psycopg.connect(DATABASE_URL) as conn:
        register_vector(conn)

        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT title, required_skills, raw_text, embedding
                FROM vacancies
                WHERE id = %s
                """,
                (vacancy_id,),
            )

            vacancy = cur.fetchone()

            if vacancy is None:
                raise ValueError(f"Vacancy {vacancy_id} not found")

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
                (vacancy_embedding, vacancy_embedding, k),
            )

            candidates = cur.fetchall()

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
            CandidateMatch(
                resume_id=resume_id,
                similarity=float(similarity),
                match_score=result.match_score,
                matching_skills=result.matching_skills,
                missing_skills=result.missing_skills,
                verdict=result.verdict,
            )
        )

    results.sort(
        key=lambda item: item.match_score,
        reverse=True,
    )
    
    return results