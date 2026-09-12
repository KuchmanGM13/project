import os

import psycopg
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from matching.service import CandidateMatch, match_candidates

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")


app = FastAPI(
    title="Vacancy-to-Candidate Matching Service",
)

class Vacancy(BaseModel):
    id: int
    title: str
    category: str

class MatchResponse(BaseModel):
    vacancy_id: int
    candidates: list[CandidateMatch]

@app.get("/vacancies", response_model=list[Vacancy])
def get_vacancies():
    with psycopg.connect(DATABASE_URL) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, title, category
                FROM vacancies
                ORDER BY id
                """
            )

            vacancies = cur.fetchall()

    return [
        {
            "id": vacancy[0],
            "title": vacancy[1],
            "category": vacancy[2],
        }
        for vacancy in vacancies
    ]

@app.get("/match/{vacancy_id}", response_model=MatchResponse)
def match(vacancy_id: int, k: int = 10):
    try:
        candidates = match_candidates(
            vacancy_id=vacancy_id,
            k=k,
        )
    except ValueError:
        raise HTTPException(
            status_code=404,
            detail=f"Vacancy {vacancy_id} not found",
        )

    return MatchResponse(
        vacancy_id=vacancy_id,
        candidates=candidates,
    )