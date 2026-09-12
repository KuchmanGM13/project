import os

import psycopg
from dotenv import load_dotenv
from matching.service import match_candidates


load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

VACANCY_IDS = [2, 3, 5, 6, 8]


with psycopg.connect(DATABASE_URL) as conn:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, title, category, required_skills
            FROM vacancies
            WHERE id = ANY(%s)
            ORDER BY id
            """,
            (VACANCY_IDS,),
        )

        vacancies = cur.fetchall()

def calculate_skill_coverage(
    matching_skills: list[str],
    required_skills: list[str],
) -> float:
    if not required_skills:
        return 0.0

    return len(matching_skills) / len(required_skills)

vacancy_coverages = []
full_match_vacancies = 0

for vacancy in vacancies:
    vacancy_id, title, category, required_skills = vacancy

    results = match_candidates(
        vacancy_id=vacancy_id,
        k=3,
    )

    coverages = []

    print(f"Vacancy: {title} | {category}")
    print(f"Required skills: {required_skills}")
    print()

    for rank, candidate in enumerate(results, start=1):
        coverage = calculate_skill_coverage(
            candidate.matching_skills,
            required_skills,
        )

        coverages.append(coverage)

        print(
            f"{rank}. CV {candidate.resume_id} | "
            f"similarity={candidate.similarity:.3f} | "
            f"score={candidate.match_score}"
        )
        print(f"   Matching: {candidate.matching_skills}")
        print(f"   Skill coverage: {coverage:.0%}")
        print()

    average_coverage = sum(coverages) / len(coverages)

    vacancy_coverages.append(average_coverage)

    if max(coverages) == 1.0:
        full_match_vacancies += 1

    print(f"Average top-3 skill coverage: {average_coverage:.1%}")
    print()

overall_coverage = sum(vacancy_coverages) / len(vacancy_coverages)

print(
    f"Overall average top-3 skill coverage: "
    f"{overall_coverage:.1%}"
)

full_match_rate = full_match_vacancies / len(vacancies)

print(
    f"Top-3 full-match rate: "
    f"{full_match_rate:.1%}"
)