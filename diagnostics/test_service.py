from matching.service import match_candidates


VACANCY_ID = 2
TOP_K = 10


results = match_candidates(
    vacancy_id=VACANCY_ID,
    k=TOP_K,
)


print(f"Retrieved and evaluated: {len(results)} candidates")
print()

for rank, candidate in enumerate(results, start=1):
    print(
        f"{rank}. CV {candidate.resume_id} | "
        f"similarity={candidate.similarity:.3f} | "
        f"score={candidate.match_score}"
    )
    print(f"   Matching: {candidate.matching_skills}")
    print(f"   Missing: {candidate.missing_skills}")
    print(f"   Verdict: {candidate.verdict}")
    print()