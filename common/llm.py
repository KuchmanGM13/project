from unittest import result

from openai import OpenAI
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from pydantic import BaseModel


class CandidateMatch(BaseModel):
    resume_id: int
    similarity: float
    match_score: int
    matching_skills: list[str]
    missing_skills: list[str]
    verdict: str


load_dotenv()

class LLMMatchResult(BaseModel):
    experience_relevance: int = Field(ge=0, le=100)
    matching_skills: list[str]
    missing_skills: list[str]
    verdict: str


class MatchResult(BaseModel):
    match_score: int
    experience_relevance: int
    matching_skills: list[str]
    missing_skills: list[str]
    verdict: str


client = OpenAI()

def validate_skills(
    matching_skills: list[str],
    required_skills: list[str],
) -> tuple[list[str], list[str]]:
    required_map = {
        skill.lower(): skill
        for skill in required_skills
    }

    matching = [
        required_map[skill.lower()]
        for skill in matching_skills
        if skill.lower() in required_map
    ]

    missing = [
        skill
        for skill in required_skills
        if skill.lower() not in {skill.lower() for skill in matching}
    ]

    return matching, missing

def evaluate_candidate(
    vacancy_title: str,
    required_skills: list[str],
    vacancy_text: str,
    resume_text: str,
) -> MatchResult:

    response = client.responses.parse(
        model="gpt-4o-mini",
        input=[
            {
                "role": "system",
                "content": (
                    "You are a recruitment assistant. "
                    "Evaluate how well a candidate matches a job vacancy. "
                    "Base your evaluation only on the provided vacancy and resume. "
                    "Do not assume skills or experience that are not supported by the resume. "
                    "matching_skills must contain only skills from the provided required_skills list. "
                    "missing_skills must contain only skills from the provided required_skills list. "
                    "Do not treat related or broader skills as equivalent unless the resume clearly "
                    "demonstrates the exact required skill. For example, SQL does not automatically "
                    "mean PostgreSQL, and Python does not automatically mean FastAPI. "
                    "Use the following scoring scale consistently: "
                    "Evaluate the relevance and strength of the candidate's experience "
                    "for this vacancy on a scale from 0 to 100. "
                    "Consider how closely the candidate's previous roles, responsibilities, "
                    "and technical experience relate to the vacancy. "
                    "Do not base this score only on the number of matching skills. "
                    "Do not include any numerical score or percentage in the verdict. "
                    "The verdict should only provide a concise qualitative explanation "
                    "of the candidate's strengths and gaps."   
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Vacancy title: {vacancy_title}\n\n"
                    f"Required skills: {required_skills}\n\n"
                    f"Vacancy description:\n{vacancy_text}\n\n"
                    f"Candidate resume:\n{resume_text}"
                ),
            },
        ],
        text_format=LLMMatchResult,
    )

    result = response.output_parsed

    matching_skills, missing_skills = validate_skills(
        result.matching_skills,
        required_skills,
    )
    
    skill_coverage = (
        len(matching_skills) / len(required_skills)
        if required_skills
        else 0
    )

    match_score = round(
        skill_coverage * 40
        + result.experience_relevance * 0.6
    )

    return MatchResult(
    match_score=match_score,
    experience_relevance=result.experience_relevance,
    matching_skills=matching_skills,
    missing_skills=missing_skills,
    verdict=result.verdict,
)