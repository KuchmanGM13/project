import json
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel


class VacancyText(BaseModel):
    raw_text: str

load_dotenv()

client = OpenAI()

BASE_DIR = Path(__file__).resolve().parent.parent
VACANCIES_PATH = BASE_DIR / "data" / "vacancies.json"


with open(VACANCIES_PATH, "r", encoding="utf-8") as f:
    vacancies = json.load(f)

print(f"Loaded {len(vacancies)} vacancies")

vacancy = vacancies[0]

response = client.responses.parse(
    model="gpt-4o-mini",
    input=[
        {
            "role": "system",
            "content": (
                "You generate realistic job vacancy descriptions. "
                "The provided title, category, and required skills are fixed. "
                "Do not change or remove the required skills. "
                "Write a natural, professional vacancy description in English."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Title: {vacancy['title']}\n"
                f"Category: {vacancy['category']}\n"
                f"Required skills: {', '.join(vacancy['required_skills'])}"
            ),
        },
    ],
    text_format=VacancyText,
)

result = response.output_parsed

print(result.raw_text)