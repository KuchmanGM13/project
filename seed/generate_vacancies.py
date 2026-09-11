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
VACANCIES_PATH = BASE_DIR / "fixtures" / "vacancies.json"
OUTPUT_PATH = BASE_DIR / "data" / "vacancies.json"

with open(VACANCIES_PATH, "r", encoding="utf-8") as f:
    vacancies = json.load(f)

print(f"Loaded {len(vacancies)} vacancies")

generated_vacancies = []

for vacancy in vacancies:
    print()
    print("Generating vacancy description for:")
    print(f"Title: {vacancy['title']}")
    print(f"Category: {vacancy['category']}")
    print(f"Required skills: {', '.join(vacancy['required_skills'])}")

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

    generated_vacancies.append({
    "title": vacancy["title"],
    "category": vacancy["category"],
    "required_skills": vacancy["required_skills"],
    "raw_text": result.raw_text
})

    print(result.raw_text)


with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(generated_vacancies, f, ensure_ascii=False, indent=4)