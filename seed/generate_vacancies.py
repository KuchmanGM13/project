import json
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()

client = OpenAI()

BASE_DIR = Path(__file__).resolve().parent.parent
VACANCIES_PATH = BASE_DIR / "data" / "vacancies.json"