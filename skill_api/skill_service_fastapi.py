import os

import pandas as pd
import spacy
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()
os.environ["TOKENIZERS_PARALLELISM"] = "false"


nlp = spacy.load("en_core_web_lg")
SKILL_CSV_PATH = "./skills.csv"
data = pd.read_csv(SKILL_CSV_PATH)
skills = list(data.columns.values)


class TextData(BaseModel):
    text: str


@app.get("/")
def home():
    """
    A dictionary to be shown on Home Page for verification of FastAPI functionality.
    """
    return {"data": {"Page:": "I am Skill service, alive 😎"}}


@app.post("/skill_extract")
async def extract_ner(Text: TextData):
    """
    Extracts skills from the text given.
    Args:
        Text : str. Text extracted from resume.

    Returns:
       All entities find by model
    """
    text = Text.dict()["text"]
    nlp_text = nlp(text)
    noun_chunks = list(nlp_text.noun_chunks)
    tokens = [token.text for token in nlp_text if not token.is_stop]
    skill_sets = []
    # check for one-grams
    for token in tokens:
        if token.lower() in skills:
            skill_sets.append(token)

    # check for bi-grams and tri-grams
    for token in noun_chunks:
        token = token.text.lower().strip()
        if token in skills:
            skill_sets.append(token)
    return [i for i in set([i.lower() for i in skill_sets])]
