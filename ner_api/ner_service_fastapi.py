import os
import spacy
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()
os.environ["TOKENIZERS_PARALLELISM"] = "false"

TEXT_SLICING = 400
V5_MODEL_PATH = r"./ner_output_5/model-best"
nlp = spacy.load(V5_MODEL_PATH)


class TextData(BaseModel):
    text: str


@app.get("/")
def home():
    """
    A dictionary to be shown on Home Page for verification of FastAPI functionality.
    """
    return {"data": {"Page:": "I am NER service, alive 😎"}}


@app.post("/ner_extract")
async def extract_ner(Text: TextData):
    """
    Extracts ner from the text given.
    Args:
        Text : str. Text extracted from resume.

    Returns:
       All entities find by model
    """
    text = Text.dict()["text"]
    doc = nlp(text, disable=["tagger"])
    ent_data = [(e.start_char, e.end_char, e.text, e.label_) for e in doc.ents]
    return ent_data
