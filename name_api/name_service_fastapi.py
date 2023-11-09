import re

from fastapi import FastAPI
from flair.data import Sentence
from flair.models import SequenceTagger
from pydantic import BaseModel

app = FastAPI()

tagger = SequenceTagger.load("./ner-english-large/ner-english-large")
TEXT_SLICING = 400


class TextData(BaseModel):
    text: str


@app.get("/")
def home():
    """
    A dictionary to be shown on Home Page for verification of FastAPI functionality.
    """
    return {"data": {"Page:": "I am name_parser service, alive 😎"}}


@app.post("/name_extract")
async def extract_name(Text: TextData):
    """
    Extracts names from the text given.
    Args:
        Text : str. Text extracted from resume or personal_details section.

    Returns:
        Name, if present. Else, None.
    """
    text = Text.dict()["text"]
    text = text[: int(TEXT_SLICING)]
    sentence = Sentence(text)

    # predict NER tags
    tagger.predict(sentence)

    # PERSON entities to be extracted
    for entity in sentence.get_spans("ner"):
        if entity.tag == "PER":
            if len(re.sub(r"[^A-Za-z]+", "", str(entity.text))) <= 2:
                continue
            return {"name": str(entity.text)}
    return {"name": ""}
