import os
import spacy
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()
os.environ["TOKENIZERS_PARALLELISM"] = "false"

TEXT_SLICING = 400
V5_MODEL_PATH = r"./ner_output_6/model-best"
nlp = spacy.load(V5_MODEL_PATH)

NER_EXTRACTION_MAPPER = {
    "NAME": "Name",
    "MOBILE_NUMBER": "Mobile_Number",
    "EMAIL": "Email",
    "SKILLS": "Skills",
    "TOTAL_EXPERIENCE": "Total_Experience",
    "CURRENT_POSITION": "Current_Position",
    "CURRENT_COMPANY": "Current_Company",
    "LOCATION": "Location",
    "DEGREE": "Degree",
    "COLLEGE_NAME": "College_Name",
    "EDU_DURATION": "Edu_Duration",
    "MARKS": "Marks",
    "EDU_LOCATION": "Edu_Location",
    "COMPANY_NAME": "Company_Name",
    "POSITION": "Position",
    "EXP_DURATION": "Exp_Duration",
    "EXP_LOCATION": "Exp_Location",
}


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
    ent_data = [
        (
            e.start_char,
            e.end_char,
            e.text,
            NER_EXTRACTION_MAPPER.get(e.label_, e.label_),
        )
        for e in doc.ents
    ]
    return ent_data
