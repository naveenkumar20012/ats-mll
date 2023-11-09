import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

load_dotenv("./constants.env")
sys.path.append(str(Path(__file__).parent.parent))

from fastapi import FastAPI, File, Form, Response, UploadFile, status

from helper_codes.inference_ui import inference_ui
from utils.logging_helpers import info_logger
from utils.cloud_utils import upload_blob_from_memory
from utils.db_models import JDescriptions as ModelJDescriptions
from utils.db_utils import get_db_session

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
KEY_PATH = os.getenv("KEY_PATH")
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = KEY_PATH
os.environ["AWS_ACCESS_KEY_ID"] = os.getenv("AWS_ACCESS_KEY_ID")
os.environ["AWS_SECRET_ACCESS_KEY"] = os.getenv("AWS_SECRET_ACCESS_KEY")

DESTINATION_FOLDER = os.getenv("DESTINATION_FOLDER")
DELETE_AFTER_INFERENCE = os.getenv("DELETE_AFTER_INFERENCE")


class TextData(BaseModel):
    text: list


@app.get("/")
def home():
    """
    A dictionary to be shown on Home Page for verification of FastAPI functionality.
    """
    return {"data": {"Page:": "I am JD_parser service, alive 😎"}}


# JD parser Main
@app.post("/upload", status_code=200)
def jd_parser(
    response: Response,
    file: UploadFile = File(...),
    filename: Optional[str] = Form("placeholder"),
    source: Optional[str] = Form("LOCAL"),
):
    """

    :param response: API response object
    :param file: jd file
    :param filename: File name of JD
    :param source: source of file
    :return: json of jd data
    """
    t1 = time.time()
    # If it's a gdrive upload, filename will be supplied in the form field
    if source != "GDRIVE":
        filename = file.filename

    filepath = DESTINATION_FOLDER + filename
    info_logger(f"Received request to process {filename}")
    try:
        with open(filepath, "wb") as jd:
            shutil.copyfileobj(file.file, jd)
        cloud_filepath = upload_blob_from_memory(filepath)
    except:
        cloud_filepath = None
        response.status_code = status.HTTP_400_BAD_REQUEST
        return {"error": "Unable to process the file"}

    result = inference_ui(filepath)
    info_logger("Predictions for", filename, json.dumps(result))

    if DELETE_AFTER_INFERENCE:
        if filename.endswith(".doc"):
            os.remove(filepath)
            docx_filepath = filepath[:-4] + filepath[-4:].replace(
                ".doc", ".docx"
            )
            os.remove(docx_filepath)
        else:
            os.remove(filepath)
    t2 = time.time()
    info_logger(f"Time Taken to predict {filename}: {t2 - t1}")
    result["taken_time"] = t2 - t1
    result["jd_file_path"] = cloud_filepath
    job_obj = ModelJDescriptions(
        gcp_filepath=cloud_filepath, json_output=result
    )
    Session = get_db_session()
    with Session() as session:
        session.add(job_obj)
        session.commit()

    return result
