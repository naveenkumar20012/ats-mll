from dotenv import load_dotenv

load_dotenv("./constants.env")

import ast
import json
import logging as logger
import os
import time
from fastapi import Body
from fastapi import FastAPI, Response, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from utils.db_models import Scoring as ModelScoring

from celery_tasks.tasks import (
    app as celery_app,
    resume_inference_V3,
    send_result_to_ats_server,
)
from utils.cloud_utils import saving_file_local_resume
from utils.db_models import ReviewedResume
from utils.db_models import Resume as ModelResume
from utils.db_utils import get_db_session, get_or_create_company
from utils.logging_helpers import info_logger, error_logger
from RP_parser.helper_codes.io_utils_2 import single_resume_read
from RP_parser.helper_codes.parser_rules import (
    extract_email_phone,
    chatgpt_parse,
)
from RP_parser.helper_codes.rating_service import (
    calculate_chatgpt_score,
)
from RP_parser.helper_files import constants as cs

Session = get_db_session()
logger.basicConfig(level="INFO")

# JD and RP related contents initialisation
KEY_PATH = os.getenv("KEY_PATH")
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = KEY_PATH
os.environ["AWS_ACCESS_KEY_ID"] = os.getenv("AWS_ACCESS_KEY_ID")
os.environ["AWS_SECRET_ACCESS_KEY"] = os.getenv("AWS_SECRET_ACCESS_KEY")

DESTINATION_FOLDER = os.getenv("DESTINATION_FOLDER")
DELETE_AFTER_INFERENCE = os.getenv("DELETE_AFTER_INFERENCE")

FINAL_DIC_RESUME = ast.literal_eval(os.getenv("final_dict_resume"))

# create instance of FastAPI
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def home():
    """
    A dictionary to be shown on Home Page for verification of FastAPI functionality.
    """
    return {"data": {"Page:": "I am alive 😎"}}


@app.post("/resume/v5", status_code=200)
def pred_resume_post_v5(
    response: Response, request: Request, payload: dict = Body(...)
):
    """
    Main API function of resume parser
    Args:
        response: API object
        payload: Payload of API

    Returns:
        taskid
    """
    start_time = time.time()
    data = payload
    filepath = data.get("filepath")
    bucket_name = data.get("bucket_name")
    company_uuid = data.get("company_uuid")
    job_data = data.get("job_data")
    num_calls_to_third_party = 0
    context_data = {
        "resume_process_id": data.get("resume_process_id"),
        "has_candidate_applied": data.get("has_candidate_applied"),
        "is_job_candidate_resume": data.get("is_job_candidate_resume"),
        "callback_url": data.get("callback_url"),
        "extra_params": data.get("extra_params"),
        "company_uuid": data.get("company_uuid"),
    }

    if company_uuid:
        company_id, num_calls_to_third_party = get_or_create_company(
            company_uuid
        )
    else:
        company_id, num_calls_to_third_party = get_or_create_company(
            "unnamed_company"
        )

    task = resume_inference_V3.delay(
        cloud_filepath=filepath,
        bucket_name=bucket_name,
        company_id=company_id,
        context_data=context_data,
        start_time=start_time,
        async_call=True,
    )
    return {"message": "PROCESSING", "task_id": task.id}


@app.post("/resume/v5_direct", status_code=200)
def pred_resume_post_v5_direct(response: Response, payload: dict = Body(...)):
    """
     API function of resume parser direct
    Args:
        response: API object
        payload: Payload of API

    Returns:
        taskid
    """
    filepath = None
    try:
        start_time = time.time()
        data = payload
        filepath = data.get("filepath")
        bucket_name = data.get("bucket_name")
        company_uuid = data.get("company_uuid")
        context_data = {
            "resume_process_id": data.get("resume_process_id"),
            "company_uuid": data.get("company_uuid"),
        }

        if company_uuid:
            company_id, num_calls_to_third_party = get_or_create_company(
                company_uuid
            )
        else:
            company_id, num_calls_to_third_party = get_or_create_company(
                "unnamed_company"
            )

        result = resume_inference_V3(
            filepath,
            bucket_name,
            company_id,
            context_data,
            start_time,
            async_call=False,
        )
        if result.get("error", False):
            status_msg = "FAILED"
            message = f"Error while parsing resume - {result.get('error')}"
        else:
            status_msg = "PASSED"
            message = "Done"
        data = {
            "data": result,
            "status": status_msg,
            "message": message,
            "force_update": result.get("force_update", False),
        }
        return json.dumps(data)
    except Exception as e:
        response.status_code = status.HTTP_400_BAD_REQUEST
        data = {
            "status": "FAILED",
            "message": f"Unable to process the file {e}",
        }
        error_logger(f"Unable to process the file {filepath}, {e}")
        return json.dumps(data)


@app.post("/resume/v5_test", status_code=200)
def pred_resume_post_v5_test(response: Response, payload: dict = Body(...)):
    """
     API function of resume parser test
    Args:
        response: API object
        payload: Payload of API

    Returns:
        taskid
    """
    start_time = time.time()
    data = payload
    filepath = data.get("filepath")
    bucket_name = data.get("bucket_name")
    company_uuid = data.get("company_uuid")
    context_data = {
        "resume_process_id": data.get("resume_process_id"),
        "has_candidate_applied": data.get("has_candidate_applied"),
        "is_job_candidate_resume": data.get("is_job_candidate_resume"),
        "callback_url": data.get("callback_url"),
        "extra_params": data.get("extra_params"),
        "company_uuid": data.get("company_uuid"),
    }

    if company_uuid:
        company_id, num_calls_to_third_party = get_or_create_company(
            company_uuid
        )
    else:
        company_id, num_calls_to_third_party = get_or_create_company(
            "unnamed_company"
        )

    result = resume_inference_V3(
        filepath,
        bucket_name,
        company_id,
        context_data,
        start_time,
        async_call=False,
    )
    return {"message": "PROCESSING", "task_id": result.id}


@app.post("/resume/v5_scoring", status_code=200)
def pred_resume_post_scoring(response: Response, payload: dict = Body(...)):
    """
    API function for resume scoring.

    Args:
        response: The API response object.
        payload: The payload of the API.

    Returns:
        dict: The task ID.
    """
    data = payload
    filepath = data.get("file_path")
    bucket_name = data.get("bucket_name")
    job_data = data.get("job_data")
    resume_text = data.get("raw_text", None)
    context_data = {
        "job_candidate_id": data.get("job_candidate_id"),
        "callback_url": data.get("callback_url"),
    }

    scoring_obj = ModelScoring(
        cloud_filepath=filepath,
        bucket_name=bucket_name,
        callback_url=context_data.get("callback_url"),
        job_candidate_id=context_data.get("job_candidate_id"),
        job_data=json.dumps(job_data),
        resume_text=resume_text,
        taken_time=None,
        status=cs.SCORING_STATUS["PENDING"],
    )

    with Session() as session:
        session.add(scoring_obj)
        session.commit()

    return True


@app.get("/resume/status/{task_id}", status_code=200)
def resume_task_status(task_id):
    """
    For knowing status of task
    Args:
        task_id: taskid of task

    Returns:
        status or result
    """
    info_logger("Querying for", task_id)
    task = celery_app.AsyncResult(task_id)
    if not task.ready():
        return JSONResponse(
            status_code=202,
            content={"task_id": str(task_id), "status": "Processing"},
        )
    result = task.get()
    return {"task_id": task_id, "status": "Success", "result": result}


@app.post("/resume/only_email", status_code=200)
def only_email(response: Response, payload: dict = Body(...)):
    data = payload
    filepath = data.get("filepath")
    bucket_name = data.get("bucket_name")
    try:
        local_filepath = saving_file_local_resume(
            filepath, bucket_name=bucket_name
        )
        info_logger(f"Resume local path: {local_filepath}")
        local_filepath = os.path.abspath(local_filepath)
        resume_text, textract_flag = single_resume_read(local_filepath)
        (
            FINAL_DIC_RESUME["phone_number"],
            FINAL_DIC_RESUME["email"],
            FINAL_DIC_RESUME["additional_phone_numbers"],
            FINAL_DIC_RESUME["additional_emails"],
        ) = extract_email_phone("\n".join(resume_text))
        return FINAL_DIC_RESUME["email"]
    except:
        response.status_code = status.HTTP_400_BAD_REQUEST
        return {"error": "Unable to process the file"}


@app.post("/resume/resume_fallback_chatgpt_via_file", status_code=200)
def resume_fallback_chatgpt_via_file(
    response: Response, payload: dict = Body(...)
):
    data = payload
    filepath = data.get("filepath")
    bucket_name = data.get("bucket_name")
    try:
        local_filepath = saving_file_local_resume(
            filepath, bucket_name=bucket_name
        )
        info_logger(f"Resume local path: {local_filepath}")
        local_filepath = os.path.abspath(local_filepath)
        resume_text, textract_flag = single_resume_read(local_filepath)
        resume_doc_text = " ".join(resume_text)
        context_data = {
            "resume_process_id": data.get("resume_process_id"),
            "has_candidate_applied": data.get("has_candidate_applied"),
            "is_job_candidate_resume": data.get("is_job_candidate_resume"),
            "callback_url": data.get("callback_url"),
            "extra_params": data.get("extra_params"),
            "company_uuid": data.get("company_uuid"),
        }
        response = {}
        if resume_doc_text:
            response = chatgpt_parse(resume_doc_text)
        if response["force_update"]:
            send_result_to_ats_server(response, context_data, "PASSED", "Done")
        return response
    except:
        response.status_code = status.HTTP_400_BAD_REQUEST
        return {"error": "Unable to process the file"}


@app.post("/resume/resume_fallback_chatgpt_via_text", status_code=200)
def resume_fallback_chatgpt_via_text(
    response: Response, payload: dict = Body(...)
):
    data = payload
    try:
        resume_text = data.get("resume_text")
        context_data = {
            "resume_process_id": data.get("resume_process_id"),
            "has_candidate_applied": data.get("has_candidate_applied"),
            "is_job_candidate_resume": data.get("is_job_candidate_resume"),
            "callback_url": data.get("callback_url"),
            "extra_params": data.get("extra_params"),
            "company_uuid": data.get("company_uuid"),
        }
        response = {}
        if resume_text:
            response = chatgpt_parse(resume_text)
        if response["force_update"]:
            send_result_to_ats_server(response, context_data, "PASSED", "Done")
        return response
    except:
        response.status_code = status.HTTP_400_BAD_REQUEST
        return {"error": "Unable to process the file"}


@app.post("/resume/only_scoring_chatgpt_via_file", status_code=200)
def only_scoring_chatgpt_via_file(
    response: Response, payload: dict = Body(...)
):
    data = payload
    filepath = data.get("filepath")
    bucket_name = data.get("bucket_name")
    try:
        local_filepath = saving_file_local_resume(
            filepath, bucket_name=bucket_name
        )
        info_logger(f"Resume local path: {local_filepath}")
        local_filepath = os.path.abspath(local_filepath)
        resume_text, textract_flag = single_resume_read(local_filepath)
        resume_doc_text = " ".join(resume_text)
        job_data = data.get("job_data")
        response = {}
        if job_data and resume_doc_text:
            response, chatgpt_response = calculate_chatgpt_score(
                job_data, resume_doc_text
            )
        return response
    except:
        response.status_code = status.HTTP_400_BAD_REQUEST
        return {"error": "Unable to calculate rating for given file"}


@app.post("/resume/only_scoring_chatgpt_via_text", status_code=200)
def only_scoring_chatgpt_via_text(
    response: Response, payload: dict = Body(...)
):
    data = payload
    try:
        resume_doc_text = data.get("resume_text")
        job_data = data.get("job_data")
        response = {}
        if job_data and resume_doc_text:
            response, chatgpt_response = calculate_chatgpt_score(
                job_data, resume_doc_text
            )
        return response
    except:
        response.status_code = status.HTTP_400_BAD_REQUEST
        return {"error": "Unable to calculate rating for given text"}


@app.post("/resume_portal/update/", status_code=200)
def save_reviewed_data(payload: dict = Body(...)):
    """
    For saving manually reviewed data to ml DB
    Args:
        payload: Payload of API

    Returns:
        status of request
    """
    try:
        data = payload
        parser_output = data.get("parser_output")
        resume_path = data.get("resume_path")
        reviewed_data = data.get("reviewed_data")
        reviewer_id = data.get("reviewer_id")
        resume_process_id = data.get("resume_process_id")
    except Exception as exception:
        return {"status": f"Invalid data due to {exception}"}
    try:
        reviewed_resume_obj = ReviewedResume(
            resume_filepath=resume_path,
            parser_output=parser_output,
            reviewed_data=reviewed_data,
            reviewer_id=reviewer_id,
            resume_process_id=resume_process_id,
        )
        Session = get_db_session()

        with Session() as session:
            session.add(reviewed_resume_obj)
            session.commit()
        return {"status": "Successfully updated data"}
    except Exception as exception:
        return {"status": f"An error occurred. {exception}"}


@app.post("/resume_portal/view/", status_code=200)
def check_saved_data(response: Response, payload: dict = Body(...)):
    """
    For check data from  ml DB to postman for data analysis
    Args:
        response: return response
        payload: Payload of API

    Returns:
        data of backend
    """
    try:
        data = payload
        resume_process_id = data.get("resume_process_id")
        Session = get_db_session()
        results = []
        with Session() as session:
            results = (
                session.query(ModelResume)
                .filter(ModelResume.resume_process_id == resume_process_id)
                .all()
            )
        for result in results:
            json_output = {
                key: value
                for key, value in json.loads(result.json_output).items()
                if value not in ["None", None]
            }
            data = {
                "resume_text": result.resume_text,
                "parsed_entities": result.resume_entities,
                "json_output": json_output,
            }
            return {"data": data}
        else:
            return {
                "status": f"No result found for given resume process id {resume_process_id} "
            }
    except Exception as exception:
        response.status_code = status.HTTP_400_BAD_REQUEST
        return {"status": f"An error occurred. {exception}"}
