from dotenv import load_dotenv

load_dotenv("./constants.env")
import numexpr
import hashlib

numexpr.set_num_threads(numexpr.detect_number_of_cores())
import ast
import datetime
import json
import logging as logger
import os
import re
import time

import requests

from RP_parser.helper_codes.edu_service_3 import edu_ner_pred
from RP_parser.helper_codes.exp_service_3 import work_exp_ner_pred

from RP_parser.helper_codes.io_utils_2 import (
    single_resume_read,
    extract_text_using_bs,
    cleaning_text,
    extract_text_from_img_pdf,
    extract_text_from_pdf_textract,
)
from RP_parser.helper_codes.parser_rules import (
    extract_email_phone,
    extract_linkedin_spe,
    extract_github,
    extract_facebook,
    extract_twitter,
    extract_instagram,
    extract_name,
    extract_skills,
    extract_email_2,
    entities_extractor_2,
    extract_current_position,
)
from utils.db_models import Resume as ModelResume
from utils.db_models import SyncResume as ModelSyncResume

from utils.logging_helpers import info_logger, error_logger
from .worker import app
from utils.cloud_utils import saving_file_local_resume
from utils.db_utils import get_db_session

logger.basicConfig(level="INFO")
celery = app
os.environ["TOKENIZERS_PARALLELISM"] = "false"
KEY_PATH = os.getenv("KEY_PATH")
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = KEY_PATH
os.environ["AWS_ACCESS_KEY_ID"] = os.getenv("AWS_ACCESS_KEY_ID")
os.environ["AWS_SECRET_ACCESS_KEY"] = os.getenv("AWS_SECRET_ACCESS_KEY")
FINAL_DIC_RESUME = ast.literal_eval(os.getenv("final_dict_resume"))
SKILL_CSV_PATH = os.getenv("SKILL_CSV_PATH")
SOFT_TIME_LIMIT = int(os.getenv("SOFT_TIME_LIMIT"))
RATING_ALLOWED = int(os.getenv("RATING_ALLOWED"))
RETRY_CHATGPT_ALLOWED = int(os.getenv("RETRY_CHATGPT_ALLOWED"))
TEXT_EXTRACTION_TYPE = ast.literal_eval(os.getenv("TEXT_EXTRACTION_TYPE"))
RATING_COMPANIES = ast.literal_eval(os.getenv("RATING_COMPANIES"))
RP_MODEL_LAST_UPDATED_AT = datetime.datetime.fromisoformat(
    str(os.getenv("RP_MODEL_LAST_UPDATED_AT"))
).replace(tzinfo=None)
AZURE_STORAGE_ACCOUNT_NAME = os.getenv("AZURE_STORAGE_ACCOUNT_NAME")
AZURE_STORAGE_ACCOUNT_KEY = os.getenv("AZURE_STORAGE_ACCOUNT_KEY")
AZURE_CONTAINER_NAME = os.getenv("AZURE_CONTAINER_NAME")

Session = get_db_session()


def send_result_to_ats_server(result, context_data, status, message):
    """
    Sends the result to the ATS server.

    Args:
        result: The result data.
        context_data: The context data.
        status: The status of the result.
        message: The message associated with the result.

    Returns:
        A tuple indicating if the request was successful and the status code.
    """
    try:
        data = {
            "data": result,
            "resume_process_id": context_data["resume_process_id"],
            "has_candidate_applied": context_data["has_candidate_applied"],
            "is_job_candidate_resume": context_data["is_job_candidate_resume"],
            "status": status,
            "message": message,
            "extra_params": context_data.get("extra_params"),
            "force_update": result.get("force_update", False),
        }
        headers = {
            "Content-Type": "application/json",
        }

        if context_data.get("callback_url"):
            # Construct the endpoint for sending the result
            ats_resume_endpoints = (
                context_data.get("callback_url") + "api/process-parsed-resume/"
            )
            # Send the request to the endpoint
            requests_status = requests.post(
                ats_resume_endpoints,
                headers=headers,
                data=json.dumps(data),
                timeout=20,
            )

            if requests_status.status_code == 200:
                # Log successful request
                info_logger(
                    f"{ats_resume_endpoints},{requests_status.status_code}, {context_data['resume_process_id']}"
                )
                return True, requests_status.status_code
            else:
                # Log unsuccessful request
                info_logger(
                    f"{ats_resume_endpoints},{requests_status.status_code}, {context_data['resume_process_id']},{json.dumps(data)}"
                )
                error_logger(
                    f"Received {requests_status.status_code} from {ats_resume_endpoints} for {context_data['resume_process_id']}"
                )
                return False, requests_status.status_code
        else:
            # Get ATS resume endpoints from environment variables
            ats_resume_endpoints = json.loads(
                os.environ["ATS_RESUME_ENDPOINTS"]
            )
            requests_statues = []
            for endpoint in ats_resume_endpoints:
                # Send the request to each endpoint
                requests_status = requests.post(
                    endpoint,
                    headers=headers,
                    data=json.dumps(data),
                    timeout=10,
                )
                requests_statues.append(requests_status.status_code)

                if requests_status.status_code == 200:
                    # Log successful request
                    info_logger(
                        f"{endpoint},{requests_status}, {context_data['resume_process_id']}"
                    )
                else:
                    # Log unsuccessful request
                    info_logger(
                        f"callback_url not found. Received {requests_status.status_code} from {endpoint} for {context_data['resume_process_id']}"
                    )
            # Check if any of the requests were successful
            return (
                200 in requests_statues,
                200 if 200 in requests_statues else requests_statues[0],
            )

    except Exception as e:
        # Log error if exception occurs
        error_logger("response to ats server is not succeed,", e)
        return False, None


def send_score_to_ats_server(result, context_data, status, message):
    """
    Sends the candidate scoring to the ATS server.

    Args:
        result (dict): The result data.
        context_data (dict): The context data.
        status (str): The status of the result.
        message (str): The message associated with the result.

    Returns:
        tuple: A tuple indicating if the request was successful and the status code.
    """
    try:
        data = {
            "resume_score": result,
            "job_candidate_id": context_data["job_candidate_id"],
            "status": status,
            "message": message,
        }

        headers = {
            "Content-Type": "application/json",
        }

        if context_data.get("callback_url"):
            # Construct the endpoint for sending the result
            ats_scoring_endpoint = (
                context_data.get("callback_url") + "/resume-score/webhook/"
            )
            # Send the request to the endpoint
            response = requests.post(
                ats_scoring_endpoint,
                headers=headers,
                data=json.dumps(data),
                timeout=10,
            )

            if response.status_code == 200:
                # Log successful request
                info_logger(
                    f"{ats_scoring_endpoint}, {response.status_code}, job_candidate_id {context_data['job_candidate_id']}"
                )
                return True, response.status_code
            else:
                # Log unsuccessful request
                info_logger(
                    f"{ats_scoring_endpoint}, {response.status_code}, job_candidate_id {context_data['job_candidate_id']}, {json.dumps(data)}"
                )
                error_logger(
                    f"Received {response.status_code} from {ats_scoring_endpoint} for job_candidate_id {context_data['job_candidate_id']}"
                )
                return False, response.status_code
        else:
            # Get ATS resume endpoints from environment variables
            ats_scoring_endpoints = json.loads(
                os.environ["ATS_RESUME_ENDPOINTS"]
            )
            response_statuses = []
            for endpoint in ats_scoring_endpoints:
                # Send the request to each endpoint
                response = requests.post(
                    endpoint,
                    headers=headers,
                    data=json.dumps(data),
                    timeout=10,
                )
                response_statuses.append(response.status_code)

                if response.status_code == 200:
                    # Log successful request
                    info_logger(
                        f"{endpoint}, {response}, job_candidate_id {context_data['job_candidate_id']}"
                    )
                else:
                    # Log unsuccessful request
                    info_logger(
                        f"callback_url not found. Received {response.status_code} from {endpoint} for job_candidate_id {context_data['job_candidate_id']}"
                    )
            # Check if any of the requests were successful
            return (
                200 in response_statuses,
                200 if 200 in response_statuses else response_statuses[0],
            )

    except Exception as e:
        # Log error if an exception occurs
        error_logger("Failed to send response to ATS server", e)
        return False, None


def generate_hash_from_file(resume_path):
    """
    Generate hash value of given file path
    Args:
        resume_path: file path of resume

    Returns:
        hash_value of file
    """
    with open(resume_path, "rb") as f:
        data = f.read()
        return hashlib.sha256(data).hexdigest()


def fetch_resume_data_from_database(resume_hash):
    """
    Fetch resume data from database and check if parsed data is expired or not
    Args:
        resume_hash:  hash value of resume

    Returns:
        resume data from database and expired status
    """
    expired = False
    try:
        with Session() as session:
            result = (
                session.query(
                    ModelSyncResume.id,
                    ModelSyncResume.time_created,
                    ModelSyncResume.time_updated,
                    ModelSyncResume.json_output,
                    ModelSyncResume.resume_text,
                )
                .filter(ModelSyncResume.hash_value == resume_hash)
                .first()
            )

            if result:
                parsed_time = (
                    result.time_updated
                    if result.time_updated
                    else result.time_created
                )
                expired = (
                    parsed_time.replace(tzinfo=None) < RP_MODEL_LAST_UPDATED_AT
                )
        return result, expired
    except Exception as e:
        if session:
            session.rollback()
            session.close()
        error_logger(f"Error while fetching data in database {e}")
        return None, expired


def format_resume_json(resume_data):
    """
    remove None and null value from dict
    Args:
        resume_data: Json dict of parsed data

    Returns:
        Dictionary without  None and null value
    """
    formatted_data = {}
    for key, value in resume_data.items():
        if value == "None" or value is None:
            continue
        formatted_data[key] = value

    return formatted_data


def create_resume_obj(
    async_call,
    context_data,
    cloud_filepath,
    resume_path,
    FINAL_DIC_RESUME_DB,
    extracted_entities,
    company_id,
    resume_text_doc_text,
    resume_hash,
    ats_response_code,
):
    """
    Create Resume Object for resume parser
    Args:
        async_call: bool flag for async_call
        context_data: context data from request
        cloud_filepath: resume cloud path
        resume_path: local resume path
        FINAL_DIC_RESUME_DB: JSON for dict
        extracted_entities: Extracted entities
        company_id: company id
        resume_text_doc_text: resume text
        resume_hash: resume file hash
        ats_response_code: ats request code
    Returns:
        DB Model object
    """
    common_data = {
        "gcp_filepath": cloud_filepath,
        "json_output": json.dumps(FINAL_DIC_RESUME_DB),
        "company_id": company_id,
        "resume_text": resume_text_doc_text,
        "resume_entities": extracted_entities,
    }

    if async_call:
        return ModelResume(
            resume_process_id=context_data["resume_process_id"],
            local_filepath=resume_path,
            is_self_parsed=True,
            ats_response_code=ats_response_code,
            **common_data,
        )
    else:
        return ModelSyncResume(
            hash_value=resume_hash, third_party_output={}, **common_data
        )


def store_resume_data(
    async_call,
    context_data,
    cloud_filepath,
    resume_path,
    FINAL_DIC_RESUME_DB,
    extracted_entities,
    company_id,
    resume_text_doc_text,
    resume_hash,
    ats_response_code=None,
    expired=False,
    id=None,
):
    """
    store resume data in ML database
    Args:
        async_call: bool flag for async_call
        context_data: context data from request
        cloud_filepath: resume cloud path
        resume_path: local resume path
        FINAL_DIC_RESUME_DB: JSON for dict
        extracted_entities: Extracted entities
        company_id: company id
        resume_text_doc_text: resume text
        resume_hash: resume file hash
        ats_response_code: ATS response code
        expired: flag for data is expired or not
        id: id from DB

    Returns:

    """
    # Create a resume object based on whether the call is async
    resume_obj = create_resume_obj(
        async_call,
        context_data,
        cloud_filepath,
        resume_path,
        FINAL_DIC_RESUME_DB,
        extracted_entities,
        company_id,
        resume_text_doc_text,
        resume_hash,
        ats_response_code,
    )

    try:
        # Add resume object to session and commit changes
        with Session() as session:
            if expired:
                session.query(ModelSyncResume).filter(
                    ModelSyncResume.id == id
                ).update(
                    {
                        "json_output": json.dumps(FINAL_DIC_RESUME_DB),
                        "resume_entities": extracted_entities,
                    }
                )
                session.commit()
                resume_obj_id = id
            else:
                session.add(resume_obj)
                session.commit()
                resume_obj_id = resume_obj.id

        return resume_obj_id
    except Exception as e:
        if session:
            session.rollback()
            session.close()
        error_logger(f"Error while storing data in database {e}")
        return None


@app.task(
    ignore_result=False,
    bind=True,
    path=("celery_tasks.ml.model", "ResumeModel"),
    name="{}.{}".format(__name__, "ResumeV3"),
    soft_time_limit=SOFT_TIME_LIMIT,
)
def resume_inference_V3(
    self,
    cloud_filepath,
    bucket_name,
    company_id,
    context_data,
    start_time,
    async_call=False,
):
    FINAL_DIC_RESUME = ast.literal_eval(os.getenv("final_dict_resume"))
    (
        resume_path,
        resume_obj_id,
        line_label,
        resume_text_doc_text,
        extracted_entities,
        retry_chatgpt,
        ats_request_status,
        resume_hash,
        expired,
        result,
        ats_response_code,
    ) = (None, None, None, "", [], False, False, None, False, None, None)
    t1 = time.time()
    try:
        resume_path = saving_file_local_resume(
            cloud_filepath, bucket_name=bucket_name
        )
        resume_path = os.path.abspath(resume_path)

        if not async_call:
            resume_hash = generate_hash_from_file(resume_path)
            if resume_hash:
                result, expired = fetch_resume_data_from_database(resume_hash)
                if result and not expired:
                    FINAL_DIC_RESUME = json.loads(result.json_output)
                    FINAL_DIC_RESUME["raw_text"] = str(
                        result.resume_text
                    ).replace("\n", " ")
                    return format_resume_json(FINAL_DIC_RESUME)

        resume_text, textract_flag = single_resume_read(resume_path)
        FINAL_DIC_RESUME["text_type"] = textract_flag

        if resume_text is None:
            raise Exception(
                "The document is corrupted. Please upload a different resume."
            )

        if len(resume_text) == 1:
            if resume_text[0] == "TimeLimitExceeded":
                raise Exception(
                    "The document taking more time than usual. So, we are removing from queue."
                )

        resume_text_doc_text = "\n".join(resume_text)
        FINAL_DIC_RESUME["raw_text"] = " ".join(resume_text)

        # entity extraction
        extracted_entities = entities_extractor_2(FINAL_DIC_RESUME["raw_text"])
        info_logger(extracted_entities)
        # section detection
        # line_label = parse_lines4(extracted_entities, resume_text_doc_text)

        # email
        emails = [
            re.sub(
                r"\s+", "", text.replace("\n", "").lower(), flags=re.UNICODE
            )
            for start, end, text, label in extracted_entities
            if label == "Email"
        ]
        text_emails = " , ".join(emails)

        # contact number
        phone_numbers = [
            re.sub(r"[^\d]", "", text)
            for start, end, text, label in extracted_entities
            if label == "Mobile_Number"
        ]

        text_phone_numbers = " , ".join(phone_numbers)
        if len(text_phone_numbers + " , " + text_emails) > 0:
            (
                FINAL_DIC_RESUME["phone_number"],
                FINAL_DIC_RESUME["email"],
                FINAL_DIC_RESUME["additional_phone_numbers"],
                FINAL_DIC_RESUME["additional_emails"],
            ) = extract_email_phone(text_emails + " " + text_phone_numbers)

        if not (FINAL_DIC_RESUME["email"]):
            (
                FINAL_DIC_RESUME["phone_number"],
                FINAL_DIC_RESUME["email"],
                FINAL_DIC_RESUME["additional_phone_numbers"],
                FINAL_DIC_RESUME["additional_emails"],
            ) = extract_email_phone(resume_text_doc_text)

        if FINAL_DIC_RESUME["email"] is None:
            # Extract email from resume text using second method
            FINAL_DIC_RESUME["email"] = extract_email_2(resume_text_doc_text)

            if FINAL_DIC_RESUME["email"] is None:
                # Extract text from image after converting pdf into image
                if resume_path.lower().endswith(".pdf"):
                    ocr_text = extract_text_from_img_pdf(resume_path)
                    if len(ocr_text) > 0:
                        # Clean the extracted text
                        ocr_text = cleaning_text(ocr_text)
                        FINAL_DIC_RESUME["ocr_text"] = "\n".join(ocr_text)

                        # Extract email and phone number using chatgpt
                        # (
                        #     FINAL_DIC_RESUME["phone_number"],
                        #     FINAL_DIC_RESUME["email"],
                        #     FINAL_DIC_RESUME["additional_phone_numbers"],
                        #     FINAL_DIC_RESUME["additional_emails"],
                        # ) = extract_email_phone_chatgpt("\n".join(ocr_text))

                        # If email is still not found, extract it using regex
                        # if FINAL_DIC_RESUME["email"] is None:

                        (
                            FINAL_DIC_RESUME["phone_number"],
                            FINAL_DIC_RESUME["email"],
                            FINAL_DIC_RESUME["additional_phone_numbers"],
                            FINAL_DIC_RESUME["additional_emails"],
                        ) = extract_email_phone("\n".join(ocr_text))

                        # If email is still not found, extract it from cleaned text
                        if FINAL_DIC_RESUME["email"] is None:
                            FINAL_DIC_RESUME["email"] = extract_email_2(
                                "\n".join(ocr_text)
                            )
                            if (
                                FINAL_DIC_RESUME["email"] is None
                                and async_call
                            ):
                                # Raise exception if email is not found and async_call flag is True
                                raise Exception("Email not found")
                    else:
                        if async_call:
                            # Raise exception if email is not found and async_call flag is True
                            raise Exception("Email not found")

                # Extract text from doc file
                elif resume_path.lower().endswith(".doc"):
                    bs_text = extract_text_using_bs(
                        resume_path, encoding="ISO-8859-1"
                    )
                    if len(bs_text) > 0:
                        # Clean the extracted text
                        bs_text = cleaning_text(bs_text)

                        # Extract email and phone number using regex
                        (
                            FINAL_DIC_RESUME["phone_number"],
                            FINAL_DIC_RESUME["email"],
                            FINAL_DIC_RESUME["additional_phone_numbers"],
                            FINAL_DIC_RESUME["additional_emails"],
                        ) = extract_email_phone("\n".join(bs_text))

                        # If email is still not found, extract it from cleaned text
                        if FINAL_DIC_RESUME["email"] is None:
                            FINAL_DIC_RESUME["email"] = extract_email_2(
                                "\n".join(bs_text)
                            )
                            if (
                                FINAL_DIC_RESUME["email"] is None
                                and async_call
                            ):
                                # Raise exception if email is not found and async_call flag is True
                                raise Exception("Email not found")
                    else:
                        if async_call:
                            # Raise exception if email is not found and async_call flag is True
                            raise Exception("Email not found")
                else:
                    if async_call:
                        # Raise exception if email is not found and async_call flag is True
                        raise Exception("Email not found")

        # name:
        names = [
            text.lower().replace("\n", " ").title()
            for start, end, text, label in extracted_entities
            if label == "Name"
        ]

        if len(names) > 0:
            FINAL_DIC_RESUME["name"] = names[0]
        else:
            FINAL_DIC_RESUME["name"] = extract_name(resume_text)

        # Location:
        Location = [
            text.replace("\n", " ")
            for start, end, text, label in extracted_entities
            if label == "Location"
        ]

        if len(Location) > 0:
            FINAL_DIC_RESUME["address"] = sorted(Location, key=len)[0]

        # All Urls
        FINAL_DIC_RESUME["linkedin_url"] = extract_linkedin_spe(
            resume_text_doc_text
        )
        FINAL_DIC_RESUME["github_url"] = extract_github(resume_text_doc_text)
        FINAL_DIC_RESUME["facebook_url"] = extract_facebook(
            resume_text_doc_text
        )
        FINAL_DIC_RESUME["twitter_url"] = extract_twitter(resume_text_doc_text)
        FINAL_DIC_RESUME["instagram_url"] = extract_instagram(
            resume_text_doc_text
        )

        # Experience
        (
            FINAL_DIC_RESUME["experience"],
            FINAL_DIC_RESUME["total_experience"],
            retry_chatgpt,
        ) = work_exp_ner_pred(extracted_entities, resume_text_doc_text)

        # Designation
        if FINAL_DIC_RESUME["experience"]:
            for pair in FINAL_DIC_RESUME["experience"]:
                if pair["current"]:
                    FINAL_DIC_RESUME["designation"] = pair["designation"]
                    break

        if not FINAL_DIC_RESUME["experience"]:
            FINAL_DIC_RESUME["designation"] = extract_current_position(
                extracted_entities
            )

        # Education
        FINAL_DIC_RESUME["education"] = edu_ner_pred(
            extracted_entities, resume_text_doc_text
        )

        # skills
        FINAL_DIC_RESUME["skills"] = extract_skills(
            extracted_entities, resume_text_doc_text
        )

        # When the experience and education details are not available. We are parsing it with textract again
        if (
            not (
                FINAL_DIC_RESUME["experience"]
                and FINAL_DIC_RESUME["education"]
            )
            and textract_flag == TEXT_EXTRACTION_TYPE["TIKA"]
        ):
            resume_text, textract_flag = extract_text_from_pdf_textract(
                resume_path
            )
            FINAL_DIC_RESUME["text_type"] = textract_flag
            resume_text_doc_text = FINAL_DIC_RESUME["raw_text"] = resume_text

            # entity extraction
            extracted_entities = entities_extractor_2(resume_text_doc_text)
            info_logger(extracted_entities)

            # name:
            names = [
                text.lower().replace("\n", " ").title()
                for start, end, text, label in extracted_entities
                if label == "Name"
            ]

            if len(names) > 0:
                FINAL_DIC_RESUME["name"] = names[0]

            # Location:
            Location = [
                text.replace("\n", " ")
                for start, end, text, label in extracted_entities
                if label == "Location"
            ]

            if len(Location) > 0:
                FINAL_DIC_RESUME["address"] = sorted(Location, key=len)[0]

            # All Urls
            FINAL_DIC_RESUME["linkedin_url"] = extract_linkedin_spe(
                resume_text_doc_text
            )
            FINAL_DIC_RESUME["github_url"] = extract_github(
                resume_text_doc_text
            )
            FINAL_DIC_RESUME["facebook_url"] = extract_facebook(
                resume_text_doc_text
            )
            FINAL_DIC_RESUME["twitter_url"] = extract_twitter(
                resume_text_doc_text
            )
            FINAL_DIC_RESUME["instagram_url"] = extract_instagram(
                resume_text_doc_text
            )

            # Experience
            (
                FINAL_DIC_RESUME["experience"],
                FINAL_DIC_RESUME["total_experience"],
                retry_chatgpt,
            ) = work_exp_ner_pred(extracted_entities, resume_text_doc_text)

            # Designation
            if FINAL_DIC_RESUME["experience"]:
                for pair in FINAL_DIC_RESUME["experience"]:
                    if pair["current"]:
                        FINAL_DIC_RESUME["designation"] = pair["designation"]
                        break

            if not FINAL_DIC_RESUME["experience"]:
                FINAL_DIC_RESUME["designation"] = extract_current_position(
                    extracted_entities
                )

            # Education
            FINAL_DIC_RESUME["education"] = edu_ner_pred(
                extracted_entities, resume_text_doc_text
            )

            # skills
            FINAL_DIC_RESUME["skills"] = extract_skills(
                extracted_entities, resume_text_doc_text
            )

        t2 = time.time()
        FINAL_DIC_RESUME["taken_time"] = t2 - t1

        # Format resume data as JSON
        formatted_data = format_resume_json(FINAL_DIC_RESUME)

        # send data to ATS server
        if async_call:
            ats_request_status, ats_response_code = send_result_to_ats_server(
                formatted_data, context_data, "PASSED", "Done"
            )
        info_logger("Result ", formatted_data)

        # Save data to ML database
        FINAL_DIC_RESUME_DB = formatted_data.copy()

        # Remove raw_text if present
        if "raw_text" in FINAL_DIC_RESUME_DB:
            del FINAL_DIC_RESUME_DB["raw_text"]

        # Add line_label, and async_call to FINAL_DIC_RESUME_DB
        FINAL_DIC_RESUME_DB["sections"] = line_label
        FINAL_DIC_RESUME_DB["async_call"] = async_call

        resume_obj_id = store_resume_data(
            async_call=async_call,
            context_data=context_data,
            cloud_filepath=cloud_filepath,
            resume_path=resume_path,
            FINAL_DIC_RESUME_DB=FINAL_DIC_RESUME_DB,
            extracted_entities=extracted_entities,
            company_id=company_id,
            resume_text_doc_text=resume_text_doc_text,
            resume_hash=resume_hash,
            ats_response_code=ats_response_code,
            expired=expired,
            id=result.id if result else None,
        )

    except Exception as exception:
        FINAL_DIC_RESUME["error"] = str(exception)
        message = f"Error while parsing resume - {exception}"

        t2 = time.time()
        FINAL_DIC_RESUME["taken_time"] = t2 - t1

        # Format resume data as JSON
        formatted_data = format_resume_json(FINAL_DIC_RESUME)

        info_logger("Error Result ", formatted_data)

        # send data to ATS server
        if async_call:
            ats_request_status, ats_response_code = send_result_to_ats_server(
                formatted_data, context_data, status="FAILED", message=message
            )

        # Save data to ML database
        FINAL_DIC_RESUME_DB = formatted_data.copy()

        # Remove raw_text if present
        if "raw_text" in FINAL_DIC_RESUME_DB:
            del FINAL_DIC_RESUME_DB["raw_text"]

        # Add line_label, async_call to FINAL_DIC_RESUME_DB
        FINAL_DIC_RESUME_DB["sections"] = line_label
        FINAL_DIC_RESUME_DB["async_call"] = async_call

        resume_obj_id = store_resume_data(
            async_call=async_call,
            context_data=context_data,
            cloud_filepath=cloud_filepath,
            resume_path=resume_path,
            FINAL_DIC_RESUME_DB=FINAL_DIC_RESUME_DB,
            extracted_entities=extracted_entities,
            company_id=company_id,
            resume_text_doc_text=resume_text_doc_text,
            resume_hash=resume_hash,
            ats_response_code=ats_response_code,
            expired=expired,
            id=result.id if result else None,
        )

        if cloud_filepath.startswith("gs://"):
            error_logger(
                message
                + " file available at https://storage.googleapis.com/"
                + cloud_filepath[5:]
            )
        elif cloud_filepath.startswith("s3://"):
            error_logger(
                message
                + " file available at https://"
                + cloud_filepath.split("/")[2]
                + ".s3.ap-south-1.amazonaws.com/"
                + "/".join(cloud_filepath.split("/")[3:])
            )
        elif cloud_filepath.startswith("azure://"):
            error_logger(
                message
                + " file available at https://"
                + AZURE_STORAGE_ACCOUNT_NAME
                + ".blob.core.windows.net/"
                + AZURE_CONTAINER_NAME
                + "/"
                + "/".join(cloud_filepath.split("/")[4:])
            )
        else:
            error_logger(message + " file available at" + cloud_filepath)

    end_time = time.time()
    info_logger(
        f"total time {end_time - start_time} for {cloud_filepath},{datetime.datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S')}, {datetime.datetime.fromtimestamp(end_time).strftime('%Y-%m-%d %H:%M:%S')}"
    )
    if (end_time - start_time) > float(os.getenv("TIME_LIMIT")):
        if cloud_filepath.startswith("gs://"):
            error_logger(
                f"total process time is {end_time - start_time} and parser time is {t2 - t1}"
                f" file available at https://storage.googleapis.com/"
                + cloud_filepath[5:]
            )
        elif cloud_filepath.startswith("s3://"):
            error_logger(
                f"total process time is {end_time - start_time} and parser time is {t2 - t1}"
                f" file available at https://"
                + cloud_filepath.split("/")[2]
                + ".s3.ap-south-1.amazonaws.com/"
                + "/".join(cloud_filepath.split("/")[3:])
            )
        elif cloud_filepath.startswith("azure://"):
            error_logger(
                f"total process time is {end_time - start_time} and parser time is {t2 - t1}"
                + " file available at https://"
                + AZURE_STORAGE_ACCOUNT_NAME
                + ".blob.core.windows.net/"
                + AZURE_CONTAINER_NAME
                + "/"
                + "/".join(cloud_filepath.split("/")[4:])
            )
        else:
            error_logger(
                f"total process time is {end_time - start_time} and parser time is {t2 - t1}"
                f" file available at" + cloud_filepath
            )
    # Return the result so it's pushed to Redis backend
    return formatted_data
