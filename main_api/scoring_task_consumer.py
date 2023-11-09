from dotenv import load_dotenv

load_dotenv("./constants.env")
import json
import os
import time
import datetime
import redis
import requests
from RP_parser.helper_files import constants as cs
from RP_parser.helper_codes.io_utils_2 import (
    single_resume_read,
    extract_text_from_docx,
)

from utils.cloud_utils import saving_file_local_resume
from utils.db_models import Scoring as ModelScoring

from RP_parser.helper_codes.rating_service_2 import bard_score
from utils.logging_helpers import info_logger, error_logger
from utils.db_utils import get_db_session

Session = get_db_session()

# Connect to Redis
REDIS_HOST = os.getenv("REDIS_HOST")  # Redis server hostname
REDIS_PORT = int(os.getenv("REDIS_PORT"))  # Redis server port
REDIS_DB = int(os.getenv("REDIS_DB"))  # Redis database number
REDIS_PASSWORD = (
    None if os.getenv("REDIS_PASSWORD") == "0" else os.getenv("REDIS_PASSWORD")
)  # Redis password (if required)

QUEUE_NAME = os.getenv("QUEUE_NAME")  # Name of the queue

r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB, password=REDIS_PASSWORD)


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
                    f"For Scoring Received {response.status_code} from {ats_scoring_endpoint} for job_candidate_id {context_data['job_candidate_id']}"
                )
                return False, response.status_code
        else:
            # Get ATS resume endpoints from environment variables
            ats_scoring_endpoints = json.loads(os.environ["ATS_RESUME_ENDPOINTS"])
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


# Consume tasks
while True:
    # Blocking pop from the right (waits for a task if the queue is empty)

    task_json = r.brpop(QUEUE_NAME)[1]

    # Convert the JSON string to a dictionary
    task = json.loads(task_json)
    (
        formatted_scoring,
        formatted_resume_summary,
        resume_text,
        ats_response_code,
        scoring_str,
        resume_summary,
    ) = ({}, {}, None, None, None, None)
    t1 = time.time()
    (
        task_id,
        job_data,
        cloud_filepath,
        bucket_name,
        resume_text,
        context_data,
        scoring_start_time,
    ) = (
        task.get("id"),
        task.get("job_data", "{}"),
        task.get("cloud_filepath"),
        task.get("bucket_name"),
        task.get("resume_text"),
        task.get("context_data", {}),
        datetime.datetime.fromisoformat(
            task.get(
                "scoring_start_time",
            )
        ).timestamp(),
    )
    info_logger(
        f"Processing {task_id}, Current length of queue {QUEUE_NAME} ,{r.llen(QUEUE_NAME)}"
    )

    # Calculate the score if job_data and resume text are present
    if job_data:
        try:
            if not resume_text:
                # Read resume from cloud storage if the resume text is not provided
                resume_path = saving_file_local_resume(
                    cloud_filepath, bucket_name=bucket_name
                )
                resume_path = os.path.abspath(resume_path)
                resume_text, textract_flag = single_resume_read(resume_path)

                if resume_text is None:
                    raise Exception(
                        "The document is corrupted. Please upload a different resume."
                    )
                else:
                    resume_text = " ".join(resume_text)

            (
                formatted_resume_summary,
                resume_summary,
                formatted_scoring,
                scoring_str,
            ) = bard_score(job_data, resume_text)
            info_logger(formatted_scoring)
            if formatted_scoring.get("matching_score", None):
                # Send the score to the ATS server
                (ats_request_status, ats_response_code,) = send_score_to_ats_server(
                    formatted_scoring, context_data, "PASSED", "Done"
                )

            # Update the ML database with the response
            with Session() as session:
                scoring_obj = (
                    session.query(ModelScoring)
                    .filter(ModelScoring.id == task_id)
                    .update(
                        {
                            "base_resume_summary": resume_summary,
                            "json_resume_summary": str(formatted_resume_summary),
                            "resume_text": resume_text,
                            "base_answer": scoring_str,
                            "json_answer": json.dumps(formatted_scoring),
                            "ats_response_code": ats_response_code,
                            "taken_time": time.time() - t1,
                            "status": cs.SCORING_STATUS["COMPLETED"],
                        }
                    )
                )
                session.commit()

            end_time = time.time()
            info_logger(
                f"For ID {task_id}, Total time {end_time - scoring_start_time} for {cloud_filepath}, {datetime.datetime.fromtimestamp(scoring_start_time).strftime('%Y-%m-%d %H:%M:%S')}, {datetime.datetime.fromtimestamp(end_time).strftime('%Y-%m-%d %H:%M:%S')}"
            )
        except Exception as exception:
            formatted_scoring["error"] = str(exception)
            # Update the ML database with the response with the error
            with Session() as session:
                scoring_obj = (
                    session.query(ModelScoring)
                    .filter(ModelScoring.id == task_id)
                    .update(
                        {
                            "base_resume_summary": resume_summary,
                            "json_resume_summary": str(formatted_resume_summary),
                            "resume_text": resume_text,
                            "base_answer": scoring_str,
                            "json_answer": json.dumps(formatted_scoring),
                            "ats_response_code": ats_response_code,
                            "taken_time": time.time() - t1,
                            "status": cs.SCORING_STATUS["ERROR"],
                        }
                    )
                )
                session.commit()
            info_logger(
                f"Scoring Calculation is failed due to {str(exception)}, {cloud_filepath},{task_id}"
            )
    else:
        formatted_scoring["error"] = "Job data is not available for this resume"
        info_logger(
            f"Scoring Calculation is failed due to Job data is not available for this resume, {cloud_filepath},{task_id}"
        )
        with Session() as session:
            scoring_obj = (
                session.query(ModelScoring)
                .filter(ModelScoring.id == task_id)
                .update(
                    {
                        "base_resume_summary": resume_summary,
                        "json_resume_summary": str(formatted_resume_summary),
                        "resume_text": resume_text,
                        "base_answer": scoring_str,
                        "json_answer": json.dumps(formatted_scoring),
                        "ats_response_code": ats_response_code,
                        "taken_time": time.time() - t1,
                        "status": cs.SCORING_STATUS["ERROR"],
                    }
                )
            )
            session.commit()
