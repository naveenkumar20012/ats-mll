import json
import os

from dotenv import load_dotenv

load_dotenv("./constants.env")
import logging as logger

logger.basicConfig(level="INFO")

import redis

from utils.db_utils import get_db_session
from utils.db_models import Scoring as ModelScoring
from utils.logging_helpers import info_logger, error_logger
from RP_parser.helper_files import constants as cs

# Connect to Redis
REDIS_HOST = os.getenv("REDIS_HOST")  # Redis server hostname
REDIS_PORT = int(os.getenv("REDIS_PORT"))  # Redis server port
REDIS_DB = int(os.getenv("REDIS_DB"))  # Redis database number
REDIS_PASSWORD = (
    None if os.getenv("REDIS_PASSWORD") == "0" else os.getenv("REDIS_PASSWORD")
)  # Redis password (if required)

QUEUE_LIMIT = int(os.getenv("QUEUE_LIMIT"))  # The limit of the queue
QUEUE_NAME = os.getenv("QUEUE_NAME")  # Name of the queue

Session = get_db_session()


def fetch_scoring_data_from_database(status):
    """
    Fetches the data from the database.
    Args:
        status: The status of the data to be fetched.

    Returns:
        The data fetched from the database.
    """
    try:
        with Session() as session:
            pending_tasks = (
                session.query(ModelScoring)
                .filter(ModelScoring.status == cs.SCORING_STATUS[status])
                .all()
            )
        return pending_tasks
    except Exception as e:
        if session:
            session.rollback()
            session.close()
        error_logger(f"Error while fetching data in database {e}")
        return None


r = redis.Redis(
    host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB, password=REDIS_PASSWORD
)

# Get the length of the queue
queue_length = int(r.llen(QUEUE_NAME))
info_logger(
    f"Current length of queue {QUEUE_NAME} ,{r.llen(QUEUE_NAME)}, queue limit is {QUEUE_LIMIT} "
)

if queue_length < QUEUE_LIMIT:
    # Push the task to the queue
    pushed_limit = QUEUE_LIMIT - queue_length
    if pushed_limit == 0:
        info_logger(f"Queue {QUEUE_NAME} is full, so not pushing any tasks")
        pass
    info_logger(
        f"Current length of queue {QUEUE_NAME} ,we can push {pushed_limit} tasks."
    )
    pushed_ids, error_ids = [], []
    pending_tasks = fetch_scoring_data_from_database("PENDING")
    for task in pending_tasks:
        if pushed_limit == 0:
            break
        pushed_limit -= 1
        try:
            task_json = {
                "id": task.id,
                # if job data str then convert to dict otherwise leave it as it is
                "job_data": task.job_data
                if isinstance(task.job_data, dict)
                else json.loads(task.job_data),
                "cloud_filepath": task.cloud_filepath,
                "bucket_name": task.bucket_name,
                "resume_text": task.resume_text,
                "context_data": {
                    "job_candidate_id": task.job_candidate_id,
                    "callback_url": task.callback_url,
                },
                "scoring_start_time": task.time_created.isoformat(),
            }
            r.lpush(QUEUE_NAME, json.dumps(task_json))
            pushed_ids.append(task.id)
        except Exception as e:
            error_ids.append(task.id)
            info_logger(f"Error while uploading the task {e} {task.id}")

    # Update the status of the task PENDING to QUEUED
    try:
        if pushed_ids:
            with Session() as session:
                update_query = (
                    session.query(ModelScoring)
                    .filter(ModelScoring.id.in_(pushed_ids))
                    .update(
                        {ModelScoring.status: cs.SCORING_STATUS["QUEUED"]},
                        synchronize_session=False,
                    )
                )
                session.commit()
        # Update the status of the task PENDING to ERROR
        if error_ids:
            with Session() as session:
                update_query = (
                    session.query(ModelScoring)
                    .filter(ModelScoring.id.in_(error_ids))
                    .update(
                        {
                            ModelScoring.json_answer: {
                                "error": "Error while pushing to queue"
                            },
                            ModelScoring.status: cs.SCORING_STATUS["ERROR"],
                        },
                        synchronize_session=False,
                    )
                )
                session.commit()
    except Exception as e:
        if session:
            session.rollback()
            session.close()
        error_logger(f"Error while updating statue in database {e}")
    info_logger("Task pushed to queue", pushed_ids)
    info_logger("Task pushed but got error", error_ids)
