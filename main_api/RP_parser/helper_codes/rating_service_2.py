import ast
import json
import os
import random
import re
import time

from RP_parser.helper_codes.palm_api import generate_text
from RP_parser.helper_files import constants as cs
from slack import WebClient
from slack.errors import SlackApiError
from sqlalchemy import func
from utils.db_models import Scoring_server as ModelServers
from utils.db_utils import get_db_session
from utils.logging_helpers import info_logger, error_logger

SESSION = get_db_session()
FAILURE_THRESHOLD = int(os.getenv("FAILURE_THRESHOLD"))
SESSION_HEADERS = cs.SESSION_HEADERS
USER_AGENTS = cs.USER_AGENTS
PROXIES = cs.PROXIES
SLACK_BOT_USER_OAUTH_TOKEN = os.getenv("SLACK_BOT_USER_OAUTH_TOKEN")
client = WebClient(token=SLACK_BOT_USER_OAUTH_TOKEN)

channel_id = (
    os.getenv("SLACK_SERVER_ALERT_CHANNEL_ID")
    if os.getenv("ENVIRONMENT") == "production"
    else os.getenv("SLACK_ATS_STAGING_CHANNEL_ID")
)


def send_slack_message(client, channel, message):
    """
    Send message to channel
    Args:
        client: client object of Slack
        channel: channel id
        message: text message
    """
    try:
        response = client.chat_postMessage(
            channel=channel,
            blocks=[
                {
                    "type": "section",
                    "text": {"type": "mrkdwn", "text": message},
                }
            ],
        )
    except SlackApiError as e:
        error_logger(f"Error occurred while sending message on slack {e}")
        assert e.response["error"]


def get_active_server_id_and_token(availability_status, max_attempts=5):
    """
    Retrieves the ID and token of an available and active server from the database.

    Args:
        availability_status: The status of server availability.
        max_attempts: The maximum number of attempts to retrieve a server.

    Returns:
        Tuple containing the server ID and token, or None if no suitable server is found.
    """
    attempt = 0

    while attempt < max_attempts:
        try:
            with SESSION() as session:
                server = (
                    session.query(ModelServers)
                    .filter(
                        ModelServers.availability
                        == cs.SERVER_AVAILABILITY[availability_status],
                        ModelServers.status == cs.SERVER_STATUS["ACTIVE"],
                    )
                    .order_by(func.random())
                    .limit(1)
                    .first()
                )

                if server is None:
                    error_logger(
                        f"No active server is available. attempt: {attempt}"
                    )

                    if attempt == max_attempts - 1:
                        message = f"No active server is available. Need to check the server status."
                        send_slack_message(client, channel_id, message)

                    attempt += 1
                    continue
                else:
                    server.availability = cs.SERVER_AVAILABILITY[
                        "NOT_AVAILABLE"
                    ]
                    session.commit()
                    info_logger(f"server id: {server.id} is selected")
                    return server.id, server.token
        except Exception as e:
            if session:
                session.rollback()
                session.close()
            error_logger(f"Failed to change server status: {e}")
            attempt += 1

    return None, None


def update_server_status(server_id, was_request_successful):
    """
    Updates the status and request success/failure counts of a server in the database.
    Args:
        server_id (int): The ID of the server.
        was_request_successful (bool): Whether the last request was successful.
    Returns:
        None
    """
    session = None
    try:
        with SESSION() as session:
            server = (
                session.query(ModelServers)
                .filter(ModelServers.id == server_id)
                .first()
            )
            server.availability = cs.SERVER_AVAILABILITY["AVAILABLE"]
            server.request_count += 1

            # Updates the success or failure count of the server based on the request status.
            if was_request_successful:
                server.success_count += 1
            else:
                server.failure_count += 1

            # Evaluates and updates the status of the server based on its failure count.
            if server.failure_count > FAILURE_THRESHOLD:
                server.status = cs.SERVER_STATUS["INACTIVE"]
            session.commit()
            info_logger(
                f"server AVAILABLE status updated successfully. server_id: {server_id}, was_request_successful: {was_request_successful},"
            )
    except Exception as e:
        if session:
            session.rollback()
            session.close()
        error_logger(f"Error while fetching data in database: {e}")


def update_server_counts(server, was_request_successful):
    """
    Updates the success or failure count of the server based on the request status.
    Args:
        server (ModelServers): The server object from the database.
        was_request_successful (bool): Whether the last request was successful.
    Returns:
        None
    """
    if was_request_successful:
        server.success_count += 1
    else:
        server.failure_count += 1
    return server


def evaluate_server_status(server):
    """
    Evaluates and updates the status of the server based on its failure count.
    Args:
        server (ModelServers): The server object from the database.
    Returns:
        None
    """
    if server.failure_count > FAILURE_THRESHOLD:
        server.status = cs.SERVER_STATUS["INACTIVE"]
    return server


def get_random_proxy():
    """
    Returns a random proxy from the list of proxies.
    Returns:
        str: A random proxy.
    """
    return random.choice(PROXIES)


def bard_request(question, json_format=True, text_length=512):
    """
    Send a request to Bard API to get an answer for a given question.

    Args:
        question (str): The question to be sent to the Bard API.
        json_format (bool): Flag indicating whether the response should be in JSON format.
        text_length (int): The maximum length of the text to be sent to the Bard API.
    Returns:
        dict: A dictionary containing the Bard API response, including the answer content and retry count.
    """
    result = {}
    try:
        for retry_count in range(5):
            server_id = None
            try:
                result = generate_text(question, text_length)
                result = parse_bard_response(result)
                result["retry_count"] = retry_count + 1

                if not json_format:
                    return result

                if result["json_answer"]:
                    break
                else:
                    time.sleep(3)
                    continue
            except Exception as e:
                info_logger(f"Failed to get answer from Bard API: {e}")
                continue
    except Exception as e:
        info_logger(f"Bard API failed with error: {e}")

    return result


def parse_bard_response(answer):
    """
    Parse the response from Bard API into a valid JSON format.

    Args:
        answer (str): The response string from Bard API.

    Returns:
        dict: A dictionary containing the base answer and the parsed JSON answer.
    """

    try:
        json_answer = json.loads(answer)
        return {"base_answer": answer, "json_answer": json_answer}
    except Exception as e:
        info_logger(f"Bard API JSON loads convert: {answer} {e}")

    # If converting to JSON using failed json.loads, try using ast.literal_eval
    try:
        json_answer = ast.literal_eval(answer)
        return {"base_answer": answer, "json_answer": json_answer}
    except Exception as e:
        info_logger(f"Bard API JSON ast convert: {answer} {e}")

    # If converting to JSON using ast.literal_eval failed, try using regex
    try:
        # Regex pattern to find the JSON object
        pattern = r"\{.*\}"
        # Find the JSON object using the regex pattern
        match = re.search(pattern, answer, re.DOTALL)
        # Extract the matched JSON object
        return {
            "base_answer": answer,
            "json_answer": json.loads(match.group().replace("%", "")),
        }
    except Exception as e:
        info_logger(f"Finding direct dict in Bard API response: {answer} {e}")

    # If converting to JSON using regex failed, try using string slicing
    try:
        # Find the starting and ending positions of the JSON string
        json_start = answer.find("{")
        json_end = answer.rfind("}") + 1
        # Extract the JSON string
        json_data = (
            answer[json_start:json_end].replace("`", '"').replace("\n", "")
        )
        # Convert to dict and return
        json_answer = ast.literal_eval(json_data)
        return {"base_answer": answer, "json_answer": json_answer}
    except Exception as e:
        info_logger(f"Bard API JSON Finding: {answer} {e}")
        return {"base_answer": answer, "json_answer": None}


def matching_score_class(matching_score):
    """
    Convert matching score to a class.
    Args:
        matching_score: matching score

    Returns:
        str: matching score class
    """

    if matching_score >= 75:
        return "high"
    elif matching_score >= 50:
        return "medium"
    else:
        return "low"


def summary_cleaning(matching_score, summary):
    """
    Clean the summary returned by the Bard API.
    Args:
        matching_score: matching score
        summary: summary returned by the Bard API

    Returns:
        str: cleaned summary
    """

    pattern = r"(?i)\b(?:medium|low|high)\b"
    replacement = matching_score_class(int(matching_score))
    replaced_summary = re.sub(pattern, replacement, summary, count=1)
    return replaced_summary


def fake_score(rating_class):
    """
    Create fake score using class
    Args:
        rating_class: rating class

    Returns:
        fake score to support old code
    """
    if rating_class == 2:
        return random.randint(75, 100)
    elif rating_class == 1:
        return random.randint(50, 74)
    else:
        return random.randint(0, 49)


def format_llm_scoring(result, retry_count):
    """
    Convert valid scores and summary to a response format for scoring.

    Args:
        result (dict): Dictionary containing skill_matching_score (int), work_experience_matching_score (int),
                       education_matching_score (int), and summary (str).
        retry_count: retry counts
    Returns:
        dict: A dictionary with keys "score" and a flag "force_update" indicating whether the
        response needs to be sent to the backend.
    """
    resume_score = {
        "matching_score": None,
        "class": None,
        "summary": None,
    }

    if result:
        if "class" in result:
            resume_score["class"] = result.get("class")
            resume_score["matching_score"] = float(
                fake_score(int(resume_score["class"]))
            )

        if "matching_score" in resume_score and "summary" in result:
            summary = result.get("summary")
            # replacing unmatched summary words with score class
            # i.e., when candidate score is low but summary has "High Match".
            # we replace them with "Low Match"
            resume_score["summary"] = summary_cleaning(
                resume_score["matching_score"], summary
            )

    if retry_count:
        resume_score["retry_count"] = retry_count

    return resume_score


def clean_score(input_str):
    """
    Convert an integer or string score to a float in the range of 0 to 100.

    Args:
        input_str (str): The input score string, e.g., "80/100", "80%", or "80".

    Returns:
        float: The cleaned score value in the range of 0 to 100.
    """
    output_score = float(str(input_str).split("/")[0].replace("%", ""))
    if output_score in range(0, 101):
        return output_score
    else:
        return None


def bard_score(job_data, resume_text):
    """
    Calculate the candidate score based on skills, education, and work experience.

    Args:
        job_data (dict): Dictionary containing job data, including job id (int), job title (str), and job description (str).
        resume_text (str): The raw text of the candidate's resume.

    Returns:
        Tuple: A tuple containing a dictionary with score information and a message about the status of the response.
        The score information dictionary has keys "score". Additionally, it also contains "scoring_retry_count" for tracking
        the number of retries.
    """
    (
        first_response_summary,
        first_response,
        formatted_response,
        second_response,
    ) = ({}, {}, {}, {})
    # for making title at first position
    job_summary = {"title": job_data.get("title", {})}
    job_summary_without_title = job_data.get("summary")
    job_summary_without_title.pop("title")
    job_summary.update(job_summary_without_title)

    if resume_text and job_summary:
        # resume summary prompt
        prompt1 = cs.bard_prompt_1.format(
            resume_text=resume_text,
            json_format="{'skills': <list>, 'education': <detailed str>, 'experience':<detailed str>}",
        )
        first_response = bard_request(
            prompt1, json_format=True, text_length=1024
        )
        # if we are not able to make JSON from first response, then we will use base answer: direct response from llm
        # and resume text
        first_response_summary = first_response.get("json_answer")

        if not first_response_summary:
            first_response_summary_text = first_response.get(
                "base_answer", resume_text
            )
        else:
            first_response_summary_text = first_response_summary
        info_logger(f"first_response_summary: {first_response_summary_text}")

        # scoring prompt
        prompt2 = cs.bard_prompt_2.format(
            job_summary=job_summary,
            resume_summary=first_response_summary_text,
            json_format='{"class": <int>, "summary":<str>}',
        )
        second_response = bard_request(
            prompt2, json_format=True, text_length=256
        )
        formatted_response = format_llm_scoring(
            second_response.get("json_answer", {}),
            second_response.get("retry_count", None),
        )

        return (
            first_response_summary,
            first_response.get("base_answer", ""),
            formatted_response,
            second_response.get("base_answer", ""),
        )
    else:
        error_logger("Job_data or Resume text is not available")
        return (
            first_response_summary,
            first_response.get(
                "base_answer", "Resume summary is not available"
            ),
            formatted_response,
            second_response.get(
                "base_answer",
                "Job_data or Resume text is not available or score not calculated",
            ),
        )
