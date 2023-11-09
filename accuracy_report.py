"""
This is standalone script for analysis.
"""
import ast
import json

from dotenv import load_dotenv

load_dotenv("./main_api/constants.env")

import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from datetime import datetime, timedelta
from tabulate import tabulate
import os
from slack import WebClient
from slack.errors import SlackApiError

# These import are from main_api because this is standalone script for analysis
from main_api.utils.db_models import (
    Resume as ModelResume,
    SyncResume as ModelSyncResume,
    JDescriptions as ModelJDescriptions,
)
from main_api.utils.logging_helpers import info_logger, error_logger

SLACK_BOT_USER_OAUTH_TOKEN = os.getenv("SLACK_BOT_USER_OAUTH_TOKEN")
client = WebClient(token=SLACK_BOT_USER_OAUTH_TOKEN)

channel_id = (
    os.getenv("SLACK_ATS_CHANNEL_ID")
    if os.getenv("ENVIRONMENT") == "production"
    else os.getenv("SLACK_ATS_STAGING_CHANNEL_ID")
)

DATABASE_URL = os.getenv("DATABASE_URL")
REVERSE_TEXT_EXTRACTION_TYPE = ast.literal_eval(
    os.getenv("REVERSE_TEXT_EXTRACTION_TYPE")
)


def get_data_from_db(table_name, start_date, end_date):
    """
    connect db and get data from resume table

    Args:
        table_name: table module name
        start_date: previous day date
        end_date: today's date

    Returns:
        list of resume output json
    """
    engine = create_engine(DATABASE_URL)
    Session = sessionmaker(engine)
    with Session() as session:
        query = (
            session.query(table_name.json_output)
            .filter(table_name.time_created.between(start_date, end_date))
            .all()
        )
        result_list = [r[0] for r in query]
    return result_list


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


def clean_df(result_list):
    """
    convert list to clean df

    Args:
        result_list:list of resume json output

    Returns: df of json

    """
    result_list_new = []
    for resumes in result_list:
        if isinstance(resumes, str):
            result_list_new.append(json.loads(resumes))
        else:
            result_list_new.append(resumes)

    result_df = pd.DataFrame(result_list_new)
    result_df = result_df.fillna(value=np.nan)
    result_df = result_df.replace("", np.NAN)
    result_df = result_df.mask(result_df.applymap(str).eq("[]"))
    return result_df


def extraction_calculation(result_df, overall_extraction_class):
    """
    Calculate total, found, not found and percentage for each entity and overall too

    Args:
        result_df: dataframe
        overall_extraction_class: list of useful columns

    Returns:
        table_data_list: list of total, found, not found and percentage for each entity ,
        extraction_percentage: overall percentage
    """
    table_data_list = []
    overall_count = 0
    for column in overall_extraction_class:
        if column not in result_df:
            null_count = 0
            not_null_count = 0
            total_count = 0
            percentage = 0
            table_data_list.append(
                [
                    column.title(),
                    total_count,
                    not_null_count,
                    null_count,
                    "{:.2%}".format(percentage),
                ]
            )
            overall_count += percentage
        else:
            null_count = result_df[column].isna().sum()
            not_null_count = result_df[column].count()
            total_count = len(result_df[column])
            percentage = not_null_count / total_count
            table_data_list.append(
                [
                    column.title(),
                    total_count,
                    not_null_count,
                    null_count,
                    "{:.2%}".format(percentage),
                ]
            )
            overall_count += percentage
    extraction_percentage = overall_count / len(overall_extraction_class)
    entity_report_table = (
        "\n\t\t:mag_right: *Entity Extraction Report* :mag:\n```"
        + tabulate(
            table_data_list,
            headers=["Entity", "Total", "Found", "Not Found", "Percentage"],
            tablefmt="simple",
        )
        + "```"
    )
    return entity_report_table, extraction_percentage


def get_time_report(time_taken):
    """
    Calculate min, max , average and total time and return in one str

    Args:
        time_taken: serious of time column

    Returns:
        return in one str with min, max, average and total time
    """
    if len(time_taken) == 0:
        return (
            "\n\t\t:hourglass_flowing_sand: *Time Report *:hourglass_flowing_sand:\n"
            + "No time found"
        )
    else:
        return (
            "\n\t\t:hourglass_flowing_sand: *Time Report *:hourglass_flowing_sand:\n"
            + "```Minimum: "
            + "{:.2f}".format(time_taken.min())
            + " seconds"
            + "\nMaximum: "
            + "{:.2f}".format(time_taken.max())
            + " seconds"
            + "\nAverage: "
            + "{:.2f}".format(time_taken.mean())
            + " seconds"
            + "\nTotal: "
            + "{:.2f}".format(time_taken.sum())
            + " seconds```"
        )


def get_section_report(sections):
    """
    Calculate total, found, not found and percentage for each section

    Args:
        sections: serious of section column

    Returns:
        return in one str with min, max, average and total time
    """
    if len(sections) == 0:
        section_report_table = (
            "\n\t\t:page_facing_up: *Section Identification Report* :page_facing_up:\n"
            + "No section found"
        )
    else:
        sections = sections.apply(lambda d: d if isinstance(d, dict) else {})
        sections_df = pd.DataFrame(sections.to_list())
        sections_df = sections_df.fillna(value=np.nan)
        sections_df = sections_df.replace("", np.NAN)
        sections_df = sections_df.mask(sections_df.applymap(str).eq("[]"))
        table_data_list = []
        for column in sections_df:
            null_count = sections_df[column].isna().sum()
            notnull_count = sections_df[column].count()
            total_count = len(sections_df[column])
            percentage = notnull_count / total_count
            table_data_list.append(
                [
                    column.title(),
                    total_count,
                    notnull_count,
                    null_count,
                    "{:.2%}".format(percentage),
                ]
            )
        section_report_table = (
            "\n\t\t:page_facing_up: *Section Identification Report* :page_facing_up:\n```"
            + tabulate(
                table_data_list,
                headers=[
                    "Section",
                    "Total",
                    "Found",
                    "Not Found",
                    "Percentage",
                ],
                tablefmt="simple",
            )
            + "```"
        )
    return section_report_table


def get_text_type_report(text_types):
    """
    Calculate type of text type used for resume parsing

    Args:
        text_types: text_type column

    Returns:
        return in text type total
    """
    if len(text_types) == 0:
        text_type_report_table = (
            "\n\t\t:page_facing_up: *Text Types Report* :page_facing_up:\n"
            + "No section found"
        )
    else:
        report_table = dict(text_types.value_counts())
        text_type_report_table = (
            "\n\t\t:page_facing_up: *Text Types Report* :page_facing_up:\n```"
        )
        for key, value in report_table.items():
            text_type_report_table += (
                REVERSE_TEXT_EXTRACTION_TYPE[key] + ": " + str(value) + "\n"
            )
        text_type_report_table += "Total: " + str(len(text_types)) + "```"
    return text_type_report_table


def get_error_report(errors):
    """
    Calculate total error, distinct errors and count

    Args:
        errors: serious of errors column

    Returns:
        return in one str with distinct errors and count
    """
    if len(errors) == 0:
        return (
            "\n\t\t:name_badge: *Error Report* :name_badge:\n"
            + "No error found"
        )

    else:
        errors = errors.replace(np.NAN, "Successfully parsed")
        errors = errors.apply(lambda x: str(x)[:75])
        error_report = (
            "\n\t\t:name_badge: *Error Report* :name_badge:\n```"
            + tabulate(
                list(dict(errors.value_counts()).items())
                + [("Total requests", len(errors))],
                headers=["Error", "Count"],
                tablefmt="simple",
            )
            + "```"
        )
        return error_report


current_time = datetime.now().replace(
    hour=0, minute=0, second=0, microsecond=0
)
one_day_interval_before = (
    current_time - timedelta(days=int(os.getenv("REPORT_PERIOD")))
).replace(hour=0, minute=0, second=0, microsecond=0)

###################### Resume Parser ########################
result_async = get_data_from_db(
    ModelResume, one_day_interval_before, current_time
)
result_async = clean_df(result_async)
result_sync = get_data_from_db(
    ModelSyncResume, one_day_interval_before, current_time
)
result_sync = clean_df(result_sync)
result = pd.concat([result_async, result_sync], ignore_index=True)

if len(result) != 0:
    entity_report_table, extraction_percentage = extraction_calculation(
        result,
        overall_extraction_class=[
            "name",
            "email",
            "total_experience",
            "skills",
            "education",
            "experience",
            "phone_number",
        ],
    )

    message = (
        f":spyparrot: *Report of Resume Parser ({os.getenv('ENVIRONMENT')})* :spyparrot:\n"
        + f"*From:* {one_day_interval_before.strftime('%d/%m/%Y %H:%M:%S')} *to:* "
        + f"{current_time.strftime('%d/%m/%Y %H:%M:%S')} \n"
        + "*Extraction Accuracy:* "
        + "{:.2%}".format(extraction_percentage)
    )

    send_slack_message(client, channel_id, message)
    # entity report
    send_slack_message(client, channel_id, entity_report_table)
    info_logger(message, entity_report_table)
    # Temporary Blocked
    # if 'sections' in result:
    #     section_report_table = get_section_report(result['sections'])
    #     send_slack_message(client, channel_id, section_report_table)
    #     info_logger(section_report_table)
    # else:
    #     section_report_table = get_section_report([])
    #     send_slack_message(client, channel_id, section_report_table)
    #     info_logger(section_report_table)
    # request types report
    request_types_report = (
        "\n\t\t:grey_question: *Request Types Report* :grey_question:\n```"
        + "Sync Requests: "
        + str(len(result_sync))
        + "\n"
        + "Async Requests: "
        + str(len(result_async))
        + "\n"
        + "Total Requests: "
        + str(len(result))
        + "```"
    )

    send_slack_message(client, channel_id, request_types_report)
    # text types report
    if "text_type" in result:
        text_type_report_table = get_text_type_report(result["text_type"])
        send_slack_message(client, channel_id, text_type_report_table)
        info_logger(text_type_report_table)
    else:
        text_type_report_table = get_text_type_report([])
        send_slack_message(client, channel_id, text_type_report_table)
        info_logger(text_type_report_table)
    # time report
    if "taken_time" in result:
        time_report = get_time_report(result["taken_time"])
        send_slack_message(client, channel_id, time_report)
        info_logger(time_report)
    else:
        time_report = get_time_report([])
        send_slack_message(client, channel_id, time_report)
        info_logger(time_report)
    # error report
    if "error" in result:
        error_report = get_error_report(result["error"])
        send_slack_message(client, channel_id, error_report)
        info_logger(error_report)
    else:
        error_report = get_error_report([])
        send_slack_message(client, channel_id, error_report)
        info_logger(error_report)
else:
    message = (
        f":spyparrot: *Report of Resume Parser ({os.getenv('ENVIRONMENT')})* :spyparrot:\n"
        + f"*From:* {one_day_interval_before.strftime('%d/%m/%Y %H:%M:%S')} *to:* "
        + f"{current_time.strftime('%d/%m/%Y %H:%M:%S')} \n"
        + "No request found in given times"
    )
    info_logger(message)
    send_slack_message(client, channel_id, message)

####################### JD Parser ########################
result = get_data_from_db(
    ModelJDescriptions, one_day_interval_before, current_time
)
result = clean_df(result)

if len(result) != 0:
    entity_report_table, extraction_percentage = extraction_calculation(
        result,
        [
            "job_title",
            "employment_type",
            "skills",
            "education",
            "work_experience",
            "location",
            "benefits_perks",
            "salary_range",
            "industry_type",
        ],
    )

    message = (
        f":spyparrot: *Report of JD Parser ({os.getenv('ENVIRONMENT')})* :spyparrot:\n"
        + f"*From:* {one_day_interval_before.strftime('%d/%m/%Y %H:%M:%S')} *to:* "
        + f"{current_time.strftime('%d/%m/%Y %H:%M:%S')} \n"
        + "*Extraction Accuracy:* "
        + "{:.2%}".format(extraction_percentage)
    )

    send_slack_message(client, channel_id, message)
    send_slack_message(client, channel_id, entity_report_table)
    info_logger(message, entity_report_table)
    if "taken_time" in result:
        time_report = get_time_report(result["taken_time"])
        send_slack_message(client, channel_id, time_report)
        info_logger(time_report)
    else:
        time_report = get_time_report([])
        send_slack_message(client, channel_id, time_report)
        info_logger(time_report)
else:
    message = (
        f":spyparrot: *Report of JD Parser ({os.getenv('ENVIRONMENT')})* :spyparrot:\n"
        + f"*From:* {one_day_interval_before.strftime('%d/%m/%Y %H:%M:%S')} *to:* "
        + f"{current_time.strftime('%d/%m/%Y %H:%M:%S')} \n"
        + "No request found in given times"
    )
    info_logger(message)
    send_slack_message(client, channel_id, message)
