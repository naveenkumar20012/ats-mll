import ast
import json
import os
import re
from difflib import SequenceMatcher

import requests
from RP_parser.helper_codes.chatgpt_service import ChatGPT
from RP_parser.helper_files import constants as cs
from bardapi import Bard
from utils.logging_helpers import info_logger, error_logger

vicuna_Api = os.getenv("VICUNA_API")
calculate_skill_score = float(os.getenv("skill_rating_ratio"))
skill_edu_ratio = float(os.getenv("skill_edu_ratio"))
exp_edu_ratio = float(os.getenv("exp_edu_ratio"))
skill_ratio = float(os.getenv("skill_ratio"))
edu_ratio = float(os.getenv("edu_ratio"))
exp_ratio = float(os.getenv("exp_ratio"))
bard_token = str(os.getenv("BARD_TOKEN"))


def calculated_skill_score(
    job_skills, candidate_skills, calculate_skill_score
):
    """
    It will calculate common skill score with job skills
    Args:
        job_skills: list of all job skills
        candidate_skills: list of all candidate skills
        calculate_skill_score: how many percentage of job skills should consider as minimum.

    Returns:
        skills score in float number 0 to 1.0
    """
    setA = set(job_skills)
    setB = set(candidate_skills)
    overlap = setA & setB
    if len(overlap) != 0:
        skill_score = float(len(overlap)) / (len(setA) * calculate_skill_score)
        return min(1.0, skill_score)
    else:
        return 0


def calculate_exp_score(jd_exp_range, candidate_exp):
    """
    Find candidate experience is in range of job's experience range or not
    Args:
        jd_exp_range: set of minimum and maximum months of experience
        candidate_exp: total experience of candidate

    Returns:
        exp score in float number 0, 0.5, 1
    """
    if candidate_exp in range(jd_exp_range[0], jd_exp_range[1]):
        return 1
    elif candidate_exp > jd_exp_range[1]:
        return 0.5
    else:
        return 0


def find_same_title(jd_title, candidate_titles):
    """
    Find candidate designation is it same as job title.
    Args:
        jd_title: job title
        candidate_titles: candidate_titles

    Returns:
        exp_score2 return 0 and 1
    """
    for cd_title in candidate_titles:
        if cd_title:
            if SequenceMatcher(None, cd_title, jd_title).ratio() > 0.75:
                return 1
    return 0


def calculate_edu_score(jd_educations, candidate_educations):
    """
    Find candidate education and job required education is mathing or not
    Args:
        jd_educations: required education for job
        candidate_educations: candidate education

    Returns:
        edu score in decimal number 0 or 1
    """
    candidate_all_title_courses = []
    for candidate_education in candidate_educations:
        if "title" in candidate_education:
            candidate_all_title_courses.append(candidate_education["title"])
        if "course" in candidate_education:
            candidate_all_title_courses.append(candidate_education["course"])
    candidate_all_title_courses = list(set(candidate_all_title_courses))
    try:
        for candidate_all_title_course in candidate_all_title_courses:
            for jd_education in jd_educations:
                if (
                    SequenceMatcher(
                        None, candidate_all_title_course, jd_education
                    ).ratio()
                    > 0.75
                ):
                    return 1
        else:
            return 0
    except Exception as ex:
        info_logger(ex)
        return 0


def calculate_rating(job_data, FINAL_DIC_RESUME):
    """
    Calculate candidate score based on skills, education and total experience
    Args:
        job_data: dictionary with all data of job
        FINAL_DIC_RESUME: dictionary with all data of candidate

    Returns:
        score in int number 0 to 5 or None
    """
    if job_data is not None:
        job_title = job_data["title"]
        job_skills = list(job_data["skill"])
        jd_exp_range = (
            int(float(job_data["min_experience"]) * 12),
            int(float(job_data["max_experience"]) * 12),
        )
        jd_educations = list(job_data["education"])
        candidate_skills, candidate_exp, candidate_educations = (
            FINAL_DIC_RESUME["skills"],
            FINAL_DIC_RESUME["total_experience"],
            FINAL_DIC_RESUME["education"],
        )
        candidate_titles = []
        for i in FINAL_DIC_RESUME["experience"]:
            if FINAL_DIC_RESUME["designation"]:
                candidate_titles.append(i["designation"])
        if candidate_skills:
            candidate_skills = [skill.lower() for skill in candidate_skills]
        if len(job_skills) == 0 and len(jd_educations) == 0:
            return None, None, None, None
        elif jd_educations or candidate_educations:
            skill_score, exp_score, all_score = 0, 0, 0
            if job_skills and candidate_skills:
                skill_score = (
                    calculated_skill_score(
                        job_skills, candidate_skills, calculate_skill_score
                    )
                    * skill_edu_ratio
                )
            if (
                jd_exp_range
                and candidate_exp
                and candidate_titles
                and job_title
            ):
                exp_score1 = (
                    calculate_exp_score(jd_exp_range, candidate_exp) * 75
                )
                exp_score2 = find_same_title(job_title, candidate_titles) * 25
                exp_score = (exp_score1 + exp_score2) * exp_edu_ratio
            elif jd_exp_range and candidate_exp:
                exp_score = (
                    calculate_exp_score(jd_exp_range, candidate_exp)
                    * exp_edu_ratio
                )
            all_score = round((skill_score + exp_score) / 2)
            return all_score, skill_score, exp_score, None
        else:
            skill_score, exp_score, edu_score, all_score = 0, 0, 0, 0
            if job_skills and candidate_skills:
                skill_score = (
                    calculated_skill_score(
                        job_skills, candidate_skills, calculate_skill_score
                    )
                    * skill_ratio
                )
            if jd_exp_range and candidate_exp:
                exp_score = (
                    calculate_exp_score(jd_exp_range, candidate_exp)
                    * exp_ratio
                )
            if jd_educations and candidate_educations:
                edu_score = (
                    calculate_edu_score(jd_educations, candidate_educations)
                    * edu_ratio
                )
            all_score = round((skill_score + exp_score + edu_score) / 3)
            return all_score, skill_score, exp_score, edu_score
    else:
        return None, None, None, None


def calculate_chatgpt_score(job_data, resume_text):
    """
    Calculate candidate score based on skills, education and work experience.

    Args:
        job_data (dict): Dictionary containing job data, including job id (int), job title (str), and job description (str).
        resume_text (str): The raw text of the candidate's resume.

    Returns:
        Tuple: A tuple containing a dictionary with score information and a message about the status of the response.
        The score information dictionary has keys "score". Additionally, it also contains "scoring_retry_count" for tracking
        the number of retries.
    """
    if (
        job_data["description"] is not None
        and job_data["title"] is not None
        and resume_text is not None
    ):
        job_description = cs.job_datas.get(str(job_data.get("id", "")), {})

        if not job_description:
            return {}, "job_data is not available"

        role = cs.rating_role.format(
            json_format='{"skill_matching_score": <int> [in range(0,100)], "work_experience_matching_score": <int> [in range(0,100)], "education_matching_score": <int> [in range(0,100)]}'
        )
        question = cs.rating_prompt2.format(
            job_title=job_data["title"],
            jd_text=job_description,
            resume_text=resume_text,
        )

        chatgpt_service = ChatGPT()
        response = chatgpt_service.chatgpt_request_openai(role, question)
        formatted_response = format_chatgpt_scoring_3(
            response.get("json_answer", {})
        )
        formatted_response.update(
            {"scoring_retry_count": response.get("retry_count", None)}
        )
        info_logger(
            f"base answer: {response['base_answer']}\n formatted_response: {formatted_response}"
        )
        return formatted_response, response.get("base_answer", "")
    else:
        error_logger(
            "Job_title or job_description or Resume test is not available"
        )
        return (
            {
                "score": {
                    "skill_matching_score": None,
                    "work_experience_matching_score": None,
                    "education_matching_score": None,
                    "resume_matching_score": None,
                }
            },
            "Job_title or job_description or Resume test is not available",
        )


def llm_request(question, json_format=False):
    result = {}
    try:
        for retry_count in range(5):
            try:
                body = json.dumps({"prompt": question})
                response = requests.post(vicuna_Api, data=body)
                response = response.json()
                text = response.get("generated_text")
                result = llm_parse_json(text)
                result["retry_count"] = retry_count + 1
                info_logger(result)

                if not json_format:
                    return result

                if result["json_answer"]:
                    break
            except Exception as e:
                info_logger(f"Failed to get Answer from ChatGPT: {e}")
                continue
    except Exception as e:
        info_logger(f"ChatGPT failed with error: {e}")
    return result


def llm_parse_json(answer):
    """parse chatGPT response in valid json"""
    try:
        json_answer = ast.literal_eval(answer)
        return {"base_answer": answer, "json_answer": json_answer}
    except Exception as e:
        info_logger(f"LLM json ast convert: {answer} {e}")
    # if chatGPT response to json ast.literal_eval failed, trying with json.loads
    try:
        json_answer = json.loads(answer)
        return {"base_answer": answer, "json_answer": json_answer}
    except Exception as e:
        info_logger(f"LLM json loads convert: {answer} {e}")
    try:
        matching_score = re.search(
            r'"matching_score": (\d{2,3})', answer
        ).group(1)
        summary = re.search(
            r'"summary"\s*:\s*({.*?})|"summary"\s*:\s*({.*?})|"summary":\s*({.*?})',
            answer,
        ).group(1)
        return {
            "base_answer": answer,
            "json_answer": {
                "matching_score": matching_score,
                "summary": summary,
            },
        }
    except Exception as e:
        info_logger(f"finding matching_score on LLM output: {answer} {e}")
        return {"base_answer": answer, "json_answer": None}


def llm_score(job_data, resume_text):
    """
    Calculate candidate score based on skills, education and work experience.

    Args:
        job_data (dict): Dictionary containing job data, including job id (int), job title (str), and job description (str).
        resume_text (str): The raw text of the candidate's resume.

    Returns:
        Tuple: A tuple containing a dictionary with score information and a message about the status of the response.
        The score information dictionary has keys "score". Additionally, it also contains "scoring_retry_count" for tracking
        the number of retries.
    """
    if (
        job_data.get("description", None) is not None
        and job_data.get("title", None) is not None
        and resume_text is not None
    ):
        job_description = cs.job_datas.get(str(job_data.get("id", "")), {})

        if not job_description:
            return {}, "job_data is not available"

        prompt1 = cs.vicuna_prompt_1.format(
            resume_text=resume_text,
            json_format="'skills': <list>, 'education': <detailed str>, 'experience':<detailed str>",
        )
        first_response = llm_request(prompt1)
        first_response = first_response.get(
            "json_answer", first_response.get("base_answer", "")
        )
        prompt2 = cs.vicuna_prompt_2.format(
            job_title=job_data["title"],
            jd_text=job_description,
            resume_summary=first_response,
            json_format='{"matching_score": <int> [in range(0,100)],"summary": <str>}',
        )
        second_response = llm_request(prompt2, True)
        formatted_response = format_llm_scoring(
            second_response.get("json_answer", {})
        )
        info_logger(
            f"base answer: {'first_response' + first_response + 'second_response' + second_response.get('base_answer', ' No answer')}\n formatted_response: {formatted_response}"
        )
        return (
            first_response,
            first_response.get("base_answer", ""),
            formatted_response,
            second_response.get("base_answer", ""),
        )
    else:
        error_logger(
            "Job_title or job_description or Resume test is not available"
        )
        return (
            {
                "score": {
                    "matching_score": None,
                    "summary": None,
                }
            },
            "Job_title or job_description or Resume test is not available",
        )


def bard_request(question, json_format=False):
    result = {}
    try:
        for retry_count in range(5):
            try:

                result = Bard(token=bard_token).get_answer(question)["content"]
                result = bard_parse_json(result)
                result["retry_count"] = retry_count + 1
                info_logger(result)

                if not json_format:
                    return result

                if result["json_answer"]:
                    break
            except Exception as e:
                info_logger(f"Failed to get Answer from ChatGPT: {e}")
                continue
    except Exception as e:
        info_logger(f"ChatGPT failed with error: {e}")
    return result


def bard_score(job_data, resume_text):
    """
    Calculate candidate score based on skills, education and work experience.

    Args:
        job_data (dict): Dictionary containing job data, including job id (int), job title (str), and job description (str).
        resume_text (str): The raw text of the candidate's resume.

    Returns:
        Tuple: A tuple containing a dictionary with score information and a message about the status of the response.
        The score information dictionary has keys "score". Additionally, it also contains "scoring_retry_count" for tracking
        the number of retries.
    """
    if (
        # job_data.get("description", None) is not None
        # and job_data.get("title", None) is not None
        # and
        resume_text
        is not None
    ):
        job_description = cs.job_datas.get(str(job_data.get("id", "")), {})

        if not job_description:
            return {}, "job_data is not available"

        prompt1 = cs.bard_prompt_1.format(
            resume_text=resume_text,
            json_format="{'skills': <list>, 'education': <detailed str>, 'experience':<detailed str>}",
        )
        first_response = bard_request(prompt1)
        first_response_summary = first_response.get(
            "json_answer", first_response.get("base_answer", resume_text)
        )
        prompt2 = cs.bard_prompt_2.format(
            job_title=job_description.get("title", job_data.get("title", "")),
            job_summary=job_description,
            resume_summary=first_response_summary,
            json_format='{"matching_score": <int> [in range(0,100)],"summary": <str>}',
        )
        second_response = bard_request(prompt2, True)
        formatted_response = format_llm_scoring(
            second_response.get("json_answer", {})
        )
        # info_logger(
        #     "resume_summary: "
        #     + str(first_response_summary)
        #     + "\n base_answer: "
        #     + first_response.get("base_answer", "")
        #     + "\n scoring: "
        #     + str(formatted_response)
        #     + "\n base_answer: "
        #     + second_response.get("base_answer", "")
        # )
        return (
            first_response_summary,
            first_response.get("base_answer", ""),
            formatted_response,
            second_response.get("base_answer", ""),
        )
    else:
        error_logger(
            "Job_title or job_description or Resume test is not available"
        )
        return (
            {},
            "resume summary is not available",
            {
                "score": {
                    "matching_score": None,
                    "summary": None,
                }
            },
            "Job_title or job_description or Resume test is not available",
        )


def bard_parse_json(answer):
    """parse chatGPT response in valid json"""
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
        info_logger(f"finding direct dict on LLM output: {answer} {e}")

    try:
        json_answer = ast.literal_eval(answer)
        return {"base_answer": answer, "json_answer": json_answer}
    except Exception as e:
        info_logger(f"LLM json ast convert: {answer} {e}")
    # if chatGPT response to json ast.literal_eval failed, trying with json.loads
    try:
        json_answer = json.loads(answer)
        return {"base_answer": answer, "json_answer": json_answer}
    except Exception as e:
        info_logger(f"LLM json loads convert: {answer} {e}")
    try:
        matching_score = re.search(
            r'"matching_score": (\d{2,3})', answer
        ).group(1)
        summary = re.search(
            r'"summary"\s*:\s*({.*?})|"summary"\s*:\s*({.*?})|"summary":\s*({.*?})',
            answer,
        ).group(1)
        return {
            "base_answer": answer,
            "json_answer": {
                "matching_score": matching_score,
                "summary": summary,
            },
        }
    except Exception as e:
        info_logger(f"finding matching_score on LLM output: {answer} {e}")
        return {"base_answer": answer, "json_answer": None}


def formate_chatgpt_scoring(result):
    """
    convert valid score and summary
    Args:
        result: {"score":50,"summary":"I am awesome summary. I can attract you."}

    Returns: valid score and summary

    """
    score, summary = None, None
    try:
        result["score"] = float(str(result["score"]).replace("%", ""))
        if result["score"] in range(0, 101):
            score = result["score"]
        if len(str(result["summary"])) > 10:
            summary = str(result["summary"])
        return score, summary
    except Exception as e:
        error_logger(f"while converting format getting error: {e}")


def formate_chatgpt_scoring_2(result):
    """
    convert valid score and summary
    Args:
        result: {"resume_matching_score": 80,"skill_matching_score": 20, "work_experience_matching_score":50,
         "education_matching_score": 60, "summary":"I am awesome summary. I can attract you."}

    Returns: response with score: {"resume_matching_score": 80,"skill_matching_score": 20, "work_experience_matching_score":50,
         "education_matching_score": 60} , summary:"I am awesome summary. I can attract you." , send_backend flag
        send_backend or not and ChatGPT retry_count
    """
    response = {
        "score": {
            "resume_matching_score": None,
            "skill_matching_score": None,
            "work_experience_matching_score": None,
            "education_matching_score": None,
        },
        "summary": None,
    }
    force_update = False
    if result:
        force_update = True
        score = {
            "resume_matching_score": None,
            "skill_matching_score": None,
            "work_experience_matching_score": None,
            "education_matching_score": None,
        }
        if "resume_matching_score" in result:
            score["resume_matching_score"] = clean_score(
                result["resume_matching_score"]
            )
        if "skill_matching_score" in result:
            score["skill_matching_score"] = clean_score(
                result["skill_matching_score"]
            )
        if "work_experience_matching_score" in result:
            score["work_experience_matching_score"] = clean_score(
                result["work_experience_matching_score"]
            )
        if "education_matching_score" in result:
            score["education_matching_score"] = clean_score(
                result["education_matching_score"]
            )

        response["score"] = score
        if len(str(result["summary"])) > 10:
            response["summary"] = str(result["summary"])

        return {
            "resume_matching_score_data": response,
            "force_update": force_update,
        }
    else:
        return {
            "resume_matching_score_data": response,
            "force_update": force_update,
        }


def format_chatgpt_scoring_3(result):
    """
    Converts valid scores and summary to response format for ChatGPT scoring.

    Args:
        result (dict): Dictionary containing skill_matching_score (int), work_experience_matching_score (int),
                       education_matching_score (int), and summary (str).

    Returns:
        dict: A dictionary with keys "score" and a flag "force_update" indicating whether the
        response needs to be sent to the backend.
    """
    response = {
        "score": {
            "skill_matching_score": None,
            "work_experience_matching_score": None,
            "education_matching_score": None,
        }
    }
    force_update = False
    if result:
        force_update = True
        score = {
            "skill_matching_score": None,
            "work_experience_matching_score": None,
            "education_matching_score": None,
        }
        if "skill_matching_score" in result:
            score["skill_matching_score"] = clean_score(
                result["skill_matching_score"]
            )
        if "work_experience_matching_score" in result:
            score["work_experience_matching_score"] = clean_score(
                result["work_experience_matching_score"]
            )
        if "education_matching_score" in result:
            score["education_matching_score"] = clean_score(
                result["education_matching_score"]
            )

        response["score"] = score

    return {
        "resume_matching_score_data": response,
        "force_update": force_update,
    }


def format_llm_scoring(result):
    """
    Converts valid scores and summary to response format for ChatGPT scoring.

    Args:
        result (dict): Dictionary containing skill_matching_score (int), work_experience_matching_score (int),
                       education_matching_score (int), and summary (str).

    Returns:
        dict: A dictionary with keys "score" and a flag "force_update" indicating whether the
        response needs to be sent to the backend.
    """
    response = {
        "score": {
            "skill_matching_score": None,
            "work_experience_matching_score": None,
            "education_matching_score": None,
            "summary": None,
        }
    }
    force_update = False
    if result:
        force_update = True
        score = {
            "skill_matching_score": None,
            "work_experience_matching_score": None,
            "education_matching_score": None,
            "summary": None,
        }
        if "matching_score" in result:
            score["skill_matching_score"] = score[
                "work_experience_matching_score"
            ] = score["education_matching_score"] = clean_score(
                result["matching_score"]
            )
        if "summary" in result:
            score["summary"] = result["summary"]

        response["score"] = score

    return {
        "resume_matching_score_data": response,
        "force_update": force_update,
    }


def clean_score(input_str):
    """
    Convert int or str score  to float in range of 0 to 100
    Args:
        input_str: like 80/100, 80%, 80

    Returns: 80.0

    """
    output_score = float(str(input_str).split("/")[0].replace("%", ""))
    if output_score in range(0, 101):
        return output_score
    else:
        return None
