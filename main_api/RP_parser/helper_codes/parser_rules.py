import ast
import datetime
import os
import re

import nltk
import requests
from dateparser.search import search_dates
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter, Retry

from RP_parser.helper_files.constants import SECTIONS
from utils.logging_helpers import error_logger
from RP_parser.helper_codes.chatgpt_service import ChatGPT

from RP_parser.helper_codes.edu_service_3 import (
    find_qualification_details_from_chatpgt_response,
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

load_dotenv("./constants.env")

TEXT_SLICING = os.getenv("TEXT_SLICING")
name_Api = os.getenv("NAME_API")
work_Api = os.getenv("WORK_API")
edu_Api = os.getenv("EDU_API")
ner_Api = os.getenv("NER_API")
skill_Api = os.getenv("SKILL_API")


def parse_lines3(texts):
    """
    find sections from text based on dictionary logic
    Args:
        texts: list of text lines of resume

    Returns:
        dictionary of line labels like {'other':[0,5],'skills':[19,28]}
    """
    line_label = {}
    start = 0
    end = len(texts)
    flags = dict.fromkeys(list(SECTIONS.keys()) + ["other"], False)
    flags["other"] = True
    for line_no, line in enumerate(texts):
        line = re.sub(r"[^\w\s]", "", line.lower()).strip()
        for clas, class_objs in SECTIONS.items():
            # if any(SequenceMatcher(None, line, class_obj).ratio() > 0.7 for class_obj in class_objs):
            if line in class_objs:
                for active_class, flag in flags.items():
                    if flag:
                        if active_class != clas:
                            if active_class in line_label.keys():
                                line_label[active_class].append(
                                    [start, line_no]
                                )
                            else:
                                line_label[active_class] = [[start, line_no]]
                            flags[active_class] = False
                        flags[clas] = True
                        start = line_no
    for active_class, flag in flags.items():
        if flag:
            if active_class in line_label.keys():
                line_label[active_class].append([start, end])
            else:
                line_label[active_class] = [[start, end]]
            flags[active_class] = False

    line_label = {
        i: [
            [
                len("\n".join(texts[0 : map[0] + 1])),
                len("\n".join(texts[0 : map[1]])),
            ]
            for map in j
        ]
        for i, j in line_label.items()
    }
    return line_label


def parse_lines4(extracted_entities, max_len):
    """
    find sections from text based on model titles.
    Args:
        extracted_entities: all extracted entities from resumes
        max_len: maximum length of document

    Returns:
         dictionary of line labels like {'other':[0,5],'skills':[19,28]}
    """
    line_labels = []
    for entity in extracted_entities:
        start_char_title = entity[-1].find("title")
        if start_char_title != -1:
            if not line_labels:
                line_labels.append(
                    [entity[-1][: start_char_title - 1], 0, entity[1]]
                )
            else:
                line_labels[-1][-1] = entity[0]
                line_labels.append(
                    [entity[-1][: start_char_title - 1], entity[0], entity[1]]
                )
        if line_labels:
            line_labels[-1][-1] = max_len
    line_label_dict = {}
    for line_label in line_labels:
        line_label_dict[line_label[0]] = [line_label[1:]]
    return line_label_dict


def entities_extractor(text, nlp, nlp2=None):
    """
    extract entities from ner model
    Args:
        text: resume text
        nlp: ner model
        nlp2: based lg model

    Returns:
        list of entities, list of relation entities, document object form spacy
    """
    ent_data, rel_data = [], []
    text = "\n".join(text)
    doc = nlp(text, disable=["tagger"])
    ent_data = [(e.start_char, e.end_char, e.text, e.label_) for e in doc.ents]
    return ent_data, rel_data, doc


def entities_extractor_2(text):
    """
    call ner api and got all entities from ner model
    Args:
        text: resume text

    Returns:
        list of entities, list of relation entities
    """
    ent_data, rel_data = [], []
    try:
        request = requests.Session()
        retries = Retry(
            total=5, backoff_factor=1, status_forcelist=[502, 503, 504]
        )
        request.mount("http://", HTTPAdapter(max_retries=retries))
        ent_data = request.post(ner_Api, json={"text": text}).json()
        return ent_data
    except Exception as e:
        error_logger("NER API is not working", e)
        return []


def get_special_parsed_text(text):
    """
    Divide the text by newline characters.
    Args:
        text: resume text in string format

    Returns:
        Returns list of strings separated by newlines.
    """
    text = text.split("\n")
    return text


##########################################################
##############             NAME             ##############
##########################################################


def extract_name(texts):
    """
    For calling name api and cleaning
    Args:
        texts: whole resume  text

    Returns:
        name
    """
    try:
        names = []
        # in first 10 lines
        name = requests.post(
            name_Api, json={"text": " ".join(texts[:10])}
        ).json()["name"]
        if name != "":
            names.append(name)
        # if not in first 10 line go for other lines
        if len(names) < 1:
            name = requests.post(
                name_Api, json={"text": " ".join(texts[10:])}
            ).json()["name"]
            if name != "":
                names.append(name)

        # name cleaning
        if len(names) > 0:
            names = [re.sub(r"[^A-Za-z .]+", "", name) for name in names]
            return max(names, key=len).title()
        else:
            return None
    except Exception as e:
        error_logger("Name API is not working", e)
        return None


def extract_email_phone(text):
    """
    Extracts unique email and unique phone from the given text.
    Args:
        text: plain text extracted from resume file.

    Returns:
        Returns first detected email and mobile as primary with add_mobile_nos and add_email or None, None and []
    """
    (
        mobile_nos,
        emails,
        add_mobile_nos,
        add_emails,
        unique_mobile_nos,
        unique_emails,
    ) = ([], [], [], [], [], [])
    mobile, email = None, None
    textlines = [
        line
        for lines in get_special_parsed_text(text)
        for line in lines.split(",")
    ]
    for line in textlines:
        email = extract_email(line)
        if email is not None:
            emails.extend(email)
        mobile = extract_mobile(line)
        if mobile is not None:
            mobile_nos.extend(mobile)
    # number cleaning
    for mobile in mobile_nos:
        mobile = re.sub(r"[^\d]", "", mobile)
        if len(mobile) == 10:
            unique_mobile_nos.append(mobile)
        elif len(mobile) == 11:
            if mobile[:1] == "0":
                unique_mobile_nos.append(mobile[1:])
        elif len(mobile) > 11:
            unique_mobile_nos.append(mobile[-10:])
    unique_mobile_nos = list(dict.fromkeys(unique_mobile_nos))
    for email in emails:
        if email[0] == "-":
            unique_emails.append(email[1:].lower())
        else:
            unique_emails.append(email.lower())
    unique_emails = list(dict.fromkeys(unique_emails))
    if len(unique_mobile_nos) > 0:
        mobile = unique_mobile_nos[0]
        if len(unique_mobile_nos) > 1:
            add_mobile_nos = unique_mobile_nos[1:]
    if len(unique_emails) > 0:
        email = unique_emails[0]
        if len(unique_emails) > 1:
            add_emails = unique_emails[1:]
    return mobile, email, add_mobile_nos, add_emails


def extract_email(text):
    """
    Extracts email from the given text.
    Args:
        text: plain text extracted from resume file.

    Returns:
        Returns detected email or None.
    """
    email_regex = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
    email = re.findall(email_regex, text)
    if email:
        return email
    else:
        return None


def extract_email_2(text):
    """
    second try for resume to find email using regex with space
    Args:
        text: resume text

    Returns:
        email or None
    """
    textlines = [
        line
        for lines in get_special_parsed_text(text)
        for line in lines.split(",")
    ]
    for line in textlines:
        email_regex = r"[\w\. -]+@[\w\. -]+(?:\. [\w]+)+"
        email = re.findall(email_regex, line)
        if email:
            email = extract_email(str(email[0]).replace(" ", ""))
            if email is not None:
                return email[0]
    # example like shubhamsharma.vs18\n@gmail.com
    for index, line in enumerate(textlines):
        email_regex = r"@[\w\.-]+(?:\.[\w]+)+"
        email = re.findall(email_regex, line)
        if email:
            email = extract_email(
                str(textlines[index - 1] + str(email[0]).replace(" ", ""))
            )
            if email is not None:
                return email[0]
    # example like bandarusantoshkumar8234@g\nmail.com
    email_regex = r"^[\w. -]+@+[\w. -]+[\r\n]+([^\r\n]+)"
    matches = re.finditer(email_regex, text, re.MULTILINE)
    for matchNum, match in enumerate(matches, start=1):
        email = str(match.group()).replace(" ", "").replace("\n", "")
        if len(email) > 10:
            email = extract_email(str(email))
            if email is not None:
                return email[0]
    return None


def extract_mobile(line):
    """
    Extracts phone numbers from the given text.

    Args:
        line: plain text extracted from resume file.

    Returns:
        Returns detected phone number or None.
    """
    match = re.search(r"\+?\d[\d -]{8,13}\d", line.strip())
    if match is not None:
        mobile = match.group(0)
        if sum(c.isdigit() for c in mobile) >= 10:
            return [mobile]
        else:
            return None
    else:
        return None


####################################################################################################################
##############             linkedin_url,github_url,twitter_url,instagram_url,facebook_url             ##############
####################################################################################################################


def extract_linkedin_spe(text):
    """
    Extract the LinkedIn links from text (better method than extract_linkedin function).

    Args:
        text: sentence in string format

    Returns:
        Detected LinkedIn link
    """
    common_words = [
        ".com",
        ".in",
        " ",
        "profile",
        "education",
        "skills",
        "experience",
        "summary",
        ":",
    ]
    text = get_special_parsed_text(text)
    for index, t in enumerate(text[:-1]):
        link = re.findall(r"linkedin.com/[a-z]+/[\S]+", t)
        if link:
            link = link[0]
            next_line = text[index + 1]
            if all(
                [
                    a.lower() not in next_line.lower().strip()
                    for a in common_words
                ]
            ):
                adjusted_link_1 = link + next_line
                adjusted_link = re.findall(
                    r"linkedin.com/[a-z]+/[\S]+", adjusted_link_1
                )
                return "https://" + adjusted_link[0]
            else:
                return "https://" + link


def extract_github(text):
    """
    Extract the GitHub links from text.

    Args:
        text: sentence in string format

    Returns:
        Detected GitHub link or None
    """
    link = re.findall(r"github.com/[\S]+", text)
    if link:
        return "https://" + link[0]
    else:
        return None


def extract_twitter(text):
    """
    Extract the twitter links from text.

    Args:
        text: sentence in string format

    Returns:
        Detected twitter link or None
    """
    link = re.findall(r"twitter.com/[\S]+", text)
    if link:
        return "https://" + link[0]
    else:
        return None


def extract_facebook(text):
    """
    Extract the facebook links from text.

    Args:
        text: sentence in string format

    Returns:
        Detected facebook link or None
    """
    link = re.findall(r"facebook.com/[\S]+", text)
    if link:
        return "https://" + link[0]
    else:
        return None


def extract_instagram(text):
    """
    Extract the instagram links from text.

    Args:
        text: sentence in string format

    Returns:
        Detected instagram link or None
    """
    link = re.findall(r"instagram.com/[\S]+", text)
    if link:
        return "https://" + link[0]
    else:
        return None


####################################################################################################################
##############             Skills              ##############
####################################################################################################################


def extract_skills(extracted_entities, text):
    """
    Helper function to extract skills from spacy nlp text
    Args:
        text: Skill section text or normal text

    Returns:
        List of skills
    """
    skill_data = list(
        set(
            [
                text
                for start, end, text, label in extracted_entities
                if label == "Skills" and 50 >= len(text) > 2
            ]
        )
    )

    if not skill_data:
        try:
            request = requests.Session()
            retries = Retry(
                total=5, backoff_factor=1, status_forcelist=[502, 503, 504]
            )
            # 502 Bad Gateway, 503 Service Unavailable, 504 Gateway Timeout
            request.mount("http://", HTTPAdapter(max_retries=retries))
            skill_data = request.post(skill_Api, json={"text": text}).json()
            return skill_data
        except Exception as e:
            error_logger("Skill API is not working", e)
            return skill_data
    else:
        return skill_data


####################################################################################################################
##############                                         Experience                                      ##############
####################################################################################################################


def extract_exp(text):
    """
    for calling exp api and get result
    Args:
        text: education section text

    Returns:
        work_exp_final_lst and total_experience will return. if not found then api give []

    """
    try:
        (
            work_exp_final_lst,
            comp_dur_lst,
            company_name,
            duration,
            total_experience,
        ) = requests.post(work_Api, json={"text": text}).json()
        return work_exp_final_lst, total_experience
    except Exception as e:
        print("Exp API is not working", e)
        return [], None


def extract_current_position(extracted_entities):
    """
    Extract current position from extracted entities
    Args:
        extracted_entities: all extracted entities from resumes

    Returns:
        Current Position

    """
    current_position = [
        text.lower().title()
        for start, end, text, label in extracted_entities
        if label == "Current_Position"
    ]

    if len(current_position) > 0:
        current_position = max(
            current_position, key=len
        )  # get the longest position
        return current_position
    else:
        return None


####################################################################################################################
##############                                         Education                                      ##############
####################################################################################################################


def extract_edu(text):
    """
    for calling edu api and get result
    Args:
        text: education section text

    Returns:
        edu_final_list will return. if not found then api give []

    """
    try:
        edu_final_list, edu_lst_map_new = requests.post(
            edu_Api, json={"text": text}
        ).json()
        return edu_final_list
    except Exception as e:
        print("Edu API is not working", e)
        return []


####################################################################################################################
##############                                     Personal Info                                      ##############
####################################################################################################################


def extract_sex(text):
    """
    Extract the Gender from text.
    Args:
        text: sentence in string format

    Returns:
          detected gender will return
    """
    text = re.sub(r"[^a-z]", " ", text.lower())
    parts = nltk.word_tokenize(text)
    if "male" in parts:
        return "male"
    elif "female" in parts:
        return "female"
    else:
        return ""


####################################################################################################################
##############                                     chatGPT                                            ##############
####################################################################################################################


def extract_email_phone_chatgpt(text):
    """
    Extracts phone numbers and email from the given text using chatGPT.

    Args:
        text: plain text extracted from resume file.

    Returns:
        Returns detected phone number and email address.
    """
    question = (
        r'Find the mobile number and valid email in the Given text and output like {"mobile": <list>,"email" : <list>} '
        r'\n "Text:" ' + text
    )
    ChatGPT_consol = ChatGPT()
    response = ChatGPT_consol.chatgpt_request(question)
    result = response.get("json_answer", None)
    if not result:
        return [], [], [], []
    unique_mobile_nos_line = [str(i) for i in result.get("mobile", [])]
    unique_emails_line = [str(i) for i in result.get("email", [])]
    (
        mobile_nos,
        emails,
        add_mobile_nos,
        add_emails,
        unique_mobile_nos,
        unique_emails,
    ) = ([], [], [], [], [], [])
    mobile, email = None, None
    for line in unique_mobile_nos_line:
        mobile = extract_mobile(line)
        if mobile is not None:
            mobile_nos.extend(mobile)
    for line in unique_emails_line:
        email = extract_email(line)
        if email is not None:
            emails.extend(email)
    for mobile in mobile_nos:
        mobile = re.sub(r"[^\d]", "", mobile)
        if len(mobile) == 10:
            unique_mobile_nos.append(mobile)
        elif len(mobile) == 11:
            if mobile[:1] == "0":
                unique_mobile_nos.append(mobile[1:])
        elif len(mobile) > 11:
            unique_mobile_nos.append(mobile[-10:])
    unique_mobile_nos = list(dict.fromkeys(unique_mobile_nos))
    unique_emails = list(dict.fromkeys(emails))
    if len(unique_mobile_nos) > 0:
        mobile = unique_mobile_nos[0]
        if len(unique_mobile_nos) > 1:
            add_mobile_nos = unique_mobile_nos[1:]
    if len(unique_emails) > 0:
        email = unique_emails[0]
        if len(unique_emails) > 1:
            add_emails = unique_emails[1:]
    return mobile, email, add_mobile_nos, add_emails


def chatgpt_parse(resume_text_doc_text):
    """
    Extracts resume data from the given text using chatGPT.

    Args:
        resume_text_doc_text: plain text extracted from resume file.
    Returns:
        Returns candidate data FINAL_DIC_RESUME
    """
    question = (
        r"""Parse resume text in the given json format
                        
            Format: {"name": None, "gender": None, "email": None, "designation": None,"location": None, "total_experience": None, "skills": None, "dob": None,"education": [{"course": None, "specialisation": None, "institute": None, "marks": None, "location": None, "start_year": 'mm yyyy', "end_year": 'mm yyyy', "current": False}] , "address": None,"profile_url": None, "experience": [{"company_name": None, "designation": None, "location": None, "start_year": 'mm yyyy', "end_year": 'mm yyyy', "current": False}] , "notice_period": None,"current_ctc": None, "expected_ctc": None, "marital_status": None, "phone_number": "", "linkedin_url": None, "twitter_url": None, "instagram_url": None, "facebook_url": None, "github_url": None, "additional_emails": [], "additional_phone_numbers": []}
            
            where start_year and end_year should be in valid %Y-%m-%d date format. default date 01 and default month 01
            
            current will be True, if start_year or end_year is present or now kind of words.
            
            In skills, provide one single list of skills
            
            In experience, Don't include projects and certificates.
            
            In education, institute can be a college name and a school name.
            
            Resume Text:"""
        + resume_text_doc_text
        + "\n Give only answer in valid json format. Use strictly given json format"
    )
    chatpgt_service = ChatGPT()
    response = chatpgt_service.chatgpt_request(question)
    result = response.get("json_answer", None)
    if not result:
        FINAL_DIC_RESUME = {
            "chatgpt_retry_count": response.get("retry_count", None)
        }
        return FINAL_DIC_RESUME

    FINAL_DIC_RESUME = ast.literal_eval(os.getenv("final_dict_resume"))
    # ChatGPT retry_count
    FINAL_DIC_RESUME["chatgpt_retry_count"] = response["retry_count"]
    # NAME
    FINAL_DIC_RESUME["name"] = str(result.get("name", None))

    # Email, Phone Number
    unique_mobile_nos_line = str(result.get("phone_number", "")).split(",")
    unique_emails_line = str(result.get("email", "")).split(",")
    (
        FINAL_DIC_RESUME["phone_number"],
        FINAL_DIC_RESUME["email"],
        FINAL_DIC_RESUME["additional_phone_numbers"],
        FINAL_DIC_RESUME["additional_emails"],
    ) = extract_email_phone(
        " , ".join(unique_mobile_nos_line + unique_emails_line)
    )

    if not FINAL_DIC_RESUME["email"]:
        raise Exception("Email not found from chatGPT")

    # Personal Info
    FINAL_DIC_RESUME.update(
        {
            "gender": extract_sex(str(result.get("gender", ""))),
            "location": str(result.get("location", None)),
            "designation": str(result.get("designation", None)),
            "address": str(result.get("address", None)),
            "profile_url": str(result.get("profile_url", None)),
            "notice_period": str(result.get("notice_period", None)),
            "current_ctc": str(result.get("current_ctc", None)),
            "expected_ctc": str(result.get("expected_ctc", None)),
            "marital_status": str(result.get("marital_status", None)),
            "dob": str(result.get("dob", None)),
        }
    )

    # All Urls
    FINAL_DIC_RESUME.update(
        {
            "linkedin_url": extract_linkedin_spe(
                str(result.get("linkedin_url", ""))
            ),
            "github_url": extract_github(str(result.get("github_url", ""))),
            "facebook_url": extract_facebook(
                str(result.get("facebook_url", ""))
            ),
            "twitter_url": extract_twitter(str(result.get("twitter_url", ""))),
            "instagram_url": extract_instagram(
                str(result.get("instagram_url", ""))
            ),
        }
    )

    # skills
    skills = result.get("skills", [])
    if isinstance(skills, str):
        skills = skills.split(",")
    FINAL_DIC_RESUME["skills"] = skills

    # Education
    education = result.get("education", [])
    if isinstance(education, dict):
        education = [education]
    FINAL_DIC_RESUME["education"] = []
    for pair in education:
        edu_one_dict = {
            "course": None,
            "specialisation": None,
            "institute": pair.get("institute", None),
            "marks": pair.get("marks", None),
            "location": pair.get("location", None),
            "start_year": date_str_date(pair.get("start_year", "")),
            "end_year": date_str_date(pair.get("end_year", "")),
            "current": bool(pair.get("current", False)),
        }
        course = find_qualification_details_from_chatpgt_response(
            pair.get("course", "")
        )
        edu_one_dict.update(
            {
                "course": course.get("course", pair.get("course")),
                "specialisation": course.get("specialisation", None),
                "title": course.get("title", None),
            }
        )
        FINAL_DIC_RESUME["education"].append(edu_one_dict)

    # Experience
    experience = result.get("experience", [])
    if isinstance(experience, dict):
        experience = [experience]
    FINAL_DIC_RESUME["experience"] = []
    for pair in experience:
        exp_one_dict = {
            "title": pair.get("company_name", None),
            "designation": pair.get("designation", None),
            "location": pair.get("location", None),
            "start_year": date_str_date(pair.get("start_year", "")),
            "end_year": date_str_date(pair.get("end_year", "")),
            "current": bool(pair.get("current", False)),
        }
        FINAL_DIC_RESUME["experience"].append(exp_one_dict)

    return FINAL_DIC_RESUME


def date_str_date(date):
    """
    convert date to proper str date
    Args:
        date: '2022'

    Returns:
        '2022-01-01'
    """
    try:
        str_date = search_dates(
            date,
            settings={
                "REQUIRE_PARTS": ["year"],
                "PREFER_DAY_OF_MONTH": "first",
                "RELATIVE_BASE": datetime.datetime(1000, 1, 1),
            },
        )[0][1].strftime("%Y-%m-%d")
        return str_date
    except:
        return None
