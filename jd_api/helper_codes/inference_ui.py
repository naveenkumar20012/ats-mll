"""
Does changes to predictions to accommodate into UI.
"""
import ast
import json
import os
import re
from pathlib import Path
import pandas as pd

# from bs4 import BeautifulSoup
from dotenv import load_dotenv


from .inference import inference

#
# from Resume_parser_v1.Full_pipeline.utils.parser_rules import extract_skills
from utils.logging_helpers import info_logger

load_dotenv("./constants.env")

EMPLOYMENTTYPE_FILE_UI = (
    str(Path(__file__).parent.parent) + "/helper_files/UI_employment_type.json"
)
INDUSTRYTYPE_FILE_UI = (
    str(Path(__file__).parent.parent) + "/helper_files/UI_industry_type.txt"
)
LOCATIONS_FILE_UI = (
    str(Path(__file__).parent.parent) + "/helper_files/UI_locations.txt"
)
REMOTE_FLAG_FILE = (
    str(Path(__file__).parent.parent) + "/helper_files/UI_remote.txt"
)
EDU_QUAL_CSV = (
    str(Path(__file__).parent.parent)
    + "/helper_files/UI_eduqual_priority_mapping.csv"
)
EDU_QUAL_MAPPING = (
    str(Path(__file__).parent.parent) + "/helper_files/UI_eduqual_mapping.json"
)  # Map to a degree name identifiable by frontend
UNNECESSARY_SKILLS = ["NIT", "IIT", "BITS"]
unnecessary_eduqual = ["msps"]

# keeps only alphanumeric characters in text
KEEP_ALPHANUM = re.compile(r"[\W_]+", re.UNICODE)

with open(EMPLOYMENTTYPE_FILE_UI, "r") as file:
    emptype_dict = json.load(file)

with open(EDU_QUAL_MAPPING, "r") as file:
    edu_qual_mapping = json.load(file)

with open(INDUSTRYTYPE_FILE_UI, "r") as file:
    indtype_lst_ui = file.read()
    indtype_lst_ui = ast.literal_eval(indtype_lst_ui)

with open(LOCATIONS_FILE_UI, "r") as file:
    loc_lst_ui = file.read()
    loc_lst_ui = ast.literal_eval(loc_lst_ui)

with open(REMOTE_FLAG_FILE, "r") as file:
    remote_flag_lst = ast.literal_eval(file.read())


def remove_char(text):
    if text[-1] == "'":
        return text[:-1]
    else:
        return text


# Get the dictionary containing key as edu_qual and value as its priority.
DF_UI_EDU = pd.read_csv(EDU_QUAL_CSV)
EDU_LST = DF_UI_EDU["Edu_qual"].map(remove_char).tolist()
RATING_LST = DF_UI_EDU["Priority"].tolist()
EDUQUAL_DICT = dict(zip(EDU_LST, RATING_LST))


def ui_employmenttype(dict_no_ui):
    """
    This function clubs the wide range of employment types into
        4 division.

    Args:
        dict_no_ui(dict) : containing results from post processed pipeline.

    Returns:
        dict_no_ui(dict) : containing results after changes related to UI.
    """
    # extracting employmenttypes predicted.
    emptype_old = dict_no_ui["employment_type"]
    emptype_final = ""  # final employment type which goes into UI.
    hierarchy_final = 0
    # If we have atleast one employment type detected.
    for ind, old_emptype in enumerate(emptype_old):
        # keep only alphanumeric characters
        old_emptype_regex = KEEP_ALPHANUM.sub("", old_emptype.lower().strip())
        if old_emptype_regex in emptype_dict:
            info_logger(old_emptype)
            hierarchy = emptype_dict[old_emptype_regex][1]
            if hierarchy >= hierarchy_final:
                emptype_final = emptype_dict[old_emptype_regex][0]
                hierarchy_final = hierarchy
        else:
            info_logger(
                f"Error Message: {old_emptype} not in employemnt_type_ui list."
            )

    # If there was no employment type detected by ui processing, remove the key-value pair.
    if emptype_final == "":
        del dict_no_ui["employment_type"]
    else:
        dict_no_ui["employment_type"] = emptype_final
    return dict_no_ui


def ui_industrytype(dict_no_ui):
    """
    This function clubs the wide range of industry types into exhaustive list the UI has.

    Args:
        dict_no_ui(dict) : containing results from post processed pipeline.

    Returns:
        dict_no_ui(dict) : containing results after changes related to UI.
    """
    # industry type list after post-processing
    indtype_old_lst = dict_no_ui["industry_type"]
    # Collect all industry-types extracted by ui processing
    old_types_modified = []
    # Handling ampersand cases
    for ind_type in indtype_old_lst:
        type_mod = KEEP_ALPHANUM.sub("", ind_type.lower().strip())
        type_mod = list(map(lambda x: x.strip(), type_mod.split("&")))
        old_types_modified.extend(type_mod)

    modified_indtype_lst_ui = []
    max_match_type = None
    for ind_type in indtype_lst_ui:
        types = ind_type.split("/")
        types = [KEEP_ALPHANUM.sub("", i.lower().strip()) for i in types]
        modified_indtype_lst_ui.append(types)
        common_elements = list(set(types).intersection(old_types_modified))
        if not max_match_type and len(common_elements) > 0:
            max_match_type = ind_type
        elif max_match_type and len(common_elements) > len(max_match_type):
            max_match_type = ind_type

    if max_match_type:
        dict_no_ui["industry_type"] = max_match_type
    else:
        del dict_no_ui["industry_type"]
    return dict_no_ui


def ui_location(dict_no_ui):
    """
    This function checks locations predicted are there in the exhaustive elist of locations
    present in UI.
    Remember that this takes only the fist location.(ignores rest)

    Args:
        dict_no_ui(dict) : containing results from post processed pipeline.

    Returns:
        dict_no_ui(dict) : containing results after changes related to UI.
    """
    loc_old_lst = dict_no_ui["location"]

    # change Bangalore to Bengaluru because Frontend list doesn't have Bangalore.
    bangalore_lst = ["bangalore", "BANGALORE", "Bangalore"]
    for bangalore in bangalore_lst:
        if bangalore in loc_old_lst:
            loc_old_lst.remove(bangalore)
            loc_old_lst.append("Bengaluru")

    else:
        pass

    ui_loc = ""
    for old_loc in loc_old_lst:
        # have only alphanumeric characters and lowercase.
        old_loc_regex = KEEP_ALPHANUM.sub("", old_loc.lower().strip())
        info_logger("old_loc_regex: ", old_loc_regex)
        for i in loc_lst_ui:
            # If element from exhaustive list is present in predicted word.
            if KEEP_ALPHANUM.sub("", i.lower().strip()) in old_loc_regex:
                # remember that this takes only the fist location.(ignores rest)
                ui_loc = i
                break
            else:
                pass

    if ui_loc != "":
        dict_no_ui["location"] = ui_loc
    else:
        # If ui_loc string is empty, then we can delete the location key-value pair.
        del dict_no_ui["location"]
    return dict_no_ui


def ui_eduqual(dict_no_ui):
    """
    This function checks locations predicted are there in the exhaustiv elist of locations
    present in UI.

    Args:
        dict_no_ui(dict) : containing results from post processed pipeline.

    Returns:
        dict_no_ui(dict) : containing results after changes related to UI.
    """
    eduqual_lst = dict_no_ui["education"]
    edu_qual_str = ""
    prioirty_final = 0  # Change to a higher value if you need lower priority

    for ind, eduqual in enumerate(eduqual_lst):
        # word has only alphanumeric characters and lowercase.
        eduqual_regex = KEEP_ALPHANUM.sub("", eduqual.lower().strip())
        if eduqual_regex in EDU_LST:
            prioirty = EDUQUAL_DICT[eduqual_regex]
            if prioirty >= prioirty_final:
                edu_qual_str = eduqual
                prioirty_final = prioirty

    if edu_qual_str == "":
        del dict_no_ui["education"]
    else:
        if edu_qual_mapping.get(edu_qual_str):
            dict_no_ui["education"] = edu_qual_mapping.get(edu_qual_str)
        else:
            dict_no_ui["education"] = edu_qual_str
    return dict_no_ui


def ui_remoteflag(dict_no_ui):
    """
    This function uses employment results to flg it employment type is remote or not.

    Args:
        dict_no_ui(dict) : containing results from post processed pipeline.

    Returns:
        dict_no_ui(dict) : containing results after changes related to UI.
    """
    # extracting employmenttypes predicted.
    remote_flag = "false"
    if "employment_type" in dict_no_ui:
        emptype_old_lst = dict_no_ui["employment_type"]
        for old_emptype in emptype_old_lst:
            old_emptype_regex = KEEP_ALPHANUM.sub(
                "", old_emptype.lower().strip()
            )
            for remote_emptype in remote_flag_lst:
                if old_emptype_regex in KEEP_ALPHANUM.sub(
                    "", remote_emptype.lower().strip()
                ):
                    remote_flag = "true"
                    break
    dict_no_ui["is_remote"] = remote_flag
    return dict_no_ui


def skills(dict_no_ui):
    """
    This function does necessary constraints to the predicted Tech Skills

    Args:
        dict_no_ui(dict) : containing results from post processed pipeline.

    Returns:
        dict_no_ui(dict) : containing results after changes related to UI.
    """
    skills_lst = dict_no_ui["skills"]
    # soup = BeautifulSoup(dict_no_ui['job_description'], features="lxml")
    # jd_text = soup.get_text('\n')
    # jd_text = jd_text.lower()
    # skills_lst += extract_skills(jd_text)
    skills_lst = list(set(skills_lst))

    # removing skills with greater than 3 words.
    skill_lst_2words = sorted(
        skills_lst, key=lambda skill: (len(skill.split(" ")))
    )

    # removing duplicates
    skills_lowercase = []
    skills_lst_nodups = []
    for skill_1 in skill_lst_2words:
        if skill_1.lower() not in skills_lowercase:
            skills_lowercase.append(skill_1.lower())
            skills_lst_nodups.append(skill_1)
        else:
            pass

    # remove unnecessary Skills
    skills_lst_necessary = []
    for i in skills_lst_nodups:
        i_regex = KEEP_ALPHANUM.sub("", i.lower().strip())
        in_unnecessary_skill = 0
        for unnecessary_skill in UNNECESSARY_SKILLS:
            if (
                KEEP_ALPHANUM.sub("", unnecessary_skill.lower().strip())
                in i_regex
            ):
                in_unnecessary_skill = 1
            else:
                pass
        if in_unnecessary_skill == 0:
            skills_lst_necessary.append(i)
        else:
            pass

    dict_no_ui["skills"] = skills_lst_necessary
    return dict_no_ui


def inference_ui(jd_path):
    """
    Final Inference which is compatible with the UI

    Args:
        jd_path(str) : gcs path for the JD.

    Returns:
        dict_no_ui(dict) : containing results after changes related to UI.
    """
    dict_no_ui = inference(jd_path)
    # info_logger("Before UI:", dict_no_ui)

    # UI processing on is_remote flag
    dict_no_ui = ui_remoteflag(dict_no_ui)

    # UI processing on employmenttype
    if "employment_type" in dict_no_ui:
        dict_no_ui = ui_employmenttype(dict_no_ui)

    # UI processing on industrytype
    if "industry_type" in dict_no_ui:
        dict_no_ui = ui_industrytype(dict_no_ui)

    # UI processing for locations
    if "location" in dict_no_ui:
        dict_no_ui = ui_location(dict_no_ui)

    # UI processing for eduqual.
    if "education" in dict_no_ui:
        dict_no_ui = ui_eduqual(dict_no_ui)

    # UI processing for non-tech skills
    if "Non_Tech_Skills" in dict_no_ui:
        del dict_no_ui["Non_Tech_Skills"]

    # UI processing for skills
    if "skills" in dict_no_ui:
        dict_no_ui = skills(dict_no_ui)

    return dict_no_ui
