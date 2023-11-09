"""
Helper functions to map education-qualification-duration.
"""
import copy
import datetime as dt
import logging
import operator
import os
import re
from collections import Counter
from heapq import *

from dateparser.search import search_dates
from dateutil.relativedelta import relativedelta
from dotenv import load_dotenv

from RP_parser.helper_files import constants as cs
from utils.logging_helpers import info_logger

logger = logging.getLogger("Resume_Parser")
TEXT_SLICING = 400
os.environ["TOKENIZERS_PARALLELISM"] = "false"

load_dotenv("constants.env")

date_parser_1 = cs.date_parser_1
date_parser_2 = cs.date_parser_2
date_parser_3 = cs.date_parser_3
current_word_regex = cs.current_word_regex
only_years = cs.only_years
only_years_1 = cs.only_years_1


def subarray_with_k_distinct(pairs, length):
    """
    finding distinct pairs with k length.
    Args:
        pairs: list of pairs
        length: length of unique pair / classes

    Returns:
        list of distinct pairs with k length.
    """
    end = len(pairs)
    unique_subset = []
    subset = []
    start, pivot_start = 0, 0
    while pivot_start < end:
        if pairs[pivot_start][0] not in [i[0][0] for i in subset]:
            if len(subset) == 0:
                start = pivot_start
            subset.append((pairs[pivot_start], pivot_start))
            if len(subset) == length:
                unique_subset.append(subset)
                subset = []
                pivot_start = start
            pivot_start += 1
        else:
            subset = []
            start += 1
            pivot_start = start
    return unique_subset


def qualification_mapping(edu_lst_map, qual_lst):
    """
    Maps the qualification with college name.

    Args:
        edu_lst_map : list of lists containing education and mapped duration. [[University_label, detected university or college name, line number, start offset in the text, end offset in the text], [duration_label, detected duration, line number, start offset in the text, end offset in the text]].
        qual_lst : list of lists containing predicted qualifications' info. eg: [qualification_label, detected qualification, line number, start offset in the text, end offset in the text]

    Returns:
        list of lists containing qualifications mapped with college-name and duration.
    """
    edu_lst_map_new = copy.deepcopy(edu_lst_map)
    # calculate distance between qual and (university, duration).  qual nearest to(less distance to) (university,
    # duration) will be mapped.
    for qual in qual_lst:
        dist = []
        for key, val in edu_lst_map.items():
            university = val[0]
            duration = val[1]
            qual_univ_dist = qual[2] - university[2]
            qual_dur_dist = qual[2] - duration[2]
            qual_dist = (abs(qual_univ_dist) + abs(qual_dur_dist)) / 2
            dist.append(qual_dist)
        index_min = min(range(len(dist)), key=dist.__getitem__)
        edu_lst_map_new[index_min].append(qual)
    return edu_lst_map_new


def find_title(course):
    """
    find specific title level for course based on cs.edu_title_new dict
    Args:
        course: course from resume text

    Returns:
        return specifics title for course
    """
    for title, titles in cs.edu_title_new.items():
        if course in titles:
            return title
    else:
        return None


def pair_sorting(pairs):
    """
    detect outlier point in odd length of total pairs
    Args:
        pairs: list of all founded entity with index

    Returns:
        return pairs with even length of pair, removed outlier data point

    """
    # O(NLog)
    pairs = sorted(pairs, key=lambda x: x[2])
    n = len(pairs)

    # mimic a double linked list
    left = [i - 1 for i in range(n)]
    right = [i + 1 for i in range(n)]
    appeared = [False] * n

    btree = []
    for i in range(0, n - 1):
        # distance of adjacent integers, and their indices
        heappush(btree, (pairs[i + 1][2] - pairs[i][2], i, i + 1))

    # roughly O(n log n), because the heap will have at most `n` items in each iteration
    result = []
    while len(btree) != 0:
        minimal = heappop(btree)
        a, b = minimal[1:3]

        # skip if either a or b appeared
        if not appeared[a] and not appeared[b]:
            result.append(pairs[a])
            result.append(pairs[b])
            appeared[a] = True
            appeared[b] = True
        else:
            continue  # this is important
        # print result

        if left[a] != -1:
            right[left[a]] = right[b]
        if right[b] != n:
            left[right[b]] = left[a]
        if left[a] != -1 and right[b] != n:
            heappush(
                btree,
                (pairs[right[b]][2] - pairs[left[a]][2], left[a], right[b]),
            )

    return result


def make_edu_one_dict(pairs):
    """
    parse list of pairs to edu_one_dict
    Args:
        pairs: list of pairs

    Returns:
        edu_one_dict for return
    """
    edu_one_dict = {
        "course": None,
        "specialisation": None,
        "institute": None,
        "marks": None,
        "Location": None,
        "start_year": None,
        "end_year": None,
        "current": False,
        "index": float("inf"),
    }
    for pair in pairs:
        if pair[0] == "College_Name":
            edu_one_dict["institute"] = pair[1]
        elif pair[0] == "Marks":
            edu_one_dict["marks"] = pair[1]
        elif pair[0] == "Edu_Location":
            edu_one_dict["location"] = pair[1]
        elif pair[0] == "Degree":
            edu_one_dict["course"] = pair[1]["course"]
            if pair[1].get("specialisation"):
                edu_one_dict["specialisation"] = pair[1]["specialisation"]
            if pair[1].get("title"):
                edu_one_dict["title"] = pair[1]["title"]
        else:
            date_duration = pair[-1]
            if len(date_duration) == 2:
                edu_one_dict["start_year"] = date_duration[0].strftime(
                    "%Y-%m-%d"
                )
                edu_one_dict["end_year"] = date_duration[1].strftime(
                    "%Y-%m-%d"
                )
                if (
                    dt.date.today() + relativedelta(day=31)
                    == date_duration[0].date()
                ):
                    edu_one_dict["current"] = True
                    edu_one_dict["end_year"] = "Present"
            elif len(date_duration) == 1:
                edu_one_dict["end_year"] = date_duration[0].strftime(
                    "%Y-%m-%d"
                )
        edu_one_dict["index"] = min(edu_one_dict["index"], pair[2])
    return edu_one_dict


def find_qual(line, start_char, end_char):
    """
    find qualification from resume text
    Args:
        line: text line which have qualification
        start_char: index of first char of qualification
        end_char: index of last char of qualification

    Returns:
        qualification pair
    """

    line = re.sub(
        r"[^\w\s]", " ", line.lower().replace("-", " ").replace("&", "and")
    )
    for course_class, courses in cs.edu_course_new.items():
        for course in courses:
            if re.search(rf"\b{course}\b", line) is None:
                pass
            else:
                edu_title = find_title(course_class)
                if edu_title == "Schooling":
                    return [
                        "Degree",
                        {"title": edu_title, "course": course_class},
                        start_char,
                        end_char,
                    ]
                elif edu_title == "Other":
                    edu_title = cs.edu_titles.get(
                        course[0]
                    ) or cs.edu_titles.get(course)
                    return [
                        "Degree",
                        {"title": edu_title, "course": course},
                        start_char,
                        end_char,
                    ]
                else:
                    for specialisation in cs.edu_specialisations_new[
                        course_class
                    ]:
                        if line.find(specialisation) == -1:
                            pass
                        else:
                            return [
                                "Degree",
                                {
                                    "title": edu_title,
                                    "course": course_class,
                                    "specialisation": specialisation,
                                },
                                start_char,
                                end_char,
                            ]

                    else:
                        return [
                            "Degree",
                            {
                                "title": edu_title,
                                "course": course_class,
                                "specialisation": None,
                            },
                            start_char,
                            end_char,
                        ]
    else:
        return None


def find_qualification_details_from_chatpgt_response(line):
    """
    Find the qualification details from the given text line that contains degree information.
    Args:
        line (str): Text line which contains degree information.

    Returns:
        dict: A dictionary containing the qualification details, including the title, course and specialisation.
              If the qualification details are not found in the input text, an empty dictionary is returned.
    """
    if not line:
        return {}

    # Clean up the input text by replacing special characters and converting to lowercase.
    line = re.sub(
        r"[^\w\s]", " ", line.lower().replace("-", " ").replace("&", "and")
    )

    # Loop through all the courses in the cs.edu_course_new dictionary and try to match them in the input text.
    for course_class, courses in cs.edu_course_new.items():
        for course in courses:
            if not re.search(rf"\b{course}\b", line):
                pass
            else:
                # If a course is found, determine the title of the qualification based on the course and return the result.
                edu_title = find_title(course_class)
                if edu_title == "Schooling":
                    return {"title": edu_title, "course": course_class}
                elif edu_title == "Other" and course[0] in cs.edu_titles:
                    edu_title = cs.edu_titles[course[0]]
                    return {"title": edu_title, "course": course}
                else:
                    # If the course has specialisations, try to match them in the input text.
                    for specialisation in cs.edu_specialisations_new[
                        course_class
                    ]:
                        if line.find(specialisation) == -1:
                            pass
                        else:
                            return {
                                "title": edu_title,
                                "course": course_class,
                                "specialisation": specialisation,
                            }

                    # If no specialisation is found, return the qualification details with None for specialisation.
                    else:
                        return {
                            "title": edu_title,
                            "course": course_class,
                            "specialisation": None,
                        }

    # If no qualification details are found in the input text, return an empty dictionary.
    else:
        return {}


def duration_cleaning(duration):
    """
    cleaning duration pair
    Args:
        duration: pairs of duration

    Returns:
        cleaned pair of duration
    """
    all_duration = []
    for label, text, start, end in duration:
        new_duration, new_word_duration = [], []
        matches = re.finditer(
            date_parser_1, text.lower(), re.IGNORECASE
        )  # [ day month year, month year]
        dates1 = []
        for matchNum, match in enumerate(matches, start=1):
            month = match.group("month")
            year = match.group("year")
            if len(year) == 2:
                year = (
                    str(int(year) + 2000)
                    if int(year) <= (dt.datetime.today().year - 2000)
                    else str(int(year) + 1900)
                )
            if month and year:
                dates1.append(
                    [
                        "Edu_Duration",
                        month + " " + year,
                        start + matchNum - 1,
                        end,
                        False,
                    ]
                )
        matches2 = re.finditer(
            date_parser_2, text.lower(), re.IGNORECASE
        )  # [month day year]
        dates2 = []
        for matchNum2, match2 in enumerate(matches2, start=1):
            month2 = match2.group("month")
            year2 = match2.group("year")
            if month2 and year2:
                dates2.append(
                    [
                        "Edu_Duration",
                        month2 + " " + year2,
                        start + matchNum2 - 1,
                        end,
                        False,
                    ]
                )
        matches3 = re.finditer(
            date_parser_3, text.lower(), re.IGNORECASE
        )  # [ year month, year month day]
        dates3 = []
        for matchNum3, match3 in enumerate(matches3, start=1):
            month3 = match3.group("month")
            year3 = match3.group("year")
            if month3 and year3:
                dates3.append(
                    [
                        "Edu_Duration",
                        month3 + " " + year3,
                        start + matchNum3 - 1,
                        end,
                        False,
                    ]
                )
        if dates1 == [] and dates2 == [] and dates3 == []:
            if re.search(r"month|year|months|years", text.lower()):
                new_word_duration.append(
                    ["Edu_Duration", text, start, end, False]
                )
            else:
                matches4 = re.finditer(only_years, text.lower(), re.IGNORECASE)
                dates4 = []
                for matchNum4, match4 in enumerate(matches4, start=1):
                    year4 = match4.group("year")
                    if year4:
                        dates4.append(
                            [
                                "Edu_Duration",
                                year4,
                                start + matchNum4 - 1,
                                end,
                                True,
                            ]
                        )

                matches5 = re.finditer(
                    only_years_1, text.lower(), re.IGNORECASE
                )
                dates5 = []
                for matchNum5, match5 in enumerate(matches5, start=1):
                    year4 = match5.group("year")
                    year5 = match5.group("year2")
                    if year4:
                        dates5.append(
                            [
                                "Exp_Duration",
                                year4,
                                start + matchNum5 - 1,
                                end,
                                True,
                            ]
                        )
                    if year5:
                        if len(year5) == 2:
                            year5 = (
                                str(int(year5) + 2000)
                                if int(year5)
                                <= (dt.datetime.today().year - 2000)
                                else str(int(year5) + 1900)
                            )
                            dates5.append(
                                [
                                    "Exp_Duration",
                                    year5,
                                    start + matchNum5 - 1,
                                    end,
                                    True,
                                ]
                            )
                if dates4 or dates5:
                    new_duration.extend(max(dates4, dates5, key=len))
        else:
            new_duration.extend(max([dates1, dates2, dates3], key=len))
        matches6 = re.finditer(current_word_regex, text.lower(), re.IGNORECASE)
        for matchNum6, match6 in enumerate(matches6, start=1):
            new_duration.append(
                ["Edu_Duration", "Present", start + match6.start(), end, False]
            )
            break
        one_date_duration, one_duration_pair = find_dates_and_make_pairs(
            new_duration.copy()
        )
        all_duration.extend(one_duration_pair)
    return all_duration


def find_dates_and_make_pairs(duration):
    """
    finding dates and make duration pairs from duration
    Args:
        duration: list of time
        [['Edu_Duration', '2022-06-16', 0, 0, 7], ['Edu_Duration', 'april 2021', 0, 49, 59], ['Edu_Duration', 'october 2020', 14, 29, 41], ['Edu_Duration', 'april 2021', 14, 44, 54], ['Edu_Duration', 'january 2020', 23, 47, 59], ['Edu_Duration', 'october 2020', 23, 62, 74]]

    Returns:
        date_duration = []  # [['Edu_Duration', '2022-06-16', 0, 0, 7, datetime.datetime(2022, 6, 30, 0, 0)], ['Edu_Duration', 'april 2021', 0, 49, 59, datetime.datetime(2021, 4, 1, 0, 0)], ['Edu_Duration', 'october 2020', 14, 29, 41, datetime.datetime(2020, 10, 1, 0, 0)], ['Edu_Duration', 'april 2021', 14, 44, 54, datetime.datetime(2021, 4, 30, 0, 0)], ['Edu_Duration', 'january 2020', 23, 47, 59, datetime.datetime(2020, 1, 1, 0, 0)], ['Edu_Duration', 'october 2020', 23, 62, 74, datetime.datetime(2020, 10, 31, 0, 0)]]
        duration_pair = []  # [['Edu_Duration', 'april 2021 <-> 2022-06-16', 0, 49, 7, (datetime.datetime(2021, 4, 1, 0, 0), datetime.datetime(2022, 6, 30, 0, 0))], ['Edu_Duration', 'october 2020 <-> april 2021', 14, 29, 54, (datetime.datetime(2020, 10, 1, 0, 0), datetime.datetime(2021, 4, 30, 0, 0))], ['Edu_Duration', 'january 2020 <-> october 2020', 23, 47, 74, (datetime.datetime(2020, 1, 1, 0, 0), datetime.datetime(2020, 10, 31, 0, 0))]]
    """
    date_duration = []
    duration_pair = []
    if len(duration) > 0:
        for date_obj_str in duration:
            date_objs = search_dates(
                date_obj_str[1],
                settings={
                    "REQUIRE_PARTS": ["year"],
                    "PREFER_DAY_OF_MONTH": "first",
                    "RELATIVE_BASE": dt.datetime(1000, 1, 1),
                },
            )
            if date_objs is not None:
                for date_obj in date_objs:
                    if (dt.datetime.today().year - 100) < date_obj[1].year:
                        new_date_obj_str = date_obj_str.copy()
                        new_date_obj_str[1] = date_obj[0]
                        new_date_obj_str.append(date_obj[1])
                        date_duration.append(new_date_obj_str)
                    else:
                        info_logger(
                            f"Got date more than 100 years old: {date_obj}"
                        )
            if date_obj_str[1] == "Present":
                new_date_obj_str = date_obj_str.copy()
                new_date_obj_str.append(dt.datetime.today())
                date_duration.append(new_date_obj_str)

    # make duration pair 'april 2021 <-> 2022-06-16','october 2020 <-> april 2021','january 2020 <-> october 2020'
    if len(date_duration) > 0:
        if len(date_duration) == 1:
            duration_pair.append(
                [
                    "Edu_Duration",
                    date_duration[0][1],
                    date_duration[0][2],
                    date_duration[0][3],
                    [date_duration[0][-1]],
                ]
            )
        elif len(date_duration) % 2 == 0:
            for start, end in zip(date_duration[0::2], date_duration[1::2]):
                start, end = (
                    min([start, end], key=lambda x: x[-1]),
                    max([start, end], key=lambda x: x[-1]),
                )
                if end[
                    -2
                ]:  # if it's only year then we will make them last day of year.
                    end[-1] = end[-1].replace(month=12, day=31)
                else:
                    # we are making last day of month. for create fair total_experience
                    end[-1] = end[-1] + relativedelta(day=31)
                duration_pair.append(
                    [
                        "Edu_Duration",
                        start[1] + " <-> " + end[1],
                        int((start[2] + end[2]) / 2),
                        start[3],
                        end[4],
                        (start[-1], end[-1]),
                    ]
                )
        else:
            date_duration_calc = pair_sorting(date_duration.copy())
            date_duration_calc = sorted(
                date_duration_calc, key=lambda x: (x[2])
            )
            for start, end in zip(
                date_duration_calc[0::2], date_duration_calc[1::2]
            ):
                start, end = (
                    min([start, end], key=lambda x: x[-1]),
                    max([start, end], key=lambda x: x[-1]),
                )
                if end[-2]:
                    end[-1] = end[-1].replace(month=12, day=31)
                else:
                    end[-1] = end[-1] + relativedelta(day=31)
                duration_pair.append(
                    [
                        "Edu_Duration",
                        start[1] + " <-> " + end[1],
                        start[2],
                        end[3],
                        (start[-1], end[-1]),
                    ]
                )

    return date_duration, duration_pair


def degree_cleaning(degree):
    """
    cleaning and make universal pair of degree
    Args:
        degree: pairs of degree

    Returns:
        universal pairs of degree
    """
    new_degree = []
    for pair in degree:
        single_degree = find_qual(pair[1], pair[2], pair[3])
        if single_degree:
            new_degree.append(single_degree)
    return new_degree


def line_no_calculation(pairs, texts):
    """
    find line number based on character index
    Args:
        pairs: entities pair
        texts: whole resume text

    Returns:
        entities pair with line no
    """
    new_pairs = []
    for pair in pairs:
        new_pair = pair
        new_pair[2:2] = [len(texts[0 : pair[3]].split("\n"))]
        new_pairs.append(new_pair)
    return new_pairs


def edu_ner_pred(extracted_entities, resume_text_doc):
    """
    Maps the education with duration and qualification.
    Args:
        extracted_entities:  extracted_entities from resumes
        resume_text_doc: resume text
    Returns:
        list of dictionaries containing college name mapped with duration and qualification.
    """
    edu_final_list = []

    if len(extracted_entities) == 0:
        return []  # , [], [], [], [], [], []
    else:
        # time_lst = []
        degree = [
            [label, text.replace("\n", " "), start, end]
            for start, end, text, label in extracted_entities
            if label == "Degree"
        ]
        college_name = [
            [label, text.replace("\n", " "), start, end]
            for start, end, text, label in extracted_entities
            if label == "College_Name"
        ]
        duration = [
            [label, text.replace("\n", " "), start, end]
            for start, end, text, label in extracted_entities
            if label == "Edu_Duration"
        ]
        # mark = [[label, text.replace('\n', ' '), start, end] for start, end, text, label in extracted_entities if
        #         label == 'Marks']
        # location = [[label, text.replace('\n', ' '), start, end]
        #             for start, end, text, label in extracted_entities if label == 'Edu_Location']

        if not (degree or college_name or duration):
            return []

        degree = sorted(degree, key=lambda x: x[2])
        degree = degree_cleaning(degree)
        college_name = sorted(college_name, key=lambda x: x[2])
        # mark = sorted(mark, key=lambda x: x[2])
        # location = sorted(location, key=lambda x: x[2])
        duration_pair = duration_cleaning(copy.deepcopy(duration))

        only_degree_pairs = line_no_calculation(
            copy.deepcopy(degree), resume_text_doc
        )

        deg_dur_lst = copy.deepcopy(degree)
        deg_dur_lst.extend(copy.deepcopy(duration_pair))
        deg_dur_lst = line_no_calculation(deg_dur_lst, resume_text_doc)
        deg_dur_lst = sorted(deg_dur_lst, key=lambda x: (x[2], x[3]))

        deg_col_dur = copy.deepcopy(degree)
        deg_col_dur.extend(copy.deepcopy(college_name))
        deg_col_dur = sorted(deg_col_dur, key=lambda x: x[2])
        deg_col_dur_lst = copy.deepcopy(duration_pair)
        deg_col_dur_lst.extend(deg_col_dur)
        deg_col_dur_lst = line_no_calculation(deg_col_dur_lst, resume_text_doc)
        deg_col_dur_lst = sorted(deg_col_dur_lst, key=lambda x: (x[2], x[3]))

        unique_class = set((lst[0] for lst in deg_col_dur_lst))
        if len(unique_class) == 3:
            print("tri_pair")
            # Data loss for two entities pair and one entity pair
            numbers_with_distance = [(deg_col_dur_lst[0], float("inf"))]
            for i in range(1, len(deg_col_dur_lst) - 1):
                left_distance = abs(
                    deg_col_dur_lst[i - 1][2] - deg_col_dur_lst[i][2]
                )
                right_distance = abs(
                    deg_col_dur_lst[i + 1][2] - deg_col_dur_lst[i][2]
                )
                numbers_with_distance.append(
                    (deg_col_dur_lst[i], left_distance + right_distance)
                )
            numbers_with_distance.append((deg_col_dur_lst[-1], float("inf")))
            minimum = min(numbers_with_distance, key=lambda item: item[1])[1]
            min_dist_list = [
                (pair[0], index, pair[1])
                for index, pair in enumerate(numbers_with_distance)
                if pair[1] == minimum
            ]
            min_val_list = subarray_with_k_distinct(deg_col_dur_lst, 3)

            common_patterns = [
                min_val
                for min_dist in min_dist_list
                for min_val in min_val_list
                if min_val[1] == (min_dist[0], min_dist[1])
            ]
            unique_pattern_counter = Counter(
                [
                    tuple([pair[0][0] for pair in pairs])
                    for pairs in common_patterns
                ]
            )
            if len(unique_pattern_counter.keys()) > 0:
                pattern = max(
                    unique_pattern_counter.items(), key=operator.itemgetter(1)
                )[0]
                used_entities = []
                for index in range(0, len(deg_col_dur_lst) - 2):
                    if (
                        deg_col_dur_lst[index][0] == pattern[0]
                        and deg_col_dur_lst[index + 1][0] == pattern[1]
                        and deg_col_dur_lst[index + 2][0] == pattern[2]
                    ):
                        edu_final_list.append(
                            make_edu_one_dict(
                                deg_col_dur_lst[index : index + 3]
                            )
                        )
                        used_entities.extend(
                            deg_col_dur_lst[index : index + 3]
                        )
                first_second_distance = (
                    used_entities[1][2] - used_entities[0][2]
                )
                second_third_distance = (
                    used_entities[2][2] - used_entities[1][2]
                )
                first_third_distance = (
                    used_entities[2][2] - used_entities[0][2]
                )
                unused_entities = [
                    pair
                    for pair in deg_col_dur_lst
                    if pair not in used_entities
                ]
                # second pair
                start = 0
                while start < len(unused_entities) - 1:
                    if (
                        unused_entities[start][0] == pattern[0]
                        and unused_entities[start + 1][0] == pattern[1]
                        and unused_entities[start + 1][2]
                        - unused_entities[start][2]
                        == first_second_distance
                    ):
                        edu_final_list.append(
                            make_edu_one_dict(
                                unused_entities[start : start + 2]
                            )
                        )
                        start += 2
                    elif (
                        unused_entities[start][0] == pattern[1]
                        and unused_entities[start + 1][0] == pattern[2]
                        and unused_entities[start + 1][2]
                        - unused_entities[start][2]
                        == second_third_distance
                    ):
                        edu_final_list.append(
                            make_edu_one_dict(
                                unused_entities[start : start + 2]
                            )
                        )
                        start += 2
                    elif (
                        unused_entities[start][0] == pattern[0]
                        and unused_entities[start + 1][0] == pattern[2]
                        and unused_entities[start + 1][2]
                        - unused_entities[start][2]
                        == first_third_distance
                    ):
                        edu_final_list.append(
                            make_edu_one_dict(
                                unused_entities[start : start + 2]
                            )
                        )
                        start += 2
                    else:
                        start += 1
                edu_final_list = sorted(
                    edu_final_list, key=lambda x: x["index"]
                )
            else:  # Founded Three class but not distinct pair of three class so removed position and go with two pair
                sorted_pair = pair_sorting(deg_dur_lst)
                unique_pattern_counter = Counter(
                    [
                        (pair1[0], pair2[0])
                        for pair1, pair2 in zip(
                            sorted_pair[0::2], sorted_pair[1::2]
                        )
                        if pair1[0] != pair2[0]
                    ]
                )
                if len(unique_pattern_counter.keys()) > 0:
                    pattern = max(
                        unique_pattern_counter.items(),
                        key=operator.itemgetter(1),
                    )[0]
                    for index in range(0, len(deg_dur_lst) - 1):
                        if (
                            deg_dur_lst[index][0] == pattern[0]
                            and deg_dur_lst[index + 1][0] == pattern[1]
                        ):
                            edu_final_list.append(
                                make_edu_one_dict(
                                    deg_dur_lst[index : index + 2]
                                )
                            )
        elif len(unique_class) == 2:
            print("dual_pair")
            # Data loss for one entity pair
            sorted_pair = pair_sorting(deg_col_dur_lst)
            unique_pattern_counter = Counter(
                [
                    (pair1[0], pair2[0])
                    for pair1, pair2 in zip(
                        sorted_pair[0::2], sorted_pair[1::2]
                    )
                    if pair1[0] != pair2[0]
                ]
            )
            if len(unique_pattern_counter.keys()) > 0:
                pattern = max(
                    unique_pattern_counter.items(), key=operator.itemgetter(1)
                )[0]
                for index in range(0, len(deg_col_dur_lst) - 1):
                    if (
                        deg_col_dur_lst[index][0] == pattern[0]
                        and deg_col_dur_lst[index + 1][0] == pattern[1]
                    ):
                        edu_final_list.append(
                            make_edu_one_dict(
                                deg_col_dur_lst[index : index + 2]
                            )
                        )

            else:  # Founded Two class but not distinct pair of two class, removed position and go with only
                # qualification
                for pair in only_degree_pairs:
                    edu_final_list.append(make_edu_one_dict([pair]))
        elif len(unique_class) == 1:
            print("one_pair")
            for pair in deg_col_dur_lst:
                edu_final_list.append(make_edu_one_dict([pair]))
    edu_final_list = sorted(edu_final_list, key=lambda d: d["index"])
    return edu_final_list
