"""
Helper functions which does company_name-duration mapping.
"""
import ast
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
os.environ["TOKENIZERS_PARALLELISM"] = "false"
load_dotenv("constants.env")

WORKEXP_CURRENT = ast.literal_eval(os.getenv("workexp_current"))

date_parser_1 = cs.date_parser_1
date_parser_2 = cs.date_parser_2
date_parser_3 = cs.date_parser_3
current_word_regex = cs.current_word_regex
only_years = cs.only_years
only_years_1 = cs.only_years_1
years_word = cs.years_word_2
months_word = cs.months_word_2


def pair_sorting(pairs):
    """
    detect outlier point in odd length of total pairs
    Args:
        pairs: list of all founded entity with index

    Returns: return pairs with even length of pair, removed outlier data point

    """
    # O(nlogn)
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


def merge_intervals(intervals):
    """
    merge overlapped time slots in timeline
    Args:
        intervals: list of intervals

    Returns: without overlapped time slots

    """
    sorted_by_lower_bound = sorted(intervals, key=lambda tup: tup[0])
    merged = []

    for higher in sorted_by_lower_bound:
        if not merged:
            merged.append(higher)
        else:
            lower = merged[-1]
            # test for intersection between lower and higher:
            # we know via sorting that lower[0] <= higher[0]
            if higher[0] <= lower[1]:
                upper_bound = max(lower[1], higher[1])
                merged[-1] = (
                    lower[0],
                    upper_bound,
                )  # replace by merged interval
            else:
                merged.append(higher)
    return merged


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


def calculate_total_experience(duration_pair):
    """
    calculate total experience based on founded duration pair , also merge overlap interval in timeline
    Args:
        duration_pair: list of duration pair

    Returns:
        total_experience in month or None
    """
    if len(duration_pair) > 0:
        total_experience = None
        intervals = [pair[-1] for pair in duration_pair]
        intervals = merge_intervals(intervals)
        for pair in intervals:
            delta = abs(relativedelta(pair[1], pair[0]))
            if total_experience:
                total_experience += delta
            else:
                total_experience = delta
        total_experience = (
            (total_experience.years * 12)
            + total_experience.months
            + round(total_experience.days / 30)
        )
        return total_experience
    else:
        return None


def duration_cleaning(duration):
    """
    cleaning duration pair follow
    Args:
       duration: pairs of duration

    Returns:
       cleaned pair of duration
    """
    new_duration, new_word_duration = [], []
    for label, text, start, end in duration:
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
                        "Exp_Duration",
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
                        "Exp_Duration",
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
                        "Exp_Duration",
                        month3 + " " + year3,
                        start + matchNum3 - 1,
                        end,
                        False,
                    ]
                )
        if dates1 == [] and dates2 == [] and dates3 == []:
            if re.search(r"month|year|months|years", text.lower()):
                new_word_duration.append(
                    ["Exp_Duration", text, start, end, False]
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
                ["Exp_Duration", "Present", start + match6.start(), end, False]
            )
            break
    return new_duration, new_word_duration


def find_dates_and_make_pairs(duration):
    """
    finding dates and make duration pairs from duration
    Args:
        duration: list of time
        [['Exp_Duration', '2022-06-16', 0, 0, 7], ['Exp_Duration', 'april 2021', 0, 49, 59], ['Exp_Duration', 'october 2020', 14, 29, 41], ['Exp_Duration', 'april 2021', 14, 44, 54], ['Exp_Duration', 'january 2020', 23, 47, 59], ['Exp_Duration', 'october 2020', 23, 62, 74]]

    Returns:
        date_duration = []  # [['Exp_Duration', '2022-06-16', 0, 0, 7, datetime.datetime(2022, 6, 30, 0, 0)], ['Exp_Duration', 'april 2021', 0, 49, 59, datetime.datetime(2021, 4, 1, 0, 0)], ['Exp_Duration', 'october 2020', 14, 29, 41, datetime.datetime(2020, 10, 1, 0, 0)], ['Exp_Duration', 'april 2021', 14, 44, 54, datetime.datetime(2021, 4, 30, 0, 0)], ['Exp_Duration', 'january 2020', 23, 47, 59, datetime.datetime(2020, 1, 1, 0, 0)], ['Exp_Duration', 'october 2020', 23, 62, 74, datetime.datetime(2020, 10, 31, 0, 0)]]
        duration_pair = []  # [['Exp_Duration', 'april 2021 <-> 2022-06-16', 0, 49, 7, (datetime.datetime(2021, 4, 1, 0, 0), datetime.datetime(2022, 6, 30, 0, 0))], ['Exp_Duration', 'october 2020 <-> april 2021', 14, 29, 54, (datetime.datetime(2020, 10, 1, 0, 0), datetime.datetime(2021, 4, 30, 0, 0))], ['Exp_Duration', 'january 2020 <-> october 2020', 23, 47, 74, (datetime.datetime(2020, 1, 1, 0, 0), datetime.datetime(2020, 10, 31, 0, 0))]]
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
        if len(date_duration) % 2 == 0:
            for start, end in zip(date_duration[0::2], date_duration[1::2]):
                start, end = (
                    min([start, end], key=lambda x: x[-1]),
                    max([start, end], key=lambda x: x[-1]),
                )
                if end[
                    -2
                ]:  # if it's only year then we will make them last day of year.
                    if dt.datetime.today().year == end[-1].year:
                        end[-1] = dt.datetime.today()
                    else:
                        end[-1] = end[-1].replace(month=12, day=31)
                else:
                    # we are making last day of month. for create fair total_experience
                    end[-1] = end[-1] + relativedelta(day=31)
                duration_pair.append(
                    [
                        "Exp_Duration",
                        start[1] + " <-> " + end[1],
                        int((start[2] + end[2]) / 2),
                        start[3],
                        end[4],
                        (start[-1], end[-1]),
                    ]
                )
        else:
            date_duration_calc = pair_sorting(copy.deepcopy(date_duration))
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
                    if dt.datetime.today().year == end[-1].year:
                        end[-1] = dt.datetime.today()
                    else:
                        end[-1] = end[-1].replace(month=12, day=31)
                else:
                    end[-1] = end[-1] + relativedelta(day=31)
                duration_pair.append(
                    [
                        "Exp_Duration",
                        start[1] + " <-> " + end[1],
                        start[2],
                        end[3],
                        (start[-1], end[-1]),
                    ]
                )

    return date_duration, duration_pair


def make_work_exp_one_dict(pairs):
    """
    parse list of pairs to work_exp_one_dict
    Args:
        pairs: list of pairs

    Returns:
        work_exp_one_dict for return
    """
    work_exp_one_dict = {
        "title": None,
        "designation": None,
        "location": None,
        "start_year": None,
        "end_year": None,
        "current": False,
        "index": float("inf"),
    }
    for pair in pairs:
        if pair[0] == "Company_Name":
            work_exp_one_dict["title"] = pair[1]
        elif pair[0] == "Position":
            work_exp_one_dict["designation"] = pair[1]
        elif pair[0] == "Exp_Location":
            work_exp_one_dict["location"] = pair[1]
        else:
            work_exp_one_dict["start_year"] = pair[-1][0].strftime("%Y-%m-%d")
            work_exp_one_dict["end_year"] = pair[-1][1].strftime("%Y-%m-%d")
            if dt.date.today() + relativedelta(day=31) == pair[-1][1].date():
                work_exp_one_dict["current"] = True
                work_exp_one_dict["end_year"] = "Present"
        work_exp_one_dict["index"] = min(work_exp_one_dict["index"], pair[2])
    return work_exp_one_dict


def line_no_calculation(pairs, texts):
    """
    find line number of every entity and return line number at 2 position
    Args:
        pairs: all pairs from extracted entities
        texts: document text
    Returns:
        list of pairs with line no of every entity
    """
    new_pairs = []
    for pair in pairs:
        new_pair = pair
        new_pair[2:2] = [len(texts[0 : pair[3]].split("\n"))]
        new_pairs.append(new_pair)
    return new_pairs


def extract_years_months(text):
    """
    Extracts the total number of years and months from the given text.

    Args:
        text (str): The input text containing information about years and months.

    Returns:
        dict: A dictionary containing the total number of years and months.
              The dictionary has the following structure:
              {
                  "years": <total years>,
                  "months": <total months>
              }
              The "years" and "months" keys are optional and will only be present
              if the respective values are greater than 0.

    Example:
     >>> extract_years_months("4years and 4 months")
     {'years': 4.0, 'months': 4.0}
     >>> extract_years_months("4 years and 4 months")
     {'years': 4.0, 'months': 4.0}
     >>> extract_years_months("4 years")
     {'years': 4.0}
     >>> extract_years_months("4 months")
     {'months': 4.0}
    """

    # Replace words that represent numbers with the actual numbers
    for word, number in cs.word_to_number.items():
        text = text.replace(word, str(number))

    # Replace "plus" with "+"
    text = text.replace("plus", "+")

    # Regular expression pattern to match years
    year_pattern = r"(\d+(\.\d+)?(?=\s*[\-+]*\s*(years?|yrs?)))"
    # Regular expression pattern to match months
    month_pattern = r"(\d+(\.\d+)?(?=\s*[\-+]*\s*months?))"

    years = re.findall(year_pattern, text, re.IGNORECASE)
    years = [float(y[0]) for y in years]

    months = re.findall(month_pattern, text, re.IGNORECASE)
    months = [float(m[0]) for m in months]

    result = {}

    # If years exist, store the sum in the result
    if years:
        result["years"] = sum(years)

    # If months exist, store the sum in the result
    if months:
        result["months"] = sum(months)

    return result


def find_total_experience(total_experience_words_pairs):
    """
    find total experience from direct text like "5 years of Experience" output is 60
    Args:
        total_experience_words_pairs: list of total_experience_words_pairs
    Returns:
        total experience in months
    """
    total_experience = []
    for total_experience_words_pair in total_experience_words_pairs:
        single_experience = extract_years_months(
            total_experience_words_pair[1].lower()
        )
        total_experience.append(
            single_experience.get("years", 0) * 12
            + single_experience.get("months", 0)
        )
    if total_experience:
        return int(max(total_experience))
    else:
        return None


def work_exp_ner_pred(extracted_entities, resume_text_doc):
    """
    Maps the company name with duration.
    Args:
        extracted_entities : payload data of request
        resume_text_doc: resume text
    Returns:
        list of dictionaries containing company_name mapped with duration.
    """

    work_exp_final_lst = []
    total_experience = None
    retry_chatgpt = False  # flag for chatgpt parser. do we require to parse again with chatgpt or not?
    if len(extracted_entities) == 0:
        retry_chatgpt = True
        return work_exp_final_lst, total_experience, retry_chatgpt
    else:
        company_name = [
            [label, text.replace("\n", " "), start, end]
            for start, end, text, label in extracted_entities
            if label == "Company_Name"
        ]
        position = [
            [label, text.replace("\n", " "), start, end]
            for start, end, text, label in extracted_entities
            if label == "Position"
        ]
        duration = [
            [label, text.replace("\n", " "), start, end]
            for start, end, text, label in extracted_entities
            if label == "Exp_Duration"
        ]
        # location = [[label, text.replace('\n', ' '), start, end]
        #             for start, end, text, label in extracted_entities if label == 'Exp_Location']
        total_experience_words = [
            [label, text.replace("\n", " "), start, end]
            for start, end, text, label in extracted_entities
            if label == "Total_Experience"
        ]
        # if company_name or position or duration, all lists are empty, return from here
        if not (company_name or position or duration):
            retry_chatgpt = True
            return work_exp_final_lst, total_experience, retry_chatgpt

        # if company_name and position length are not same or  no company found so raise retry_chatgpt flag
        if len(company_name) != len(position) or len(company_name) == 0:
            retry_chatgpt = True

        company_name = sorted(company_name, key=lambda x: (x[2]))
        position = sorted(position, key=lambda x: (x[2]))
        # location = sorted(location, key=lambda x: (x[2]))
        duration = sorted(duration, key=lambda x: (x[2]))
        new_duration, new_word_duration = duration_cleaning(
            copy.deepcopy(duration)
        )
        date_duration, duration_pair = find_dates_and_make_pairs(
            copy.deepcopy(new_duration)
        )

        # Total_experience
        if total_experience_words:
            total_experience = find_total_experience(total_experience_words)
        if not total_experience:
            if duration_pair:
                total_experience = calculate_total_experience(duration_pair)
            else:
                total_experience = find_total_experience(new_word_duration)

        only_company_name_pairs = line_no_calculation(
            copy.deepcopy(company_name), resume_text_doc
        )

        comp_dur_lst = copy.deepcopy(company_name)
        comp_dur_lst.extend(copy.deepcopy(duration_pair))
        comp_dur_lst = sorted(comp_dur_lst, key=lambda x: x[2])
        comp_dur_lst = line_no_calculation(
            comp_dur_lst, resume_text_doc
        )  # add line no
        comp_dur_lst = sorted(comp_dur_lst, key=lambda x: (x[2], x[3]))

        comp_pos_lst = copy.deepcopy(company_name)
        comp_pos_lst.extend(copy.deepcopy(position))
        comp_pos_lst = sorted(comp_pos_lst, key=lambda x: x[2])
        com_pos_dur_lst = copy.deepcopy(duration_pair)
        com_pos_dur_lst.extend(comp_pos_lst)
        com_pos_dur_lst = line_no_calculation(
            com_pos_dur_lst, resume_text_doc
        )  # add line no
        com_pos_dur_lst = sorted(com_pos_dur_lst, key=lambda x: (x[2], x[3]))
        unique_class = set((lst[0] for lst in com_pos_dur_lst))
        if len(unique_class) == 3:
            print("tri_pair")
            # Data loss for two entities pair and one entity pair
            numbers_with_distance = [(com_pos_dur_lst[0], float("inf"))]
            for i in range(1, len(com_pos_dur_lst) - 1):
                left_distance = abs(
                    com_pos_dur_lst[i - 1][2] - com_pos_dur_lst[i][2]
                )
                right_distance = abs(
                    com_pos_dur_lst[i + 1][2] - com_pos_dur_lst[i][2]
                )
                numbers_with_distance.append(
                    (com_pos_dur_lst[i], left_distance + right_distance)
                )
            numbers_with_distance.append((com_pos_dur_lst[-1], float("inf")))
            minimum = min(numbers_with_distance, key=lambda item: item[1])[1]
            min_dist_list = [
                (pair[0], index, pair[1])
                for index, pair in enumerate(numbers_with_distance)
                if pair[1] == minimum
            ]
            min_val_list = subarray_with_k_distinct(com_pos_dur_lst, 3)

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
                for index in range(0, len(com_pos_dur_lst) - 2):
                    if (
                        com_pos_dur_lst[index][0] == pattern[0]
                        and com_pos_dur_lst[index + 1][0] == pattern[1]
                        and com_pos_dur_lst[index + 2][0] == pattern[2]
                    ):
                        work_exp_final_lst.append(
                            make_work_exp_one_dict(
                                com_pos_dur_lst[index : index + 3]
                            )
                        )
                        used_entities.extend(
                            com_pos_dur_lst[index : index + 3]
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
                    for pair in com_pos_dur_lst
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
                        work_exp_final_lst.append(
                            make_work_exp_one_dict(
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
                        work_exp_final_lst.append(
                            make_work_exp_one_dict(
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
                        work_exp_final_lst.append(
                            make_work_exp_one_dict(
                                unused_entities[start : start + 2]
                            )
                        )
                        start += 2
                    else:
                        start += 1
                work_exp_final_lst = sorted(
                    work_exp_final_lst, key=lambda x: x["index"]
                )
            else:  # Founded Three class but not distinct pair of three class so removed position and go with two pair
                sorted_pair = pair_sorting(comp_dur_lst)
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
                    for index in range(0, len(comp_dur_lst) - 1):
                        if (
                            comp_dur_lst[index][0] == pattern[0]
                            and comp_dur_lst[index + 1][0] == pattern[1]
                        ):
                            work_exp_final_lst.append(
                                make_work_exp_one_dict(
                                    comp_dur_lst[index : index + 2]
                                )
                            )

        elif len(unique_class) == 2:
            print("dual_pair")
            # Data loss for one entity pair
            sorted_pair = pair_sorting(com_pos_dur_lst)
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
                for index in range(0, len(com_pos_dur_lst) - 1):
                    if (
                        com_pos_dur_lst[index][0] == pattern[0]
                        and com_pos_dur_lst[index + 1][0] == pattern[1]
                    ):
                        work_exp_final_lst.append(
                            make_work_exp_one_dict(
                                com_pos_dur_lst[index : index + 2]
                            )
                        )

            else:  # Founded Two class but not distinct pair of two class so removed position and go with only company
                for pair in only_company_name_pairs:
                    work_exp_one_dict = {
                        "title": None,
                        "designation": None,
                        "start_year": None,
                        "end_year": None,
                        "current": False,
                        "index": float("inf"),
                    }
                    if pair[0] == "Company_Name":
                        work_exp_one_dict["title"] = pair[1]
                    elif pair[0] == "Position":
                        work_exp_one_dict["designation"] = pair[1]
                    elif pair[0] == "Exp_Location":
                        work_exp_one_dict["location"] = pair[1]
                    else:
                        work_exp_one_dict["start_year"] = pair[-1][0].strftime(
                            "%Y-%m-%d"
                        )
                        work_exp_one_dict["end_year"] = pair[-1][1].strftime(
                            "%Y-%m-%d"
                        )
                        if (
                            dt.date.today() + relativedelta(day=31)
                            == pair[-1][1].date()
                        ):
                            work_exp_one_dict["current"] = True
                            work_exp_one_dict["end_year"] = "Present"
                    work_exp_one_dict["index"] = min(
                        work_exp_one_dict["index"], pair[2]
                    )
                    work_exp_final_lst.append(work_exp_one_dict)
        elif len(unique_class) == 1:
            print("one_pair")
            if list(unique_class)[0] != "Position":
                for pair in com_pos_dur_lst:
                    work_exp_one_dict = {
                        "title": None,
                        "designation": None,
                        "start_year": None,
                        "end_year": None,
                        "current": False,
                        "index": float("inf"),
                    }
                    if pair[0] == "Company_Name":
                        work_exp_one_dict["title"] = pair[1]
                    elif pair[0] == "Position":
                        work_exp_one_dict["designation"] = pair[1]
                    elif pair[0] == "Exp_Location":
                        work_exp_one_dict["location"] = pair[1]
                    else:
                        work_exp_one_dict["start_year"] = pair[-1][0].strftime(
                            "%Y-%m-%d"
                        )
                        work_exp_one_dict["end_year"] = pair[-1][1].strftime(
                            "%Y-%m-%d"
                        )
                        if (
                            dt.date.today() + relativedelta(day=31)
                            == pair[-1][1].date()
                        ):
                            work_exp_one_dict["current"] = True
                            work_exp_one_dict["end_year"] = "Present"
                    work_exp_one_dict["index"] = min(
                        work_exp_one_dict["index"], pair[2]
                    )
                    work_exp_final_lst.append(work_exp_one_dict)

    work_exp_final_lst = sorted(work_exp_final_lst, key=lambda d: d["index"])
    if total_experience:
        if total_experience > 500:
            info_logger(
                f"total_experience is higher than limit {total_experience} so resetting None"
            )
            total_experience = None
    return work_exp_final_lst, total_experience, retry_chatgpt
