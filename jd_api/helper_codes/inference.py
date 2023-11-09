"""
Inference module which is used in fastapi main.py.
"""

import logging as logger
import os
import re

import requests
from dotenv import load_dotenv
from google.cloud import storage

from pathlib import Path

import sys

sys.path.append(str(Path(__file__).parent.parent))

from helper_codes import helpers
from ModelLoader import ModelLoader

logger.basicConfig(level="DEBUG")

load_dotenv("./constants.env")

PROJECT_NAME = os.getenv("PROJECT_NAME")
BUCKET_NAME = os.getenv("BUCKET_NAME")
# GCS_FOLDER_PATH = "jds_denny/"
DESTINATION_FOLDER = os.getenv("DESTINATION_FOLDER")
# keeps only alphanumeric characters in text
KEEP_ALPHANUM = re.compile(r"[\W_]+", re.UNICODE)

if not os.path.exists(DESTINATION_FOLDER):
    os.mkdir(DESTINATION_FOLDER)


def saving_file_local(filepath):
    """
    Take filepath, filepath a gcs path, return filepath stored
    in local directory.

    Args:
        filepath(str): Absolute path of the file.

    Returns:
        Absolute path of file wrt to local directory.
    """
    filename = filepath.split("/")[-1]
    if filepath.startswith("gs://"):  # that means environment is "prod"

        storage_client = storage.Client()

        bucket = storage_client.bucket(BUCKET_NAME)

        # Construct a client side representation of a blob.
        # Note `Bucket.blob` differs from `Bucket.get_blob` as it doesn't retrieve
        # any content from Google Cloud Storage. As we don't need additional data,
        # using `Bucket.blob` is preferred here.
        blob_name = "/".join(filepath.split("/")[3:])
        blob = bucket.blob(blob_name)
        dest_file_path = os.path.join(DESTINATION_FOLDER + filename)
        logger.info(f"dest_file_path :{dest_file_path}")
        blob.download_to_filename(dest_file_path)

        logger.info(
            "Blob {} downloaded to {}.".format(filepath, dest_file_path)
        )

    else:  # If file is in local, no need to copy to local.
        dest_file_path = filepath

    return dest_file_path


def extract_job(lst):
    """
    call jd api to get ner entities
    Args:
        lst: list of words

    Returns:
        found entities in dictionary
    """
    try:
        NLP_NER = ModelLoader.getInstance().get_model()
        dic_final = {}
        docs_list = list(NLP_NER.pipe(lst))
        for ind, doc in enumerate(docs_list):  # for each datapoint
            pred_json = {}
            text = lst[ind]
            for (
                ent
            ) in doc.ents:  # for each predicted entity # the predicted word
                if ent.label_ in pred_json.keys():
                    pred_json[ent.label_].append(
                        [ent.start_char, ent.end_char, ent.text, text]
                    )
                else:
                    pred_json[ent.label_] = [
                        [ent.start_char, ent.end_char, ent.text, text]
                    ]
            dic_final = helpers.predicted_json_manipulation(
                ind, pred_json, dic_final
            )
        return dic_final
    except Exception as e:
        print(
            "\n\nException occured while extracting JD entities. {}\n\n".format(
                e
            )
        )
        return {}


def inference(jd_path):
    """
    Does inference on one JD using trained NER model.

    Args:
        jd_path(str) : Path to the JD

    Returns:
        A dictionary containing gkey as label and value as list of predicted entities.
    """
    # final output dictionary {Label: [Words]}
    dic_final = {}
    jd_file = os.path.basename(jd_path)
    html_version = None

    if jd_file.endswith(".txt"):
        lst = helpers.jd_txt(jd_path)
    elif jd_file.endswith(".pdf"):
        lst, html_version = helpers.jd_pdf(jd_path)
    elif jd_file.endswith(".docx"):
        lst, html_version = helpers.jd_docx(jd_path)
    elif jd_file.endswith(".doc"):
        lst, html_version = helpers.jd_doc(jd_path)
    else:
        return "Extension doesn't match : {}".format(jd_file)

    # html_version = None
    if html_version:
        full_jd_text = html_version
    else:
        full_jd_text = "\n".join(lst)

    lst = [item for item in lst if not "job_link" in item and item != "\n"]

    dict_final = extract_job(lst)

    for key, val in dic_final.items():
        val_alpha = []
        job_title_label = helpers.get_true_label_name("Job_Title")
        work_ex_label = helpers.get_true_label_name("Work_Exp")
        if key == job_title_label:
            new_val = []
            # Since a job can have only one Job-title. Take the first element predicted as job-title.
            new_val = val[0]
        elif key == work_ex_label:
            val_alpha = []
            new_val = []
            for ind, ele in enumerate(val):
                ele_0_alpha = KEEP_ALPHANUM.sub("", ele[0])
                if ele_0_alpha in val_alpha:
                    pass
                else:
                    val_alpha.append(ele_0_alpha)
                    new_val.append(ele)

            # To have only one work exp in the final pred.
            # exp_maxdiff: max difference between min and max workexp
            # exp_diff: difference between min and max for each datapoint.
            if len(new_val) > 1:
                for ind, workexp in enumerate(new_val):
                    if ind == 0:
                        if workexp[1]["max"] is None:
                            exp_maxdiff = workexp[1]["min"]
                            final_workexp = [workexp]
                            # exps.append({"Max_None": workexp[1]["min_exp"]})
                        else:
                            exp_maxdiff = workexp[1]["max"] - workexp[1]["min"]
                            final_workexp = [workexp]
                            # exps.append("Min_Max" : workexp[1]["max"] - workexp[1]["min_exp"])
                    else:
                        if workexp[1]["max"] == None:
                            exp_diff = workexp[1]["min"]
                            if exp_diff > exp_maxdiff:
                                exp_maxdiff = exp_diff
                                final_workexp = [workexp]

                        else:
                            exp_diff = workexp[1]["max"] - workexp[1]["min"]
                            if exp_diff > exp_maxdiff:
                                exp_maxdiff = exp_diff
                                final_workexp = [workexp]
                new_val = [final_workexp[0][1]]
            else:
                new_val = [new_val[0][1]]
        # If label is not Work_exp and Job-title
        else:
            new_val = val.copy()
            for ele in val:
                ele_alpha = KEEP_ALPHANUM.sub("", ele)
                if ele_alpha in val_alpha:
                    new_val.remove(ele)
                else:
                    val_alpha.append(ele_alpha)

        dict_final[key] = new_val

    dict_final["job_description"] = full_jd_text
    return dict_final
