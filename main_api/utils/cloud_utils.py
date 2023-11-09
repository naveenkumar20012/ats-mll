from __future__ import print_function

import logging as logger
import os
import time

import boto3
from dotenv import load_dotenv
from google.cloud import storage
from azure.storage.blob import BlobServiceClient, ContentSettings
from utils.logging_helpers import info_logger

logger.basicConfig(level="INFO")
load_dotenv("./constants.env")

PROJECT_NAME = os.getenv("PROJECT_NAME")
BUCKET_NAME_get = os.getenv("BUCKET_NAME_get")
BUCKET_NAME_post = os.getenv("BUCKET_NAME_post")

DESTINATION_FOLDER = os.getenv("DESTINATION_FOLDER")
JD_DESTINATION_BLOB = os.getenv("JD_DESTINATION_BLOB")
AWS_BUCKET_NAME = os.getenv("AWS_BUCKET_NAME")
CLOUD_BACKEND = os.getenv("CLOUD_BACKEND")
RETRY_COUNTS = int(os.getenv("RETRY_COUNTS"))
RETRY_TIME_GAP = int(os.getenv("RETRY_TIME_GAP"))
AZURE_STORAGE_ACCOUNT_NAME = os.getenv("AZURE_STORAGE_ACCOUNT_NAME")
AZURE_STORAGE_ACCOUNT_KEY = os.getenv("AZURE_STORAGE_ACCOUNT_KEY")
AZURE_CONTAINER_NAME = os.getenv("AZURE_CONTAINER_NAME")
AZURE_JD_CONTAINER_NAME = os.getenv("AZURE_JD_CONTAINER_NAME")
s3 = boto3.client("s3")
blob_service_client = BlobServiceClient(
    account_url=f"https://{AZURE_STORAGE_ACCOUNT_NAME}.blob.core.windows.net",
    credential=AZURE_STORAGE_ACCOUNT_KEY,
)

if not os.path.exists("." + DESTINATION_FOLDER):
    os.mkdir("." + DESTINATION_FOLDER)


def upload_blob_from_memory(filepath, bucket_name=None, s3=s3):
    """
    File stored from a local path to gcp container or aws container based on CLOUD_BACKEND from .env.

    Args:
        filepath(str): Absolute path of the file.
        bucket_name: Target bucket name

    Returns:
        gcp_path.
    """
    if CLOUD_BACKEND == "gcp":
        if bucket_name is None:
            bucket_name = BUCKET_NAME_post
        filename = (
            time.strftime("%Y%m%d-%H%M%S") + "_" + filepath.split("/")[-1]
        )
        destination_blob_name = JD_DESTINATION_BLOB + filename
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(destination_blob_name)
        blob.upload_from_filename(filepath)
        info_logger(
            f"On GCP, File {filepath} uploaded to {destination_blob_name}."
        )
        return (
            r"https://storage.googleapis.com/"
            + BUCKET_NAME_post
            + "/"
            + destination_blob_name
        )
    elif CLOUD_BACKEND == "aws":
        if bucket_name is None:
            bucket_name = AWS_BUCKET_NAME
        filename = (
            time.strftime("%Y%m%d-%H%M%S") + "_" + filepath.split("/")[-1]
        )
        destination_blob_name = JD_DESTINATION_BLOB + filename
        s3.upload_file(filepath, bucket_name, destination_blob_name)
        info_logger(
            f"On AWS, File {filepath} uploaded to {destination_blob_name}."
        )
        return (
            r"https://"
            + bucket_name
            + ".s3.ap-south-1.amazonaws.com/"
            + destination_blob_name
        )
    elif CLOUD_BACKEND == "azure":
        filename = (
            time.strftime("%Y%m%d-%H%M%S") + "_" + filepath.split("/")[-1]
        )
        destination_blob_name = (
            AZURE_JD_CONTAINER_NAME + "/" + JD_DESTINATION_BLOB + filename
        )
        blob_client = blob_service_client.get_blob_client(
            container=AZURE_CONTAINER_NAME, blob=destination_blob_name
        )

        with open(filepath, "rb") as data:
            blob_client.upload_blob(
                data,
                overwrite=True,
                content_settings=ContentSettings(
                    content_type="application/octet-stream"
                ),
            )

        logger.info(
            f"On Azure, File {filepath} uploaded to {destination_blob_name}."
        )
        return (
            r"https://"
            + AZURE_STORAGE_ACCOUNT_NAME
            + ".blob.core.windows.net/"
            + AZURE_CONTAINER_NAME
            + "/"
            + destination_blob_name
        )


def saving_file_local_resume(filepath, bucket_name=None, s3=s3):
    """
    Downloads the file from cloud storage and saves it in a local directory.

    Args:
        filepath (str): The absolute path of the file.
        bucket_name (str): The name of the target bucket if file is in cloud storage.
        s3 (boto3.client): The S3 client if file is in S3 bucket.

    Returns:
        str: The absolute path of the downloaded file wrt to the local directory.
    """
    filename = os.path.basename(filepath)

    if filepath.startswith("gs://"):
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob_name = "/".join(filepath.split("/")[3:])
        blob = bucket.blob(blob_name)
        destination_file_path = os.path.join(DESTINATION_FOLDER, filename)

        if not os.path.exists(destination_file_path):
            blob.download_to_filename(destination_file_path)

        info_logger(
            "From GCP, Blob {} downloaded to {}.".format(
                filepath, destination_file_path
            )
        )

    elif filepath.startswith("s3://"):
        destination_file_path = os.path.join(DESTINATION_FOLDER, filename)
        FILE_NAME = "/".join(filepath.split("/")[3:])

        if not os.path.exists(destination_file_path):
            for retry_count in range(RETRY_COUNTS):
                try:
                    s3.download_file(
                        bucket_name, FILE_NAME, destination_file_path
                    )
                    info_logger(
                        "From AWS, Blob {} downloaded to {}.".format(
                            filepath, destination_file_path
                        )
                    )
                    break
                except Exception as exception:
                    info_logger(
                        f"From s3, Got Exception {exception} retrying {retry_count + 1} times."
                    )
                    time.sleep(RETRY_TIME_GAP)
            else:
                raise Exception(f"Unable to download file")
    elif filepath.startswith("azure://"):
        destination_file_path = os.path.join(DESTINATION_FOLDER, filename)
        FILE_NAME = "/".join(filepath.split("/")[4:])
        if not os.path.exists(destination_file_path):
            for retry_count in range(RETRY_COUNTS):
                try:
                    blob_client = blob_service_client.get_blob_client(
                        container=AZURE_CONTAINER_NAME, blob=FILE_NAME
                    )
                    blob_data = blob_client.download_blob()
                    with open(destination_file_path, "wb") as local_file:
                        local_file.write(blob_data.readall())
                    break
                except Exception as exception:
                    info_logger(
                        f"From Azure, Got Exception {exception} retrying {retry_count + 1} times."
                    )
                    time.sleep(RETRY_TIME_GAP)
    else:  # If the file is in local, no need to download.
        destination_file_path = filepath
    return destination_file_path
