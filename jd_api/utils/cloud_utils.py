import logging as logger
import os
import time

import boto3
from dotenv import load_dotenv
from google.cloud import storage
from azure.storage.blob import BlobServiceClient, ContentSettings

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
    file stored from a local path to gcp container or aws container based on CLOUD_BACKEND from .env.

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
        logger.info(
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
        logger.info(
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
