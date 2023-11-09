"""
Helper functions to deal with how to log intermediate outputs
"""
import logging
import os
from logging.handlers import TimedRotatingFileHandler
from datetime import datetime
import sentry_sdk

LOG_DIRECTORY = os.getenv("LOG_DIRECTORY")
log_file = f"{LOG_DIRECTORY}/{datetime.now().strftime('%Y-%m-%d')}.log"
os.makedirs(LOG_DIRECTORY, exist_ok=True)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

formatter = logging.Formatter("%(asctime)s %(levelname)s: %(message)s")

file_handler = TimedRotatingFileHandler(
    filename=log_file, when="midnight", backupCount=7
)
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)

logger.addHandler(file_handler)

if (
    os.getenv("ENVIRONMENT") == "production"
    or os.getenv("ENVIRONMENT") == "staging"
):
    sentry_sdk.init(
        os.getenv("SENTRY_API"),
        traces_sample_rate=1.0,
        environment=os.getenv("ENVIRONMENT"),
    )
else:
    print(
        f"ENVIRONMENT not found for logger. Current one is {os.getenv('ENVIRONMENT')}"
    )


def info_logger(*args):
    logger.info(args)


def error_logger(*args):
    logger.error(args)
