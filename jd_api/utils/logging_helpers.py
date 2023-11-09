"""
Helper functions to deal with how to log intermediate outputs
"""
import logging as logger
import os

import sentry_sdk

logger.basicConfig(
    format="%(asctime)s: %(levelname)s: %(message)s",
    datefmt="%m/%d/%Y %I:%M:%S %p",
    level="DEBUG",
)

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
        f'ENVIRONMENT not found for logger. Current one is {os.getenv("ENVIRONMENT")}'
    )


def info_logger(*args):
    logger.info(args)


def error_logger(*args):
    logger.error(args)
