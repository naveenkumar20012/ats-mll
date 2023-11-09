import os

from celery import Celery
import ssl

BROKER_URI = os.environ["BROKER_URI"]
BACKEND_URI = os.environ["BACKEND_URI"]

app = Celery(
    "celery_app",
    broker=BROKER_URI,
    backend=BACKEND_URI,
    # broker_use_ssl={'ssl_cert_reqs': ssl.CERT_NONE},
    # redis_backend_use_ssl={'ssl_cert_reqs': ssl.CERT_NONE},
    include=["celery_tasks.tasks"],
)
