FROM python:3.10-slim

RUN pip install --upgrade pip
COPY ./jd_api/requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt

COPY ./jd_api /

CMD [ "uvicorn", "jd_service_fastapi:app", "--host", "0.0.0.0", "--port", "8053"]