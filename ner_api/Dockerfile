FROM python:3.10-slim

RUN pip install --upgrade pip
COPY ./ner_api/requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt

COPY ./ner_api /

CMD [ "uvicorn", "ner_service_fastapi:app", "--host", "0.0.0.0", "--port", "8052"]