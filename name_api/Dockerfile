FROM python:3.10-slim

RUN apt-get update  \
    && apt-get install -y build-essential cmake pkg-config libpq-dev git && apt-get clean all

RUN pip install --upgrade pip
COPY ./name_api/requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt

COPY ./name_api /

CMD [ "uvicorn", "name_service_fastapi:app", "--host", "0.0.0.0", "--port", "8051"]