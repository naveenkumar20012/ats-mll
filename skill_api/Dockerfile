FROM python:3.10-slim

RUN pip install --upgrade pip
COPY ./skill_api/requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt

COPY ./skill_api /

CMD [ "uvicorn", "skill_service_fastapi:app", "--host", "0.0.0.0", "--port", "8054"]