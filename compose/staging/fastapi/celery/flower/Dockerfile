FROM python:3.10-slim

ENV PYTHONUNBUFFERED 1
ENV PYTHONDONTWRITEBYTECODE 1

RUN apt-get update \
  && apt-get install -y curl unzip python-dev libxml2-dev libxslt1-dev antiword unrtf poppler-utils tesseract-ocr flac ffmpeg lame libmad0 libsox-fmt-mp3 sox libjpeg-dev swig \
  && apt-get clean
RUN pip install --upgrade pip
COPY ./main_api/requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt

RUN python -m nltk.downloader stopwords
RUN python -m nltk.downloader punkt


COPY ./compose/staging/fastapi/celery/flower/start /start
RUN sed -i 's/\r$//g' /start
RUN chmod +x /start

COPY ./main_api /
RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
RUN unzip awscliv2.zip
RUN ./aws/install

ENTRYPOINT ["/start"]