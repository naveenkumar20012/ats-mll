# shellcheck disable=SC2164
source /home/abhishek.shingadiya/fastapi-venv/bin/activate
cd /home/abhishek.shingadiya/ats-ml/ner_api
uvicorn ner_service_fastapi_2:app --port 8052 --workers 16 --limit-concurrency 8
