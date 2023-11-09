# shellcheck disable=SC2164
source /home/abhishek.shingadiya/fastapi-venv/bin/activate
cd /home/abhishek.shingadiya/ats-ml/name_api
uvicorn name_service_fastapi:app --port 8051
