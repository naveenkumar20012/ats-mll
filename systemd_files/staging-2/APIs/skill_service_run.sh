# shellcheck disable=SC2164
source /home/abhishek.shingadiya/fastapi-venv/bin/activate
cd /home/abhishek.shingadiya/ats-ml/skill_api
uvicorn skill_service_fastapi:app --port 8053 --workers 1