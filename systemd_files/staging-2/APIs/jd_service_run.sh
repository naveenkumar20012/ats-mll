# shellcheck disable=SC2164
source /home/abhishek.shingadiya/fastapi-venv/bin/activate
cd /home/abhishek.shingadiya/ats-ml/jd_api
uvicorn jd_service_fastapi:app --port 8054

