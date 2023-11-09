# shellcheck disable=SC2164
source ../fastapi-venv/bin/activate
cd ./RP_parser/helper_codes
uvicorn ner_service_fastapi_2:app --port 8052 --workers 16 --limit-concurrency 8
