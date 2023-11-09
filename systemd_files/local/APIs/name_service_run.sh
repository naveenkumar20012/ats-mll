# shellcheck disable=SC2164
source ../fastapi-venv/bin/activate
cd ./RP_parser/helper_codes
uvicorn name_service_fastapi:app --port 8051
