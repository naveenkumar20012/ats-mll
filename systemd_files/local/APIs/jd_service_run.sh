# shellcheck disable=SC2164
source ../fastapi-venv/bin/activate
uvicorn JD_parser.helper_codes.jd_parser_fastapi:app --port 8053
