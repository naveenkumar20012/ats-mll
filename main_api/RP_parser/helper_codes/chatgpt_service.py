import json

from utils.logging_helpers import info_logger, error_logger
import os
import re
import time
import ast
import requests
from dotenv import load_dotenv
import openai

load_dotenv("./constants.env")
CHATGPT_API = str(os.getenv("CHATGPT_API"))
CHATGPT_USERNAME = str(os.getenv("CHATGPT_USERNAME"))
CHATGPT_PASSWORD = str(os.getenv("CHATGPT_PASSWORD"))
openai.api_key = str(os.getenv("OPENAI_KEY"))


class ChatGPT:
    """Wrapper for ChatGPT functions"""

    endpoint = CHATGPT_API
    username = CHATGPT_USERNAME
    password = CHATGPT_PASSWORD

    def chatgpt_request(self, question):
        """chatGPT request"""

        result = {}
        try:
            for retry_count in range(5):
                try:
                    body = {"message": question}
                    response = requests.post(
                        self.endpoint,
                        json=body,
                        auth=(self.username, self.password),
                    )
                    response = response.json()
                    text = response.get("text")
                    result = self.parse_json(text)
                    result["retry_count"] = retry_count + 1
                    info_logger(result)
                    if result["json_answer"]:
                        break
                except Exception as e:
                    info_logger(f"Failed to get Answer from ChatGPT: {e}")
                    continue
        except Exception as e:
            info_logger(f"ChatGPT failed with error: {e}")
        return result

    def chatgpt_request_openai(self, role, question):
        """chatGPT request"""
        result, response_text = {}, ""
        try:
            for retry_count in range(3):
                try:
                    completion = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role": "system", "content": role},
                            {"role": "user", "content": question},
                        ],
                        temperature=0,
                    )
                    response_text = completion["choices"][0]["message"][
                        "content"
                    ]
                    result = self.parse_json(response_text)
                    result["retry_count"] = retry_count + 1
                    info_logger(result)
                    if result["json_answer"]:
                        break
                    else:
                        info_logger(
                            f"Retry Count: {retry_count + 1} Answer: {response_text} "
                        )
                except Exception as e:
                    info_logger(
                        f"Failed to get Answer from ChatGPT: {e} answer from chatgpt is {response_text}"
                    )
                    continue

        except Exception as e:
            info_logger(f"ChatGPT failed with error: {e}")

        return result

    @staticmethod
    def parse_json(answer):
        """parse chatGPT response in valid json"""
        try:
            json_answer = ast.literal_eval(answer)
            return {"base_answer": answer, "json_answer": json_answer}
        except Exception as e:
            info_logger(f"ChatGPT json ast convert: {answer} {e}")
        # if chatGPT response to json ast.literal_eval failed, trying with json.loads
        try:
            json_answer = json.loads(answer)
            return {"base_answer": answer, "json_answer": json_answer}
        except Exception as e:
            info_logger(f"ChatGPT json loads convert: {answer} {e}")
            return {"base_answer": answer, "json_answer": None}
