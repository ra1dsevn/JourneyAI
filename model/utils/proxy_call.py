import requests
import json
import logging
import os
import openai
import time

class OpenaiCall:
    def __init__(self):
        # Configure the base URL and API key
        # openai.api_base = 'https://api.gptapi.us/v1/'
        # openai.api_key = 'sk-xxx'
        openai.api_base = 'https://xiaoai.plus/v1'
        openai.api_key = 'sk-xxx'

    # 在 OpenaiCall.chat 中
    def chat(self, messages, model="gpt-4o-mini", temperature=0, max_retries=5):
        for attempt in range(max_retries):
            try:
                logging.info(f"Attempt {attempt + 1}: Calling OpenAI API with model={model}, messages={messages}")
                response = openai.ChatCompletion.create(
                    model=model,
                    messages=messages,
                    temperature=temperature
                )
                logging.info(f"Received response: {response}")
                return response.choices[0].message.content
            except Exception as e:
                if attempt < max_retries - 1:
                    delay = 2 ** attempt + 5  # 增加基础延迟
                    logging.warning(f"Error on attempt {attempt + 1}: {str(e)}. Retrying after {delay} seconds...")
                    time.sleep(delay)
                    continue
                logging.error(f"Failed after {max_retries} attempts: {str(e)}")
                raise e

    def stream_chat(self, messages, model="gpt-4o-mini", temperature=0):
        for chunk in openai.ChatCompletion.create(
            model=model,
            messages=messages,
            temperature=temperature,
            stream=True
        ):
            if hasattr(chunk.choices[0], 'delta') and hasattr(chunk.choices[0].delta, 'content'):
                yield chunk.choices[0].delta.content

    def embedding(self, input_data):
        response = openai.Embedding.create(
            input=input_data,
            model="text-embedding-3-small"
        )
        return response
