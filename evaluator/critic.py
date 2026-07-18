import os
import time
import requests
from typing import Callable

class CriticAgent(object):
    def __init__(self,
                 system_prompt: str = None):
        self.system_prompt = system_prompt
        # Ollama serving the WritingBench critic model.
        self.url = os.environ.get("OLLAMA_URL", "http://localhost:11434") + "/api/chat"
        self.model = os.environ.get(
            "CRITIC_MODEL",
            "hf.co/mradermacher/WritingBench-Critic-Model-Qwen-7B-i1-GGUF:Q4_K_M",
        )

    def call_critic(self,
            messages: str,
            top_p: float = 0.95,
            temperature: float = 1.0,
            max_length: int = 2048):

        data = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "top_p": top_p,
                "num_predict": int(max_length),
            },
        }

        attempt = 0
        max_attempts = 5
        wait_time = 1

        while attempt < max_attempts:
            try:
                response = requests.post(self.url, json=data, timeout=600)

                if response.status_code == 200:
                    return response.json()["message"]["content"]
                else:
                    print(f"Attempt {attempt+1}: Failed with status {response.status_code}, retrying...")

            except requests.exceptions.RequestException as e:
                print(f"Attempt {attempt+1}: Ollama call failed due to network error: {e}, retrying...")

            time.sleep(wait_time)
            attempt += 1

        raise Exception("Max attempts exceeded. Failed to get a successful response.")

    def basic_success_check(self, response):
        if not response:
            print(response)
            return False
        else:
            return True

    def run(self,
            prompt: str,
            top_p: float = 0.95,
            temperature: float = 1.0,
            max_length: int = 2048,
            max_try: int = 5,
            success_check_fn: Callable = None):

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user","content": prompt}
        ]
        success = False
        try_times = 0
        response = ""

        while try_times < max_try:
            response = self.call_critic(
                messages=messages,
                top_p=top_p,
                temperature=temperature,
                max_length=max_length,
            )

            if success_check_fn is None:
                success_check_fn = lambda x: True

            if success_check_fn(response):
                success = True
                break
            else:
                try_times += 1

        return response, success
