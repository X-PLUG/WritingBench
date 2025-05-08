import requests
import time
from typing import Callable


class ClaudeAgent(object):
    def __init__(self,
                 system_prompt: str = None):
        self.system_prompt = system_prompt
        self.api_key = '' # Yor API KEY
        self.url = '' # Your URL path
        self.model = '' # Model name
    
    def call_claude(self,
             messages: str,
             top_p: float = 0.95,
             temperature: float = 1.0,
             max_length: int = 2048):
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        data = {
            "model": f"{self.model}",  
            "messages": messages,  
            "max_tokens": max_length,
            "top_p": top_p,
            "temperature": temperature
        }

        attempt = 0
        max_attempts = 5
        wait_time = 1

        while attempt < max_attempts:
            try:
                response = requests.post(self.url, headers=headers, json=data)

                if response.status_code == 200:
                    return response.json()["choices"][0]["message"]["content"]
                else:
                    print(f"Attempt {attempt+1}: Failed with status {response.status_code}, retrying...")
            
            except requests.exceptions.RequestException as e:
                print(f"Attempt {attempt+1}: Request failed due to network error: {e}, retrying...")

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

        while try_times < max_try:
            response = self.call_claude(
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
