import os
import requests
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from dotenv import load_dotenv
load_dotenv()

HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
MODEL_ID = "gpt2"
API_URL = f"https://router.huggingface.co/hf-inference/models/{MODEL_ID}"
headers = {"Authorization": f"Bearer {HF_TOKEN}"}

print(f"Testing HF API for {MODEL_ID}...")
response = requests.post(API_URL, headers=headers, json={"inputs": ["hello world"], "options": {"wait_for_model": True}})
print(f"Status Code: {response.status_code}")
print(f"Response: {response.text}")
