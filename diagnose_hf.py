import os
import requests
from dotenv import load_dotenv

load_dotenv()

HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
print(f"Token: {HF_TOKEN[:4]}...{HF_TOKEN[-4:]}")

def test_url(url, payload):
    print(f"Testing URL: {url}")
    try:
        response = requests.post(
            url, 
            headers={"Authorization": f"Bearer {HF_TOKEN}"}, 
            json=payload
        )
        print(f"Status: {response.status_code}")
        print(f"Response: {response.text[:200]}")
    except Exception as e:
        print(f"Error: {e}")

payload = {"inputs": "hello world", "options": {"wait_for_model": True}}

# 1. Old API
test_url("https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2", payload)

# 2. Router API
test_url("https://router.huggingface.co/hf-inference/models/sentence-transformers/all-MiniLM-L6-v2", payload)

# 3. Router API (alternate)
test_url("https://router.huggingface.co/models/sentence-transformers/all-MiniLM-L6-v2", payload)

# 4. Check Token
print("\nChecking Token permissions...")
try:
    whoami = requests.get("https://huggingface.co/api/whoami-v2", headers={"Authorization": f"Bearer {HF_TOKEN}"})
    print(f"Whoami Status: {whoami.status_code}")
    print(f"Whoami: {whoami.text}")
except Exception as e:
    print(f"Whoami Error: {e}")
