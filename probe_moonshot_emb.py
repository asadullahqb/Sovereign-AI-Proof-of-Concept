import os
import requests
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from dotenv import load_dotenv
load_dotenv()

API_KEY = os.getenv("MOONSHOT_API_KEY")
BASE_URL = "https://api.moonshot.cn/v1/embeddings"
headers = {"Authorization": f"Bearer {API_KEY}"}

print(f"Testing Moonshot Embeddings endpoint: {BASE_URL}")
# Try standard OpenAI embedding format
payload = {
    "input": "hello world",
    "model": "moonshot-v1-8k" # or maybe they have a specific embedding model?
}

try:
    resp = requests.post(BASE_URL, headers=headers, json=payload)
    print(f"Status: {resp.status_code}")
    print(f"Response: {resp.text}")
except Exception as e:
    print(f"Error: {e}")
