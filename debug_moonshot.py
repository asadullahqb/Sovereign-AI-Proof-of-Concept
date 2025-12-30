import os
import requests
from dotenv import load_dotenv

load_dotenv()
KEY = os.getenv("MOONSHOT_API_KEY")
print(f"Testing Moonshot Key: {KEY[:8]}...")

url = "https://api.moonshot.cn/v1/chat/completions"
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {KEY}"
}
data = {
    "model": "moonshot-v1-8k",
    "messages": [
        {"role": "user", "content": "Hello!"}
    ]
}

try:
    resp = requests.post(url, headers=headers, json=data)
    print(f"Status: {resp.status_code}")
    print(f"Response: {resp.text}")
except Exception as e:
    print(f"Error: {e}")
