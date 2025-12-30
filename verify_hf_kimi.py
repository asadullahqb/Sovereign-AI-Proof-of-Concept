import os
import sys
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
print(f"Testing with Token: {HF_TOKEN[:4]}...{HF_TOKEN[-4:]}")

# Try the Router URL structure found in search results
# Option 1: Generic Router
BASE_URL_1 = "https://router.huggingface.co/hf-inference/v1"
# Option 2: Specific Provider (Featherless) if needed, but let's try generic first or standard inference
BASE_URL_2 = "https://api-inference.huggingface.co/models/moonshotai/Kimi-K2-Thinking/v1" 
# Option 3: Router with OpenAI compatibility
BASE_URL_3 = "https://router.huggingface.co/v1"

MODEL = "moonshotai/Kimi-K2-Thinking"

def test_endpoint(url, model_name):
    print(f"\nTesting Endpoint: {url} with model {model_name}")
    try:
        client = OpenAI(
            base_url=url,
            api_key=HF_TOKEN
        )
        resp = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": "Hello, are you Kimi?"}],
            max_tokens=50
        )
        print("Success!")
        print(f"Response: {resp.choices[0].message.content}")
        return True
    except Exception as e:
        print(f"Failed: {e}")
        return False

# 1. Try generic router with specific model tag if needed (e.g. :featherless-ai)
# Search result said: model="moonshotai/Kimi-K2-Thinking:featherless-ai"
if not test_endpoint(BASE_URL_3, "moonshotai/Kimi-K2-Thinking"):
    test_endpoint(BASE_URL_3, "moonshotai/Kimi-K2-Thinking:featherless-ai")

# 2. Try standard inference API (often doesn't support v1/chat/completions directly for all models)
# test_endpoint("https://api-inference.huggingface.co/v1", "moonshotai/Kimi-K2-Thinking")
