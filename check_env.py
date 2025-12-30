
from dotenv import load_dotenv
import os
load_dotenv()
key = os.getenv("OPENAI_API_KEY")
print(f"Key loaded: '{key}'")
print(f"Key length: {len(key) if key else 0}")
