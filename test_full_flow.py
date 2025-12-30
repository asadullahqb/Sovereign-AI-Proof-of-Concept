import os
import sys
# Add project root to sys.path so we can import from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from dotenv import load_dotenv
load_dotenv()

# Simulate UI logic
print("Simulating UI logic...")
MOONSHOT_KEY = os.getenv("MOONSHOT_API_KEY")
OPENAI_KEY = os.getenv("OPENAI_API_KEY")

if MOONSHOT_KEY:
    print("Moonshot Key found.")
if OPENAI_KEY:
    print("OpenAI Key found (will be used for embeddings).")

from src.vectorstore import get_vectorstore
from src.chain import build_conversational_chain
from src.analytics import executive_summary

print("Testing Ingestion...")
chunks = ["This is a test document about YTL AI Labs.", "The revenue was $1M in 2024."]
try:
    # This should use OpenAIEmbeddings now
    vs = get_vectorstore(chunks)
    print("Vectorstore created successfully.")
except Exception as e:
    print(f"Vectorstore FAILED: {e}")
    sys.exit(1)

print("Testing Chain...")
try:
    chain = build_conversational_chain(vs)
    # Mock retrieval
    res = chain.invoke({"question": "What is the revenue?"})
    print(f"Chain Answer: {res.get('answer')}")
except Exception as e:
    print(f"Chain FAILED: {e}")
    sys.exit(1)

print("Testing Analytics...")
try:
    summary = executive_summary(chunks)
    print(f"Summary: {summary}")
except Exception as e:
    print(f"Analytics FAILED: {e}")
    sys.exit(1)

print("ALL TESTS PASSED.")
