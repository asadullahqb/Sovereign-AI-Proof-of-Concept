
import os
import sys
from dotenv import load_dotenv

# Load env vars
load_dotenv()

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))

print("Attempting imports...")
try:
    from src.ingestion import load_pdfs, smart_chunk
    from src.vectorstore import get_vectorstore
    from src.chain import build_conversational_chain
    from src.analytics import executive_summary
    print("Imports successful.")
except Exception as e:
    print(f"Import failed: {e}")
    sys.exit(1)

print("Checking API Key...")
from src.config import get_settings
s = get_settings()
print(f"Provider: {s.llm_provider()}")
if s.llm_provider() == "none":
    print("WARNING: No provider found.")
else:
    print("Provider found.")

print("Attempting to initialize LLM...")
try:
    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(api_key=s.openai_api_key)
    print("LLM initialized.")
    print("Invoking LLM...")
    response = llm.invoke("Hello")
    print(f"LLM Response: {response.content}")
except Exception as e:
    print(f"LLM initialization/invocation failed: {e}")
    # Print full traceback
    import traceback
    traceback.print_exc()
