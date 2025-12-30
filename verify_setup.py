import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from dotenv import load_dotenv
load_dotenv()

print("Testing imports...")
try:
    from src.config import get_settings
    s = get_settings()
    print(f"Provider: {s.llm_provider()}")
    
    from src.vectorstore import _get_embeddings
    
    print("Testing Embeddings call...")
    emb = _get_embeddings()
    vec = emb.embed_query("hello world")
    print(f"Embeddings success. Vector length: {len(vec)}")
    
    print("Testing LLM call...")
    if s.llm_provider() == "moonshot":
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(
            model="moonshot-v1-8k",
            api_key=s.moonshot_api_key,
            base_url="https://api.moonshot.cn/v1",
            temperature=0
        )
        resp = llm.invoke("Hello, say 'test passed'.")
        print(f"LLM Response: {resp.content}")
    
    print("Verification complete.")

except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
