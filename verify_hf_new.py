import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from dotenv import load_dotenv
load_dotenv()

from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings

print("Testing langchain_community Embeddings...")
try:
    emb = HuggingFaceInferenceAPIEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        api_key=os.getenv("HUGGINGFACEHUB_API_TOKEN")
    )
    vec = emb.embed_query("hello world")
    print(f"Success! Vector length: {len(vec)}")
except Exception as e:
    print(f"Error: {e}")
