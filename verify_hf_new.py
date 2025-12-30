import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from dotenv import load_dotenv
load_dotenv()

from langchain_huggingface import HuggingFaceEndpointEmbeddings

print("Testing langchain_huggingface Embeddings...")
try:
    emb = HuggingFaceEndpointEmbeddings(
        model="sentence-transformers/all-MiniLM-L6-v2",
        huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
    )
    vec = emb.embed_query("hello world")
    print(f"Success! Vector length: {len(vec)}")
except Exception as e:
    print(f"Error: {e}")
