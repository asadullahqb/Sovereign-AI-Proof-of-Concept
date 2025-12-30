import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from langchain_community.embeddings.fastembed import FastEmbedEmbeddings

print("Testing FastEmbedEmbeddings...")
try:
    emb = FastEmbedEmbeddings()
    vec = emb.embed_query("hello world")
    print(f"Success! Vector length: {len(vec)}")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
