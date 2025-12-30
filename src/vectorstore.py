from typing import Any, List
import os
import sys
import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, AzureOpenAIEmbeddings
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from src.config import get_settings

def _get_embeddings():
    s = get_settings()
    
    if s.openai_api_key:
        return OpenAIEmbeddings(api_key=s.openai_api_key)
        
    api_key = s.huggingfacehub_api_token
    if api_key:
        # Use HuggingFaceEndpointEmbeddings as the modern replacement
        # It handles the Inference API (serverless) or dedicated endpoints
        return HuggingFaceEndpointEmbeddings(
            model="sentence-transformers/all-MiniLM-L6-v2",
            task="feature-extraction",
            huggingfacehub_api_token=api_key,
        )
    return None

def get_retriever(chunks: List[str]) -> Any:
    """
    Returns a retriever object. 
    Tries to use dense embeddings (Chroma) first. 
    If that fails (quota/permissions/DLLs), falls back to sparse retrieval (BM25).
    """
    try:
        embeddings = _get_embeddings()
        if not embeddings:
            raise ValueError("No valid embedding configuration found.")

        # Test embeddings with a single query to fail fast
        # (This catches 429 Insufficient Quota or 403 Forbidden immediately)
        embeddings.embed_query("test")
        
        base_dir = os.path.dirname(os.path.abspath(__file__))
        vectorstore_dir = os.path.join(base_dir, '..', 'data', 'chroma')
        os.makedirs(vectorstore_dir, exist_ok=True)
        
        vs = Chroma.from_texts(
            texts=chunks, 
            embedding=embeddings, 
            persist_directory=vectorstore_dir
        )
        return vs.as_retriever(search_kwargs={"k": 4})
        
    except Exception as e:
        print(f"Dense embeddings failed ({e}). Falling back to BM25 (Keyword Search).")
        # Suppressed UI warning per user request
        # if "streamlit" in sys.modules:
        #     st.warning(f"Embeddings failed ({str(e)[:100]}...). Switched to Keyword Search (BM25). RAG will still work but might be less semantic.")
        
        # Fallback to BM25
        # BM25Retriever expects Documents or texts
        return BM25Retriever.from_texts(chunks)

# Backward compatibility alias (though it returns a retriever now, not a vectorstore)
# We should update callers to expect a retriever.
def get_vectorstore(chunks: List[str]) -> Any:
    return get_retriever(chunks)
