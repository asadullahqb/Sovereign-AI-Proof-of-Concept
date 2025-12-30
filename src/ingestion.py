from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_pdfs(paths: List[str]) -> List[str]:
    texts: List[str] = []
    for p in paths:
        docs = PyPDFLoader(p).load()
        for d in docs:
            texts.append(d.page_content)
    return texts

def smart_chunk(texts: List[str], chunk_size: int = 1000, chunk_overlap: int = 150) -> List[str]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks: List[str] = []
    for t in texts:
        chunks.extend(splitter.split_text(t))
    return chunks
