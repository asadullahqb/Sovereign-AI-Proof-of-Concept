from typing import Any
try:
    from langchain.chains import ConversationalRetrievalChain
except ImportError as e:
    import langchain
    # This will be caught by UI and displayed
    raise ImportError(f"Could not import langchain.chains. langchain path: {langchain.__path__ if hasattr(langchain, '__path__') else 'unknown'}, version: {langchain.__version__ if hasattr(langchain, '__version__') else 'unknown'}. Original error: {e}")

from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate
from src.llm_utils import get_llm_model

def build_conversational_chain(retriever_or_vectorstore: Any, language: str = "en") -> Any:
    llm = get_llm_model()
    
    if hasattr(retriever_or_vectorstore, "as_retriever"):
        retriever = retriever_or_vectorstore.as_retriever(search_kwargs={"k": 4})
    else:
        retriever = retriever_or_vectorstore
        
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key='answer')

    # Define prompts
    if language == "ms":
        template = """Gunakan maklumat berikut untuk menjawab soalan di penghujung. Jika anda tidak tahu jawapannya, katakan sahaja anda tidak tahu, jangan cuba reka jawapan.
Jawab dalam Bahasa Melayu rasmi (Bahasa Melayu Malaysia). 
PENTING: Jangan gunakan Bahasa Indonesia.
- Gunakan 'boleh' BUKAN 'bisa'.
- Gunakan 'kerana' BUKAN 'karena'.
- Gunakan 'awak' atau 'anda' untuk 'you'.
- Gunakan 'ini' untuk 'this'.

{context}

Soalan: {question}
Jawapan Berguna:"""
    else:
        template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

{context}

Question: {question}
Helpful Answer:"""

    QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"], template=template)

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm, 
        retriever=retriever, 
        memory=memory,
        combine_docs_chain_kwargs={"prompt": QA_CHAIN_PROMPT}
    )
    return chain
