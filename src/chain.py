from typing import Any
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from src.config import get_settings
from src.llm_utils import TieredRaceChatModel

def _get_llm():
    s = get_settings()
    # provider = s.llm_provider() # We are moving away from single provider selection
    
    # 1. Primary (Cheap): Kimi K2 Thinking via HF
    # Requires HF Token
    hf_model = None
    if s.huggingfacehub_api_token:
        # Use ChatOpenAI client compatible with HF Router
        hf_model = ChatOpenAI(
            model="moonshotai/Kimi-K2-Thinking",
            api_key=s.huggingfacehub_api_token,
            base_url="https://router.huggingface.co/v1",
            temperature=0
        )
    
    # 2. Secondary (Paid/Fast): OpenAI or Moonshot
    paid_model = None
    if s.openai_api_key:
        paid_model = ChatOpenAI(
            model="gpt-3.5-turbo", # or gpt-4o-mini
            api_key=s.openai_api_key,
            temperature=0
        )
    elif s.moonshot_api_key:
        paid_model = ChatOpenAI(
            model="moonshot-v1-8k",
            api_key=s.moonshot_api_key,
            base_url="https://api.moonshot.cn/v1",
            temperature=0
        )

    # 3. Fallback/Standard logic if keys are missing
    if hf_model and paid_model:
        # The user wants: Try Kimi (HF) first. If > 20ms, try Paid.
        # If Paid fails, wait for Kimi.
        return TieredRaceChatModel(
            primary_model=hf_model,
            secondary_model=paid_model,
            latency_budget=3.0 # 3 seconds
        )
    elif hf_model:
        return hf_model
    elif paid_model:
        return paid_model
    
    # Final fallback if nothing else (shouldn't happen if is_valid() checked)
    # Return dummy or raise
    raise ValueError("No valid API keys found for any provider (HF, OpenAI, Moonshot).")

def build_conversational_chain(retriever_or_vectorstore: Any, language: str = "en") -> Any:
    llm = _get_llm()
    
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
