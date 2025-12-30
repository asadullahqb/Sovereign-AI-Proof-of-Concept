import os
import sys
# Add project root to sys.path so we can import from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Explicitly load dotenv here to ensure env vars are available before UI renders
from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import concurrent.futures
from typing import List
from src.config import get_settings

# Translation Dictionary
TRANSLATIONS = {
    "en": {
        "config": "Configuration",
        "smart_model": "Smart Model Selection",
        "smart_model_help": (
            "**Strategy:**\n"
            "1. Tries **Kimi K2 Thinking** (Hugging Face) first.\n"
            "2. If it exceeds **3 seconds**, it attempts **OpenAI** or **Moonshot AI** (if keys provided) for faster response.\n"
            "3. If paid models fail (e.g. no credits), it falls back to waiting for Kimi K2.\n\n"
            "*Prioritizes Speed & Cost Efficiency.*"
        ),
        "api_keys": "API Keys (Required)",
        "hf_token": "Hugging Face Token",
        "hf_help": "Required for Kimi K2 Thinking & Embeddings",
        "openai_key": "OpenAI API Key",
        "moonshot_key": "Moonshot API Key",
        "fast_fallback": "Optional: For fast fallback",
        "upload_pdf": "Upload PDF",
        "process": "Process",
        "doc_intel": "Document Intelligence",
        "chat": "Chat",
        "set_keys": "Set API keys in the sidebar.",
        "your_question": "Your question",
        "ask": "Ask",
        "init_pipeline": "Initializing processing pipeline...",
        "load_ai": "Loading AI modules...",
        "save_local": "Saving {n} file(s) locally...",
        "extract_text": "Extracting text from PDF documents...",
        "split_text": "Splitting text into semantic chunks...",
        "gen_embed": "Generating vector embeddings (This may take a moment)...",
        "build_chain": "Building conversational AI chain...",
        "gen_summary": "Generating executive summary and analytics...",
        "gen_summary_bg": "Started executive summary generation in background...",
        "finalizing_summary": "Finalizing executive summary...",
        "complete": "Processing Complete!",
        "error_proc": "Error during processing: {e}",
        "error_gen": "Error during generation: {e}",
        "you": "You",
        "assistant": "Assistant",
        "upload_prompt": "Upload a PDF and press Process."
    },
    "ms": {
        "config": "Konfigurasi",
        "smart_model": "Pemilihan Model Pintar",
        "smart_model_help": (
            "**Strategi:**\n"
            "1. Mencuba **Kimi K2 Thinking** (Hugging Face) dahulu.\n"
            "2. Jika melebihi **3 saat**, ia mencuba **OpenAI** atau **Moonshot AI** (jika kunci diberikan) untuk respons lebih pantas.\n"
            "3. Jika model berbayar gagal, ia kembali menunggu Kimi K2.\n\n"
            "*Mengutamakan Kelajuan & Kecekapan Kos.*"
        ),
        "api_keys": "Kunci API (Diperlukan)",
        "hf_token": "Token Hugging Face",
        "hf_help": "Diperlukan untuk Kimi K2 Thinking & Embeddings",
        "openai_key": "Kunci API OpenAI",
        "moonshot_key": "Kunci API Moonshot",
        "fast_fallback": "Pilihan: Untuk sandaran pantas",
        "upload_pdf": "Muat Naik PDF",
        "process": "Proses",
        "doc_intel": "Kecerdasan Dokumen",
        "chat": "Sembang",
        "set_keys": "Tetapkan kunci API di bar sisi.",
        "your_question": "Soalan anda",
        "ask": "Tanya",
        "init_pipeline": "Memulakan saluran pemprosesan...",
        "load_ai": "Memuatkan modul AI...",
        "save_local": "Menyimpan {n} fail secara tempatan...",
        "extract_text": "Mengekstrak teks daripada dokumen PDF...",
        "split_text": "Memecahkan teks kepada bahagian semantik...",
        "gen_embed": "Menjana benaman vektor (Ini mungkin mengambil masa)...",
        "build_chain": "Membina rantaian AI perbualan...",
        "gen_summary": "Menjana ringkasan eksekutif dan analitik...",
        "complete": "Pemprosesan Selesai!",
        "error_proc": "Ralat semasa pemprosesan: {e}",
        "error_gen": "Ralat semasa penjanaan: {e}",
        "you": "Anda",
        "assistant": "Pembantu",
        "upload_prompt": "Muat naik PDF dan tekan Proses."
    }
}

def setup_session_state() -> None:
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None
    if "chain" not in st.session_state:
        st.session_state.chain = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "summary" not in st.session_state:
        st.session_state.summary = None
    if "language" not in st.session_state:
        st.session_state.language = "en"

def build_layout() -> None:
    st.set_page_config(page_title="Enterprise RAG & Analytics POC", layout="wide")
    setup_session_state()
    
    # Language Toggle in Sidebar (Top)
    with st.sidebar:
        # Create a toggle for language
        is_ms = st.toggle("Bahasa Melayu", value=(st.session_state.language == "ms"))
        new_lang = "ms" if is_ms else "en"
        
        # If language changed, clear chain to force rebuild with new prompt
        if new_lang != st.session_state.language:
            st.session_state.language = new_lang
            st.session_state.chain = None # Force rebuild
            st.rerun() # Rerun to apply translation immediately
            
    t = TRANSLATIONS[st.session_state.language]

    with st.sidebar:
        st.header(t["config"])
        st.subheader(
            t["smart_model"], 
            help=t["smart_model_help"]
        )
        
        with st.expander(t["api_keys"], expanded=True):
            # 1. HF Token (Required for Kimi K2 Thinking)
            default_hf = os.getenv("HUGGINGFACEHUB_API_TOKEN", "")
            hf_k = st.text_input(t["hf_token"], type="password", value=default_hf, help=t["hf_help"])
            if hf_k: os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_k
            
            # 2. OpenAI Key (Optional - for Speed Fallback)
            default_openai = os.getenv("OPENAI_API_KEY", "")
            openai_k = st.text_input(t["openai_key"], type="password", value=default_openai, help=t["fast_fallback"])
            if openai_k: os.environ["OPENAI_API_KEY"] = openai_k
            
            # 3. Moonshot Key (Optional - for Speed Fallback)
            default_moonshot = os.getenv("MOONSHOT_API_KEY", "")
            moonshot_k = st.text_input(t["moonshot_key"], type="password", value=default_moonshot, help=t["fast_fallback"])
            if moonshot_k: os.environ["MOONSHOT_API_KEY"] = moonshot_k

        uploaded = st.file_uploader(t["upload_pdf"], type=["pdf"], accept_multiple_files=True)
        process = st.button(t["process"])

    s = get_settings()
    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader(t["doc_intel"])
        if process and uploaded:
            status_container = st.empty()
            with st.spinner(t["init_pipeline"]):
                try:
                    # Lazy import heavy modules only when needed
                    status_container.info(t["load_ai"])
                    from src.ingestion import load_pdfs, smart_chunk
                    from src.vectorstore import get_vectorstore
                    from src.chain import build_conversational_chain
                    from src.analytics import executive_summary
                    
                    paths: List[str] = []
                    os.makedirs("data/raw", exist_ok=True)
                    
                    status_container.info(t["save_local"].format(n=len(uploaded)))
                    for uf in uploaded:
                        p = os.path.join("data/raw", uf.name)
                        with open(p, "wb") as f:
                            f.write(uf.getbuffer())
                        paths.append(p)
                    
                    status_container.info(t["extract_text"])
                    texts = load_pdfs(paths)
                    
                    # Start Summary Generation in Parallel
                    status_container.info(t["gen_summary_bg"])
                    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
                    future_summary = executor.submit(executive_summary, texts)
                    
                    status_container.info(t["split_text"])
                    chunks = smart_chunk(texts)
                    
                    status_container.info(t["gen_embed"])
                    st.session_state.vectorstore = get_vectorstore(chunks)
                    
                    status_container.info(t["build_chain"])
                    st.session_state.chain = build_conversational_chain(st.session_state.vectorstore, language=st.session_state.language)
                    
                    status_container.info(t["finalizing_summary"])
                    st.session_state.summary = future_summary.result()
                    executor.shutdown()
                    
                    status_container.success(t["complete"])
                except Exception as e:
                    status_container.empty()
                    import traceback
                    # Helpful diagnostics for deployment
                    err_msg = f"{e}\n{traceback.format_exc()}"
                    try:
                        import langchain
                        err_msg += f"\nLangChain Version: {langchain.__version__}"
                    except:
                        err_msg += "\nLangChain not found"
                    
                    st.error(t["error_proc"].format(e=err_msg))
        if st.session_state.summary:
            from src.analytics import generate_visualization
            fig = generate_visualization(st.session_state.summary)
            st.plotly_chart(fig, use_container_width=True)
            st.json(st.session_state.summary)
        else:
            st.write(t["upload_prompt"])
    with col2:
        st.subheader(t["chat"])
        if not s.is_valid():
            st.info(t["set_keys"])
        prompt = st.text_input(t["your_question"])
        ask = st.button(t["ask"])
        
        # Rebuild chain if language changed but chain exists from previous session
        if st.session_state.chain is None and st.session_state.vectorstore is not None:
             from src.chain import build_conversational_chain
             st.session_state.chain = build_conversational_chain(st.session_state.vectorstore, language=st.session_state.language)
             
        if ask and st.session_state.chain and prompt:
            with st.spinner("Thinking..." if st.session_state.language == "en" else "Sedang berfikir..."):
                try:
                    resp = st.session_state.chain.invoke({"question": prompt})
                    answer = resp.get("answer", "")
                    st.session_state.chat_history.append(("user", prompt))
                    st.session_state.chat_history.append(("assistant", answer))
                except Exception as e:
                    st.error(t["error_gen"].format(e=e))
        for role, msg in st.session_state.chat_history:
            if role == "user":
                st.markdown(f"**{t['you']}:** {msg}")
            else:
                st.markdown(f"**{t['assistant']}:** {msg}")

if __name__ == "__main__":
    build_layout()
