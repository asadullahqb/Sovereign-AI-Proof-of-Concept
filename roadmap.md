# Roadmap: Enterprise RAG & Analytics Platform (The "RM15k" POC)

## 1. Project Vision
**Goal:** Build a deployable, production-grade RAG application that allows users to upload complex PDF documents (e.g., Annual Reports, Technical Manuals) and not only "chat" with them but receive proactive structured insights.

**Differentiation:**
* **Junior Dev:** Builds a simple chatbot.
* **Lead Data Scientist (You):** Builds a platform that combines GenAI with deterministic analytics (Option B flavor), uses Docker for portability, and implements proper error handling and typing.

**Tech Stack:**
* **Frontend:** Streamlit (for rapid prototyping + professional UI).
* **Orchestration:** LangChain.
* **LLM:** OpenAI (GPT-4o) or Azure OpenAI (configurable).
* **Vector DB:** ChromaDB (local/persistent) or FAISS.
* **Containerization:** Docker & Docker Compose.

---

## Phase 1: Environment & Scaffolding
**Objective:** Set up a robust Python environment that mirrors a professional engineering standard (not just a notebook).

* [ ] **Step 1.1: Initialize Repository**
    * Create `requirements.txt` with: `streamlit`, `langchain`, `langchain-openai`, `langchain-community`, `chromadb`, `pypdf`, `python-dotenv`, `tiktoken`, `pandas`, `plotly`.
    * Create a `.env.example` file for API keys.
    * Create a `src/` directory for modular code (separating UI from Logic).

* [ ] **Step 1.2: Base Utility Modules**
    * Create `src/utils.py` for logging configuration and helper functions.
    * Create `src/config.py` to handle environment variables (ensure it fails gracefully if keys are missing).

---

## Phase 2: The "Intelligence" Engine (Backend)
**Objective:** Build the RAG pipeline. [cite_start]This mirrors your work on the "RAG System for Leading Power & Utilities" client.

* [ ] **Step 2.1: Document Ingestion**
    * Create `src/ingestion.py`.
    * Implement a function to load PDFs using `PyPDFLoader`.
    * **Crucial:** Implement "Smart Chunking" (RecursiveCharacterTextSplitter) with overlap to preserve context.

* [ ] **Step 2.2: Vector Store Setup**
    * Create `src/vectorstore.py`.
    * Implement a function `get_vectorstore(chunks)` that:
        * Generates embeddings (use `OpenAIEmbeddings` for quality).
        * Stores them in ChromaDB (persist locally so we don't re-embed on every reload).

* [ ] **Step 2.3: Retrieval Chain**
    * Create `src/chain.py`.
    * Build a Conversational Retrieval Chain that maintains chat history (memory).
    * *Refinement:* Add a system prompt that enforces a "Senior Consultant" persona (professional, concise, citation-heavy).

---

## Phase 3: The "Option B" Analytics Layer (The Polish)
[cite_start]**Objective:** Show off your "Applied Analytics" and "Pattern Recognition" skills [cite: 31, 35] by adding structured data extraction, not just chat.

* [ ] **Step 3.1: The "Executive Summary" Agent**
    * In `src/analytics.py`, create a function that runs *immediately* upon file upload.
    * It should ask the LLM to extract specific metadata into a JSON format:
        * **Document Sentiment:** (Positive/Neutral/Negative).
        * **Key Entities:** (Companies, Dates, Dollar Amounts).
        * **Complexity Score:** (1-10 based on technical density).

* [ ] **Step 3.2: Visualization Generator**
    * Use `Plotly` to render a simple chart based on the extracted data (e.g., if the doc contains financial tables, try to extract a "Revenue vs Year" trend). *Even if simple, this visual element proves you can handle multi-modal outputs.*

---

## Phase 4: Frontend & UX (Streamlit)
**Objective:** Create a clean, "Lead-level" interface.

* [ ] **Step 4.1: Layout Design**
    * **Sidebar:** File Uploader, API Key input (for security), and "Model Settings" (Temperature slider).
    * **Main Panel:**
        * **Top Section:** "Document Intelligence" Dashboard (The Option B analytics from Phase 3).
        * **Bottom Section:** Chat Interface (Standard chat bubbles).

* [ ] **Step 4.2: Session State Management**
    * Ensure chat history and vector store persist between button clicks.
    * Add a "Clear Conversation" button.

---

## Phase 5: Production Engineering
[cite_start]**Objective:** Prove you know DevOps/Containerization[cite: 64, 67].

* [ ] **Step 5.1: Dockerization**
    * Create a `Dockerfile`.
    * Use a slim python base image (`python:3.10-slim`).
    * Optimize caching (copy requirements first, then install, then copy code).

* [ ] **Step 5.2: Deployment Configuration**
    * Create a `docker-compose.yml` (optional, but good for local testing).
    * Add a `README.md` with specific instructions on how to run it:
        1.  `docker build -t rockgap-poc .`
        2.  `docker run -p 8501:8501 rockgap-poc`

---

## Phase 6: Final Polish & "Why Hire Me"
**Objective:** Subliminal messaging.

* [ ] **Step 6.1: The "About" Modal**
    * Add a small info icon in the app header.
    * When clicked, it displays: *"Designed by Asadullah Qamar Bhatti | Lead Data Scientist | Stack: LangChain, OpenAI, Docker, Streamlit"*.
    * Link to your LinkedIn/GitHub.

* [ ] **Step 6.2: Pre-loaded Questions**
    * After processing a PDF, generate 3 "Suggested Questions" based on the content to guide the user (showing off agentic behavior).