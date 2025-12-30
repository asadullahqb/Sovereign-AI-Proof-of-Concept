# Enterprise RAG & Analytics Platform (POC)

A production-grade Proof of Concept for an Enterprise RAG (Retrieval-Augmented Generation) system. This application allows users to upload PDF documents, extract structured insights (Executive Summary, Sentiment, Key Entities), and chat with the content using a professional "Senior Consultant" persona.

Built with **Streamlit**, **LangChain**, **OpenAI/Azure OpenAI**, and **ChromaDB**.

## Features

- **Document Ingestion**: Robust PDF loading and "Smart Chunking" to preserve context.
- **Hybrid Vector Store**: Uses ChromaDB for local persistence of embeddings.
- **Dual-Mode AI**: Supports both **OpenAI** (Direct) and **Azure OpenAI** (Enterprise) configurations.
- **Analytics Layer**: Automatically generates an "Executive Summary" and visualizations (Plotly) upon upload.
- **Conversational Memory**: Remembers context across the chat session.
- **Lazy Loading**: Optimized UI startup time.

## Prerequisites

- **Python 3.10+**
- **Docker** (Optional, for containerized deployment)
- API Keys for **OpenAI** OR **Azure OpenAI**

## Installation

1.  **Clone the repository** (if applicable) or navigate to the project root.

2.  **Create a virtual environment**:
    ```bash
    # Windows
    python -m venv .venv
    .\.venv\Scripts\activate

    # Mac/Linux
    python3 -m venv .venv
    source .venv/bin/activate
    ```

3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## Configuration

1.  **Environment Variables**:
    Copy `.env.example` to `.env`:
    ```bash
    cp .env.example .env
    ```

2.  **Edit `.env`** with your credentials.
    *   **Option A: OpenAI (Direct)**
        *   Fill `OPENAI_API_KEY`.
    *   **Option B: Azure OpenAI**
        *   Fill `AZURE_OPENAI_API_KEY`
        *   Fill `AZURE_OPENAI_ENDPOINT`
        *   Fill `AZURE_OPENAI_API_VERSION` (e.g., `2024-02-15-preview`)
        *   Fill `AZURE_OPENAI_DEPLOYMENT` (Chat model name, e.g., `gpt-4o`)
        *   Fill `AZURE_OPENAI_EMBEDDING_DEPLOYMENT` (Embedding model name, e.g., `text-embedding-3-large`)

    *Alternatively, you can enter these keys directly in the application Sidebar.*

## Usage

### Local Run
Start the application using Streamlit:
```bash
streamlit run src/ui.py
```
Access the app at `http://localhost:8501`.

### Docker Run
Build and run using Docker Compose:
```bash
docker build -t ytl-ai-poc .
docker compose up
```
*Note: You may need to map the port or use `docker compose run --service-ports app streamlit run src/ui.py` if the default command is overridden.*

## Project Structure

```
├── data/               # Local data storage (raw PDFs, ChromaDB)
├── src/
│   ├── analytics.py    # Extraction & Visualization logic
│   ├── chain.py        # RAG Retrieval Chain construction
│   ├── config.py       # Environment configuration & validation
│   ├── ingestion.py    # PDF Loading & Chunking
│   ├── ui.py           # Streamlit Frontend
│   ├── utils.py        # Logging & Helpers
│   └── vectorstore.py  # Embedding & Database management
├── tests/              # Validation scripts
├── .env                # Secrets (Gitignored)
├── Dockerfile          # Container config
├── requirements.txt    # Python dependencies
└── roadmap.md          # Project goals
```

## Troubleshooting

- **Import Errors**: Ensure you are running `streamlit run src/ui.py` from the *root* directory, not inside `src/`.
- **Azure Errors**: Verify your `AZURE_OPENAI_DEPLOYMENT` matches exactly what is in your Azure Portal.
- **ChromaDB Issues**: If you face SQLite errors, try deleting the `data/chroma` folder to reset the database.
