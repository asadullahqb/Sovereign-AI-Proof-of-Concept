import os
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

@dataclass
class Settings:
    openai_api_key: Optional[str]
    langchain_api_key: Optional[str]
    langchain_tracing_v2: Optional[str]
    moonshot_api_key: Optional[str]
    huggingfacehub_api_token: Optional[str]
    hf_repo_id: Optional[str]

    def llm_provider(self) -> str:
        # Check specific flags first
        if os.getenv("USE_KIMI_VIA_HF") == "true":
            return "huggingface_kimi"
            
        if self.moonshot_api_key:
            return "moonshot"
        if self.huggingfacehub_api_token:
            return "huggingface"
        if self.openai_api_key:
            return "openai"
        return "none"

    def is_valid(self) -> bool:
        return self.llm_provider() != "none"

    def validate(self) -> None:
        provider = self.llm_provider()
        if provider == "none":
            raise RuntimeError("No LLM provider configured. Set OPENAI_API_KEY or other variables.")

def get_settings() -> Settings:
    return Settings(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        langchain_api_key=os.getenv("LANGCHAIN_API_KEY"),
        langchain_tracing_v2=os.getenv("LANGCHAIN_TRACING_V2"),
        moonshot_api_key=os.getenv("MOONSHOT_API_KEY"),
        huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
        hf_repo_id=os.getenv("HF_REPO_ID", "mistralai/Mistral-7B-Instruct-v0.3"),
    )
