from typing import Any, Dict, List
import json
from langchain_openai import ChatOpenAI
from langchain_community.llms import HuggingFaceEndpoint
from src.config import get_settings
import plotly.express as px
from src.llm_utils import TieredRaceChatModel
import streamlit as st
import sys

def _get_llm():
    s = get_settings()
    
    # 1. Primary (Cheap): Kimi K2 Thinking via HF
    hf_model = None
    if s.huggingfacehub_api_token:
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
            model="gpt-3.5-turbo",
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
        return TieredRaceChatModel(
            primary_model=hf_model,
            secondary_model=paid_model,
            latency_budget=3.0 # 3 seconds
        )
    elif hf_model:
        return hf_model
    elif paid_model:
        return paid_model
    
    raise ValueError("No valid API keys found for any provider (HF, OpenAI, Moonshot).")

def executive_summary(texts: List[str]) -> Dict[str, Any]:
    s = get_settings()
    if not s.is_valid():
        return {
            "DocumentSentiment": "Neutral (Settings Invalid)",
            "KeyEntities": {"Companies": [], "Dates": [], "Amounts": []},
            "ComplexityScore": 5,
            "RevenueByYear": {}
        }
    
    try:
        llm = _get_llm()
        prompt = (
            "Analyze the following document content and return a valid JSON object with keys: "
            "DocumentSentiment (Positive/Neutral/Negative), "
            "KeyEntities (Companies, Dates, Amounts), "
            "ComplexityScore (1-10), "
            "RevenueByYear (object mapping year to number if available). "
            "Ensure the response is strictly valid JSON without any markdown formatting or explanations. "
        )
        content = "\n\n".join(texts[:20])
        resp = llm.invoke(prompt + "\n\n" + content)
        
        content_str = resp.content if hasattr(resp, 'content') else str(resp)
        if "```json" in content_str:
            content_str = content_str.split("```json")[1].split("```")[0]
        elif "```" in content_str:
            content_str = content_str.split("```")[1].split("```")[0]
        return json.loads(content_str.strip())
        
    except Exception as e:
        print(f"Executive Summary Generation Failed: {e}")
        # Return fallback data on error
        return {
            "DocumentSentiment": f"Error during generation: {str(e)[:100]}",
            "KeyEntities": {"Companies": [], "Dates": [], "Amounts": []},
            "ComplexityScore": 0,
            "RevenueByYear": {}
        }

def generate_visualization(summary: Dict[str, Any]) -> Any:
    rev = summary.get("RevenueByYear") or {}
    if isinstance(rev, dict) and len(rev) > 0:
        years = list(rev.keys())
        values = [rev[y] for y in years]
        fig = px.line(x=years, y=values, markers=True, title="Revenue vs Year")
        fig.update_layout(xaxis_title="Year", yaxis_title="Revenue")
        return fig
    ents = summary.get("KeyEntities") or {}
    companies = ents.get("Companies") or []
    dates = ents.get("Dates") or []
    amounts = ents.get("Amounts") or []
    labels = ["Companies", "Dates", "Amounts"]
    counts = [len(companies), len(dates), len(amounts)]
    fig = px.bar(x=labels, y=counts, title="Extracted Entities Count")
    fig.update_layout(xaxis_title="Entity Type", yaxis_title="Count")
    return fig
