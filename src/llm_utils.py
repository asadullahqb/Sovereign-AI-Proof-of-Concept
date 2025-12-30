from typing import Any, List, Optional, Tuple
import time
import concurrent.futures
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage
from langchain_core.language_models import BaseChatModel
from langchain_core.outputs import ChatResult, ChatGeneration
from src.config import get_settings
import streamlit as st
import os

@st.cache_resource
def load_local_tiny_model():
    """
    Loads a tiny local model using HuggingFacePipeline.
    Cached to prevent reloading on every interaction.
    """
    try:
        from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
        import torch

        model_id = "Qwen/Qwen2.5-0.5B-Instruct"
        print(f"Loading local model: {model_id}...")
        
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            trust_remote_code=True
        )
        
        pipe = pipeline(
            "text-generation", 
            model=model, 
            tokenizer=tokenizer, 
            max_new_tokens=512,
            temperature=0.1,
            do_sample=True
        )
        
        llm = HuggingFacePipeline(pipeline=pipe)
        # ChatHuggingFace expects a specific LLM type or wrapping
        chat_model = ChatHuggingFace(llm=llm)
        # Monkey patch model_name for logging - handle Pydantic model vs attribute
        try:
            chat_model.model_name = model_id
        except:
            pass # Ignore if it fails, it's just for logging
            
        print("Local model loaded successfully.")
        return chat_model
    except Exception as e:
        print(f"Failed to load local model: {e}")
        # Try simplified loading if ChatHuggingFace fails (e.g. tokenizer issues)
        try:
             # Fallback to just wrapping the pipeline directly if ChatHuggingFace wrapper complains
             # But we need BaseChatModel interface. 
             # Let's verify the error: "Expected llm to be one of ... received HuggingFacePipeline"
             # This error usually means ChatHuggingFace in this version doesn't support the community pipeline.
             # We installed langchain-huggingface, so we should use its classes.
             pass
        except:
             pass
        return None

class MultiTierRaceStrategy(BaseChatModel):
    """
    A Chat Model that races multiple models in a tiered strategy.
    
    Structure:
    - tiers: List of (model, latency_budget_seconds)
    - The last tier has infinite budget (waits until completion or failure).
    
    Strategy:
    1. Start Tier 1.
    2. Wait for Tier 1's budget.
    3. If Tier 1 finishes, return.
    4. If timeout, Start Tier 2 (keep Tier 1 running).
    5. Wait for Tier 2's budget.
    6. ... and so on.
    7. Return whichever finishes first.
    """
    
    tiers: List[Tuple[BaseChatModel, float]]
    # Each element is (Model, Budget before starting next tier)
    # The last tier's budget is ignored (effectively infinite).

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResult:
        
        # If only one tier, just run it
        if len(self.tiers) == 1:
            res = self.tiers[0][0].generate([messages], stop=stop, **kwargs)
            return ChatResult(generations=res.generations[0], llm_output=res.llm_output)

        futures = []
        processed_futures = set()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(self.tiers)) as executor:
            # Launch Tier 1
            current_tier_idx = 0
            
            while current_tier_idx < len(self.tiers):
                model, budget = self.tiers[current_tier_idx]
                
                print(f"Launching Tier {current_tier_idx + 1}: {model.__class__.__name__} ({getattr(model, 'model_name', 'unknown')})...")
                
                f = executor.submit(self._safe_invoke, model, messages, stop, **kwargs)
                futures.append(f)
                
                # Determine how long to wait for THIS tier before launching the next
                is_last = (current_tier_idx == len(self.tiers) - 1)
                wait_time = None if is_last else budget
                
                try:
                    # Wait for ANY of the currently running futures to complete
                    done, _ = concurrent.futures.wait(
                        futures, 
                        timeout=wait_time, 
                        return_when=concurrent.futures.FIRST_COMPLETED
                    )
                    
                    # Check ONLY the new results
                    for d in done:
                        if d in processed_futures:
                            continue
                            
                        processed_futures.add(d)
                        try:
                            res = d.result()
                            if res:
                                print(f"Winner: Tier associated with {getattr(model, 'model_name', 'unknown')}")
                                return res
                        except Exception as e:
                            print(f"Tier failed: {e}")
                            # Continue to next tier or wait for others
                    
                    if not is_last and not any(f.done() and not f.exception() for f in futures):
                         print(f"Tier {current_tier_idx + 1} exceeded {budget}s budget or failed. Proceeding to next tier...")

                except Exception as e:
                    print(f"Error in race loop: {e}")
                
                current_tier_idx += 1
            
            # If all launched and we are here, we wait for the remaining ones
            print("All tiers launched. Waiting for first successful completion...")
            
            # We loop until everything is done or we get a result
            while len(processed_futures) < len(futures):
                done, _ = concurrent.futures.wait(
                    futures, 
                    return_when=concurrent.futures.FIRST_COMPLETED
                )
                
                # Filter for unprocessed
                new_done = [f for f in done if f not in processed_futures]
                if not new_done:
                    # Should not happen if logic is correct, but break to avoid infinite loop
                    break
                    
                for f in new_done:
                    processed_futures.add(f)
                    try:
                        return f.result()
                    except Exception as e:
                        print(f"Tier failed: {e}")
            
            raise RuntimeError("All models failed.")

    def _safe_invoke(self, model, messages, stop, **kwargs):
        try:
            # Call generate on the inner model to get LLMResult
            res = model.generate([messages], stop=stop, **kwargs)
            # We need to return ChatResult to satisfy _generate signature
            return ChatResult(generations=res.generations[0], llm_output=res.llm_output)
        except Exception as e:
            # Log error but don't crash thread immediately, let the waiter handle it
            raise e

    @property
    def _llm_type(self) -> str:
        return "multi-tier-race-chat-model"

def get_llm_model():
    """
    Centralized LLM factory.
    Returns a MultiTierRaceStrategy model.
    
    Order:
    1. TRM Variant (Tiny/Open Source) - e.g. SmolLM2
    2. Hugging Face (Kimi K2)
    3. Paid (Moonshot/OpenAI)
    """
    s = get_settings()
    tiers = []
    
    # Tier 1: TRM Variant (Tiny/Open Source)
    # Using Local Qwen 0.5B (Tiny) to avoid HF Router limits
    local_model = load_local_tiny_model()
    if local_model:
        # Give it a reasonable budget for CPU inference (10s)
        # It takes ~5s to generate a short response on CPU
        tiers.append((local_model, 10.0))
    elif s.huggingfacehub_api_token:
        # Fallback to HF Router if local fails (e.g. memory issues)
        trm_variant = ChatOpenAI(
            model="microsoft/Phi-3-mini-4k-instruct", 
            openai_api_key=s.huggingfacehub_api_token,
            openai_api_base="https://router.huggingface.co/v1",
            temperature=0,
            max_retries=1
        )
        tiers.append((trm_variant, 1.5)) 
    
    # Tier 2: Hugging Face Primary (Reliable Open Source)
    # Switching to Mistral 7B as it is consistently available on free tier
    if s.huggingfacehub_api_token:
        hf_model = ChatOpenAI(
            model="mistralai/Mistral-7B-Instruct-v0.3",
            openai_api_key=s.huggingfacehub_api_token,
            openai_api_base="https://router.huggingface.co/v1",
            temperature=0,
            max_retries=1
        )
        tiers.append((hf_model, 3.0))
        
    # Tier 3: Paid / Reliable Fallback
    paid_model = None
    if s.moonshot_api_key:
        paid_model = ChatOpenAI(
            model="moonshot-v1-8k",
            openai_api_key=s.moonshot_api_key,
            openai_api_base="https://api.moonshot.cn/v1",
            temperature=0
        )
    elif s.openai_api_key:
        paid_model = ChatOpenAI(
            model="gpt-3.5-turbo",
            openai_api_key=s.openai_api_key,
            temperature=0
        )
        
    if paid_model:
        tiers.append((paid_model, 0)) # 0 budget means irrelevant for last tier
    
    if not tiers:
        raise ValueError("No valid API keys found for any provider (HF, Moonshot, OpenAI).")
        
    return MultiTierRaceStrategy(tiers=tiers)
