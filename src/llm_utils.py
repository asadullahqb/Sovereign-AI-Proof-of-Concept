from typing import Any, List, Optional
import time
import concurrent.futures
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage
from langchain_core.language_models import BaseChatModel
from langchain_core.outputs import ChatResult, ChatGeneration
from src.config import get_settings

class TieredRaceChatModel(BaseChatModel):
    """
    A Chat Model that races a 'Cheap/Primary' model against a 'Fast/Paid' model.
    
    Strategy:
    1. Start the 'Cheap' model (e.g. Kimi K2 via HF).
    2. Wait for `latency_budget` seconds (e.g. 0.02s).
    3. If Cheap model hasn't finished (it likely hasn't), start the 'Paid' model (OpenAI/Moonshot).
    4. Return whichever completes first.
    5. Exception Handling:
       - If Paid model fails (e.g. 429 Insufficient Quota), we silently ignore it and wait for Cheap model.
       - If Cheap model fails, we wait for Paid model.
       - If both fail, raise the last exception.
    """
    
    primary_model: BaseChatModel
    secondary_model: Optional[BaseChatModel] = None
    latency_budget: float = 3.0 # 3 seconds

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResult:
        if not self.secondary_model:
            # Must return ChatResult, not AIMessage
            res = self.primary_model.generate([messages], stop=stop, **kwargs)
            return ChatResult(generations=res.generations[0], llm_output=res.llm_output)

        # We use a ThreadPool to race them
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            # Start Primary
            future_primary = executor.submit(
                self._safe_invoke, self.primary_model, messages, stop, **kwargs
            )
            
            # Wait for latency budget
            try:
                # If primary is miraculously fast
                return future_primary.result(timeout=self.latency_budget)
            except concurrent.futures.TimeoutError:
                pass # Expected, continue to launch secondary
            except Exception as e:
                # Primary failed immediately? Launch secondary
                print(f"Primary failed immediately: {e}")
                # Don't return yet, launch secondary
            
            print(f"Primary exceeded {self.latency_budget}s. Launching Secondary...")
            future_secondary = executor.submit(
                self._safe_invoke, self.secondary_model, messages, stop, **kwargs
            )
            
            # Now wait for whichever finishes first
            done, not_done = concurrent.futures.wait(
                [future_primary, future_secondary], 
                return_when=concurrent.futures.FIRST_COMPLETED
            )
            
            for f in done:
                try:
                    result = f.result()
                    # If we got a valid result, return it
                    if result:
                        print(f"Winner: {'Primary' if f == future_primary else 'Secondary'}")
                        return result
                except Exception as e:
                    print(f"Task failed: {e}")
                    # If this was secondary (paid) failure, we must wait for primary
                    # If this was primary failure, we must wait for secondary
            
            # If we are here, one finished but failed, OR we are waiting for the other
            # Wait for the remaining one
            errors = []
            for f in done:
                try:
                    f.result()
                except Exception as e:
                    errors.append(str(e))

            for f in not_done:
                try:
                    print("Waiting for the slower/remaining model...")
                    result = f.result()
                    if result:
                        return result
                except Exception as e:
                    print(f"Remaining task failed: {e}")
                    errors.append(str(e))
            
            raise RuntimeError(f"All models failed. Errors: {'; '.join(errors)}")

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
        return "tiered-race-chat-model"

def get_llm_model():
    """
    Centralized LLM factory that returns the configured TieredRaceChatModel
    or a single model depending on available keys.
    """
    s = get_settings()
    
    # 1. Primary (Cheap): Kimi K2 Thinking via HF
    # Requires HF Token
    hf_model = None
    if s.huggingfacehub_api_token:
        # Use ChatOpenAI client compatible with HF Router
        # NOTE: We use openai_api_base instead of base_url to ensure compatibility 
        # with older langchain_openai versions and prevent defaulting to OpenAI's real API.
        hf_model = ChatOpenAI(
            model="moonshotai/Kimi-K2-Thinking",
            openai_api_key=s.huggingfacehub_api_token,
            openai_api_base="https://router.huggingface.co/v1",
            temperature=0
        )
    
    # 2. Secondary (Paid/Fast): Moonshot or OpenAI
    # We prefer Moonshot if available, as OpenAI often has quota issues in this env
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
            model="gpt-3.5-turbo", # or gpt-4o-mini
            openai_api_key=s.openai_api_key,
            temperature=0
        )

    # 3. Fallback/Standard logic if keys are missing
    if hf_model and paid_model:
        # The user wants: Try Kimi (HF) first. If > 3s, try Paid.
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
