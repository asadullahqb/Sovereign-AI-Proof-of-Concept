from typing import Any, List, Optional
import time
import concurrent.futures
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage
from langchain_core.language_models import BaseChatModel
from langchain_core.outputs import ChatResult, ChatGeneration

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
