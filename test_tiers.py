from src.llm_utils import get_llm_model
from langchain_core.messages import HumanMessage
import sys

def test_tiers():
    print("Initializing Multi-Tier LLM...")
    try:
        llm = get_llm_model()
        print(f"LLM initialized: {llm}")
        print(f"Number of tiers: {len(llm.tiers)}")
        for i, (model, budget) in enumerate(llm.tiers):
            name = getattr(model, 'model_name', 'unknown')
            print(f"Tier {i+1}: {name} (Budget: {budget}s)")
    except Exception as e:
        print(f"Failed to initialize LLM: {e}")
        return

    print("\nInvoking LLM to trigger race strategy...")
    messages = [HumanMessage(content="Hello")]
    try:
        res = llm.invoke(messages)
        print("Response received!")
    except Exception as e:
        print(f"Invocation failed as expected (due to keys): {e}")

if __name__ == "__main__":
    test_tiers()
