import streamlit as st
from src.llm_utils import load_local_tiny_model
from langchain_core.messages import HumanMessage
import time

def test_local_only():
    print("Testing Local Model Only...")
    start_load = time.time()
    model = load_local_tiny_model()
    print(f"Model loaded in {time.time() - start_load:.2f}s")
    
    if not model:
        print("Failed to load model.")
        return

    print("Invoking model...")
    start_gen = time.time()
    try:
        # ChatHuggingFace requires messages
        messages = [HumanMessage(content="Hello, how are you?")]
        res = model.invoke(messages)
        print(f"Generation took {time.time() - start_gen:.2f}s")
        print("Result:", res.content)
    except Exception as e:
        print(f"Generation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_local_only()
