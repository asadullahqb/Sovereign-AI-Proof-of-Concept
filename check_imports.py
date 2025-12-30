
try:
    from langchain_community.llms import HuggingFaceEndpoint
    print("HuggingFaceEndpoint: OK")
except ImportError as e:
    print(f"HuggingFaceEndpoint: Failed - {e}")

try:
    from langchain_community.vectorstores import Chroma
    print("Chroma: OK")
except ImportError as e:
    print(f"Chroma: Failed - {e}")

try:
    from langchain.chains import ConversationalRetrievalChain
    print("ConversationalRetrievalChain: OK")
except ImportError as e:
    print(f"ConversationalRetrievalChain: Failed - {e}")

try:
    from langchain_core.prompts import PromptTemplate
    print("PromptTemplate: OK")
except ImportError as e:
    print(f"PromptTemplate: Failed - {e}")

try:
    from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
    print("HuggingFaceInferenceAPIEmbeddings: OK")
except ImportError as e:
    print(f"HuggingFaceInferenceAPIEmbeddings: Failed - {e}")
