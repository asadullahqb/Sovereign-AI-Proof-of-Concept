import os
import sys
import unittest
from unittest.mock import MagicMock, patch

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class TestBackendPipeline(unittest.TestCase):
    
    def setUp(self):
        # Mock environment variables to avoid real API calls failing config validation
        self.env_patcher = patch.dict(os.environ, {"OPENAI_API_KEY": "sk-fake-key"})
        self.env_patcher.start()

    def tearDown(self):
        self.env_patcher.stop()

    def test_ingestion_and_chunking(self):
        """Test if PDF loading and chunking logic works (with mocked loader)."""
        from src.ingestion import smart_chunk
        
        # Mock PyPDFLoader to avoid needing a real PDF file
        with patch('src.ingestion.PyPDFLoader') as MockLoader:
            mock_instance = MockLoader.return_value
            # Create a mock document object
            mock_doc = MagicMock()
            mock_doc.page_content = "This is a test document content. " * 50
            mock_instance.load.return_value = [mock_doc]
            
            # Since load_pdfs takes paths, we can verify it calls loader
            from src.ingestion import load_pdfs
            texts = load_pdfs(["dummy.pdf"])
            
            self.assertEqual(len(texts), 1)
            self.assertTrue("test document" in texts[0])
            
            # Test chunking
            chunks = smart_chunk(texts, chunk_size=100, chunk_overlap=10)
            self.assertTrue(len(chunks) > 1)
            self.assertTrue(isinstance(chunks[0], str))

    @patch('src.vectorstore.Chroma')
    @patch('src.vectorstore.OpenAIEmbeddings')
    def test_vectorstore_creation(self, MockEmbeddings, MockChroma):
        """Test vectorstore creation with mocked Chroma and Embeddings."""
        from src.vectorstore import get_vectorstore
        
        chunks = ["chunk1", "chunk2"]
        vs = get_vectorstore(chunks)
        
        # Verify Embeddings were initialized
        MockEmbeddings.assert_called()
        # Verify Chroma.from_texts was called
        MockChroma.from_texts.assert_called()

    def test_chain_creation_robust(self):
        """Test conversational chain builder with robust patching."""
        # Use simple import patching
        import src.chain
        with patch.object(src.chain, 'ConversationalRetrievalChain') as MockChain:
            with patch.object(src.chain, 'ChatOpenAI') as MockChat:
                from src.chain import build_conversational_chain
                mock_vs = MagicMock()
                chain = build_conversational_chain(mock_vs)
                MockChat.assert_called()
                MockChain.from_llm.assert_called()

    @patch('src.analytics.ChatOpenAI')
    def test_analytics_summary(self, MockChat):
        """Test analytics summary generation."""
        from src.analytics import executive_summary
        
        # Mock the LLM response to return valid JSON
        mock_llm_instance = MockChat.return_value
        mock_llm_instance.invoke.return_value.content = '{"DocumentSentiment": "Positive", "KeyEntities": ["TestCorp"], "ComplexityScore": 5, "RevenueByYear": {"2023": 100}}'
        
        summary = executive_summary(["some text"])
        
        self.assertIsInstance(summary, dict)
        self.assertEqual(summary["DocumentSentiment"], "Positive")
        self.assertEqual(summary["ComplexityScore"], 5)

if __name__ == '__main__':
    unittest.main()
