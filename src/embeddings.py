import os
from typing import List, Optional, Union
from dotenv import load_dotenv
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import time

# Load environment variables
load_dotenv()


class EmbeddingGenerator:
    """
    Generates embeddings using Gemini (primary) and Sentence Transformers (backup).
    Implements fallback logic for robustness.
    """
    
    def __init__(
        self, 
        gemini_api_key: Optional[str] = None,
        local_model_name: str = "all-MiniLM-L6-v2",
        use_gemini: bool = True
    ):
        """
        Initialize embedding generators.
        
        Args:
            gemini_api_key: Gemini API key (if None, loads from env)
            local_model_name: Sentence Transformers model name
            use_gemini: Whether to use Gemini as primary (True) or local only (False)
        """
        self.use_gemini = use_gemini
        self.gemini_available = False
        self.local_model_name = local_model_name
        
        # Initialize Gemini
        if self.use_gemini:
            try:
                api_key = gemini_api_key or os.getenv('GEMINI_API_KEY')
                if not api_key:
                    print("⚠️  GEMINI_API_KEY not found in environment")
                    self.use_gemini = False
                else:
                    genai.configure(api_key=api_key)
                    self.gemini_model = os.getenv('EMBEDDING_MODEL_GEMINI', 'models/embedding-001')
                    self.gemini_available = True
                    print(f"✅ Gemini embeddings initialized: {self.gemini_model}")
            except Exception as e:
                print(f"⚠️  Failed to initialize Gemini: {e}")
                self.use_gemini = False
        
        # Initialize local model (always as backup)
        try:
            print(f"Loading local model: {self.local_model_name}...")
            self.local_model = SentenceTransformer(self.local_model_name)
            self.local_dimension = self.local_model.get_sentence_embedding_dimension()
            print(f"✅ Local embeddings initialized: {self.local_model_name} (dim: {self.local_dimension})")
        except Exception as e:
            print(f"❌ Failed to load local model: {e}")
            raise RuntimeError("Cannot proceed without at least one embedding model")
        
        # Set primary embedding dimension
        if self.gemini_available:
            # Gemini embedding-001 has 768 dimensions
            self.embedding_dimension = 768
            self.primary_model = "gemini"
        else:
            self.embedding_dimension = self.local_dimension
            self.primary_model = "local"
        
        print(f"\n📊 Primary model: {self.primary_model} (dimension: {self.embedding_dimension})")
    
    def embed_text_gemini(self, text: str) -> Optional[List[float]]:
        """
        Generate embedding using Gemini.
        
        Args:
            text: Input text to embed
            
        Returns:
            List of floats (embedding vector) or None if failed
        """
        if not self.gemini_available:
            return None
        
        try:
            # Add retry logic for rate limits
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    result = genai.embed_content(
                        model=self.gemini_model,
                        content=text,
                        task_type="retrieval_document"
                    )
                    return result['embedding']
                except Exception as e:
                    if "429" in str(e) or "quota" in str(e).lower():
                        if attempt < max_retries - 1:
                            wait_time = (attempt + 1) * 2
                            print(f"⏳ Rate limit hit, waiting {wait_time}s...")
                            time.sleep(wait_time)
                            continue
                    raise e
        except Exception as e:
            print(f"⚠️  Gemini embedding failed: {e}")
            return None
    
    def embed_text_local(self, text: str) -> List[float]:
        """
        Generate embedding using local Sentence Transformers model.
        
        Args:
            text: Input text to embed
            
        Returns:
            List of floats (embedding vector)
        """
        try:
            embedding = self.local_model.encode(text, convert_to_numpy=True)
            return embedding.tolist()
        except Exception as e:
            print(f"❌ Local embedding failed: {e}")
            raise e
    
    def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding with automatic fallback.
        Tries Gemini first (if available), falls back to local model.
        
        Args:
            text: Input text to embed
            
        Returns:
            List of floats (embedding vector)
        """
        # Try Gemini first
        if self.gemini_available:
            embedding = self.embed_text_gemini(text)
            if embedding is not None:
                return embedding
            else:
                print("⚠️  Falling back to local model")
        
        # Fallback to local model
        return self.embed_text_local(text)
    
    def embed_batch(self, texts: List[str], show_progress: bool = True) -> List[List[float]]:
        """
        Generate embeddings for a batch of texts.
        
        Args:
            texts: List of texts to embed
            show_progress: Whether to show progress
            
        Returns:
            List of embedding vectors
        """
        embeddings = []
        total = len(texts)
        
        for idx, text in enumerate(texts):
            if show_progress and (idx + 1) % 10 == 0:
                print(f"Progress: {idx + 1}/{total} embeddings generated")
            
            embedding = self.embed_text(text)
            embeddings.append(embedding)
        
        if show_progress:
            print(f"✅ Completed: {total}/{total} embeddings generated")
        
        return embeddings
    
    def get_embedding_dimension(self) -> int:
        """Return the dimension of embeddings."""
        return self.embedding_dimension
    
    def get_primary_model(self) -> str:
        """Return the name of the primary model being used."""
        return self.primary_model
