"""
Complete Retrieval System
- Vector search (Qdrant)
- BM25 keyword search
- Hybrid fusion (RRF)
- Cross-encoder reranking
"""
import os
import pickle
import numpy as np
from typing import List, Dict, Optional, Union
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

7
class VectorStore:
    """Vector store using Qdrant."""
    
    def __init__(self, collection_name: str = "documents", embedding_dim: int = 768):
        """
        Initialize vector store.
        
        Args:
            collection_name: Name of the collection
            embedding_dim: Dimension of embeddings (768 for Gemini, 384 for MiniLM)
        """
        self.client = QdrantClient(path="./qdrant_data")
        self.collection_name = collection_name
        self.embedding_dim = embedding_dim
        
        # Create collection if not exists
        try:
            collection_info = self.client.get_collection(collection_name)
            print(f"✅ Connected to collection: {collection_name}")
            
            # Verify dimension matches
            if collection_info.config.params.vectors.size != embedding_dim:
                print(f"⚠️  Dimension mismatch! Expected {embedding_dim}, got {collection_info.config.params.vectors.size}")
                print(f"   Recreating collection with correct dimension...")
                self.client.delete_collection(collection_name)
                self._create_collection()
        except:
            self._create_collection()
    
    def _create_collection(self):
        """Create a new collection."""
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=self.embedding_dim,
                distance=Distance.COSINE
            )
        )
        print(f"✅ Created collection: {self.collection_name} (dim: {self.embedding_dim})")
    
    def add_documents(self, chunks: List[Dict], embeddings: Union[List, np.ndarray]):
        """
        Add documents to vector store.
        
        Args:
            chunks: List of document chunks with metadata
            embeddings: List or array of embeddings
        """
        points = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            # Convert embedding to list if needed
            if isinstance(embedding, np.ndarray):
                vector = embedding.tolist()
            elif isinstance(embedding, list):
                vector = embedding
            else:
                raise TypeError(f"Unsupported embedding type: {type(embedding)}")
            
            # Verify dimension
            if len(vector) != self.embedding_dim:
                raise ValueError(f"Embedding dimension {len(vector)} doesn't match collection dimension {self.embedding_dim}")
            
            points.append(PointStruct(
                id=i,
                vector=vector,
                payload=chunk
            ))
        
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
        print(f"✅ Indexed {len(points)} documents in vector store")
    
    def search(self, query_vector: Union[List, np.ndarray], limit: int = 30, score_threshold: float = 0.6) -> List[Dict]:
        """
        Search vector store.
        
        Args:
            query_vector: Query embedding
            limit: Maximum number of results
            score_threshold: Minimum similarity score
            
        Returns:
            List of search results with scores
        """
        # Convert to list if needed
        if isinstance(query_vector, np.ndarray):
            query_vector = query_vector.tolist()
        
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=limit,
            score_threshold=score_threshold
        )
        
        return [
            {
                **result.payload,
                'score': result.score
            }
            for result in results
        ]
    
    def clear(self):
        """Clear all documents."""
        self.client.delete_collection(self.collection_name)
        print(f"✅ Cleared collection: {self.collection_name}")


class BM25Retriever:
    """BM25 keyword search."""
    
    def __init__(self, index_path: str = "./bm25_index.pkl"):
        """Initialize BM25 retriever."""
        self.index_path = index_path
        self.bm25 = None
        self.corpus_chunks = []
        
        if os.path.exists(index_path):
            self.load_index()
            print(f"✅ Loaded BM25 index: {len(self.corpus_chunks)} documents")
    
    def tokenize(self, text: str) -> List[str]:
        """Simple tokenization."""
        return text.lower().split()
    
    def build_index(self, chunks: List[Dict]):
        """Build BM25 index."""
        self.corpus_chunks = chunks
        tokenized_corpus = [self.tokenize(chunk['chunk_text']) for chunk in chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)
        self.save_index()
        print(f"✅ Built BM25 index: {len(chunks)} documents")
    
    def search(self, query: str, top_k: int = 30) -> List[Dict]:
        """Search using BM25."""
        if not self.bm25:
            return []
        
        query_tokens = self.tokenize(query)
        scores = self.bm25.get_scores(query_tokens)
        
        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                result = self.corpus_chunks[idx].copy()
                result['score'] = float(scores[idx])
                results.append(result)
        
        return results
    
    def save_index(self):
        """Save index to disk."""
        with open(self.index_path, 'wb') as f:
            pickle.dump({
                'bm25': self.bm25,
                'corpus_chunks': self.corpus_chunks
            }, f)
    
    def load_index(self):
        """Load index from disk."""
        with open(self.index_path, 'rb') as f:
            data = pickle.load(f)
            self.bm25 = data['bm25']
            self.corpus_chunks = data['corpus_chunks']


class HybridRetriever:
    """Hybrid retrieval with RRF fusion."""
    
    def __init__(self, embedder, vector_store: VectorStore, bm25_retriever: BM25Retriever):
        """Initialize hybrid retriever."""
        self.embedder = embedder
        self.vector_store = vector_store
        self.bm25_retriever = bm25_retriever
    
    def search(
        self,
        query: str,
        top_k: int = 50,
        vector_k: int = 20,
        bm25_k: int = 20,
        vector_threshold: float = 0.6,
        rrf_k: int = 60
    ) -> List[Dict]:
        """
        Hybrid search with RRF fusion.
        
        Pipeline:
        1. Vector search (semantic)
        2. BM25 search (keyword)
        3. RRF fusion
        """
        # Vector search
        query_embedding = self.embedder.embed_text(query)
        vector_results = self.vector_store.search(
            query_vector=query_embedding,
            limit=vector_k,
            score_threshold=vector_threshold
        )
        
        # BM25 search
        bm25_results = self.bm25_retriever.search(query, top_k=bm25_k)
        
        # RRF fusion
        rrf_scores = {}
        chunk_data = {}
        
        # Process vector results
        for rank, result in enumerate(vector_results, 1):
            chunk_id = result['chunk_id']
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + 1.0 / (rrf_k + rank)
            chunk_data[chunk_id] = result
        
        # Process BM25 results
        for rank, result in enumerate(bm25_results, 1):
            chunk_id = result['chunk_id']
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + 1.0 / (rrf_k + rank)
            if chunk_id not in chunk_data:
                chunk_data[chunk_id] = result
        
        # Sort by RRF score
        sorted_chunks = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Build final results
        fused_results = []
        for chunk_id, rrf_score in sorted_chunks[:top_k]:
            result = chunk_data[chunk_id].copy()
            result['rrf_score'] = rrf_score
            fused_results.append(result)
        
        return fused_results


class Reranker:
    """Cross-encoder reranking."""
    
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """Initialize reranker."""
        print(f"Loading reranker: {model_name}...")
        self.model = CrossEncoder(model_name)
        print("✅ Reranker loaded")
    
    def rerank(self, query: str, documents: List[Dict], top_k: int = 10) -> List[Dict]:
        """Rerank documents using cross-encoder."""
        if not documents:
            return []
        
        # Create query-document pairs
        pairs = [[query, doc['chunk_text']] for doc in documents]
        
        # Score pairs
        scores = self.model.predict(pairs, show_progress_bar=False)
        
        # Add scores and sort
        reranked = []
        for doc, score in zip(documents, scores):
            doc_copy = doc.copy()
            doc_copy['rerank_score'] = float(score)
            reranked.append(doc_copy)
        
        reranked.sort(key=lambda x: x['rerank_score'], reverse=True)
        
        return reranked[:top_k]


# Test
if __name__ == "__main__":
    print("Retriever module - all components loaded")
