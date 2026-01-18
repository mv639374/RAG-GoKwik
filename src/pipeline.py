"""
Complete RAG Pipeline
Orchestrates: Document Processing → Indexing → Retrieval → LLM Generation
"""
import os
import time
from typing import Dict, List
from document_parser import DocumentParser
from embeddings import EmbeddingGenerator
from retriever import VectorStore, BM25Retriever, HybridRetriever, Reranker
from llm_agent import LLMAgent


class RAGPipeline:
    """End-to-end RAG system."""
    
    def __init__(
        self,
        data_folder: str = "data",
        chunk_size: int = 1024,
        chunk_overlap: int = 200,
        use_reranking: bool = True
    ):
        """
        Initialize RAG pipeline.
        
        Args:
            data_folder: Folder containing documents
            chunk_size: Chunk size for documents
            chunk_overlap: Overlap between chunks
            use_reranking: Whether to use cross-encoder reranking
        """
        print("="*80)
        print("INITIALIZING RAG PIPELINE")
        print("="*80)
        
        self.data_folder = data_folder
        self.use_reranking = use_reranking
        
        # Initialize components
        print("\n🔧 Loading components...")
        
        self.parser = DocumentParser(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.embedder = EmbeddingGenerator()
        
        # Get embedding dimension based on primary model
        if self.embedder.primary_model == 'gemini':
            embedding_dim = 768
        else:
            embedding_dim = self.embedder.local_dimension

        
        self.vector_store = VectorStore(embedding_dim=embedding_dim)  # Fixed: use correct dimension
        self.bm25_retriever = BM25Retriever()
        self.hybrid_retriever = HybridRetriever(
            embedder=self.embedder,
            vector_store=self.vector_store,
            bm25_retriever=self.bm25_retriever
        )
        
        if use_reranking:
            self.reranker = Reranker()
        else:
            self.reranker = None
        
        self.llm = LLMAgent()
        
        print("\n✅ All components loaded!")
        print("="*80)
    
    def ingest_documents(self, force_reingest: bool = False):
        """
        Ingest documents from data folder.
        
        Args:
            force_reingest: Force reprocessing even if already indexed
        """
        print("\n📂 Ingesting documents...")
        
        # Check if already indexed
        if not force_reingest and os.path.exists("./bm25_index.pkl"):
            print("✅ Documents already indexed. Use force_reingest=True to reindex.")
            return
        
        # Process documents
        doc_chunks_map = self.parser.process_folder(self.data_folder)
        
        if not doc_chunks_map:
            print("❌ No documents found in data folder!")
            return
        
        # Flatten chunks
        all_chunks = []
        for chunks in doc_chunks_map.values():
            all_chunks.extend(chunks)
        
        print(f"\n📊 Total chunks: {len(all_chunks)}")
        
        # Generate embeddings
        print("\n🔄 Generating embeddings...")
        texts = [chunk['chunk_text'] for chunk in all_chunks]
        embeddings = self.embedder.embed_batch(texts)
        
        # Index in vector store
        print("🔄 Indexing in vector store...")
        self.vector_store.add_documents(all_chunks, embeddings)
        
        # Build BM25 index
        print("🔄 Building BM25 index...")
        self.bm25_retriever.build_index(all_chunks)
        
        print(f"\n✅ Ingestion complete!")
        print(f"   Documents: {len(doc_chunks_map)}")
        print(f"   Total chunks: {len(all_chunks)}")
    
    def query(
        self,
        query: str,
        top_k: int = 10,
        retrieval_k: int = 30,
        verbose: bool = True
    ) -> Dict:
        """
        Execute RAG query.
        
        Args:
            query: User question
            top_k: Final number of chunks for LLM
            retrieval_k: Number of chunks to retrieve before reranking
            verbose: Print progress
            
        Returns:
            Answer with sources and metadata
        """
        timing = {}
        start_total = time.time()
        
        if verbose:
            print(f"\n{'='*80}")
            print(f"QUERY: {query}")
            print('='*80)
        
        # Stage 1: Hybrid Retrieval
        if verbose:
            print("\n🔍 Stage 1: Hybrid Retrieval")
        
        start = time.time()
        retrieved_chunks = self.hybrid_retriever.search(
            query=query,
            top_k=retrieval_k
        )
        timing['retrieval'] = time.time() - start
        
        if verbose:
            print(f"   Retrieved: {len(retrieved_chunks)} chunks ({timing['retrieval']:.3f}s)")
        
        if not retrieved_chunks:
            return {
                'answer': "No relevant documents found.",
                'sources': [],
                'confidence': 'none',
                'timing': timing
            }
        
        # Stage 2: Reranking
        if self.use_reranking and self.reranker:
            if verbose:
                print(f"\n🎯 Stage 2: Reranking")
            
            start = time.time()
            final_chunks = self.reranker.rerank(
                query=query,
                documents=retrieved_chunks,
                top_k=top_k
            )
            timing['reranking'] = time.time() - start
            
            if verbose:
                print(f"   Reranked to top-{len(final_chunks)} ({timing['reranking']:.3f}s)")
        else:
            final_chunks = retrieved_chunks[:top_k]
            timing['reranking'] = 0.0
        
        # Stage 3: LLM Generation
        if verbose:
            print(f"\n🤖 Stage 3: Answer Generation")
        
        start = time.time()
        llm_result = self.llm.answer_query(
            query=query,
            context_chunks=final_chunks
        )
        timing['llm'] = time.time() - start
        timing['total'] = time.time() - start_total
        
        if verbose:
            print(f"   Generated answer ({timing['llm']:.3f}s)")
            print(f"\n⏱️  Total: {timing['total']:.3f}s")
        
        result = {
            'answer': llm_result['answer'],
            'sources': llm_result['sources'],
            'confidence': llm_result['confidence'],
            'timing': timing,
            'query': query
        }
        
        return result

