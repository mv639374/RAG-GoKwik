import os
import re
from typing import List, Dict, Optional, Tuple
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()


class LLMAgent:
    """
    LLM-powered answer generation with citation support.
    Uses Gemini for generating grounded answers from retrieved context.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gemini-2.0-flash",  # Use one from the list
        temperature: float = 0.3,
        no_answer_threshold: float = 0.5
    ):
        """
        Initialize LLM agent.
        
        Args:
            api_key: Gemini API key (if None, loads from env)
            model_name: Gemini model name (without 'models/' prefix)
            temperature: Response creativity (0-1, lower = more focused)
            no_answer_threshold: Min rerank score to attempt answering
        """
        self.temperature = temperature
        self.no_answer_threshold = no_answer_threshold
        
        # Initialize Gemini
        api_key = api_key or os.getenv('GEMINI_API_KEY')
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment")
        
        try:
            genai.configure(api_key=api_key)
            
            # Try different models in order of preference
            models_to_try = [
                model_name,
                "gemini-2.0-flash",
                "gemini-flash-latest",
                "gemini-2.5-flash",
                "gemini-pro-latest"
            ]
            
            model_loaded = False
            last_error = None
            
            for try_model in models_to_try:
                try:
                    # Remove 'models/' prefix if present
                    clean_model_name = try_model.replace('models/', '')
                    
                    print(f"Trying model: {clean_model_name}...")
                    
                    # Initialize model
                    self.model = genai.GenerativeModel(clean_model_name)
                    
                    # Store the model name
                    self.model_name = clean_model_name
                    model_loaded = True
                    
                    print(f"✅ Successfully loaded: {clean_model_name}")
                    break
                    
                except Exception as e:
                    last_error = e
                    continue
            
            if not model_loaded:
                raise Exception(f"Could not load any Gemini model. Last error: {last_error}")
            
            print(f"✅ LLM Agent initialized")
            print(f"   Model: {self.model_name}")
            print(f"   Temperature: {temperature}")
            print(f"   No-answer threshold: {no_answer_threshold}")
            
        except Exception as e:
            print(f"❌ Failed to initialize Gemini: {e}")
            raise e
    
    # ========================================================================
    # STEP 8.1: GEMINI INTEGRATION WITH PROMPT TEMPLATE
    # ========================================================================
    
    def create_prompt(
        self,
        query: str,
        context_chunks: List[Dict],
        include_citations: bool = True
    ) -> str:
        """
        Create prompt with context chunks and instructions.
        
        Args:
            query: User query
            context_chunks: List of retrieved document chunks
            include_citations: Whether to request citations
            
        Returns:
            Formatted prompt string
        """
        # Format context with source IDs
        context_parts = []
        for i, chunk in enumerate(context_chunks, 1):
            source_id = f"[{i}]"
            doc_name = chunk.get('doc_name', 'Unknown')
            text = chunk.get('chunk_text', '')
            
            context_parts.append(f"{source_id} Document: {doc_name}\n{text}\n")
        
        context_text = "\n".join(context_parts)
        
        # Build prompt with strict instructions
        citation_instruction = ""
        if include_citations:
            citation_instruction = """
When referencing information, cite the source using [1], [2], etc. based on the context provided.
"""
        
        prompt = f"""You are a helpful AI assistant that answers questions based ONLY on the provided context.

IMPORTANT INSTRUCTIONS:
1. Answer ONLY using information from the context below
2. If the context does not contain enough information to answer the question, say "I don't have enough information in the provided documents to answer this question."
3. Do NOT use any external knowledge or make assumptions
4. Be concise and accurate
5. If multiple documents contain relevant information, synthesize them{citation_instruction}

CONTEXT:
{context_text}

QUESTION: {query}

ANSWER:"""
        
        return prompt
    
    def generate_answer(
        self,
        query: str,
        context_chunks: List[Dict],
        max_tokens: int = 500
    ) -> str:
        """
        Generate answer using Gemini.
        
        Args:
            query: User query
            context_chunks: Retrieved context chunks
            max_tokens: Maximum response length
            
        Returns:
            Generated answer text
        """
        # Create prompt
        prompt = self.create_prompt(query, context_chunks, include_citations=True)
        
        # Generate response
        try:
            response = self.model.generate_content(
                prompt,
                generation_config=genai.GenerationConfig(
                    temperature=self.temperature,
                    max_output_tokens=max_tokens,
                )
            )
            
            # Extract text from response
            if hasattr(response, 'text'):
                answer = response.text.strip()
            elif hasattr(response, 'parts') and response.parts:
                answer = ''.join(part.text for part in response.parts).strip()
            else:
                answer = "I encountered an issue generating a response. Please try rephrasing your question."
            
            return answer
        
        except Exception as e:
            print(f"❌ Error generating answer: {e}")
            return "I encountered an error while generating the answer. Please try again."
    
    # ========================================================================
    # STEP 8.2: CITATION EXTRACTION
    # ========================================================================
    
    def extract_citations(
        self,
        answer: str,
        context_chunks: List[Dict]
    ) -> Tuple[str, List[Dict]]:
        """
        Extract citations from answer and map to source documents.
        
        Args:
            answer: Generated answer with citations like [1], [2]
            context_chunks: List of context chunks
            
        Returns:
            Tuple of (answer, list of cited sources)
        """
        # Find all citation markers [1], [2], etc.
        citation_pattern = r'\[(\d+)\]'
        citations = re.findall(citation_pattern, answer)
        
        # Extract unique cited sources
        cited_sources = []
        seen_indices = set()
        
        for citation in citations:
            idx = int(citation) - 1  # Convert to 0-based index
            
            if 0 <= idx < len(context_chunks) and idx not in seen_indices:
                chunk = context_chunks[idx]
                cited_sources.append({
                    'citation_id': int(citation),
                    'doc_name': chunk.get('doc_name', 'Unknown'),
                    'chunk_id': chunk.get('chunk_id', ''),
                    'excerpt': chunk.get('chunk_text', '')[:200] + '...',
                    'full_text': chunk.get('chunk_text', '')
                })
                seen_indices.add(idx)
        
        # If no citations found, include all sources
        if not cited_sources:
            cited_sources = [
                {
                    'citation_id': i + 1,
                    'doc_name': chunk.get('doc_name', 'Unknown'),
                    'chunk_id': chunk.get('chunk_id', ''),
                    'excerpt': chunk.get('chunk_text', '')[:200] + '...',
                    'full_text': chunk.get('chunk_text', '')
                }
                for i, chunk in enumerate(context_chunks[:3])
            ]
        
        return answer, cited_sources
    
    # ========================================================================
    # STEP 8.3: NO-ANSWER LOGIC
    # ========================================================================
    
    def check_answer_confidence(
        self,
        context_chunks: List[Dict]
    ) -> Tuple[bool, float]:
        """
        Check if we have sufficient confidence to answer.
        
        Args:
            context_chunks: Retrieved context chunks with scores
            
        Returns:
            Tuple of (should_answer, max_score)
        """
        if not context_chunks:
            return False, 0.0
        
        # Get maximum rerank score
        max_score = 0.0
        for chunk in context_chunks:
            score = chunk.get('rerank_score', chunk.get('score', 0.0))
            max_score = max(max_score, score)
        
        # Check against threshold
        should_answer = max_score >= self.no_answer_threshold
        
        return should_answer, max_score
    
    def generate_no_answer_response(
        self,
        query: str,
        max_score: float
    ) -> Dict:
        """
        Generate response when confidence is too low.
        
        Args:
            query: User query
            max_score: Maximum relevance score found
            
        Returns:
            Response dictionary with no-answer message
        """
        return {
            'answer': "I don't have enough relevant information in the provided documents to answer this question confidently.",
            'sources': [],
            'confidence': 'low',
            'max_relevance_score': max_score,
            'query': query
        }
    
    # ========================================================================
    # COMPLETE ANSWER GENERATION PIPELINE
    # ========================================================================
    
    def answer_query(
        self,
        query: str,
        context_chunks: List[Dict],
        max_tokens: int = 500,
        force_answer: bool = False
    ) -> Dict:
        """
        Complete answer generation pipeline with confidence check and citations.
        
        Args:
            query: User query
            context_chunks: Retrieved and reranked context chunks
            max_tokens: Maximum response length
            force_answer: Skip confidence check (for testing)
            
        Returns:
            Dictionary with answer, sources, and metadata
        """
        # Check confidence
        if not force_answer:
            should_answer, max_score = self.check_answer_confidence(context_chunks)
            
            if not should_answer:
                return self.generate_no_answer_response(query, max_score)
        else:
            max_score = context_chunks[0].get('rerank_score', 1.0) if context_chunks else 0.0
        
        # Generate answer
        answer = self.generate_answer(query, context_chunks, max_tokens)
        
        # Extract citations
        answer, sources = self.extract_citations(answer, context_chunks)
        
        # Build response
        return {
            'answer': answer,
            'sources': sources,
            'confidence': 'high' if max_score >= 0.7 else 'medium',
            'max_relevance_score': max_score,
            'query': query,
            'num_context_chunks': len(context_chunks)
        }
    
    def get_model_info(self) -> Dict:
        """Get information about the LLM."""
        return {
            'model_name': self.model_name,
            'temperature': self.temperature,
            'no_answer_threshold': self.no_answer_threshold
        }
