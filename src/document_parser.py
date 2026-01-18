"""
Document Parser - Handles document loading and chunking
Supports: PDF, DOCX, TXT
"""
import os
import re
from typing import List, Dict
from PyPDF2 import PdfReader
from docx import Document as DocxDocument


class DocumentParser:
    """Parse documents and split into chunks."""
    
    def __init__(self, chunk_size: int = 1024, chunk_overlap: int = 200):
        """
        Initialize document parser.
        
        Args:
            chunk_size: Size of text chunks in characters
            chunk_overlap: Overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        self.supported_extensions = {'.pdf', '.docx', '.txt'}
    
    def load_document(self, file_path: str) -> str:
        """
        Load document content based on file type.
        
        Args:
            file_path: Path to document
            
        Returns:
            Extracted text content
        """
        ext = os.path.splitext(file_path)[1].lower()
        
        if ext == '.pdf':
            return self._load_pdf(file_path)
        elif ext == '.docx':
            return self._load_docx(file_path)
        elif ext == '.txt':
            return self._load_txt(file_path)
        else:
            raise ValueError(f"Unsupported file type: {ext}")
    
    def _load_pdf(self, file_path: str) -> str:
        """Load PDF document."""
        reader = PdfReader(file_path)
        text = []
        for page in reader.pages:
            text.append(page.extract_text())
        return '\n'.join(text)
    
    def _load_docx(self, file_path: str) -> str:
        """Load DOCX document."""
        doc = DocxDocument(file_path)
        return '\n'.join([para.text for para in doc.paragraphs])
    
    def _load_txt(self, file_path: str) -> str:
        """Load TXT document."""
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    def clean_text(self, text: str) -> str:
        """Clean extracted text."""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s.,!?;:()\-\'"]+', '', text)
        return text.strip()
    
    def chunk_text(self, text: str) -> List[str]:   
        """
        Split text into overlapping chunks.
        
        Args:
            text: Text to chunk
            
        Returns:
            List of text chunks
        """
        if len(text) <= self.chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end]
            
            # Try to break at sentence boundary
            if end < len(text):
                last_period = chunk.rfind('.')
                last_newline = chunk.rfind('\n')
                break_point = max(last_period, last_newline)
                
                if break_point > self.chunk_size // 2:
                    chunk = text[start:start + break_point + 1]
                    end = start + break_point + 1
            
            chunks.append(chunk.strip())
            start = end - self.chunk_overlap
        
        return chunks
    
    def process_document(self, file_path: str) -> List[Dict]:
        """
        Complete document processing pipeline.
        
        Args:
            file_path: Path to document
            
        Returns:
            List of chunk dictionaries with metadata
        """
        # Load and clean
        raw_text = self.load_document(file_path)
        clean_text = self.clean_text(raw_text)
        
        # Chunk
        chunks = self.chunk_text(clean_text)
        
        # Add metadata
        doc_name = os.path.basename(file_path)
        
        processed_chunks = []
        for i, chunk_text in enumerate(chunks):
            processed_chunks.append({
                'doc_name': doc_name,
                'chunk_id': f"{doc_name}_chunk_{i}",
                'chunk_text': chunk_text,
                'chunk_index': i,
                'total_chunks': len(chunks)
            })
        
        return processed_chunks
    
    def process_folder(self, folder_path: str) -> Dict[str, List[Dict]]:
        """
        Process all documents in a folder.
        
        Args:
            folder_path: Path to folder containing documents
            
        Returns:
            Dictionary mapping filenames to chunks
        """
        results = {}
        
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            
            if os.path.isfile(file_path):
                ext = os.path.splitext(filename)[1].lower()
                
                if ext in self.supported_extensions:
                    try:
                        chunks = self.process_document(file_path)
                        results[filename] = chunks
                        print(f"✅ Processed {filename}: {len(chunks)} chunks")
                    except Exception as e:
                        print(f"❌ Error processing {filename}: {e}")
        
        return results


