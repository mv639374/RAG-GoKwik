# RAG Document Q&A System

This project is a Retrieval-Augmented Generation (RAG) app that answers questions strictly based on a local corpus of documents (PDF/DOCX/TXT). It retrieves relevant chunks, reranks them, and then uses an LLM to generate grounded answers with citations. The app ships with Streamlit UI for interactive use.

---

### 1. How to Run the App

1. **Clone and enter the project**

   ```bash
   git clone <your-repo-url>
   cd gokwik
   ```
2. **Create and activate virtualenv**

   ```bash
   python3 -m venv venv
   source venv/bin/activate        # on Windows: venv\Scripts\activate
   ```
3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```
4. **Set environment variables**

   Create a `.env` file in the project root:

   ```txt
   GEMINI_API_KEY=your_gemini_api_key_here
   ```
5. **Add documents**

   Put your `.pdf`, `.docx`, or `.txt` files into the `data/` folder:

   ```bash
   mkdir -p data
   # copy your docs into data/
   ```
6. **Run the Streamlit UI**

   ```bash
   streamlit run app.py
   ```

---

### 2. Tools and Models Used

- **Document parsing**
  - PyPDF2 (PDF)
  - python-docx (DOCX)
- **Embeddings**
  - Google Gemini embeddings (`models/embedding-001`) for primary 768-d vectors
  - Local `all-MiniLM-L6-v2` for secondary embeddings and reranking support
- **Vector search**
  - Qdrant (local, file-based, cosine similarity)
- **Keyword search**
  - BM25 (rank-bm25)
- **Reranking**
  - Cross-encoder: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- **LLM**
  - `gemini-2.0-flash` for answer generation
- **UI**
  - Streamlit for a simple web interface

---

### 3. Code Structure (src/ + app.py)

The core logic is intentionally split into five focused modules under `src/` plus a thin UI layer.

#### `src/document_parser.py`

Responsible for loading raw documents and turning them into overlapping text chunks.

**Main class:**

- `DocumentParser`
  - `__init__(chunk_size: int = 1024, chunk_overlap: int = 200)`
    - Configures how big each chunk is and how much overlap exists between adjacent chunks (helps with context continuity).
  - `load_documents(data_folder: str) -> List[ParsedDocument]`
    - Walks the `data/` folder, loads `.pdf`, `.docx`, `.txt`, and returns a list of parsed documents with their raw text.
  - `chunk_document(doc: ParsedDocument) -> List[Chunk]`
    - Splits a single document’s text into fixed-size overlapping chunks.
  - `process_all(data_folder: str) -> List[Chunk]`
    - Convenience method: load all documents and return a flat list of chunks ready for embedding and indexing.

The parser hides all format-specific details so the rest of the pipeline deals purely with text chunks.

---

#### `src/embeddings.py`

Encapsulates both cloud and local embedding generation and exposes a uniform interface.

**Main class:**

- `EmbeddingGenerator`
  - `__init__(primary_model: str = "gemini")`
    - Initializes Gemini embeddings and the local `all-MiniLM-L6-v2` model. Stores dimensions for each.
  - `embed_texts(texts: List[str], use_primary: bool = True) -> np.ndarray`
    - Batches input texts and returns embedding vectors. Uses Gemini by default, or local model if requested.
  - `embed_query(query: str) -> np.ndarray`
    - Shortcut for embedding a single query string.
  - `local_dimension: int`
    - Attribute describing the dimension of the local MiniLM embeddings.
  - `primary_model: str`
    - Attribute used by the pipeline to know which embedding dimension to pass into Qdrant.

This abstraction keeps all embedding details in one place, making it easy to switch models later. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/87718418/3fd78510-8104-4ecd-b1f5-67fea5c4ccee/embeddings.py)

---

#### `src/vector_store.py` / `src/retriever.py`

Depending on the refactor, you’ll either see separate `vector_store.py` + `bm25_retriever.py` + `hybrid_retriever.py` + `reranker.py`, or a combined `retriever.py` that wires them together. The behavior is the same: vector search, BM25, fusion, and reranking. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/87718418/d95c862c-8ee4-4b74-bc2e-6b89d0b1bf7e/vector_store.py)

**If using combined `retriever.py`:**

**Main classes:**

- `VectorStore`
  - `__init__(embedding_dim: int)`
    - Sets up a local Qdrant collection (`./qdrant_data`) with the given dimension.
  - `index(chunks: List[Chunk], embeddings: np.ndarray)`
    - Bulk-inserts chunk embeddings into Qdrant with metadata (doc name, chunk id, text).
  - `search(query_embedding: np.ndarray, top_k: int = 20) -> List[ScoredChunk]`
    - Vector similarity search over stored embeddings.
- `BM25Retriever`
  - `__init__()`
    - Builds or loads a BM25 index over all chunks from disk (e.g., `bm25_index.pkl`).
  - `index(chunks: List[Chunk])`
    - Create a BM25 index from chunk texts.
  - `search(query: str, top_k: int = 20) -> List[ScoredChunk]`
    - Keyword-based retrieval using BM25 scores.
- `HybridRetriever`
  - `__init__(embedder, vector_store, bm25_retriever)`
    - Accepts the embedding generator and both retrieval backends.
  - `hybrid_search(query: str, vector_k: int, bm25_k: int, fusion_k: int) -> List[ScoredChunk]`
    - Runs vector search and BM25 search, then merges results using Reciprocal Rank Fusion (RRF). Ensures diverse but relevant candidates.
- `Reranker`
  - `__init__()`
    - Loads the cross-encoder model (`ms-marco-MiniLM-L-6-v2`).
  - `rerank(query: str, candidates: List[ScoredChunk], top_k: int = 10) -> List[ScoredChunk]`
    - Applies the cross-encoder to re-score candidates and returns top reranked chunks.

These classes collectively handle retrieval, fusion, and reranking. They are deliberately separated so you can swap out any layer without touching the others. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/87718418/4e791328-9274-458f-bc9b-42edb53e35d2/bm25_retriever.py)

---

#### `src/llm_agent.py`

Responsible for taking top-ranked chunks and producing a grounded, cited answer.

**Main class:**

- `LLMAgent`
  - `__init__(model_name: str = "gemini-2.0-flash", temperature: float = 0.3, no_answer_threshold: float = 0.5)`
    - Configures the LLM, creativity, and a threshold below which the agent prefers to say “I don’t know” instead of hallucinating.
  - `build_prompt(query: str, chunks: List[Chunk]) -> str`
    - Composes a prompt that includes the user question plus a compact context window built from the retrieved chunks. Also hints the model to use inline citations.
  - `generate_answer(query: str, chunks: List[Chunk]) -> Dict`
    - Calls Gemini with the constructed prompt and returns a structured result: answer text, raw model output, and timing metadata.
  - `extract_citations(answer: str, context_chunks: List[Chunk]) -> Tuple[str, List[Source]]`
    - Parses inline markers (like ` [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/87718418/c53bd529-5cb4-49d6-9dcc-1a3b069553f2/GKAIIntern_Project.pdf)`, ` [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/87718418/4e791328-9274-458f-bc9b-42edb53e35d2/bm25_retriever.py)`), maps them back to the original chunks, and returns:
      - Cleaned answer
      - A list of sources including `doc_name`, `citation_id`, `excerpt`, `full_text`.
  - `assess_confidence(answer: str, chunks: List[Chunk]) -> str`
    - Heuristic to tag each answer as `high`, `medium`, or `low` confidence based on coverage and retrieval scores.

This module enforces the “no hallucination” rule: if the context does not support the answer, it prefers a safe, honest response. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/87718418/4f6e3981-e0cf-423f-a20d-10120c979d8f/llm_agent.py)

---

#### `src/rag_pipeline.py` / `src/pipeline.py`

This is the orchestrator that ties everything together into a single RAG pipeline.

**Main class:**

- `RAGPipeline`
  - `__init__(data_folder: str = "data", chunk_size: int = 1024, chunk_overlap: int = 200, use_reranking: bool = True)`
    - Instantiates `DocumentParser`, `EmbeddingGenerator`, `VectorStore`, `BM25Retriever`, `HybridRetriever`, `Reranker`, and `LLMAgent`. Decides embedding dimension based on primary model (Gemini vs local) and configures chunking parameters. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/87718418/90586abe-47a5-4c07-ab1c-3884c18a7dea/rag_pipeline.py)
  - `ingest_documents(force_reingest: bool = False)`
    - Parses all docs from `data/`, chunks them, computes embeddings, indexes them into Qdrant and BM25. Can skip ingestion if indexes already exist, unless `force_reingest` is true.
  - `retrieve(query: str, retrieval_k: int) -> List[ScoredChunk]`
    - Runs the full retrieval pipeline: hybrid search (vector + BM25) and optional reranking to select the most relevant chunks.
  - `query(query: str, top_k: int = 5, retrieval_k: int = 30, verbose: bool = False) -> Dict`
    - End-to-end call:
      1. Retrieve and rerank chunks.
      2. Call the LLM agent to generate an answer.
      3. Extract citations and compute timing breakdown (retrieval, reranking, LLM, total).
      4. Return a structured response with `answer`, `sources`, `confidence`, and `timing`.

This class is what both the CLI and Streamlit app call. It gives you a single entry point for the whole system. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/87718418/90586abe-47a5-4c07-ab1c-3884c18a7dea/rag_pipeline.py)

---

#### `app.py` (Streamlit UI)

A thin UI layer on top of `RAGPipeline`.

**Key parts:**

- App layout:
  - Sets page title, icon, layout, and some basic CSS for styling the answer box, metrics, and sidebar.
- `load_pipeline()`
  - Cached loader that initializes `RAGPipeline` and runs `ingest_documents()` once at startup, then reuses the same pipeline instance across user interactions.
- `display_answer(result: Dict)`
  - Renders the answer, confidence badge, timing metrics, and sources list. Shows:
    - Answer in a highlighted box
    - Confidence (high/medium/low) with emoji
    - Retrieval, reranking, LLM, and total time
    - Expanders for each source with excerpt and full chunk text
- Main UI flow:
  - Sidebar:
    - Advanced settings (results count, retrieval pool size)
    - Startup timing breakdown (how long each component took to initialize)
    - System info and example queries
  - Center:
    - Text input for the question
    - Clear button to reset query and answer
    - Search button to trigger the pipeline
    - Answer and citations rendered below the divider

The UI is intentionally simple: it exposes the core behavior of the RAG system without hiding the internal reasoning (sources and timing are visible) but keeps the interface clean and easy to use.

---

### 4. What Can Be Improved Over Time

This version is intentionally minimal and focused, but there are several natural extensions:

1. **Better comparison handling**

   - Add a dedicated prompt template for “compare X vs Y” questions.
   - Possibly split comparison queries into sub-questions and merge the answers.
2. **Richer evaluation and monitoring**

   - Integrate a small evaluation dashboard (e.g., Streamlit tab) to visualize metrics (MRR, latency) over time.
   - Log real user queries and outcomes (with anonymization) for continuous improvement.
3. **Semantic chunking**

   - Replace fixed-size chunking with sentence- or paragraph-aware segmentation using a simple NLP pass, to avoid cutting important context mid-thought.
4. **Caching and cost optimization**

   - Cache embeddings and LLM responses for repeated queries.
   - Add optional local-only mode (no external LLM) for offline experimentation.
5. **Multi-user / multi-collection support**

   - Support multiple “workspaces” or “collections”, each with its own document set.
   - Add simple auth if deployed in a shared environment.
6. **File uploads from the UI**

   - Allow users to upload new documents directly from the Streamlit app and trigger incremental ingestion without restarting the service.

Even without these extras, the current setup already behaves like a small, production-ready RAG system suitable for internal knowledge bases and technical documentation search.
