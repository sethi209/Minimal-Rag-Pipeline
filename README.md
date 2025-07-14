# Minimal Retrieval-Augmented Generation (RAG) Pipeline

This repository contains my submission for the **Minimal Retrieval-Augmented Generation (RAG) Pipeline** technical challenge.

---

## Overview

This repository implements a minimal yet robust **Retrieval-Augmented Generation (RAG) pipeline** using Python and modern LLM tooling. The system:
- Exposes a FastAPI endpoint for question answering
- Retrieves context from local documents using embeddings and vector search
- Generates answers via an LLM (Llama-3.1-8b-Instant via Groq API)
- Provides a simple Streamlit UI for interactive querying
- Logs all interactions for traceability in LangSmith
- Returns structured JSON responses with:
    - Answer text
    - Retrieved context chunks (summaries, metadata, similarity scores)
    - Metadata on embeddings and prompts used

---

## Architecture & Design Decisions

### Retrieval & Embeddings

- **Embedding model**: BAAI/bge-base-en-v1.5 via Hugging Face
- **Vector store**: FAISS
- **Document splitting**:
    - Tool: RecursiveCharacterTextSplitter
    - Chunk size: 1000 characters
    - Overlap: 200 characters
- **Document loader**: PyPDFDirectoryLoader

### Rationale:

- BGE embeddings have excellent performance for semantic retrieval at moderate size.
- Chunking ensures context fits within token limits and improves retrieval granularity.
- FAISS offers fast, local vector search without external dependencies.

### LLM Choice

- **Model:** Groq LLaMA-3.1-8B-Instant
- **Provider:** Groq API via `langchain_groq`

**Reasoning**:
- Good balance of quality and speed
- Easy API integration
- Smaller footprint for a minimal proof-of-concept

### Prompt Engineering

Prompt used:

```
Answer the questions based on the provided context only.
Please provide the most accurate response based on the question.

<context>
{context}
<context>

Question:{input}
```

- Explicit instruction to only use the provided context.
- Prevents hallucination.
- Simple yet effective for minimal RAG demonstrations.

---

## Documents & Preprocessing

- **Data Source:** Local PDFs stored in `research_papers/`

- **Loading:** Using LangChain's `PyPDFDirectoryLoader`

- **Chunking:** Documents split into overlapping chunks with `RecursiveCharacterTextSplitter` (chunk size: 1000, overlap: 200) before embedding

--- 

## API Usage

### Endpoint

```bash
POST /query
```

**Request body**:

```json
{
  "query": "Your question here"
}
```

**Response Body:**

```json
{
  "query": "...",
  "answer": "...",
  "embedding_model": "...",
  "prompt_used": "...",
  "context_documents": [
    {
      "score": float,
      "metadata": {...},
      "summary": "..."
    }
  ]
}
```

### Error Handling

- **400** → Empty query  
- **404** → No relevant documents found  
- **502** → LLM returned empty response  
- **500** → Unexpected server errors  

_All errors are returned as JSON with meaningful messages._

---

## Observability & LangSmith Tracing

The pipeline is fully instrumented for LangSmith tracing:

- Tracks all calls to the retrieval chain  
- Stores LLM interactions, prompts, and responses  
- Provides lineage of retrieved documents and answers  

**Env variables used:**

```ini
LANGCHAIN_API_KEY=...
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=...
```

Additionally, structured JSON logs are produced locally:

```json
{
  "timestamp": "2025-07-14 12:34:56",
  "level": "INFO",
  "message": {
    "event": "retrieval_result",
    "score": 0.89,
    "metadata": { ... }
  }
}
```

---

## Streamlit UI

A minimal Streamlit UI is provided for local testing and demonstrations.

### Features:

- Text box to input questions  
- Displays:  
  - LLM answer  
  - Retrieved documents’:  
    - Similarity scores  
    - Metadata  
    - Summaries  

### Running the UI

Once containers are running:  

- On your browser: http://localhost:8501

Or run locally:

```bash
streamlit run app_ui.py
```

---

## Security Considerations

- All API keys loaded from environment variables:  
  - `GROQ_API_KEY`  
  - `HF_TOKEN`  
  - `LANGCHAIN_API_KEY`  

- `.env` excluded from version control via `.gitignore` and `.dockerignore`.

- Future production hardening recommendations:  
  - Rate limiting  
  - Auth on endpoints  
  - Secrets management (e.g. Vault, AWS Secrets Manager)  
  - API monitoring and anomaly detection

---

## Testing

### Test Coverage

**Integration tests:**

- Successful API query  
- Handling empty queries  
- Simulated empty retrieval results  

**Retrieval chain tests:**

- RAG chain returns a valid answer field  

**Vector store tests:**

- Retrieval returns expected number of documents  

- Run tests via Docker Compose:

```bash
docker-compose run tests
```

### Future Testing Plans

- Test retrieval precision/recall on labeled queries
- Load testing for API concurrency
- Monitoring token usage and cost estimation

---

## Containerization

### Local Setup

1. Clone repository.

2. Create a `.env` file with:

```ini
GROQ_API_KEY=...
HF_TOKEN=...
LANGCHAIN_API_KEY=...
LANGCHAIN_PROJECT=...
```

3. Build and run services:

```bash
docker-compose up --build
```

- API → http://localhost:8000
- UI → http://localhost:8501

### Docker Compose Services

- app → FastAPI backend
- ui → Streamlit frontend
- tests → Pytest runner

---

## Trade-offs & Simplifications

- Chose local PDFs instead of larger datasets for simplicity.  
- Used only one embedding model for consistency and speed.  
- Limited LLM calls to a single model to minimize external API costs.  
- Summaries for retrieved docs are simple content snippets rather than generated summaries.  
- No advanced retrieval strategies like RRF or hybrid search due to time constraints.

---

## AI Tools Disclosure

I used AI tools during this challenge as follows:

- **GPT-4 / ChatGPT:** Helped me refine Python snippets, write test cases faster, and debug minor issues.   
- **My own prior templates:** I maintained templates for RAG patterns, chunking, and embeddings from earlier GenAI experiments, which formed the backbone of this project. I adapted and tailored these for the challenge.

_All AI-generated code was reviewed and adjusted to ensure correctness and alignment with the challenge requirements._

---

# Future Work

Given more time, I’d expand this pipeline to:

- Integrate hybrid retrieval (dense + sparse)  
- Add caching of embeddings and LLM outputs  
- Include automated evaluation pipelines (BLEU, ROUGE, retrieval metrics)  
- Implement frontend improvements (e.g. feedback capture)  
- Deploy with CI/CD and infrastructure-as-code

---