# AI Handbook Generator — Write-up

## What I Built

An AI-powered chat application that accepts PDF uploads, answers contextual questions about them, and generates 20,000+ word structured handbooks — all through a single conversational interface.

The application follows the exact architecture specified in the assignment:

```
PDF Upload → LightRAG (Supabase) → Chat UI (Grok 4.1) → Handbook (20k+ words)
```

## Approach

### Architecture Decisions

**LLM — Grok 4.1 Fast (xAI API):**
Chosen for its 2M token context window and OpenAI-compatible API. Used for both knowledge graph entity extraction (via LightRAG) and handbook section generation. The `grok-4-1-fast-non-reasoning` variant provides the best speed-to-cost ratio at $0.20/1M input tokens.

**RAG — LightRAG with Knowledge Graph:**
Instead of simple vector similarity search, LightRAG builds a knowledge graph from uploaded PDFs — extracting entities, relationships, and semantic connections. This enables richer, more contextual answers than flat chunk retrieval. The graph is stored locally (NetworkX) while vectors and metadata are persisted in Supabase PostgreSQL with pgvector.

**Database — Supabase (PostgreSQL + pgvector):**
Used as the persistent storage backend for LightRAG's vector store, key-value store, and document status tracking. The pgvector extension enables HNSW-indexed vector similarity search. Free tier handles the workload comfortably.

**Handbook Generation — AgentWrite/LongWriter with Parallel Batching:**
Based on the LongWriter research paper's AgentWrite technique. The pipeline:
1. **Plan** — Grok generates a structured 22-section writing plan
2. **Parallel Write** — Sections are written in 4 batches of 5-6, each batch running concurrently. Each batch receives a summary of prior batches for coherence.
3. **Compile** — All sections assembled with table of contents, headings, and references
4. **Export** — Saved as both Markdown and professionally formatted PDF

This parallel approach reduces generation time from ~15 minutes (sequential) to ~2 minutes.

### Technical Stack

| Component | Technology |
|-----------|-----------|
| Frontend | Gradio 6.x |
| LLM | Grok 4.1 (xAI API) |
| RAG | LightRAG (knowledge graph) |
| Database | Supabase (PostgreSQL + pgvector) |
| Embeddings | all-MiniLM-L6-v2 (local) |
| PDF Processing | pdfplumber |

## Features Implemented

### Must-Have
- PDF upload and parsing (multiple files supported)
- Knowledge graph creation via LightRAG
- Chat interface with contextual Q&A
- 20,000+ word handbook generation via chat

### Optional
- Multiple PDF support
- Live progress indicator during indexing and generation
- Export to Markdown and PDF
- Conversation history (in-session)
- Download button for generated handbooks
- Cancel button for indexing
- Persistent knowledge graph (survives restarts)

## Challenges

### 1. Embedding Model Compatibility
Initially used BAAI/bge-m3 for embeddings, but it caused `IndexError` crashes on short entity names with special characters. Switched to all-MiniLM-L6-v2 with a defensive wrapper that sanitizes text (strips newlines, non-printable chars) and falls back to individual encoding when batch encoding fails.

### 2. LightRAG + Supabase Integration
LightRAG's `PGGraphStorage` requires Apache AGE extension, which Supabase doesn't support. Solved by using `NetworkXStorage` for the graph (local file) while keeping PostgreSQL for vectors, key-value, and document status — a hybrid approach that works within Supabase's free tier.

### 3. Knowledge Graph Query Failures
LightRAG's keyword extraction sometimes produces overly generic terms (e.g., "paper" instead of specific entities), causing zero search results. Implemented a multi-layer fallback: hybrid → naive → local → direct vector search with raw Grok call. This ensures every query gets an answer.

### 4. Generation Speed
Sequential AgentWrite (22 sections × 20s each = 7+ minutes) was too slow. Redesigned as parallel batch writing — sections within each batch are generated concurrently, with batch summaries maintaining coherence between batches. Final time: ~2 minutes for 20,000+ words.

### 5. PDF Formatting
The fpdf2 library doesn't natively handle markdown. Built a custom renderer that handles headings (H1-H4), bold text, italic text, bullet points, numbered lists, horizontal rules, and a professional title page.

## Project Structure

```
AI-Handbook-Generator/
├── main.py                     # Entry point
├── app/
│   ├── config.py               # Environment configuration
│   ├── ui.py                   # Gradio interface
│   ├── handlers.py             # Chat event handlers
│   ├── rag_engine.py           # LightRAG + Grok integration
│   ├── handbook_generator.py   # AgentWrite/LongWriter pipeline
│   ├── pdf_processor.py        # PDF extraction
│   └── db.py                   # PostgreSQL validation
├── requirements.txt
├── .env.example
└── README.md
```
