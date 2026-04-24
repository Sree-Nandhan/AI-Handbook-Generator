# PaperLens — Write-up

## What I Built

PaperLens is an AI-powered chat application that accepts PDF research paper uploads, answers contextual questions about them with proper citations, and generates 20,000+ word structured handbooks — all through a conversational interface.

```
PDF Upload → LightRAG (Supabase) → Chat UI (Grok 4.1) → Handbook (20k+ words)
```

## Approach

### Architecture

**LLM — Grok 4.1 Fast (xAI API):**
Used for both knowledge graph entity extraction (via LightRAG) and handbook generation. The `grok-4-1-fast-non-reasoning` variant provides fast responses at $0.20/1M input tokens.

**RAG — LightRAG with Knowledge Graph:**
LightRAG builds a knowledge graph from uploaded PDFs — extracting entities, relationships, and semantic connections. Vectors and metadata are persisted in Supabase PostgreSQL with pgvector. Q&A uses direct vector search with paper title attribution to ensure accurate source referencing.

**Database — Supabase (PostgreSQL + pgvector):**
Persistent storage for LightRAG's vector store, key-value store, and document status. HNSW-indexed vector similarity search via pgvector extension. Database is cleared on every app startup to prevent stale data from previous sessions.

**Handbook Generation — AgentWrite/LongWriter with Parallel Batching:**
Based on the LongWriter research paper's AgentWrite technique:
1. **Plan** — Grok generates a structured 22-chapter writing plan
2. **Context Fetch** — RAG context fetched for all chapters in parallel
3. **Parallel Write** — Chapters written in batches of 12 concurrently, with batch summaries for coherence
4. **Compile** — Assembled with table of contents, chapter headings (Chapter N / N.1 subsections), and references from source PDFs
5. **Export** — Saved as Markdown and professionally formatted PDF with tables, headers/footers, and page numbers

**Frontend — Gradio 6 (ChatInterface + Sidebar):**
Two-phase layout: upload page → chat page. Uses `gr.ChatInterface` for built-in chat management with autofocus. Sidebar houses PDF upload and progress indicators. Download button appears after handbook generation.

### Key Design Decisions

- **Clean DB on startup** — Prevents hallucinations from stale knowledge graph data mixing with new uploads
- **Research Master Q&A** — Grok receives paper titles in its prompt so it correctly attributes information to the right source paper
- **Direct vector search as primary Q&A** — Bypasses LightRAG's knowledge graph for Q&A to avoid cross-paper entity confusion; LightRAG used as fallback
- **Greeting detection** — Simple messages like "hello" get instant responses without hitting RAG
- **Word count enforcement** — Sections under 60% of target are automatically regenerated
- **Paper title extraction** — Extracts research paper title from first page for handbook metadata

## Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Frontend | Gradio 6 (ChatInterface) | Chat UI with history |
| LLM | Grok 4.1 (xAI API) | Generation + entity extraction |
| RAG | LightRAG | Knowledge graph from PDFs |
| Database | Supabase (pgvector) | Vector storage + embeddings |
| Embeddings | all-MiniLM-L6-v2 | Local document embeddings |
| PDF Processing | pdfplumber + fpdf2 | Extract text + export PDFs |

## Features Implemented

### Must-Have
- PDF upload and parsing (multiple files)
- Knowledge graph creation via LightRAG + Supabase
- Chat interface with contextual Q&A
- 20,000+ word handbook generation via chat

### Optional
- Multiple PDF support with progress indicators
- Live progress bar during indexing (with estimated time)
- Export to both Markdown and PDF
- Conversation history (in-session)
- Download button for generated handbooks
- Fun loading messages during Q&A processing
- Handbook re-generation flow ("same" / "regenerate")
- Handbook history listing ("show handbooks")
- PDF table rendering with proportional column widths
- Chapter-based structure with numbered subsections (N.1, N.2)

## Challenges

### 1. Cross-Paper Attribution
When multiple PDFs are indexed, LightRAG merges all entities into one knowledge graph, losing source paper attribution. Solved by using direct vector search as the primary Q&A path and injecting paper titles into the Grok prompt so it correctly attributes information.

### 2. Stale Knowledge Base
Previous sessions' data persisted in Supabase, causing hallucinations referencing papers not uploaded in the current session. Fixed by clearing all LightRAG tables on every startup.

### 3. Embedding Batch Failures
The sentence-transformers model would fail on batch encoding with "list index out of range" errors, falling back to slow individual encoding. Fixed by cleaning input text (stripping non-printable chars, type coercion) and increasing batch size.

### 4. PDF Formatting
fpdf2 doesn't handle markdown natively. Built a custom renderer handling: headings (H1-H4), bold text, italic text, bullet points, numbered lists, tables with proportional column widths, horizontal rules, page headers/footers, and chapter-based page breaks.

### 5. Generation Speed
Sequential AgentWrite (22 sections × 20s = 7+ minutes) was too slow. Redesigned with parallel batch writing (12 sections per batch), concurrent context fetching, and dynamic max_tokens. Final time: ~60-90 seconds for 20,000+ words.

## Project Structure

```
AI-Handbook-Generator/
├── main.py                     # Entry point
├── app/
│   ├── config.py               # Environment configuration
│   ├── ui.py                   # Gradio ChatInterface + Sidebar
│   ├── handlers.py             # Chat handlers (Q&A, handbook, greetings)
│   ├── rag_engine.py           # LightRAG + vector search + Grok
│   ├── handbook_generator.py   # AgentWrite pipeline + PDF export
│   ├── pdf_processor.py        # PDF extraction + title detection
│   └── db.py                   # PostgreSQL validation
├── outputs/
│   └── handbooks/              # Generated handbooks (PDF + MD + metadata)
├── uploads/                    # Uploaded PDFs
├── requirements.txt
├── .env.example
├── WRITEUP.md
└── README.md
```
