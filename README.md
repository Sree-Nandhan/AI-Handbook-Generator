# PaperLens — AI Handbook Generator

An AI-powered chat application that lets you upload PDF research papers, ask contextual questions, and generate structured 20,000+ word handbooks through conversation.

## Quick Start

```bash
# Clone
git clone https://github.com/Sree-Nandhan/AI-Handbook-Generator.git
cd AI-Handbook-Generator

# Setup
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Configure
cp .env.example .env
# Edit .env with your xAI API key and Supabase credentials

# Run
python main.py
# Open http://localhost:7860
```

## Prerequisites

- Python 3.10+
- [Supabase](https://supabase.com) account (free tier works)
- [xAI](https://console.x.ai/) API key for Grok 4.1

### Supabase Setup

1. Create a new project at [supabase.com](https://supabase.com)
2. Go to **SQL Editor** and run:
   ```sql
   CREATE EXTENSION IF NOT EXISTS vector;
   ```
3. Go to **Settings → Database** and note your:
   - Host (e.g., `db.xxxxx.supabase.co`)
   - Password
   - Port (`5432`)

### Environment Variables

Copy `.env.example` to `.env` and fill in:

```env
XAI_API_KEY=your-xai-api-key
POSTGRES_HOST=db.xxxxx.supabase.co
POSTGRES_PASSWORD=your-supabase-db-password
```

## Features

| Feature | Description |
|---------|-------------|
| **PDF Upload** | Upload multiple research papers (drag & drop) |
| **Knowledge Graph** | LightRAG builds a knowledge graph from PDFs |
| **Q&A Chat** | Ask questions, get citation-rich answers |
| **Handbook Generation** | Generate 20,000+ word handbooks via chat |
| **PDF Export** | Download handbooks as professionally formatted PDFs |
| **Progress Indicators** | Live progress for indexing and generation |
| **Session Management** | Clean DB on every startup, no stale data |

## Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Frontend | Gradio 6 (ChatInterface + Sidebar) | Chat UI with history |
| LLM | Grok 4.1 (xAI API) | Generation + entity extraction |
| RAG | LightRAG | Knowledge graph from PDFs |
| Database | Supabase (PostgreSQL + pgvector) | Vector storage |
| Embeddings | all-MiniLM-L6-v2 (local) | Document embeddings |
| PDF Processing | pdfplumber + fpdf2 | Extract text, export PDFs |

## Usage

1. **Upload PDFs** — Drop research papers on the upload page
2. **Click "Index Documents"** — Wait for progress bar to complete
3. **Ask questions** — e.g., "What methodology does the paper use?"
4. **Generate handbook** — e.g., "Create a handbook on Machine Learning"
5. **Download** — Click the download button that appears after generation

### Special Commands

- `Create a handbook on [topic]` — Generate a 20K+ word handbook
- `show handbooks` — List previously generated handbooks
- `same` / `regenerate` — Regenerate with the last topic

## Architecture

```
PDF Upload → pdfplumber (text extraction)
  → LightRAG (knowledge graph + Supabase/pgvector)
    → Chat Q&A (vector search + Grok 4.1 Research Master)
    → Handbook Generation (AgentWrite/LongWriter pipeline)
      → PDF Export (fpdf2 with tables, chapters, references)
```

### AgentWrite/LongWriter Pipeline

1. **Plan** — Grok creates a 22-chapter writing plan
2. **Context Fetch** — RAG context retrieved for each chapter in parallel
3. **Parallel Write** — Chapters written in batches of 12 concurrently
4. **Compile** — Assembled with TOC, chapter headings, references
5. **Export** — Saved as Markdown + professionally formatted PDF

## Project Structure

```
AI-Handbook-Generator/
├── main.py                     # Entry point
├── app/
│   ├── config.py               # Environment configuration
│   ├── ui.py                   # Gradio ChatInterface + Sidebar
│   ├── handlers.py             # Chat event handlers
│   ├── rag_engine.py           # LightRAG + Grok integration
│   ├── handbook_generator.py   # AgentWrite/LongWriter pipeline
│   ├── pdf_processor.py        # PDF extraction + title detection
│   └── db.py                   # PostgreSQL validation
├── requirements.txt
├── .env.example
├── WRITEUP.md
└── README.md
```
