# AI Handbook Generator

An AI-powered chat application that lets you upload PDF documents, ask contextual questions, and generate structured 20,000+ word handbooks through conversation.

## Features

- **PDF Upload & Processing**: Upload multiple PDF documents for analysis
- **Knowledge Graph RAG**: Uses LightRAG to build a knowledge graph from your documents
- **Contextual Chat**: Ask questions and get accurate answers grounded in your documents
- **20,000+ Word Handbook Generation**: Generate comprehensive handbooks using the AgentWrite/LongWriter pipeline
- **Export**: Download generated handbooks as Markdown and PDF files

## Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Frontend | Gradio | Simple chat interface |
| LLM | Grok 4.1 (xAI API) | Long-context generation with LongWriter |
| RAG System | LightRAG | Knowledge graph creation from PDFs |
| Database | Supabase (PostgreSQL + pgvector) | Vector storage for embeddings |
| PDF Processing | pdfplumber | Extract text from uploads |
| Embeddings | BAAI/bge-m3 (local) | Document embedding for retrieval |

## Setup

### 1. Prerequisites

- Python 3.10+
- A [Supabase](https://supabase.com) account (free tier)
- An [xAI](https://console.x.ai/) API key

### 2. Supabase Setup

1. Create a new project at [supabase.com](https://supabase.com)
2. Go to **SQL Editor** and run:
   ```sql
   CREATE EXTENSION IF NOT EXISTS vector;
   ```
3. Note your **Project URL**, **anon key**, and **database password**

### 3. Install & Configure

```bash
# Clone the repository
git clone <your-repo-url>
cd LunarTech_Handbook_Generator

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your xAI API key and Supabase credentials
```

### 4. Run

```bash
python main.py
```

Open your browser to `http://localhost:7860`

## Usage

All interaction happens through a single chat interface:

1. **Upload PDFs**: Attach PDF files using the file upload button and click Send
2. **Ask Questions**: Type questions about your documents to get contextual answers
3. **Generate Handbook**: Type "Create a handbook on [topic]" to generate a 20,000+ word handbook
4. **Download**: Use the download button to get the generated handbook as PDF

## How It Works

### AgentWrite / LongWriter Pipeline

The handbook generation uses a two-phase approach inspired by the LongWriter research:

1. **Planning Phase**: Grok 4.1 breaks the topic into 30-40 section-level subtasks, each with a main point and target word count
2. **Writing Phase**: Each section is generated iteratively, with:
   - The full plan for structural awareness
   - Previously written text for coherence
   - LightRAG knowledge graph context for accuracy
3. **Compilation**: All sections are assembled into a structured Markdown document with a table of contents, then exported as both MD and PDF

### Architecture

```
PDF Upload → pdfplumber (text extraction)
    → LightRAG (knowledge graph + Supabase/pgvector)
        → Chat Q&A (hybrid retrieval via Grok 4.1)
        → Handbook Generation (AgentWrite + RAG context)
            → Markdown + PDF Export
```

## Project Structure

```
LunarTech_Handbook_Generator/
├── main.py                     # Entry point
├── app/
│   ├── __init__.py             # Package init
│   ├── config.py               # Configuration + validation
│   ├── ui.py                   # Gradio layout, theme, wiring
│   ├── handlers.py             # Chat event handlers
│   ├── rag_engine.py           # LightRAG + Grok 4.1 + PostgreSQL
│   ├── handbook_generator.py   # AgentWrite/LongWriter pipeline
│   ├── pdf_processor.py        # PDF text extraction + cleaning
│   └── db.py                   # PostgreSQL env validation
├── requirements.txt            # Python dependencies
├── .env.example                # Environment variable template
└── README.md                   # This file
```

## Approach & Challenges

### Approach
- Used **LightRAG** for knowledge graph-based RAG, providing better contextual understanding of document relationships than simple vector similarity
- Implemented the **AgentWrite** iterative generation pattern to achieve 20,000+ word output within LLM context limits
- Chose **Grok 4.1** for its large 256K context window and cost-effective pricing ($0.20/1M input tokens)
- Used **Supabase PostgreSQL with pgvector** as the persistent vector storage backend
- Local **BAAI/bge-m3** embeddings for zero-cost, high-quality document retrieval

### Challenges
- **Knowledge graph construction**: LightRAG requires careful configuration of PostgreSQL backends; ensuring proper pgvector extension setup on Supabase
- **Context management**: As the handbook grows, passing all prior text becomes impractical; implemented sliding window with summary of earlier sections
- **Plan consistency**: LLM plan output varies in format; used robust regex parsing with multiple fallback patterns
