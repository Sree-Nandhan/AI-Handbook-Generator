# AI Handbook Generator

An AI-powered chat application that lets you upload PDF documents, ask contextual questions, and generate structured 20,000+ word handbooks through conversation.

## Features

- **PDF Upload & Processing**: Upload multiple PDF documents for analysis
- **Knowledge Graph RAG**: Uses LightRAG to build a knowledge graph from your documents
- **Contextual Chat**: Ask questions and get accurate answers grounded in your documents
- **20,000+ Word Handbook Generation**: Generate comprehensive handbooks using the AgentWrite pipeline
- **Export**: Download generated handbooks as Markdown files

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Frontend | Gradio |
| LLM | Google Gemini 2.0 Flash (free tier) |
| RAG System | LightRAG (knowledge graph) |
| Database | Supabase (PostgreSQL + pgvector) |
| PDF Processing | pdfplumber |

## Setup

### 1. Prerequisites

- Python 3.10+
- A [Supabase](https://supabase.com) account (free tier)
- A [Google AI Studio](https://aistudio.google.com/app/apikey) API key (free)

### 2. Supabase Setup

1. Create a new project at [supabase.com](https://supabase.com)
2. Go to **SQL Editor** and run:
   ```sql
   CREATE EXTENSION IF NOT EXISTS vector;
   ```
3. Go to **Settings > Database** and copy your connection string (URI format)

### 3. Install & Configure

```bash
# Clone the repository
git clone <your-repo-url>
cd LunarTech

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your API keys and Supabase connection details
```

### 4. Run

```bash
python app.py
```

Open your browser to `http://localhost:7860`

## Usage

1. **Upload PDFs**: Use the sidebar to upload one or more PDF documents
2. **Index Documents**: Click "Index Documents" to process and store them in the knowledge graph
3. **Chat**: Ask questions about your documents in the chat interface
4. **Generate Handbook**: Enter a topic and click "Generate Handbook" to create a 20,000+ word structured handbook

## How It Works

### AgentWrite Pipeline

The handbook generation uses a two-phase approach inspired by the LongWriter research:

1. **Planning Phase**: The LLM breaks the handbook topic into 30-50 paragraph-level subtasks, each with a main point and target word count
2. **Writing Phase**: Each paragraph is generated iteratively, with:
   - The full plan for structural awareness
   - Previously written text for coherence
   - RAG-retrieved context from your documents for accuracy
3. **Compilation**: All paragraphs are assembled into a structured Markdown document with a table of contents

### Architecture

```
PDF Upload -> Text Extraction (pdfplumber)
    -> LightRAG Knowledge Graph (Supabase/pgvector)
        -> Chat Q&A (hybrid retrieval)
        -> Handbook Generation (AgentWrite + RAG context)
            -> Markdown Export
```

## Project Structure

```
LunarTech/
├── app.py                  # Gradio UI and event handlers
├── config.py               # Configuration and environment variables
├── pdf_processor.py        # PDF text extraction
├── rag_engine.py           # LightRAG integration with Gemini + PostgreSQL
├── handbook_generator.py   # AgentWrite plan + write pipeline
├── supabase_client.py      # PostgreSQL environment setup
├── requirements.txt        # Python dependencies
├── .env.example            # Environment variable template
└── README.md               # This file
```

## Approach & Challenges

### Approach
- Used LightRAG for knowledge graph-based RAG instead of simple vector similarity, providing better contextual understanding of document relationships
- Implemented the AgentWrite iterative generation pattern to achieve 20,000+ word output within LLM context limits
- Chose Gemini 2.0 Flash for its generous free tier and 1M token context window
- Used Supabase PostgreSQL with pgvector as the vector storage backend

### Challenges
- **Rate limiting**: Gemini free tier has 15 RPM limit. Implemented exponential backoff retry logic and pacing between generation calls
- **Context management**: As the handbook grows, passing all prior text becomes impractical. Implemented sliding window (last 3000 words) with summary of earlier sections
- **Plan consistency**: LLM plan output can vary in format. Used robust regex parsing with fallback re-prompting
