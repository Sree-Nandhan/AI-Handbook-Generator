"""
RAG Engine — LightRAG knowledge graph with Supabase/PostgreSQL storage
and Grok 4.1 as the LLM backend.
"""

import asyncio
import os
import logging
import numpy as np
from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import openai_complete_if_cache
from app.config import (
    XAI_API_KEY, XAI_BASE_URL, XAI_MODEL,
    LIGHTRAG_WORKING_DIR, EMBEDDING_MODEL_NAME, EMBEDDING_DIM,
)

try:
    from lightrag.utils import EmbeddingFunc
except ImportError:
    from lightrag import EmbeddingFunc

logger = logging.getLogger("handbook.rag")

# ---------------------------------------------------------------------------
# LLM function (Grok 4.1 via xAI OpenAI-compatible API)
# ---------------------------------------------------------------------------

async def grok_complete(
    prompt, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs
) -> str:
    """Call Grok 4.1 through xAI's OpenAI-compatible endpoint."""
    kwargs.pop("response_format", None)
    kwargs.pop("mode", None)
    logger.debug(f"LLM call: {len(prompt)} chars")
    return await openai_complete_if_cache(
        model=XAI_MODEL,
        prompt=prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        api_key=XAI_API_KEY,
        base_url=XAI_BASE_URL,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Embedding function (local sentence-transformers)
# ---------------------------------------------------------------------------

_embed_model = None


def _clean_for_embedding(text: str) -> str:
    """Sanitize text for sentence-transformers encode."""
    import re
    # Replace newlines/tabs with spaces
    text = re.sub(r"[\n\r\t]+", " ", text)
    # Collapse multiple spaces
    text = re.sub(r"\s+", " ", text)
    # Strip non-printable chars
    text = "".join(c for c in text if c.isprintable() or c == " ")
    text = text.strip()
    return text if text else "empty"


def _encode_sync(texts: list[str]) -> np.ndarray:
    """Synchronous encode with defensive handling."""
    global _embed_model
    if _embed_model is None:
        from sentence_transformers import SentenceTransformer
        logger.info(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
        _embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        logger.info("Embedding model loaded")

    # Clean all texts and ensure they're plain strings (not numpy arrays or other types)
    cleaned = []
    for t in texts:
        if isinstance(t, str):
            cleaned.append(_clean_for_embedding(t))
        else:
            cleaned.append(_clean_for_embedding(str(t)))

    try:
        emb = _embed_model.encode(cleaned, normalize_embeddings=True, batch_size=64,
                                   show_progress_bar=False)
        return np.array(emb, dtype=np.float32)
    except Exception as e:
        logger.warning(f"Batch encode failed ({len(cleaned)} texts): {e}, encoding individually")
        results = []
        for text in cleaned:
            try:
                emb = _embed_model.encode([text], normalize_embeddings=True)
                results.append(emb[0])
            except Exception:
                results.append(np.zeros(EMBEDDING_DIM))
        return np.array(results, dtype=np.float32)


async def local_embed(texts: list[str]) -> np.ndarray:
    """Generate embeddings using a local sentence-transformers model."""
    if not texts:
        return np.array([]).reshape(0, EMBEDDING_DIM)

    return await asyncio.to_thread(_encode_sync, texts)


# ---------------------------------------------------------------------------
# RAG Engine
# ---------------------------------------------------------------------------

class RAGEngine:
    """LightRAG knowledge graph RAG with Supabase/PostgreSQL backends."""

    def __init__(self):
        self._initialized = False
        self._rag: LightRAG | None = None
        self._has_documents = False
        self._references: dict[str, str] = {}  # filename -> references text
        self._source_filenames: list[str] = []  # uploaded PDF names

    async def initialize(self) -> None:
        """Initialize LightRAG with PostgreSQL storage backends.
        Clears all previous data on startup to prevent stale/hallucinated responses.
        """
        if self._initialized:
            return

        os.makedirs(LIGHTRAG_WORKING_DIR, exist_ok=True)

        # ALWAYS clear previous session data on startup — prevents hallucinations
        # from stale knowledge graph entries mixing with new uploads
        logger.info("Clearing previous session data...")
        await self._clear_all_data()

        logger.info("Initializing LightRAG with PostgreSQL backends...")

        self._rag = LightRAG(
            working_dir=LIGHTRAG_WORKING_DIR,
            llm_model_func=grok_complete,
            llm_model_max_async=32,
            embedding_func=EmbeddingFunc(
                embedding_dim=EMBEDDING_DIM,
                max_token_size=8192,
                func=local_embed,
            ),
            graph_storage="NetworkXStorage",
            vector_storage="PGVectorStorage",
            kv_storage="PGKVStorage",
            doc_status_storage="PGDocStatusStorage",
            embedding_batch_num=16,
            embedding_func_max_async=16,
        )

        await self._rag.initialize_storages()
        self._initialized = True
        self._has_documents = False

        # Save embedding model name for consistency checks
        model_file = os.path.join(LIGHTRAG_WORKING_DIR, ".embedding_model")
        with open(model_file, "w") as f:
            f.write(EMBEDDING_MODEL_NAME)

        logger.info("LightRAG initialized — clean slate, ready for uploads")

    async def insert(self, text: str) -> None:
        """Insert document text into the knowledge graph."""
        if not self._initialized:
            raise RuntimeError("RAG engine not initialized")
        logger.info(f"Inserting document: {len(text):,} chars")
        await self._rag.ainsert(text)
        self._has_documents = True
        logger.info("Document indexed")

    async def query(self, question: str, mode: str = "hybrid") -> str:
        """Query the knowledge graph — uses direct vector search with paper attribution
        as the primary path for accurate source attribution."""
        logger.info(f"Query: {question[:80]}...")

        # Primary path: direct vector search + Grok with paper titles
        # This ensures accurate attribution to the correct uploaded papers
        try:
            result = await self._direct_answer(question)
            if result and len(result.strip()) > 50:
                logger.info(f"Query result (direct): {len(result)} chars")
                return result
        except Exception as e:
            logger.warning(f"Direct answer failed: {e}")

        # Fallback: LightRAG knowledge graph query
        try:
            result = await self._rag.aquery(question, param=QueryParam(mode=mode))
            if result and "[no-context]" not in result and len(result.strip()) >= 20:
                logger.info(f"Query result (RAG {mode}): {len(result)} chars")
                return result
        except Exception as e:
            logger.warning(f"RAG query failed: {e}")

        # Last resort
        return (
            "I couldn't find a specific answer in the uploaded papers. "
            "Try asking about specific topics, methods, or findings."
        )

    async def get_context(self, topic: str) -> str:
        """Retrieve knowledge graph context for handbook generation."""
        if not self._has_documents:
            return ""
        try:
            return await self._rag.aquery(topic, param=QueryParam(mode="hybrid"))
        except Exception as e:
            logger.warning(f"Context retrieval failed: {e}")
            return ""

    async def _direct_answer(self, question: str) -> str:
        """Fallback: pull raw chunks from DB and answer directly via Grok."""
        import asyncpg
        from app.config import (
            POSTGRES_HOST, POSTGRES_PORT, POSTGRES_USER,
            POSTGRES_PASSWORD, POSTGRES_DATABASE,
        )

        # Get raw document chunks from Supabase
        conn = await asyncpg.connect(
            host=POSTGRES_HOST, port=POSTGRES_PORT, user=POSTGRES_USER,
            password=POSTGRES_PASSWORD, database=POSTGRES_DATABASE, ssl="require",
        )

        # Embed the question and find closest chunks
        q_emb = await local_embed([question])
        vec_str = "[" + ",".join(str(x) for x in q_emb[0]) + "]"

        rows = await conn.fetch(
            "SELECT content FROM lightrag_vdb_chunks "
            "ORDER BY content_vector <=> $1::vector LIMIT 8",
            vec_str,
        )
        await conn.close()

        if not rows:
            return ""

        context = "\n\n".join(r["content"] for r in rows)

        # Build paper list for attribution
        paper_list = ""
        if self._source_filenames:
            papers = [f"  Paper {i+1}: \"{name}\"" for i, name in enumerate(self._source_filenames)]
            paper_list = (
                f"The user uploaded these specific papers:\n"
                + "\n".join(papers)
                + "\n\nIMPORTANT: Only reference these papers. Do NOT invent or confuse paper names. "
                f"Match each excerpt to its source paper based on content, authors, and topic.\n\n"
            )

        # Direct Grok call with the context
        answer = await grok_complete(
            prompt=(
                f"You are a Research Master AI. A user uploaded research papers and asked:\n\n"
                f'"{question}"\n\n'
                f"{paper_list}"
                f"Here are the most relevant excerpts from their uploaded papers:\n\n"
                f"{context[:8000]}\n\n"
                f"Provide a detailed, scholarly answer following these rules:\n"
                f"- ONLY reference the papers listed above — do not invent paper names\n"
                f"- Clearly label which paper each piece of information comes from\n"
                f"- Reference specific authors, sections, figures, tables, or equations\n"
                f"- Use academic formatting: numbered references, proper citations\n"
                f"- Include specific data points, statistics, and metrics from the papers\n"
                f"- Be comprehensive but well-organized with clear headings\n"
                f"- Never say you don't have enough information — synthesize what's available"
            ),
            system_prompt=(
                "You are a Research Master — an expert academic assistant specializing in analyzing "
                "uploaded research papers. You provide precise, citation-rich answers. "
                "CRITICAL: Only reference papers the user actually uploaded. Never confuse or invent "
                "paper titles. Match content to its correct source paper based on authors and topic. "
                "Format responses with clear structure: headings, bullet points, and numbered references."
            ),
        )
        return answer

    async def _check_embedding_consistency(self):
        """Detect if the embedding model changed since last indexing. If so, wipe stale data."""
        model_file = os.path.join(LIGHTRAG_WORKING_DIR, ".embedding_model")
        current_model = EMBEDDING_MODEL_NAME

        if os.path.exists(model_file):
            with open(model_file, "r") as f:
                saved_model = f.read().strip()
            if saved_model == current_model:
                return  # Same model, no action needed
            logger.warning(
                f"Embedding model changed: {saved_model} -> {current_model}. "
                f"Clearing stale data to prevent mismatched vectors."
            )
            await self._clear_all_data()

        # Save current model name
        with open(model_file, "w") as f:
            f.write(current_model)

    async def _clear_all_data(self):
        """Drop all LightRAG tables and clear local graph."""
        from app.config import (
            POSTGRES_HOST, POSTGRES_PORT, POSTGRES_USER,
            POSTGRES_PASSWORD, POSTGRES_DATABASE,
        )
        import asyncpg

        try:
            conn = await asyncpg.connect(
                host=POSTGRES_HOST, port=POSTGRES_PORT, user=POSTGRES_USER,
                password=POSTGRES_PASSWORD, database=POSTGRES_DATABASE, ssl="require",
            )
            tables = await conn.fetch(
                "SELECT tablename FROM pg_tables WHERE tablename LIKE 'lightrag_%'"
            )
            for t in tables:
                await conn.execute(f"DROP TABLE IF EXISTS {t['tablename']} CASCADE")
            await conn.close()
            logger.info(f"Dropped {len(tables)} stale LightRAG tables")
        except Exception as e:
            logger.error(f"Failed to clear stale DB: {e}")

        # Clear local graph file
        graph_file = os.path.join(LIGHTRAG_WORKING_DIR, "graph_chunk_entity_relation.graphml")
        if os.path.exists(graph_file):
            os.remove(graph_file)
            logger.info("Cleared local graph file")

    def add_source_filename(self, filename: str) -> None:
        """Track uploaded PDF filenames for title generation."""
        name = filename.rsplit(".", 1)[0]  # strip .pdf
        if name not in self._source_filenames:
            self._source_filenames.append(name)

    def get_source_title(self) -> str:
        """Return a title derived from uploaded source filenames."""
        if not self._source_filenames:
            return ""
        if len(self._source_filenames) == 1:
            return self._source_filenames[0]
        return self._source_filenames[0]  # use first paper as primary title

    def add_references(self, filename: str, references: str) -> None:
        """Store extracted references from a source document."""
        self._references[filename] = references

    def get_references(self) -> str:
        """Return all stored references, formatted for the handbook."""
        if not self._references:
            return ""
        parts = []
        for fname, refs in self._references.items():
            parts.append(refs)
        return "\n\n".join(parts)

    @property
    def has_documents(self) -> bool:
        return self._has_documents
