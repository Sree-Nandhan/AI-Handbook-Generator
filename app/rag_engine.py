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
        """Initialize LightRAG with PostgreSQL storage backends."""
        if self._initialized:
            return

        os.makedirs(LIGHTRAG_WORKING_DIR, exist_ok=True)

        # Verify embedding model consistency — if model changed, wipe stale data
        await self._check_embedding_consistency()

        logger.info("Initializing LightRAG with PostgreSQL backends...")

        self._rag = LightRAG(
            working_dir=LIGHTRAG_WORKING_DIR,
            llm_model_func=grok_complete,
            llm_model_max_async=24,
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

        # Check if knowledge graph already has data from previous sessions
        graph_file = os.path.join(LIGHTRAG_WORKING_DIR, "graph_chunk_entity_relation.graphml")
        if os.path.exists(graph_file) and os.path.getsize(graph_file) > 1000:
            self._has_documents = True
            size_kb = os.path.getsize(graph_file) // 1024
            logger.info(f"LightRAG initialized — found existing KG ({size_kb} KB)")
        else:
            logger.info("LightRAG initialized — empty knowledge graph")

    async def insert(self, text: str) -> None:
        """Insert document text into the knowledge graph."""
        if not self._initialized:
            raise RuntimeError("RAG engine not initialized")
        logger.info(f"Inserting document: {len(text):,} chars")
        await self._rag.ainsert(text)
        self._has_documents = True
        logger.info("Document indexed")

    async def query(self, question: str, mode: str = "hybrid") -> str:
        """Query the knowledge graph for an answer."""
        if not self._has_documents:
            return "No documents indexed yet. Upload and index a PDF first."
        logger.info(f"Query [{mode}]: {question[:80]}...")
        result = await self._rag.aquery(question, param=QueryParam(mode=mode))

        # If LightRAG returns no-context, try naive mode (pure vector search)
        if not result or "[no-context]" in result or len(result.strip()) < 20:
            logger.warning("Hybrid query returned no context, falling back to naive mode")
            try:
                result = await self._rag.aquery(question, param=QueryParam(mode="naive"))
            except Exception:
                pass

        # If still no context, try local mode
        if not result or "[no-context]" in result or len(result.strip()) < 20:
            logger.warning("Naive query also failed, trying local mode")
            try:
                result = await self._rag.aquery(question, param=QueryParam(mode="local"))
            except Exception:
                pass

        # Final fallback — direct LLM call with global context
        if not result or "[no-context]" in result or len(result.strip()) < 20:
            logger.warning("All RAG modes failed, using direct LLM with global context")
            try:
                result = await self._direct_answer(question)
            except Exception as e:
                logger.error(f"Direct answer failed: {e}")
                result = (
                    "I couldn't find a specific answer in the knowledge graph. "
                    "Try asking about specific topics, methods, or findings from the paper."
                )

        logger.info(f"Query result: {len(result)} chars")
        return result

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

        # Direct Grok call with the context
        answer = await grok_complete(
            prompt=(
                f"Based on the following document excerpts, answer this question:\n\n"
                f"Question: {question}\n\n"
                f"Document excerpts:\n{context[:6000]}\n\n"
                f"Provide a detailed, accurate answer based on the documents."
            ),
            system_prompt="You are a helpful research assistant. Answer based on the provided documents.",
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
