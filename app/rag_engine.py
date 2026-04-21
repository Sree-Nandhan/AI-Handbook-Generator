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

    # Clean all texts first
    cleaned = [_clean_for_embedding(t) for t in texts]

    try:
        emb = _embed_model.encode(cleaned, normalize_embeddings=True, batch_size=32)
        return np.array(emb)
    except Exception as e:
        logger.warning(f"Batch encode failed ({len(cleaned)} texts): {e}, encoding individually")
        results = []
        for text in cleaned:
            try:
                emb = _embed_model.encode([text], normalize_embeddings=True)
                results.append(emb[0])
            except Exception:
                results.append(np.zeros(EMBEDDING_DIM))
        return np.array(results)


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

    async def initialize(self) -> None:
        """Initialize LightRAG with PostgreSQL storage backends."""
        if self._initialized:
            return

        os.makedirs(LIGHTRAG_WORKING_DIR, exist_ok=True)
        logger.info("Initializing LightRAG with PostgreSQL backends...")

        self._rag = LightRAG(
            working_dir=LIGHTRAG_WORKING_DIR,
            llm_model_func=grok_complete,
            llm_model_max_async=4,
            embedding_func=EmbeddingFunc(
                embedding_dim=EMBEDDING_DIM,
                max_token_size=8192,
                func=local_embed,
            ),
            graph_storage="NetworkXStorage",
            vector_storage="PGVectorStorage",
            kv_storage="PGKVStorage",
            doc_status_storage="PGDocStatusStorage",
            embedding_batch_num=8,
            embedding_func_max_async=4,
        )

        await self._rag.initialize_storages()
        self._initialized = True
        logger.info("LightRAG initialized")

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

    @property
    def has_documents(self) -> bool:
        return self._has_documents
