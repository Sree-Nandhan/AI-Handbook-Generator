import asyncio
import numpy as np
from lightrag import LightRAG, QueryParam
from lightrag.llm.gemini import gemini_model_complete, gemini_embed
from lightrag.utils import wrap_embedding_func_with_attrs
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
from config import (
    GEMINI_API_KEY, WORKING_DIR, LLM_MODEL_NAME,
    EMBEDDING_MODEL_NAME, EMBEDDING_DIM, CHUNK_TOKEN_SIZE, CHUNK_OVERLAP,
)
from supabase_client import setup_postgres_env_for_lightrag


class RAGEngine:
    def __init__(self):
        self.rag: LightRAG | None = None
        self._initialized = False
        self._has_documents = False

    async def initialize(self) -> None:
        if self._initialized:
            return

        setup_postgres_env_for_lightrag()

        @retry(
            wait=wait_exponential(min=2, max=60),
            stop=stop_after_attempt(5),
            retry=retry_if_exception_type(Exception),
        )
        async def llm_model_func(
            prompt,
            system_prompt=None,
            history_messages=[],
            keyword_extraction=False,
            **kwargs,
        ) -> str:
            return await gemini_model_complete(
                prompt,
                system_prompt=system_prompt,
                history_messages=history_messages,
                api_key=GEMINI_API_KEY,
                model_name=LLM_MODEL_NAME,
                **kwargs,
            )

        @wrap_embedding_func_with_attrs(
            embedding_dim=EMBEDDING_DIM,
            max_token_size=2048,
            model_name=EMBEDDING_MODEL_NAME,
        )
        @retry(
            wait=wait_exponential(min=2, max=60),
            stop=stop_after_attempt(5),
            retry=retry_if_exception_type(Exception),
        )
        async def embedding_func(texts: list[str]) -> np.ndarray:
            return await gemini_embed.func(
                texts,
                api_key=GEMINI_API_KEY,
                model=EMBEDDING_MODEL_NAME,
            )

        self.rag = LightRAG(
            working_dir=WORKING_DIR,
            llm_model_name=LLM_MODEL_NAME,
            llm_model_func=llm_model_func,
            embedding_func=embedding_func,
            embedding_func_max_async=4,
            embedding_batch_num=8,
            llm_model_max_async=2,
            chunk_token_size=CHUNK_TOKEN_SIZE,
            chunk_overlap_token_size=CHUNK_OVERLAP,
            graph_storage="PGGraphStorage",
            vector_storage="PGVectorStorage",
            doc_status_storage="PGDocStatusStorage",
            kv_storage="PGKVStorage",
        )
        await self.rag.initialize_storages()
        self._initialized = True

    async def insert_document(self, text: str) -> None:
        if not self._initialized:
            raise RuntimeError("RAG engine not initialized. Call initialize() first.")
        await self.rag.ainsert(text)
        self._has_documents = True

    async def query(self, question: str, mode: str = "hybrid") -> str:
        if not self._initialized:
            raise RuntimeError("RAG engine not initialized. Call initialize() first.")
        if not self._has_documents:
            return "No documents have been indexed yet. Please upload and index PDFs first."
        result = await self.rag.aquery(
            question,
            param=QueryParam(mode=mode),
        )
        return result

    async def query_for_handbook_context(self, section_topic: str) -> str:
        return await self.query(section_topic, mode="hybrid")

    async def shutdown(self) -> None:
        if self.rag:
            await self.rag.finalize_storages()
