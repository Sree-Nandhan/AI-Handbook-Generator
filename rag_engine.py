import asyncio
import numpy as np
from cerebras.cloud.sdk import Cerebras
from sentence_transformers import SentenceTransformer
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
from config import CEREBRAS_API_KEY, CEREBRAS_MODEL_NAME, EMBEDDING_MODEL_NAME


class RAGEngine:
    """Fast vector RAG: chunk → embed locally → cosine search → Cerebras answer."""

    def __init__(self):
        self._initialized = False
        self._embed_model: SentenceTransformer | None = None
        self._client: Cerebras | None = None
        self._chunks: list[str] = []
        self._chunk_embeddings: np.ndarray | None = None

    async def initialize(self) -> None:
        if self._initialized:
            return
        self._embed_model = await asyncio.to_thread(SentenceTransformer, EMBEDDING_MODEL_NAME)
        self._client = Cerebras(api_key=CEREBRAS_API_KEY)
        self._initialized = True

    async def insert_document(self, text: str) -> None:
        chunks = self._chunk_text(text, chunk_size=200, overlap=30)
        self._chunks.extend(chunks)
        embeddings = await asyncio.to_thread(self._embed_model.encode, chunks)
        if self._chunk_embeddings is None:
            self._chunk_embeddings = np.array(embeddings)
        else:
            self._chunk_embeddings = np.vstack([self._chunk_embeddings, np.array(embeddings)])

    async def query(self, question: str, mode: str = "hybrid", top_k: int = 5) -> str:
        if not self._chunks:
            return "No documents indexed yet. Upload and index a PDF first."

        q_emb = await asyncio.to_thread(self._embed_model.encode, [question])
        q_emb = np.array(q_emb)
        sims = np.dot(self._chunk_embeddings, q_emb.T).squeeze()
        top_indices = np.argsort(sims)[-top_k:][::-1]
        context = "\n\n".join(self._chunks[i] for i in top_indices)

        return await self._answer_with_context(question, context)

    async def query_for_handbook_context(self, topic: str, top_k: int = 5) -> str:
        if not self._chunks:
            return ""
        q_emb = await asyncio.to_thread(self._embed_model.encode, [topic])
        q_emb = np.array(q_emb)
        sims = np.dot(self._chunk_embeddings, q_emb.T).squeeze()
        top_indices = np.argsort(sims)[-top_k:][::-1]
        return "\n\n".join(self._chunks[i] for i in top_indices)

    @retry(
        wait=wait_exponential(min=2, max=30),
        stop=stop_after_attempt(5),
        retry=retry_if_exception_type(Exception),
    )
    async def _answer_with_context(self, question: str, context: str) -> str:
        response = await asyncio.to_thread(
            self._client.chat.completions.create,
            model=CEREBRAS_MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a helpful research assistant. Answer the user's question "
                        "accurately based on the provided document context. Be detailed."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQuestion: {question}",
                },
            ],
            max_tokens=1024,
            temperature=0.3,
        )
        return response.choices[0].message.content

    async def shutdown(self) -> None:
        pass

    @staticmethod
    def _chunk_text(text: str, chunk_size: int = 200, overlap: int = 30) -> list[str]:
        words = text.split()
        chunks = []
        i = 0
        while i < len(words):
            chunk = " ".join(words[i : i + chunk_size])
            if chunk.strip():
                chunks.append(chunk)
            i += chunk_size - overlap
        return chunks
