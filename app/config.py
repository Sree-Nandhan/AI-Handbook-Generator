"""
Application configuration — loads all settings from .env file.
"""

import os
import logging
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("handbook")

# ---------------------------------------------------------------------------
# xAI Grok 4.1
# ---------------------------------------------------------------------------
XAI_API_KEY = os.getenv("XAI_API_KEY", "")
XAI_BASE_URL = os.getenv("XAI_BASE_URL", "https://api.x.ai/v1")
XAI_MODEL = os.getenv("XAI_MODEL", "grok-4.1")

# ---------------------------------------------------------------------------
# Supabase
# ---------------------------------------------------------------------------
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")

# ---------------------------------------------------------------------------
# PostgreSQL (for LightRAG storage backends)
# ---------------------------------------------------------------------------
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "")
POSTGRES_PORT = int(os.getenv("POSTGRES_PORT", "5432"))
POSTGRES_USER = os.getenv("POSTGRES_USER", "postgres")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "")
POSTGRES_DATABASE = os.getenv("POSTGRES_DATABASE", "postgres")

# ---------------------------------------------------------------------------
# LightRAG
# ---------------------------------------------------------------------------
LIGHTRAG_WORKING_DIR = os.getenv("LIGHTRAG_WORKING_DIR", "./rag_storage")
CHUNK_TOKEN_SIZE = int(os.getenv("LIGHTRAG_CHUNK_TOKEN_SIZE", "2400"))
CHUNK_OVERLAP = int(os.getenv("LIGHTRAG_CHUNK_OVERLAP", "200"))

# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "384"))

# ---------------------------------------------------------------------------
# AgentWrite / LongWriter
# ---------------------------------------------------------------------------
MIN_WORDS_PER_PARAGRAPH = int(os.getenv("MIN_WORDS_PER_PARAGRAPH", "900"))
MAX_WORDS_PER_PARAGRAPH = int(os.getenv("MAX_WORDS_PER_PARAGRAPH", "1300"))
TARGET_HANDBOOK_WORDS = int(os.getenv("TARGET_HANDBOOK_WORDS", "25000"))

# ---------------------------------------------------------------------------
# File paths
# ---------------------------------------------------------------------------
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "./uploads")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "./outputs")
APP_HOST = os.getenv("APP_HOST", "0.0.0.0")
APP_PORT = int(os.getenv("APP_PORT", "7860"))


def validate():
    """Validate that all critical environment variables are set."""
    errors = []
    if not XAI_API_KEY or XAI_API_KEY.startswith("your-"):
        errors.append("XAI_API_KEY")
    if not POSTGRES_HOST or POSTGRES_HOST.startswith("your-"):
        errors.append("POSTGRES_HOST")
    if not POSTGRES_PASSWORD or POSTGRES_PASSWORD.startswith("your-"):
        errors.append("POSTGRES_PASSWORD")
    if errors:
        raise EnvironmentError(
            f"Missing required config: {', '.join(errors)}. "
            f"Copy .env.example to .env and fill in your credentials."
        )
    logger.info("Configuration validated")
