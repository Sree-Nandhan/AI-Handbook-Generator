import os
from dotenv import load_dotenv

load_dotenv()

# Google Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# Supabase / PostgreSQL
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")

# LightRAG settings
WORKING_DIR = "./rag_storage"
LLM_MODEL_NAME = "gemini-2.0-flash"
EMBEDDING_MODEL_NAME = "models/text-embedding-004"
EMBEDDING_DIM = 768
CHUNK_TOKEN_SIZE = 1200
CHUNK_OVERLAP = 100

# AgentWrite settings
MIN_WORDS_PER_PARAGRAPH = 200
MAX_WORDS_PER_PARAGRAPH = 1000
TARGET_HANDBOOK_WORDS = 20000

# File paths
UPLOAD_DIR = "./uploads"
OUTPUT_DIR = "./outputs"
