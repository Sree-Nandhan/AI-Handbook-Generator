import os
from dotenv import load_dotenv

load_dotenv()

# Cerebras LLM
CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY", "")
CEREBRAS_MODEL_NAME = "qwen-3-235b-a22b-instruct-2507"

# Supabase / PostgreSQL
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")

# LightRAG settings
WORKING_DIR = "./rag_storage"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
CHUNK_TOKEN_SIZE = 1200
CHUNK_OVERLAP = 100

# AgentWrite settings
MIN_WORDS_PER_PARAGRAPH = 800
MAX_WORDS_PER_PARAGRAPH = 1200
TARGET_HANDBOOK_WORDS = 20000

# File paths
UPLOAD_DIR = "./uploads"
OUTPUT_DIR = "./outputs"
