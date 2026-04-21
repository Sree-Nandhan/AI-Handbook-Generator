"""
Supabase/PostgreSQL environment validation for LightRAG.

LightRAG reads POSTGRES_* environment variables directly.
This module validates they are properly configured.
"""

import os
import logging

logger = logging.getLogger("handbook.db")


def validate_postgres():
    """Ensure all POSTGRES_* env vars are set for LightRAG storage backends."""
    required = {
        "POSTGRES_HOST": os.getenv("POSTGRES_HOST", ""),
        "POSTGRES_PORT": os.getenv("POSTGRES_PORT", ""),
        "POSTGRES_USER": os.getenv("POSTGRES_USER", ""),
        "POSTGRES_PASSWORD": os.getenv("POSTGRES_PASSWORD", ""),
        "POSTGRES_DATABASE": os.getenv("POSTGRES_DATABASE", ""),
    }
    missing = [k for k, v in required.items() if not v or v.startswith("your-")]
    if missing:
        raise EnvironmentError(
            f"Missing PostgreSQL env vars: {', '.join(missing)}. "
            f"Update .env with your Supabase database credentials."
        )
    logger.info(f"PostgreSQL config OK — host: {required['POSTGRES_HOST']}")
