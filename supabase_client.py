import os
from dotenv import load_dotenv

load_dotenv()


def setup_postgres_env_for_lightrag():
    """Ensure POSTGRES_* env vars are set for LightRAG's PG storage backends.
    These are read directly by LightRAG internals."""
    required_vars = [
        "POSTGRES_HOST",
        "POSTGRES_PORT",
        "POSTGRES_USER",
        "POSTGRES_PASSWORD",
        "POSTGRES_DATABASE",
    ]
    missing = [v for v in required_vars if not os.getenv(v)]
    if missing:
        raise EnvironmentError(
            f"Missing PostgreSQL env vars: {', '.join(missing)}. "
            f"Check your .env file."
        )
