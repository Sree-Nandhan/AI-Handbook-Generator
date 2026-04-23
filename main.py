"""
Entry point — starts the AI Handbook Generator.
"""

import os
import asyncio
import logging
from app.config import OUTPUT_DIR, APP_HOST, APP_PORT, validate
from app.rag_engine import RAGEngine
from app.handbook_generator import HandbookGenerator
from app import ui

logger = logging.getLogger("handbook")


async def startup():
    """Validate config, initialize RAG engine and handbook generator."""
    validate()
    engine = RAGEngine()
    await engine.initialize()
    generator = HandbookGenerator(engine)
    return engine, generator


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    logger.info("Starting AI Handbook Generator...")

    engine, generator = asyncio.run(startup())

    app = ui.build(engine, generator)
    app.queue()
    app.launch(
        server_name=APP_HOST,
        server_port=APP_PORT,
        theme=ui.THEME,
        css=ui.CSS,
        js="() => { document.querySelector('body').classList.remove('dark'); }",
    )


if __name__ == "__main__":
    main()
