"""
main.py — Expert Network Search Copilot
========================================
Run:
    uvicorn main:app --reload --port 8001

Docs:
    http://localhost:8001/docs
"""
from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)

from app.routes import router


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Pre-warm the Text-to-SQL agent
    from app.routes import get_agent
    try:
        get_agent()
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning("Agent warm-up failed (will retry on first request): %s", e)
    yield


app = FastAPI(
    title="Expert Network Search Copilot",
    description=(
        "Hybrid expert search API combining Pinecone vector search with a "
        "LangGraph Text-to-SQL agent. Supports natural language queries, "
        "multi-signal re-ranking, explainable scoring, and conversational follow-ups."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)
