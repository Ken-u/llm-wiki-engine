"""FastAPI application entry point."""

from __future__ import annotations

from contextlib import asynccontextmanager
from collections.abc import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_config
from app.database import init_db


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    await init_db()
    from app.ingest.queue import ingest_queue
    ingest_queue.start()
    recovered = await ingest_queue.recover_interrupted_jobs()
    if recovered:
        import logging
        logging.getLogger(__name__).info("Recovered %d interrupted ingest jobs", recovered)

    from app.projects.git_sync import start_sync_scheduler, stop_sync_scheduler
    start_sync_scheduler()

    yield

    stop_sync_scheduler()
    ingest_queue.stop()


app = FastAPI(
    title="LLM Wiki Engine",
    description="Compile-time knowledge synthesis engine",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from app.auth.router import router as auth_router  # noqa: E402
from app.projects.router import router as projects_router  # noqa: E402
from app.documents.router import router as documents_router  # noqa: E402
from app.ingest.router import router as ingest_router  # noqa: E402
from app.wiki.router import router as wiki_router  # noqa: E402
from app.search.router import router as search_router  # noqa: E402
from app.chat.router import router as chat_router  # noqa: E402
from app.agents.router import router as agents_router  # noqa: E402
from app.agents.public_router import router as public_agents_router  # noqa: E402
from app.admin.router import router as admin_router  # noqa: E402
from app.feedback.router import router as feedback_router  # noqa: E402

app.include_router(auth_router)
app.include_router(projects_router)
app.include_router(documents_router)
app.include_router(ingest_router)
app.include_router(wiki_router)
app.include_router(search_router)
app.include_router(chat_router)
app.include_router(agents_router)
app.include_router(public_agents_router)
app.include_router(admin_router)
app.include_router(feedback_router)


@app.get("/health")
async def health():
    return {"status": "ok"}
