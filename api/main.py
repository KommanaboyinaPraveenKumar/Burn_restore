"""
api/main.py
===========
FastAPI application entrypoint for Grounded Document Restoration.
Serves REST API endpoints, Swagger documentation, and the interactive web workbench.
"""

import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from api.dependencies import get_restoration_engine, set_restoration_engine
from api.routes.restoration import router as restoration_router
from inference.restoration_engine import DEFAULT_CHECKPOINT, RestorationEngine

_api_dir = Path(__file__).resolve().parent
_static_dir = _api_dir / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initializes and caches the RestorationEngine during application startup."""
    print("[API Lifespan] Starting up: initializing RestorationEngine singleton...")
    checkpoint_p = DEFAULT_CHECKPOINT
    if not os.path.isfile(checkpoint_p):
        print(f"[API Lifespan] WARNING: checkpoint not found: {checkpoint_p} - serving an UNTRAINED model. "
              "Set BURN_CKPT to a trained checkpoint.")

    engine = RestorationEngine(checkpoint_path=checkpoint_p)
    set_restoration_engine(engine)
    print(f"[API Lifespan] RestorationEngine ready on device: {engine.device}")
    yield
    print("[API Lifespan] Shutting down application.")


app = FastAPI(
    title="Grounded Side-Burnt Document Text Restoration API",
    description=(
        "Research prototype: reads faded strokes at the burn edge of each severed line and "
        "suggests length-bounded endings from the surrounding text (T5 + ResNet-34)."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

# Cross-Origin Resource Sharing (CORS) for local frontend clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,  # wildcard origins must not be combined with credentials
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include REST routes
app.include_router(restoration_router)

# Mount static web assets if directory exists
if _static_dir.is_dir():
    app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")


@app.get("/", include_in_schema=False)
@app.get("/ui", include_in_schema=False)
async def serve_interactive_ui():
    """Serves the interactive research workbench single-page application."""
    index_path = _static_dir / "index.html"
    if index_path.is_file():
        return FileResponse(str(index_path))
    return JSONResponse(
        status_code=status.HTTP_404_NOT_FOUND,
        content={"detail": "Interactive frontend index.html not found."},
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Catches uncaught exceptions and formats a structured JSON response."""
    import logging
    logging.getLogger("burn_restore.api").exception("Unhandled error on %s", request.url.path)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error_type": type(exc).__name__,
            "detail": "Internal server error (see server log).",
            "path": request.url.path,
        },
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="127.0.0.1", port=8000, reload=False)

