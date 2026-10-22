"""
Tetrax ASR FastAPI Server
Entry point for real-time WebSocket streaming, REST transcription, and web console.
"""

from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, Response
from fastapi.middleware.cors import CORSMiddleware
import logging

from tetrax_asr.config import api_config, TETRAX_DIR
from tetrax_asr.inference.engine import TetraxWhisperEngine
from tetrax_asr.api.routes import (
    transcriptions_router,
    stream_router,
    models_router,
    health_router,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TetraxASR.API")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load the Whisper model into memory
    logger.info("Initializing Tetrax Whisper inference engine...")
    engine = TetraxWhisperEngine()
    app.state.engine = engine
    logger.info(f"Engine loaded. Base model: {engine.base_model_id}, Fine-tuned LoRA: {engine.is_finetuned}")
    yield
    # Shutdown
    logger.info("Shutting down Tetrax ASR engine...")


app = FastAPI(
    title="Tetrax ASR API",
    description="Real-Time Indian-English Speech Intelligence API for Tetrax CPaaS",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=api_config.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files & Web UI
static_dir = TETRAX_DIR / "web" / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

templates_dir = TETRAX_DIR / "web" / "templates"


@app.get("/", response_class=HTMLResponse)
async def serve_console():
    index_file = templates_dir / "index.html"
    if index_file.exists():
        return HTMLResponse(content=index_file.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>Tetrax ASR API is running.</h1><p>Visit /docs for API documentation.</p>")


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return Response(status_code=204)


# Mount API Routers
app.include_router(transcriptions_router)
app.include_router(stream_router)
app.include_router(models_router)
app.include_router(health_router)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "tetrax_asr.api.main:app",
        host=api_config.host,
        port=api_config.port,
        reload=False
    )
