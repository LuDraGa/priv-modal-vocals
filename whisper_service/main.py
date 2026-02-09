"""Modal app entry point for WhisperX STT API.

This module defines the Modal application with ASGI/FastAPI endpoints for:
- Speech-to-Text transcription with word-level timestamps
- Supported languages list
- Health checks

The app uses:
- Modal Volume for fast model loading (~5-8s vs 10+ min download)
- GPU (A10G with 24GB VRAM) for inference
- Memory snapshotting for faster cold starts
- WhisperX large-v3-turbo (6x faster than large-v3)
- Wav2Vec2 forced alignment for frame-accurate word timestamps
"""

import modal

# ============================================================================
# Modal Configuration
# ============================================================================

# Create Modal app
app = modal.App("whisperx-apis")

# Create/reference Modal Volume for model storage
volume = modal.Volume.from_name("whisperx-models-v1", create_if_missing=True)

# Define container image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "structlog>=25.5.0",
        "fastapi[standard]>=0.108.0",
        "pydantic>=2.5.0",
        "python-multipart",  # For file uploads
    )
    .run_commands("apt-get update && apt-get install -y ffmpeg git")
    # Install WhisperX from official GitHub repo (fixes yanked PyPI version issue)
    .pip_install("git+https://github.com/m-bain/whisperX.git@v3.7.6")
    .add_local_python_source("whisper_service")  # Mount whisper_service package
)

# ============================================================================
# Modal ASGI App (FastAPI)
# ============================================================================

@app.function(
    image=image,
    gpu="A10G",  # 24GB VRAM
    volumes={"/models": volume},  # Mount Volume at /models
    min_containers=0,  # Scale to zero when idle
    timeout=600,  # 10 min max per request
    enable_memory_snapshot=True,  # Faster cold starts
    scaledown_window=120,  # Keep alive for 2min
)
@modal.asgi_app()
def fastapi_app():
    """Create and configure FastAPI application.

    This function is called once per container startup.
    The returned FastAPI app handles all HTTP requests.
    """
    from fastapi import FastAPI
    import structlog
    import os
    from whisper_service.routes import create_routes
    from whisper_service.engine import WhisperXEngine

    # Set HuggingFace cache to volume path
    os.environ["HF_HUB_CACHE"] = "/models/hf_cache"
    os.environ["HF_HOME"] = "/models/hf_cache"

    # Configure structlog
    structlog.configure(
        processors=[
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer(),
        ],
    )
    logger = structlog.get_logger()

    # Create FastAPI app
    web_app = FastAPI(
        title="WhisperX STT API",
        description=(
            "Speech-to-Text API with word-level timestamps for karaoke-style highlighting. "
            "Powered by WhisperX large-v3-turbo with Wav2Vec2 forced alignment."
        ),
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # Initialize engine (global per container)
    engine = WhisperXEngine(cache_dir="/models/hf_cache")

    # Startup event: Pre-load models
    @web_app.on_event("startup")
    async def startup():
        """Pre-load WhisperX models on container startup."""
        logger.info("fastapi.startup")
        try:
            engine.load_models()
            logger.info(
                "fastapi.startup.models_loaded",
                gpu_available=engine.is_gpu_available(),
                alignment_available=engine.is_alignment_available(),
            )
        except Exception as e:
            logger.error("fastapi.startup.failed", error=str(e))
            raise

    # Register routes
    create_routes(app=web_app, engine=engine)

    logger.info("fastapi.app_created")
    return web_app
