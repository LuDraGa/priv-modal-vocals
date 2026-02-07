"""Modal app entry point for Coqui TTS API.

This module defines the Modal application with ASGI/FastAPI endpoints for:
- Text-to-Speech synthesis (built-in speakers)
- Speaker list retrieval (with caching)
- Voice cloning synthesis

The app uses:
- Modal Volume v2 for fast model loading (8s vs 60s download)
- GPU (T4) for inference
- Memory snapshotting for faster cold starts
- Stale-while-revalidate caching for speaker metadata
"""

import modal

# ============================================================================
# Modal Configuration
# ============================================================================

# Create Modal app
app = modal.App("coqui-apis")

# Create/reference Modal Volume for model storage
volume = modal.Volume.from_name("coqui-models-v2", create_if_missing=True)

# Define container image with all dependencies (matches download_models.py)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch>=2.1.0",
        "torchaudio>=2.1.0",
        "transformers<5.0",  # Pin to 4.x for coqui-tts compatibility
        "coqui-tts[codec]>=0.27.3",  # Include codec extra for PyTorch 2.9+ audio I/O
        "structlog>=25.5.0",
        "numpy<2.4",
        "fastapi[standard]>=0.108.0",
        "pydantic>=2.5.0",
        "python-multipart",  # For file uploads
    )
    .run_commands("apt-get update && apt-get install -y ffmpeg")  # Required by Coqui
    .add_local_python_source("coqui_service")  # Mount coqui_service package (Modal 1.0 API)
)

# ============================================================================
# Global State (Loaded Once Per Container)
# ============================================================================

# These will be initialized at container startup
_tts_engine = None
_speaker_cache = None
_logger = None


def get_logger():
    """Get logger instance (initialized in fastapi_app)."""
    return _logger


def get_tts_engine():
    """Get or initialize TTS engine (singleton per container)."""
    global _tts_engine
    if _tts_engine is None:
        from coqui_service.engine import TTSEngine
        import structlog

        logger = structlog.get_logger()
        logger.info("tts_engine.initializing")
        _tts_engine = TTSEngine(model_path="/models/coqui")
        _tts_engine.load_model()
        logger.info("tts_engine.initialized")

    return _tts_engine


def get_speaker_cache():
    """Get or initialize speaker cache (singleton per container)."""
    global _speaker_cache
    if _speaker_cache is None:
        from coqui_service.utils.speaker_cache import SpeakerCache
        import structlog

        logger = structlog.get_logger()
        logger.info("speaker_cache.initializing")
        _speaker_cache = SpeakerCache(volume_path="/models/coqui")
        logger.info("speaker_cache.initialized")

    return _speaker_cache


# ============================================================================
# Modal ASGI App
# ============================================================================

@app.function(
    image=image,
    gpu="T4",  # T4 GPU (16GB VRAM, sufficient for XTTS v2)
    volumes={"/models": volume},  # Mount Volume at /models
    min_containers=0,  # Scale to zero when idle (cost-optimized)
    timeout=300,  # 5 min max per request
    enable_memory_snapshot=True,  # Faster cold starts (8s → 3s)
    scaledown_window=120,  # Keep container alive for 2min after request
)
@modal.asgi_app()
def fastapi_app():
    """Create and configure FastAPI application.

    This function is called once per container startup.
    The returned FastAPI app handles all HTTP requests.
    """
    from fastapi import FastAPI
    import structlog
    from coqui_service.routes import create_routes

    global _logger

    # Configure structlog
    structlog.configure(
        processors=[
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer(),
        ],
    )
    logger = structlog.get_logger()
    _logger = logger

    # Create FastAPI app
    web_app = FastAPI(
        title="Coqui TTS API",
        description="Text-to-Speech and Voice Cloning API powered by Coqui XTTS v2",
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # Startup event: Pre-load model
    @web_app.on_event("startup")
    async def startup():
        """Pre-load TTS model and speaker cache on container startup."""
        logger.info("fastapi.startup")
        try:
            # Initialize engine (loads model from Volume)
            engine = get_tts_engine()
            logger.info("fastapi.startup.engine_loaded", speakers=len(engine.get_speakers()))

            # Initialize speaker cache
            cache = get_speaker_cache()
            logger.info("fastapi.startup.cache_initialized")

        except Exception as e:
            logger.error("fastapi.startup.failed", error=str(e))
            raise

    # Register routes
    create_routes(
        app=web_app,
        engine=get_tts_engine(),
        speaker_cache=get_speaker_cache(),
        volume=volume,
    )

    logger.info("fastapi.app_created")
    return web_app
