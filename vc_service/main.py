"""Modal app entry point for OpenVoice v2 Voice Conversion API.

This module defines the Modal application with ASGI/FastAPI endpoints for:
- Voice conversion: convert source audio into a target voice (POST /voice-convert)
- Health check (GET /health)

The app uses:
- Modal Volume v2 for fast model loading (openvoice-models-v1)
- GPU (T4) for inference
- Memory snapshotting for faster cold starts
- Separate deployment from coqui_service (independent scaling)
"""

import modal

# ============================================================================
# Modal Configuration
# ============================================================================

app = modal.App("openvoice-vc-api")

volume = modal.Volume.from_name("openvoice-models-v1", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.4.1",
        "torchaudio==2.4.1",
        "transformers<5.0",
        "coqui-tts[codec]>=0.27.3",
        "structlog>=25.5.0",
        "numpy<2.4",
        "fastapi[standard]>=0.108.0",
        "pydantic>=2.5.0",
        "python-multipart",
    )
    .run_commands("apt-get update && apt-get install -y ffmpeg")
    .add_local_python_source("vc_service")
    .add_local_python_source("shared")
)

# ============================================================================
# Global State (Loaded Once Per Container)
# ============================================================================

_vc_engine = None


def get_vc_engine():
    """Get or initialize VCEngine (singleton per container)."""
    global _vc_engine
    if _vc_engine is None:
        from vc_service.engine import VCEngine
        import structlog

        logger = structlog.get_logger()
        logger.info("vc_engine.initializing")
        _vc_engine = VCEngine(model_path="/models/openvoice")
        _vc_engine.load_model()
        logger.info("vc_engine.initialized")

    return _vc_engine


# ============================================================================
# Modal ASGI App
# ============================================================================

@app.function(
    image=image,
    gpu="T4",
    volumes={"/models": volume},
    min_containers=0,
    timeout=300,
    enable_memory_snapshot=True,
    scaledown_window=120,
)
@modal.asgi_app()
def fastapi_app():
    """Create and configure FastAPI application.

    Called once per container startup. Returns the FastAPI app
    that handles all HTTP requests for the lifetime of the container.
    """
    from fastapi import FastAPI
    import structlog
    from vc_service.routes import create_routes

    structlog.configure(
        processors=[
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer(),
        ],
    )
    logger = structlog.get_logger()

    web_app = FastAPI(
        title="OpenVoice v2 Voice Conversion API",
        description=(
            "Convert any audio to a target voice using OpenVoice v2. "
            "Preserves content, emotion, rhythm, and speaking style — only timbre changes."
        ),
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # CORS middleware
    from fastapi.middleware.cors import CORSMiddleware
    web_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=[
            "X-Sample-Rate", "X-Duration-Sec", "X-Engine", "X-Mode",
            "X-Reference-Count", "X-Validation-Warnings",
        ],
    )

    @web_app.on_event("startup")
    async def startup():
        logger.info("fastapi.startup")
        try:
            get_vc_engine()
            logger.info("fastapi.startup.engine_loaded")
        except Exception as e:
            logger.error("fastapi.startup.failed", error=str(e))
            raise

    create_routes(app=web_app, engine=get_vc_engine())

    logger.info("fastapi.app_created")
    return web_app
