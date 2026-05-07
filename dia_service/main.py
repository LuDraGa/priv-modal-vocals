"""Modal app entry point for Dia2 TTS API.

This service is intentionally independent from Coqui, WhisperX, and OpenVoice:
- expressive single-speaker and dialogue TTS
- one-shot prefix-audio conditioning
- reusable voice profile registry

V1 is batch WAV only. Realtime streaming and true voice conversion are deferred.
"""

import modal

app = modal.App("dia2-tts-api")

volume = modal.Volume.from_name("dia2-models-v1", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .run_commands("apt-get update && apt-get install -y ffmpeg git libsndfile1")
    .run_commands("git clone --depth 1 https://github.com/nari-labs/dia2.git /opt/dia2")
    .pip_install(
        "torch==2.8.0",
        "torchaudio==2.8.0",
        "numpy>=2.1.0,<2.4",
        "fastapi[standard]>=0.108.0",
        "pydantic>=2.5.0",
        "structlog>=25.5.0",
        "python-multipart",
        "huggingface-hub>=0.24.7",
        "soundfile>=0.12.1",
        "transformers>=4.55.3",
        "safetensors==0.5.3",
        "sphn>=0.2.0",
        "whisper-timestamped>=1.14.2",
    )
    .add_local_python_source("dia_service")
    .add_local_python_source("shared")
)

_dia_engine = None
_profile_store = None


class LazyDiaEngine:
    """Proxy that loads Dia2 only when generation is requested."""

    @property
    def _loaded(self):
        return _dia_engine is not None and _dia_engine._loaded

    @property
    def device(self):
        if _dia_engine is None:
            return "not_loaded"
        return _dia_engine.device or "unknown"

    def generate(self, **kwargs):
        return get_dia_engine().generate(**kwargs)


def get_dia_engine():
    """Get or initialize Dia2 engine once per container."""
    global _dia_engine
    if _dia_engine is None:
        import structlog

        from dia_service.engine import DiaEngine

        logger = structlog.get_logger()
        logger.info("dia_engine.initializing")
        _dia_engine = DiaEngine(cache_dir="/models/dia2/hf_cache")
        _dia_engine.load_model()
        logger.info("dia_engine.initialized")
    return _dia_engine


def get_profile_store():
    """Get or initialize file-backed voice profile store."""
    global _profile_store
    if _profile_store is None:
        from dia_service.profile_store import VoiceProfileStore

        _profile_store = VoiceProfileStore(root_path="/models/dia2/profiles")
    return _profile_store


@app.function(
    image=image,
    gpu="A10G",
    volumes={"/models": volume},
    min_containers=0,
    timeout=600,
    enable_memory_snapshot=True,
    scaledown_window=60,
)
@modal.asgi_app()
def fastapi_app():
    """Create and configure FastAPI application."""
    import structlog
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware

    from dia_service.routes import create_routes

    structlog.configure(
        processors=[
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer(),
        ],
    )
    logger = structlog.get_logger()

    web_app = FastAPI(
        title="Dia2 Expressive TTS API",
        description=(
            "Batch expressive TTS and dialogue generation powered by Nari Labs Dia2-1B. "
            "Includes reusable voice profiles for prefix-audio conditioning."
        ),
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    web_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=[
            "X-Sample-Rate",
            "X-Duration-Sec",
            "X-Compute-Sec",
            "X-Engine",
            "X-Model",
            "X-Mode",
            "X-Voice-Profile-Id",
        ],
    )

    @web_app.on_event("startup")
    async def startup():
        logger.info("fastapi.startup")
        try:
            get_profile_store()
            logger.info("fastapi.startup.ready")
        except Exception as e:
            logger.error("fastapi.startup.failed", error=str(e))
            raise

    create_routes(
        app=web_app,
        engine=LazyDiaEngine(),
        profile_store=get_profile_store(),
        volume=volume,
    )

    logger.info("fastapi.app_created")
    return web_app
