"""Modal app entry point for Dia 1.6B high-fidelity TTS API."""

import modal

app = modal.App("dia16-tts-api")

volume = modal.Volume.from_name("dia16-models-v1", create_if_missing=True)
huggingface_secret = modal.Secret.from_name(
    "huggingface-secret",
    required_keys=["HF_TOKEN"],
)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .run_commands("apt-get update && apt-get install -y ffmpeg git libsndfile1")
    .env({"HF_HUB_DISABLE_XET": "1", "HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .pip_install(
        "torch==2.8.0",
        "torchaudio==2.8.0",
        "numpy>=2.1.0,<2.4",
        "fastapi[standard]>=0.108.0",
        "pydantic>=2.5.0",
        "structlog>=25.5.0",
        "python-multipart",
        "huggingface-hub>=0.24.7",
        "hf-transfer>=0.1.9",
        "soundfile>=0.12.1",
        "transformers>=4.56.0",
        "accelerate>=1.0.0",
        "safetensors>=0.5.3",
    )
    .add_local_python_source("dia16_service")
    .add_local_python_source("shared")
)

_dia16_engine = None
_profile_store = None


class LazyDia16Engine:
    """Proxy that loads Dia 1.6B only when generation is requested."""

    @property
    def _loaded(self):
        return _dia16_engine is not None and _dia16_engine._loaded

    @property
    def device(self):
        if _dia16_engine is None:
            return "not_loaded"
        return _dia16_engine.device or "unknown"

    def generate(self, **kwargs):
        return get_dia16_engine().generate(**kwargs)


def get_dia16_engine():
    """Get or initialize Dia16 engine once per container."""
    global _dia16_engine
    if _dia16_engine is None:
        import structlog

        from dia16_service.engine import Dia16Engine

        logger = structlog.get_logger()
        logger.info("dia16_engine.initializing")
        _dia16_engine = Dia16Engine(cache_dir="/models/dia16/hf_cache")
        _dia16_engine.load_model()
        logger.info("dia16_engine.initialized")
    return _dia16_engine


def get_profile_store():
    """Get or initialize file-backed voice profile store."""
    global _profile_store
    if _profile_store is None:
        from dia16_service.profile_store import VoiceProfileStore

        _profile_store = VoiceProfileStore(root_path="/models/dia16/profiles")
    return _profile_store


@app.function(
    image=image,
    gpu="L40S",
    volumes={"/models": volume},
    secrets=[huggingface_secret],
    min_containers=0,
    timeout=900,
    enable_memory_snapshot=True,
    scaledown_window=60,
)
@modal.asgi_app()
def fastapi_app():
    """Create and configure FastAPI application."""
    import structlog
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware

    from dia16_service.routes import create_routes

    structlog.configure(
        processors=[
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer(),
        ],
    )
    logger = structlog.get_logger()

    web_app = FastAPI(
        title="Dia 1.6B High-Fidelity TTS API",
        description=(
            "Batch high-fidelity TTS and dialogue generation powered by Nari Labs "
            "Dia-1.6B-0626. Includes reusable audio-prompt voice profiles."
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
            "X-Predefined-Voice-Id",
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
        engine=LazyDia16Engine(),
        profile_store=get_profile_store(),
        volume=volume,
    )

    logger.info("fastapi.app_created")
    return web_app
