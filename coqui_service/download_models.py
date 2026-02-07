"""Download Coqui XTTS v2 models to Modal Volume.

This script should be run once to download the XTTS v2 model (1.8GB)
to the Modal Volume for fast loading in production containers.

Usage:
    modal run coqui_service/download_models.py
"""

import os
import modal

# Create Modal app for model download
app = modal.App("coqui-model-download")

# Reference the same volume as production
volume = modal.Volume.from_name("coqui-models-v2", create_if_missing=True)

# Same image as production (includes TTS dependencies)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch>=2.1.0",
        "torchaudio>=2.1.0",
        "transformers<5.0",  # Pin to 4.x for coqui-tts compatibility
        "coqui-tts[codec]>=0.27.3",  # Include codec extra for PyTorch 2.9+ audio I/O
        "structlog>=25.5.0",
        "numpy<2.4",
    )
    .run_commands("apt-get update && apt-get install -y ffmpeg")
)


@app.function(
    image=image,
    volumes={"/models": volume},
    timeout=1200,  # 20 minutes (model download can be slow)
)
def download_xtts_v2():
    """Download XTTS v2 model to Modal Volume.

    This downloads ~1.8GB of model files and saves them to the Volume.
    After this completes, production containers can load the model in ~8s
    instead of downloading it every time (60s).
    """
    from TTS.api import TTS
    import structlog

    logger = structlog.get_logger()

    # Set TTS_HOME to Volume path
    os.environ["TTS_HOME"] = "/models/coqui"

    # Agree to Coqui TOS (required for non-interactive environments)
    os.environ["COQUI_TOS_AGREED"] = "1"

    logger.info("model_download.start", model="xtts_v2", path="/models/coqui")

    try:
        # Download XTTS v2 model
        # This will download to /models/coqui/tts_models--multilingual--multi-dataset--xtts_v2/
        tts = TTS(
            model_name="tts_models/multilingual/multi-dataset/xtts_v2",
            progress_bar=True,
            gpu=False,  # CPU is fine for download
        )

        # Verify model loaded
        assert tts is not None, "TTS model failed to load"
        assert hasattr(tts, "speakers"), "TTS model has no speakers"

        speaker_count = len(tts.speakers) if tts.speakers else 0
        logger.info(
            "model_download.complete",
            speaker_count=speaker_count,
            has_model=True,
        )

        # Commit changes to Volume (persist the downloaded model)
        logger.info("model_download.committing")
        volume.commit()
        logger.info("model_download.committed")

        return {
            "success": True,
            "model": "xtts_v2",
            "speakers": speaker_count,
            "path": "/models/coqui",
        }

    except Exception as e:
        logger.error("model_download.failed", error=str(e))
        raise


@app.local_entrypoint()
def main():
    """Local entrypoint for running the download."""
    print("=" * 60)
    print("Downloading Coqui XTTS v2 model to Modal Volume")
    print("=" * 60)
    print()
    print("This will download ~1.8GB of model files.")
    print("The download will be stored in Modal Volume: coqui-models-v2")
    print()

    result = download_xtts_v2.remote()

    print()
    print("=" * 60)
    print("Model Download Complete!")
    print("=" * 60)
    print(f"Model: {result['model']}")
    print(f"Speakers: {result['speakers']}")
    print(f"Path: {result['path']}")
    print()
    print("You can now deploy the API:")
    print("  modal deploy coqui_service/main.py")
    print()
    print("Or test locally:")
    print("  modal serve coqui_service/main.py")
    print()
