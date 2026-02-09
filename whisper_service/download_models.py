"""Download WhisperX models to Modal Volume.

This script should be run once to download the required models to the Modal Volume:
- WhisperX large-v3-turbo model (~3GB)
- Wav2Vec2 alignment model (~1.2GB)
- VAD (Voice Activity Detection) model (~300MB)

Usage:
    modal run whisper_service/download_models.py
"""

import os
import modal

# Create Modal app for model download
app = modal.App("whisperx-model-download")

# Reference the same volume as production
volume = modal.Volume.from_name("whisperx-models-v1", create_if_missing=True)

# Image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "structlog>=25.5.0",
    )
    .run_commands("apt-get update && apt-get install -y ffmpeg git")
    # Install WhisperX from official GitHub repo (fixes yanked PyPI version issue)
    .pip_install("git+https://github.com/m-bain/whisperX.git@v3.7.6")
)


@app.function(
    image=image,
    volumes={"/models": volume},
    timeout=1800,  # 30 minutes (model downloads can be slow)
    gpu="A10G",  # Use GPU to test model loading
)
def download_whisperx_models():
    """Download WhisperX models to Modal Volume.

    Downloads:
    1. WhisperX large-v3-turbo model (~3GB)
    2. Wav2Vec2 alignment model for English (~1.2GB)
    3. VAD model (~300MB)

    Note: Modal automatically commits volume changes in background (2025 update).
    No need to call volume.commit() manually.
    """
    import structlog
    import torch

    # Workaround for PyTorch 2.8+ weights_only security
    # PyTorch 2.6+ changed default from weights_only=False to True
    # pyannote/whisperx models use omegaconf in checkpoints, which isn't whitelisted
    # Since these are trusted model sources (HuggingFace), we force weights_only=False
    _original_torch_load = torch.load

    def _patched_torch_load(*args, **kwargs):
        """Patched torch.load that forces weights_only=False for trusted models."""
        # Force False even if explicitly passed by libraries (lightning_fabric, etc.)
        kwargs['weights_only'] = False
        return _original_torch_load(*args, **kwargs)

    # Apply monkey patch globally
    torch.load = _patched_torch_load

    import whisperx

    logger = structlog.get_logger()

    # Set cache directories to Volume paths
    # HuggingFace hub cache (for alignment models)
    os.environ["HF_HUB_CACHE"] = "/models/hf_cache"
    os.environ["HF_HOME"] = "/models/hf_cache"
    # Torch cache
    os.environ["TORCH_HOME"] = "/models/torch_cache"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"

    logger.info(
        "model_download.start",
        device=device,
        compute_type=compute_type,
    )

    # ========================================================================
    # 1. Download WhisperX large-v3-turbo model
    # ========================================================================
    logger.info("model_download.whisperx.start", model="large-v3-turbo")

    try:
        # WhisperX will download to HF_HOME
        model = whisperx.load_model(
            "large-v3-turbo",
            device=device,
            compute_type=compute_type,
        )
        logger.info(
            "model_download.whisperx.complete",
            model="large-v3-turbo",
            device=device,
        )
    except Exception as e:
        logger.error("model_download.whisperx.failed", error=str(e))
        raise

    # ========================================================================
    # 2. Download Wav2Vec2 alignment model (English)
    # ========================================================================
    logger.info("model_download.alignment.start", language="en")

    try:
        # Load alignment model for English (downloads from HuggingFace)
        alignment_model, metadata = whisperx.load_align_model(
            language_code="en",
            device=device,
        )
        logger.info(
            "model_download.alignment.complete",
            language="en",
            metadata=metadata,
        )
    except Exception as e:
        logger.error("model_download.alignment.failed", error=str(e))
        raise

    # ========================================================================
    # Note: Volume changes are auto-committed in background (Modal 2025)
    # ========================================================================
    logger.info("model_download.complete_auto_commit")

    return {
        "success": True,
        "whisper_model": "large-v3-turbo",
        "alignment_model": "wav2vec2 (en)",
        "device": device,
        "cache_path": "/models/hf_cache",
    }


@app.local_entrypoint()
def main():
    """Local entrypoint for running the download."""
    print("=" * 70)
    print("Downloading WhisperX Models to Modal Volume")
    print("=" * 70)
    print()
    print("This will download the following models (~4GB total):")
    print("  1. WhisperX large-v3-turbo (~3GB)")
    print("  2. Wav2Vec2 alignment model (~1.2GB)")
    print()
    print("Volume: whisperx-models-v1")
    print("GPU: A10G (24GB VRAM)")
    print()
    print("Note: Modal auto-commits volume changes in background (no manual commit needed)")
    print("This may take 10-20 minutes depending on network speed...")
    print()

    result = download_whisperx_models.remote()

    print()
    print("=" * 70)
    print("Model Download Complete!")
    print("=" * 70)
    print(f"Whisper Model: {result['whisper_model']}")
    print(f"Alignment Model: {result['alignment_model']}")
    print(f"Device: {result['device']}")
    print(f"Cache Path: {result['cache_path']}")
    print()
    print("Models are cached in Volume and will auto-commit in background.")
    print()
    print("Next steps:")
    print("  1. Test locally:  source .venv/bin/activate && python3 -m modal serve whisper_service/main.py")
    print("  2. Deploy:        source .venv/bin/activate && python3 -m modal deploy whisper_service/main.py")
    print()
