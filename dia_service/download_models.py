"""Download Dia2-1B assets to Modal Volume.

Run before first deployment:
    modal run dia_service/download_models.py

This stores Hugging Face assets in the dia2-models-v1 Modal Volume so runtime
containers do not redownload model weights on cold start.
"""

import os
import sys

import modal

from dia_service.constants import DIA_MODEL_ID, MIMI_MODEL_ID

app = modal.App("dia2-download-models")

volume = modal.Volume.from_name("dia2-models-v1", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .run_commands("apt-get update && apt-get install -y ffmpeg git libsndfile1")
    .run_commands("git clone --depth 1 https://github.com/nari-labs/dia2.git /opt/dia2")
    .pip_install(
        "torch==2.8.0",
        "torchaudio==2.8.0",
        "numpy>=2.1.0,<2.4",
        "huggingface-hub>=0.24.7",
        "soundfile>=0.12.1",
        "transformers>=4.55.3",
        "safetensors==0.5.3",
        "sphn>=0.2.0",
        "whisper-timestamped>=1.14.2",
    )
    .add_local_python_source("dia_service")
)


@app.function(
    image=image,
    volumes={"/models": volume},
    timeout=1800,
)
def download_models():
    """Download Dia2-1B and initialize runtime assets."""
    os.environ["HF_HOME"] = "/models/dia2/hf_cache"
    os.environ["HF_HUB_CACHE"] = "/models/dia2/hf_cache"
    os.environ["TORCH_HOME"] = "/models/dia2/hf_cache/torch"
    if "/opt/dia2" not in sys.path:
        sys.path.insert(0, "/opt/dia2")

    from huggingface_hub import snapshot_download

    print(f"Downloading Dia2 assets from {DIA_MODEL_ID}...")
    snapshot_download(repo_id=DIA_MODEL_ID, cache_dir="/models/dia2/hf_cache")
    print(f"Downloading Mimi codec assets from {MIMI_MODEL_ID}...")
    snapshot_download(repo_id=MIMI_MODEL_ID, cache_dir="/models/dia2/hf_cache")
    print("Dia2 model files cached. Runtime initialization is deferred to modal serve/deploy.")

    volume.commit()
    return {
        "success": True,
        "model": DIA_MODEL_ID,
        "mimi": MIMI_MODEL_ID,
        "cache_dir": "/models/dia2/hf_cache",
    }


@app.local_entrypoint()
def main():
    """Local entrypoint for model download."""
    result = download_models.remote()
    print("Dia2 model download complete.")
    print(result)
