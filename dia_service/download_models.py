"""Download Dia2 assets to Modal Volume.

Run before first deployment:
    modal run dia_service/download_models.py

This stores Hugging Face assets in the dia2-models-v1 Modal Volume so runtime
containers do not redownload model weights on cold start.
"""

import os
import sys

import modal

from dia_service.constants import DIA_MODELS, MIMI_LOCAL_DIR, MIMI_MODEL_ID

app = modal.App("dia2-download-models")

volume = modal.Volume.from_name("dia2-models-v1", create_if_missing=True)
huggingface_secret = modal.Secret.from_name(
    "huggingface-secret",
    required_keys=["HF_TOKEN"],
)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .run_commands("apt-get update && apt-get install -y ffmpeg git libsndfile1")
    .run_commands("git clone --depth 1 https://github.com/nari-labs/dia2.git /opt/dia2")
    .env({"HF_HUB_DISABLE_XET": "1", "HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .pip_install(
        "torch==2.8.0",
        "torchaudio==2.8.0",
        "numpy>=2.1.0,<2.4",
        "huggingface-hub>=0.24.7",
        "hf-transfer>=0.1.9",
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
    secrets=[huggingface_secret],
    timeout=1800,
)
def download_models():
    """Download Dia2-1B, Dia2-2B, and shared Mimi assets."""
    os.environ["HF_HOME"] = "/models/dia2/hf_cache"
    os.environ["HF_HUB_CACHE"] = "/models/dia2/hf_cache"
    os.environ["TORCH_HOME"] = "/models/dia2/hf_cache/torch"
    if "/opt/dia2" not in sys.path:
        sys.path.insert(0, "/opt/dia2")

    from huggingface_hub import snapshot_download

    downloaded_models = []
    for model_size, model_info in DIA_MODELS.items():
        print(f"Downloading Dia2 {model_size} assets from {model_info['id']}...")
        snapshot_download(
            repo_id=model_info["id"],
            cache_dir="/models/dia2/hf_cache",
            local_dir=model_info["local_dir"],
            token=os.environ["HF_TOKEN"],
            max_workers=4,
        )
        downloaded_models.append(
            {
                "model_size": model_size,
                "model": model_info["id"],
                "local_dir": model_info["local_dir"],
            }
        )

    print(f"Downloading Mimi codec assets from {MIMI_MODEL_ID}...")
    snapshot_download(
        repo_id=MIMI_MODEL_ID,
        cache_dir="/models/dia2/hf_cache",
        local_dir=MIMI_LOCAL_DIR,
        token=os.environ["HF_TOKEN"],
        max_workers=1,
    )
    print("Dia2 model files cached. Runtime initialization is deferred to modal serve/deploy.")

    volume.commit()
    return {
        "success": True,
        "models": downloaded_models,
        "mimi": MIMI_MODEL_ID,
        "cache_dir": "/models/dia2/hf_cache",
        "mimi_local_dir": MIMI_LOCAL_DIR,
    }


@app.local_entrypoint()
def main():
    """Local entrypoint for model download."""
    result = download_models.remote()
    print("Dia2 model download complete.")
    print(result)
