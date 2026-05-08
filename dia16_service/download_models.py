"""Download Dia 1.6B assets to Modal Volume.

Run before first deployment:
    modal run dia16_service/download_models.py
"""

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import modal

from dia16_service.constants import (
    DIA16_DAC_LOCAL_DIR,
    DIA16_DAC_MODEL_ID,
    DIA16_LOCAL_DIR,
    DIA16_MODEL_ID,
    DIA16_PREDEFINED_VOICES_DIR,
    DIA16_PREDEFINED_VOICES_REPO,
)

app = modal.App("dia16-download-models")

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
        "huggingface-hub>=0.24.7",
        "hf-transfer>=0.1.9",
        "soundfile>=0.12.1",
        "transformers>=4.56.0",
        "accelerate>=1.0.0",
        "safetensors>=0.5.3",
    )
    .add_local_python_source("dia16_service")
)


@app.function(
    image=image,
    volumes={"/models": volume},
    secrets=[huggingface_secret],
    timeout=1800,
)
def download_models():
    """Download Dia 1.6B assets."""
    os.environ["HF_HOME"] = "/models/dia16/hf_cache"
    os.environ["HF_HUB_CACHE"] = "/models/dia16/hf_cache"
    os.environ["TORCH_HOME"] = "/models/dia16/hf_cache/torch"

    from huggingface_hub import snapshot_download

    print(f"Downloading Dia 1.6B assets from {DIA16_MODEL_ID}...")
    snapshot_download(
        repo_id=DIA16_MODEL_ID,
        cache_dir="/models/dia16/hf_cache",
        local_dir=DIA16_LOCAL_DIR,
        token=os.environ["HF_TOKEN"],
        max_workers=4,
    )
    print(f"Downloading Dia 1.6B DAC assets from {DIA16_DAC_MODEL_ID}...")
    snapshot_download(
        repo_id=DIA16_DAC_MODEL_ID,
        cache_dir="/models/dia16/hf_cache",
        local_dir=DIA16_DAC_LOCAL_DIR,
        token=os.environ["HF_TOKEN"],
        max_workers=2,
    )

    audio_tokenizer_config = Path(DIA16_LOCAL_DIR) / "audio_tokenizer_config.json"
    config = json.loads(audio_tokenizer_config.read_text())
    config["audio_tokenizer_name_or_path"] = DIA16_DAC_LOCAL_DIR
    audio_tokenizer_config.write_text(json.dumps(config, indent=2))

    predefined_count = _download_predefined_voices()

    volume.commit()
    return {
        "success": True,
        "model": DIA16_MODEL_ID,
        "dac": DIA16_DAC_MODEL_ID,
        "predefined_voice_count": predefined_count,
        "cache_dir": "/models/dia16/hf_cache",
        "local_dir": DIA16_LOCAL_DIR,
        "dac_local_dir": DIA16_DAC_LOCAL_DIR,
        "predefined_voices_dir": DIA16_PREDEFINED_VOICES_DIR,
    }


def _download_predefined_voices() -> int:
    """Download third-party predefined voice prompt WAV/TXT pairs."""
    target_dir = Path(DIA16_PREDEFINED_VOICES_DIR)
    target_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir) / "Dia-TTS-Server"
        print(f"Downloading predefined Dia16 voices from {DIA16_PREDEFINED_VOICES_REPO}...")
        subprocess.run(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "--filter=blob:none",
                "--sparse",
                DIA16_PREDEFINED_VOICES_REPO,
                str(repo_dir),
            ],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(repo_dir), "sparse-checkout", "set", "voices"],
            check=True,
        )

        source_dir = repo_dir / "voices"
        if not source_dir.exists():
            print("No predefined voices directory found.")
            return 0

        for existing_file in target_dir.glob("*"):
            if existing_file.is_file():
                existing_file.unlink()

        copied = 0
        for source_file in source_dir.iterdir():
            if source_file.suffix.lower() not in {".wav", ".txt"}:
                continue
            shutil.copy2(source_file, target_dir / source_file.name)
            if source_file.suffix.lower() == ".wav":
                copied += 1

    print(f"Predefined Dia16 voices copied: {copied}")
    return copied


@app.local_entrypoint()
def main():
    """Local entrypoint for model download."""
    result = download_models.remote()
    print("Dia 1.6B model download complete.")
    print(result)
