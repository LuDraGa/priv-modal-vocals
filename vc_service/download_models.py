"""Download OpenVoice v2 model to Modal Volume.

Run before first deployment:
    modal run vc_service/download_models.py

This downloads the OpenVoice v2 voice conversion model from HuggingFace
(via Coqui TTS) into the openvoice-models-v1 Modal Volume.
"""

import modal
import os

app = modal.App("openvoice-download-models")

volume = modal.Volume.from_name("openvoice-models-v1", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.4.1",
        "torchaudio==2.4.1",
        "transformers<5.0",
        "coqui-tts[codec]>=0.27.3",
        "numpy<2.4",
    )
    .run_commands("apt-get update && apt-get install -y ffmpeg")
)


@app.function(
    image=image,
    volumes={"/models": volume},
    timeout=600,
)
def download_models():
    """Download OpenVoice v2 model weights to Modal Volume."""
    os.environ["TTS_HOME"] = "/models/openvoice"

    from TTS.api import TTS

    print("Downloading OpenVoice v2 voice conversion model...")
    TTS(
        model_name="voice_conversion_models/multilingual/multi-dataset/openvoice_v2",
        progress_bar=True,
        gpu=False,
    )

    volume.commit()
    print("Download complete. Volume committed to openvoice-models-v1.")


@app.local_entrypoint()
def main():
    download_models.remote()
