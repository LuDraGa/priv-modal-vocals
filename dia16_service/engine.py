"""Dia 1.6B engine wrapper.

This service uses the Hugging Face Transformers Dia implementation and keeps it
isolated from the Dia2 runtime.
"""

from __future__ import annotations

import io
import json
import os
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import structlog

from dia16_service.constants import (
    DIA16_DAC_LOCAL_DIR,
    DIA16_LOCAL_DIR,
    DIA16_MODEL_ID,
    DIA16_MODEL_NAME,
)

logger = structlog.get_logger()


@dataclass
class Dia16GenerationResult:
    """Normalized Dia 1.6B generation output."""

    pcm_bytes: bytes
    sample_rate: int
    duration_sec: float
    compute_sec: float
    model_name: str


class Dia16Engine:
    """Nari Labs Dia 1.6B inference wrapper."""

    def __init__(
        self,
        *,
        cache_dir: str = "/models/dia16/hf_cache",
        local_dir: str = DIA16_LOCAL_DIR,
        device: Optional[str] = None,
        dtype: str = "bfloat16",
    ):
        self.cache_dir = Path(cache_dir)
        self.local_dir = Path(local_dir)
        self.device = device
        self.dtype = dtype
        self.processor = None
        self.model = None
        self.sample_rate = 44100
        self._loaded = False

    def load_model(self) -> None:
        """Load Dia 1.6B from preloaded Modal Volume assets."""
        if self._loaded:
            logger.info("dia16_engine.already_loaded")
            return

        os.environ["HF_HOME"] = str(self.cache_dir)
        os.environ["HF_HUB_CACHE"] = str(self.cache_dir)
        os.environ["TORCH_HOME"] = str(self.cache_dir / "torch")
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

        try:
            import torch
            from transformers import AutoProcessor, DiaForConditionalGeneration

            runtime_device = self.device
            if runtime_device is None:
                runtime_device = "cuda" if torch.cuda.is_available() else "cpu"

            if not self._is_complete_model_dir(self.local_dir):
                raise FileNotFoundError(
                    f"Dia 1.6B model files missing at {self.local_dir}. "
                    "Run modal run dia16_service/download_models.py"
                )
            self._patch_audio_tokenizer_config()

            torch_dtype = getattr(torch, self.dtype)
            logger.info(
                "dia16_engine.loading",
                repo_id=DIA16_MODEL_ID,
                local_dir=str(self.local_dir),
                device=runtime_device,
                dtype=self.dtype,
                cache_dir=str(self.cache_dir),
            )

            self.processor = AutoProcessor.from_pretrained(
                str(self.local_dir),
                local_files_only=True,
                trust_remote_code=True,
            )
            self.model = DiaForConditionalGeneration.from_pretrained(
                str(self.local_dir),
                torch_dtype=torch_dtype,
                local_files_only=True,
                trust_remote_code=True,
            ).to(runtime_device)
            self.model.eval()
            self.device = runtime_device
            self.sample_rate = int(getattr(self.processor, "sampling_rate", 44100))
            self._loaded = True

            logger.info(
                "dia16_engine.loaded",
                model=DIA16_MODEL_NAME,
                sample_rate=self.sample_rate,
                device=self.device,
            )

        except Exception as e:
            logger.error("dia16_engine.load_failed", error=str(e))
            raise RuntimeError(f"Failed to load Dia 1.6B model: {e}") from e

    def _is_complete_model_dir(self, path: Path) -> bool:
        if not path.exists():
            return False
        config_exists = (path / "config.json").exists() or (path / "preprocessor_config.json").exists()
        weights_exist = any(path.glob("*.safetensors")) or (path / "pytorch_model.bin").exists()
        return config_exists and weights_exist

    def _patch_audio_tokenizer_config(self) -> None:
        """Force the Dia processor to use the preloaded local DAC tokenizer."""
        config_path = self.local_dir / "audio_tokenizer_config.json"
        if not config_path.exists():
            return
        config = json.loads(config_path.read_text())
        if config.get("audio_tokenizer_name_or_path") == DIA16_DAC_LOCAL_DIR:
            return
        config["audio_tokenizer_name_or_path"] = DIA16_DAC_LOCAL_DIR
        config_path.write_text(json.dumps(config, indent=2))
        logger.info(
            "dia16_engine.audio_tokenizer_patched",
            dac_model_path=DIA16_DAC_LOCAL_DIR,
        )

    def generate(
        self,
        *,
        script: str,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.95,
        cfg_scale: float = 2.0,
        max_new_tokens: int = 1024,
        seed: Optional[int] = None,
        prompt_audio_path: Optional[str] = None,
        prompt_transcript: Optional[str] = None,
    ) -> Dia16GenerationResult:
        """Generate speech from text and optional Dia 1.6B audio prompt."""
        del cfg_scale  # Kept for cross-service API compatibility.

        if not self._loaded or self.model is None or self.processor is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        try:
            import torch

            if seed is not None:
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)

            generation_text = script
            prompt_audio = None
            prompt_len = None
            if prompt_audio_path:
                prompt_audio = self._load_prompt_audio(prompt_audio_path)
                generation_text = self._conditioning_text(
                    reference_transcript=prompt_transcript or "",
                    script=script,
                )

            logger.info(
                "dia16_engine.generate.start",
                text_len=len(generation_text),
                has_prompt_audio=prompt_audio is not None,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
                seed=seed,
            )

            start = time.monotonic()
            inputs = self._prepare_inputs(text=generation_text, audio=prompt_audio)
            if prompt_audio is not None:
                prompt_len = self.processor.get_audio_prompt_len(inputs["decoder_attention_mask"])

            with torch.inference_mode():
                outputs = self.model.generate(
                    **inputs,
                    do_sample=True,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    max_new_tokens=max_new_tokens,
                )
            compute_sec = time.monotonic() - start

            audio_array = self._decode_outputs(outputs, audio_prompt_len=prompt_len)
            pcm_bytes = self._to_pcm16(audio_array)
            duration_sec = len(pcm_bytes) / (self.sample_rate * 2)

            logger.info(
                "dia16_engine.generate.complete",
                audio_size=len(pcm_bytes),
                duration_sec=duration_sec,
                compute_sec=compute_sec,
            )

            return Dia16GenerationResult(
                pcm_bytes=pcm_bytes,
                sample_rate=self.sample_rate,
                duration_sec=duration_sec,
                compute_sec=compute_sec,
                model_name=DIA16_MODEL_NAME,
            )

        except Exception as e:
            logger.error("dia16_engine.generate.failed", error=str(e))
            raise RuntimeError(f"Dia 1.6B generation failed: {e}") from e

    def _prepare_inputs(self, *, text: str, audio: Optional[np.ndarray]):
        kwargs = {"text": [text], "padding": True, "return_tensors": "pt"}
        if audio is not None:
            kwargs["audio"] = audio
        inputs = self.processor(**kwargs)
        return {key: value.to(self.device) if hasattr(value, "to") else value for key, value in inputs.items()}

    def _decode_outputs(self, outputs, *, audio_prompt_len):
        if audio_prompt_len is None:
            decoded = self.processor.batch_decode(outputs)
        else:
            decoded = self.processor.batch_decode(outputs, audio_prompt_len=audio_prompt_len)

        if isinstance(decoded, (list, tuple)):
            decoded = decoded[0]
        if isinstance(decoded, dict):
            for key in ("audio", "array", "waveform"):
                if key in decoded:
                    decoded = decoded[key]
                    break
        return decoded

    def _load_prompt_audio(self, audio_path: str) -> np.ndarray:
        import soundfile as sf
        import torch
        import torchaudio.functional as ta_functional

        audio, sample_rate = sf.read(audio_path, dtype="float32")
        audio_array = np.asarray(audio)
        if audio_array.ndim > 1:
            audio_array = audio_array.mean(axis=1)
        if sample_rate != self.sample_rate:
            audio_tensor = torch.from_numpy(audio_array)
            audio_tensor = ta_functional.resample(
                audio_tensor,
                orig_freq=sample_rate,
                new_freq=self.sample_rate,
            )
            audio_array = audio_tensor.numpy()
        return audio_array.astype(np.float32)

    def _conditioning_text(self, *, reference_transcript: str, script: str) -> str:
        reference = reference_transcript.strip()
        if reference and not reference.startswith("[S1]") and not reference.startswith("[S2]"):
            reference = f"[S1] {reference}"
        if not reference:
            return script
        return f"{reference} {script.strip()}"

    def _to_pcm16(self, waveform) -> bytes:
        if hasattr(waveform, "detach"):
            waveform = waveform.detach().cpu().numpy()

        audio_array = np.asarray(waveform)
        if audio_array.ndim > 1:
            audio_array = audio_array.reshape(-1)
        if audio_array.size == 0:
            return b""

        if np.issubdtype(audio_array.dtype, np.integer):
            max_value = np.iinfo(audio_array.dtype).max
            audio_array = audio_array.astype(np.float32) / max_value
        audio_array = np.nan_to_num(audio_array.astype(np.float32))
        audio_array = np.clip(audio_array, -1.0, 1.0)
        return (audio_array * 32767.0).astype("<i2").tobytes()

    def pcm_to_wav_bytes(self, pcm_bytes: bytes) -> bytes:
        """Debug helper to wrap generated PCM into WAV bytes."""
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(self.sample_rate)
            wav_file.writeframes(pcm_bytes)
        return buffer.getvalue()
