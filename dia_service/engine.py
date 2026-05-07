"""Dia2 engine wrapper.

This isolates the Dia2 runtime API from FastAPI route code and keeps generation
metadata consistent across endpoints.
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import structlog

from dia_service.constants import DIA_MODEL_ID, DIA_MODEL_NAME, MIMI_MODEL_ID

logger = structlog.get_logger()


@dataclass
class DiaGenerationResult:
    """Normalized Dia2 generation output."""

    pcm_bytes: bytes
    sample_rate: int
    duration_sec: float
    compute_sec: float


class DiaEngine:
    """Nari Labs Dia2 inference wrapper."""

    def __init__(
        self,
        *,
        cache_dir: str = "/models/dia2/hf_cache",
        repo_id: str = DIA_MODEL_ID,
        device: Optional[str] = None,
        dtype: str = "bfloat16",
    ):
        self.cache_dir = Path(cache_dir)
        self.repo_id = repo_id
        self.device = device
        self.dtype = dtype
        self.model = None
        self.sample_rate = 24000
        self._loaded = False
        self._asset_snapshot = None

    def load_model(self) -> None:
        """Load Dia2 from Hugging Face cache / Modal Volume."""
        if self._loaded:
            logger.info("dia_engine.already_loaded")
            return

        os.environ["HF_HOME"] = str(self.cache_dir)
        os.environ["HF_HUB_CACHE"] = str(self.cache_dir)
        os.environ["TORCH_HOME"] = str(self.cache_dir / "torch")
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        if "/opt/dia2" not in sys.path:
            sys.path.insert(0, "/opt/dia2")

        try:
            import torch
            from dia2 import Dia2

            from dia_service.compat import patch_torch_cudnn_conv

            patch_torch_cudnn_conv()

            runtime_device = self.device
            if runtime_device is None:
                runtime_device = "cuda" if torch.cuda.is_available() else "cpu"

            snapshot_path = self._find_cached_snapshot()
            mimi_path = self._find_cached_snapshot(MIMI_MODEL_ID)
            config_path = snapshot_path / "config.json"
            weights_path = snapshot_path / "model.safetensors"

            logger.info(
                "dia_engine.loading",
                repo_id=self.repo_id,
                asset_snapshot=str(snapshot_path),
                mimi_snapshot=str(mimi_path),
                device=runtime_device,
                dtype=self.dtype,
                cache_dir=str(self.cache_dir),
            )

            self.model = Dia2.from_local(
                config_path=config_path,
                weights_path=weights_path,
                tokenizer_id=str(snapshot_path),
                mimi_id=str(mimi_path),
                device=runtime_device,
                dtype=self.dtype,
            )
            self._asset_snapshot = snapshot_path
            self.device = runtime_device
            self.sample_rate = int(self.model.sample_rate)
            self._loaded = True

            logger.info(
                "dia_engine.loaded",
                model=DIA_MODEL_NAME,
                sample_rate=self.sample_rate,
                device=self.device,
            )

        except Exception as e:
            logger.error("dia_engine.load_failed", error=str(e))
            raise RuntimeError(f"Failed to load Dia2 model: {e}") from e

    def _find_cached_snapshot(self, repo_id: str = DIA_MODEL_ID) -> Path:
        """Return the local Hugging Face snapshot path for cached repo assets."""
        repo_cache_dir = self.cache_dir / f"models--{repo_id.replace('/', '--')}"
        snapshots_dir = repo_cache_dir / "snapshots"
        if not snapshots_dir.exists():
            raise FileNotFoundError(
                f"Cache missing for {repo_id} at {snapshots_dir}. "
                "Run modal run dia_service/download_models.py"
            )

        candidates = sorted(
            snapshots_dir.iterdir(),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        for snapshot in candidates:
            if repo_id == DIA_MODEL_ID:
                if (snapshot / "config.json").exists() and (snapshot / "model.safetensors").exists():
                    return snapshot
            elif any(snapshot.iterdir()):
                return snapshot

        raise FileNotFoundError(
            f"No complete cached snapshot for {repo_id} under {snapshots_dir}"
        )

    def generate(
        self,
        *,
        script: str,
        temperature: float = 0.8,
        top_k: int = 50,
        cfg_scale: float = 2.0,
        seed: Optional[int] = None,
        prefix_speaker_1: Optional[str] = None,
        prefix_speaker_2: Optional[str] = None,
    ) -> DiaGenerationResult:
        """Generate speech from a Dia2 script and optional prefix audio."""
        if not self._loaded or self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        try:
            import torch
            from dia2 import GenerationConfig, SamplingConfig

            if seed is not None:
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)

            config = GenerationConfig(
                cfg_scale=cfg_scale,
                audio=SamplingConfig(temperature=temperature, top_k=top_k),
                text=SamplingConfig(temperature=temperature, top_k=top_k),
                use_cuda_graph=False,
                use_torch_compile=False,
            )

            logger.info(
                "dia_engine.generate.start",
                text_len=len(script),
                has_prefix_s1=bool(prefix_speaker_1),
                has_prefix_s2=bool(prefix_speaker_2),
                temperature=temperature,
                top_k=top_k,
                cfg_scale=cfg_scale,
                seed=seed,
            )

            start = time.monotonic()
            result = self.model.generate(
                script,
                config=config,
                prefix_speaker_1=prefix_speaker_1,
                prefix_speaker_2=prefix_speaker_2,
                include_prefix=False,
                verbose=False,
            )
            compute_sec = time.monotonic() - start

            sample_rate = int(getattr(result, "sample_rate", self.sample_rate))
            pcm_bytes = self._to_pcm16(result.waveform)
            duration_sec = len(pcm_bytes) / (sample_rate * 2)

            logger.info(
                "dia_engine.generate.complete",
                audio_size=len(pcm_bytes),
                duration_sec=duration_sec,
                compute_sec=compute_sec,
            )

            return DiaGenerationResult(
                pcm_bytes=pcm_bytes,
                sample_rate=sample_rate,
                duration_sec=duration_sec,
                compute_sec=compute_sec,
            )

        except Exception as e:
            logger.error("dia_engine.generate.failed", error=str(e))
            raise RuntimeError(f"Dia2 generation failed: {e}") from e

    def _to_pcm16(self, waveform) -> bytes:
        """Convert a waveform tensor/array to mono PCM16 bytes."""
        if hasattr(waveform, "detach"):
            waveform = waveform.detach().cpu().numpy()

        audio_array = np.asarray(waveform)
        if audio_array.ndim > 1:
            audio_array = audio_array.reshape(-1)

        audio_array = np.clip(audio_array, -1.0, 1.0)
        pcm = (audio_array * 32767).astype(np.int16)
        return pcm.tobytes()
