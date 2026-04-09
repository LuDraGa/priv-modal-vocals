"""Voice conversion engine wrapper for OpenVoice v2.

Handles model loading from Modal Volume and voice conversion.
Converts source audio to match a target voice while preserving
content, emotion, rhythm, and speaking style.
"""

import os
import tempfile
import wave
from pathlib import Path
from typing import List
import numpy as np
import structlog

logger = structlog.get_logger()


class VCEngine:
    """OpenVoice v2 voice conversion engine wrapper."""

    def __init__(self, model_path: str = "/models/openvoice"):
        self.model_path = Path(model_path)
        self.tts = None
        # OpenVoice v2 outputs at 24kHz
        self.sample_rate = 24000
        self._loaded = False

    def load_model(self) -> None:
        """Load OpenVoice v2 model from Modal Volume.

        Raises:
            RuntimeError: If model loading fails
        """
        if self._loaded:
            logger.info("vc_engine.already_loaded")
            return

        try:
            from TTS.api import TTS

            os.environ["TTS_HOME"] = str(self.model_path)

            logger.info("vc_engine.loading", model_path=str(self.model_path))

            self.tts = TTS(
                model_name="voice_conversion_models/multilingual/multi-dataset/openvoice_v2",
                progress_bar=False,
                gpu=True,
            )

            self._loaded = True
            logger.info("vc_engine.loaded", sample_rate=self.sample_rate)

        except Exception as e:
            logger.error("vc_engine.load_failed", error=str(e))
            raise RuntimeError(f"Failed to load VC model: {e}") from e

    def convert(
        self,
        source_audio_bytes: bytes,
        reference_audio_bytes: List[bytes],
    ) -> bytes:
        """Convert voice in source audio to match the reference voice.

        Preserves content, emotion, rhythm, and speaking style from source.
        Only the voice timbre/tone is changed to match the reference.

        Args:
            source_audio_bytes: The audio to convert (WAV/MP3/M4A)
            reference_audio_bytes: 1-3 reference audio files of the target voice.
                                   Multiple files are concatenated for richer tone extraction.

        Returns:
            Raw PCM audio bytes (16-bit mono, 24000Hz)

        Raises:
            RuntimeError: If model not loaded or conversion fails
            ValueError: If inputs are invalid
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        if not source_audio_bytes:
            raise ValueError("Source audio is required")
        if not reference_audio_bytes:
            raise ValueError("At least one reference audio file is required")

        temp_files: List[str] = []

        try:
            # Write source audio to temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
                f.write(source_audio_bytes)
                source_path = f.name
                temp_files.append(source_path)

            # Write reference audio files to temp files
            ref_paths: List[str] = []
            for audio_bytes in reference_audio_bytes:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
                    f.write(audio_bytes)
                    ref_paths.append(f.name)
                    temp_files.append(f.name)

            # Prepare single reference path (concatenate if multiple)
            ref_path = self._prepare_reference(ref_paths, temp_files)

            logger.info(
                "vc_engine.convert.start",
                source_size=len(source_audio_bytes),
                ref_count=len(reference_audio_bytes),
            )

            # Run voice conversion
            audio = self.tts.voice_conversion(
                source_wav=source_path,
                target_wav=ref_path,
            )

            pcm_bytes = self._to_pcm16(audio)

            logger.info(
                "vc_engine.convert.complete",
                audio_size=len(pcm_bytes),
            )

            return pcm_bytes

        except Exception as e:
            logger.error("vc_engine.convert.failed", error=str(e))
            raise RuntimeError(f"Voice conversion failed: {e}") from e

        finally:
            for path in temp_files:
                if path and os.path.exists(path):
                    try:
                        os.remove(path)
                    except OSError:
                        logger.warning("vc_engine.cleanup_failed", path=path)

    def _prepare_reference(self, ref_paths: List[str], temp_files: List[str]) -> str:
        """Return a single reference WAV path.

        If multiple reference files provided, concatenates them into one
        to give the tone encoder broader sampling of the target voice.

        Args:
            ref_paths: List of reference WAV file paths
            temp_files: Shared list to register any new temp files for cleanup

        Returns:
            Path to the (possibly concatenated) reference WAV
        """
        if len(ref_paths) == 1:
            return ref_paths[0]

        all_frames: List[bytes] = []
        base_params = None

        for path in ref_paths:
            try:
                with wave.open(path, "rb") as wf:
                    if base_params is None:
                        base_params = wf.getparams()
                    all_frames.append(wf.readframes(wf.getnframes()))
            except Exception:
                continue  # Skip any unreadable files

        # Fall back to first file if concatenation not possible
        if not all_frames or base_params is None:
            logger.warning("vc_engine.concat_failed", fallback="first_ref")
            return ref_paths[0]

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            concat_path = f.name
            temp_files.append(concat_path)

        with wave.open(concat_path, "wb") as out:
            out.setparams(base_params)
            for frames in all_frames:
                out.writeframes(frames)

        logger.info("vc_engine.refs_concatenated", count=len(all_frames))
        return concat_path

    def _to_pcm16(self, audio) -> bytes:
        """Convert audio array to PCM16 bytes.

        Args:
            audio: Audio array (numpy or torch tensor)

        Returns:
            PCM16 bytes
        """
        if hasattr(audio, "detach"):
            audio = audio.detach().cpu().numpy()

        audio_array = np.asarray(audio)

        if audio_array.ndim > 1:
            audio_array = audio_array.reshape(-1)

        audio_array = np.clip(audio_array, -1.0, 1.0)
        pcm = (audio_array * 32767).astype(np.int16)
        return pcm.tobytes()
