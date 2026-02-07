"""TTS engine wrapper for Coqui XTTS v2.

Handles model loading from Modal Volume, synthesis, and voice cloning.
"""

import os
import tempfile
from pathlib import Path
from typing import List, Optional
import numpy as np
import structlog

logger = structlog.get_logger()


class TTSEngine:
    """Coqui XTTS v2 engine wrapper."""

    def __init__(self, model_path: str = "/models/coqui"):
        self.model_path = Path(model_path)
        self.tts = None
        self.sample_rate = 22050  # XTTS v2 default
        self._loaded = False

    def load_model(self) -> None:
        """Load XTTS v2 model from Modal Volume.

        Raises:
            RuntimeError: If model loading fails
        """
        if self._loaded:
            logger.info("tts_engine.already_loaded")
            return

        try:
            from TTS.api import TTS

            # Set TTS_HOME to Volume path
            os.environ["TTS_HOME"] = str(self.model_path)

            logger.info("tts_engine.loading", model_path=str(self.model_path))

            # Load XTTS v2 model
            self.tts = TTS(
                model_name="tts_models/multilingual/multi-dataset/xtts_v2",
                progress_bar=False,
                gpu=True,  # Use GPU if available (Modal T4)
            )

            # Infer sample rate from model
            self.sample_rate = self._infer_sample_rate()

            self._loaded = True
            logger.info(
                "tts_engine.loaded",
                sample_rate=self.sample_rate,
                has_speakers=hasattr(self.tts, "speakers"),
                speaker_count=len(self.tts.speakers) if hasattr(self.tts, "speakers") else 0,
            )

        except Exception as e:
            logger.error("tts_engine.load_failed", error=str(e))
            raise RuntimeError(f"Failed to load TTS model: {e}") from e

    def _infer_sample_rate(self) -> int:
        """Infer sample rate from loaded model."""
        if not self.tts:
            return 22050

        # Try multiple paths to find sample rate
        for path in [
            ("synthesizer", "output_sample_rate"),
            ("synthesizer", "sample_rate"),
            ("tts_config", "audio", "sample_rate"),
        ]:
            current = self.tts
            try:
                for key in path:
                    if isinstance(key, str) and hasattr(current, key):
                        current = getattr(current, key)
                    elif isinstance(current, dict) and key in current:
                        current = current[key]
                    else:
                        current = None
                        break

                if isinstance(current, int):
                    return current
            except Exception:
                continue

        return 22050  # Default fallback

    def synthesize_builtin(
        self,
        text: str,
        speaker_id: str,
        language: str = "en",
    ) -> bytes:
        """Synthesize speech using a built-in speaker.

        Args:
            text: Text to synthesize
            speaker_id: Built-in speaker name (e.g., "Aaron Dreschner")
            language: Language code (e.g., "en", "es", "fr")

        Returns:
            Raw PCM audio bytes (16-bit, mono)

        Raises:
            RuntimeError: If synthesis fails
            ValueError: If speaker_id is invalid
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Validate speaker
        if hasattr(self.tts, "speakers") and self.tts.speakers:
            speakers = self.tts.speakers
            valid_speakers = speakers if isinstance(speakers, list) else list(speakers.keys())
            if speaker_id not in valid_speakers:
                raise ValueError(
                    f"Invalid speaker: {speaker_id}. Valid speakers: {', '.join(valid_speakers[:5])}..."
                )

        try:
            logger.info(
                "tts_engine.synthesize.start",
                speaker=speaker_id,
                language=language,
                text_len=len(text),
            )

            # Synthesize with built-in speaker
            audio = self.tts.tts(
                text=text,
                speaker=speaker_id,
                language=language,
                split_sentences=False,  # We handle chunking externally
            )

            # Convert to PCM16 bytes
            pcm_bytes = self._to_pcm16(audio)

            logger.info(
                "tts_engine.synthesize.complete",
                audio_size=len(pcm_bytes),
            )

            return pcm_bytes

        except Exception as e:
            logger.error("tts_engine.synthesize.failed", error=str(e), speaker=speaker_id)
            raise RuntimeError(f"Synthesis failed: {e}") from e

    def synthesize_clone(
        self,
        text: str,
        reference_audio_bytes: bytes,
        language: str = "en",
    ) -> bytes:
        """Synthesize speech using voice cloning.

        Args:
            text: Text to synthesize
            reference_audio_bytes: Reference audio file bytes (WAV, MP3, M4A)
            language: Language code

        Returns:
            Raw PCM audio bytes (16-bit, mono)

        Raises:
            RuntimeError: If synthesis fails
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Write reference audio to temp file
        temp_file = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
                f.write(reference_audio_bytes)
                temp_file = f.name

            logger.info(
                "tts_engine.clone.start",
                language=language,
                text_len=len(text),
                ref_audio_size=len(reference_audio_bytes),
            )

            # Synthesize with voice cloning
            audio = self.tts.tts(
                text=text,
                speaker_wav=temp_file,
                language=language,
                split_sentences=False,
            )

            # Convert to PCM16 bytes
            pcm_bytes = self._to_pcm16(audio)

            logger.info(
                "tts_engine.clone.complete",
                audio_size=len(pcm_bytes),
            )

            return pcm_bytes

        except Exception as e:
            logger.error("tts_engine.clone.failed", error=str(e))
            raise RuntimeError(f"Voice cloning failed: {e}") from e

        finally:
            # Clean up temp file
            if temp_file and os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except OSError:
                    logger.warning("tts_engine.cleanup_failed", path=temp_file)

    def get_speakers(self) -> List[str]:
        """Get list of available built-in speakers.

        Returns:
            List of speaker names

        Raises:
            RuntimeError: If model not loaded
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        if not hasattr(self.tts, "speakers") or not self.tts.speakers:
            return []

        speakers = self.tts.speakers

        if isinstance(speakers, list):
            return speakers
        elif isinstance(speakers, dict):
            return list(speakers.keys())

        return []

    def _to_pcm16(self, audio) -> bytes:
        """Convert audio array to PCM16 bytes.

        Args:
            audio: Audio array (numpy or tensor)

        Returns:
            PCM16 bytes
        """
        # Handle torch tensor
        if hasattr(audio, "detach"):
            audio = audio.detach().cpu().numpy()

        # Convert to numpy array
        audio_array = np.asarray(audio)

        # Flatten if multi-dimensional
        if audio_array.ndim > 1:
            audio_array = audio_array.reshape(-1)

        # Clip to [-1.0, 1.0] range
        audio_array = np.clip(audio_array, -1.0, 1.0)

        # Convert to 16-bit PCM
        pcm = (audio_array * 32767).astype(np.int16)

        return pcm.tobytes()
