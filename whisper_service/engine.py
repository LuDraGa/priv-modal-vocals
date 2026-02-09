"""WhisperX engine wrapper with forced phoneme alignment.

This module provides the core transcription engine using WhisperX with:
- Word-level timestamps via Wav2Vec2 forced alignment
- Voice Activity Detection (VAD) for accuracy
- Auto-language detection with manual override
- Multi-language support
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Any
import structlog

logger = structlog.get_logger()


# WhisperX supported languages (from faster-whisper)
# See: https://github.com/openai/whisper/blob/main/whisper/tokenizer.py
SUPPORTED_LANGUAGES = {
    "en": "English",
    "zh": "Chinese",
    "de": "German",
    "es": "Spanish",
    "ru": "Russian",
    "ko": "Korean",
    "fr": "French",
    "ja": "Japanese",
    "pt": "Portuguese",
    "tr": "Turkish",
    "pl": "Polish",
    "ca": "Catalan",
    "nl": "Dutch",
    "ar": "Arabic",
    "sv": "Swedish",
    "it": "Italian",
    "id": "Indonesian",
    "hi": "Hindi",
    "fi": "Finnish",
    "vi": "Vietnamese",
    "he": "Hebrew",
    "uk": "Ukrainian",
    "el": "Greek",
    "ms": "Malay",
    "cs": "Czech",
    "ro": "Romanian",
    "da": "Danish",
    "hu": "Hungarian",
    "ta": "Tamil",
    "no": "Norwegian",
    "th": "Thai",
    "ur": "Urdu",
    "hr": "Croatian",
    "bg": "Bulgarian",
    "lt": "Lithuanian",
    "la": "Latin",
    "mi": "Maori",
    "ml": "Malayalam",
    "cy": "Welsh",
    "sk": "Slovak",
    "te": "Telugu",
    "fa": "Persian",
    "lv": "Latvian",
    "bn": "Bengali",
    "sr": "Serbian",
    "az": "Azerbaijani",
    "sl": "Slovenian",
    "kn": "Kannada",
    "et": "Estonian",
    "mk": "Macedonian",
    "br": "Breton",
    "eu": "Basque",
    "is": "Icelandic",
    "hy": "Armenian",
    "ne": "Nepali",
    "mn": "Mongolian",
    "bs": "Bosnian",
    "kk": "Kazakh",
    "sq": "Albanian",
    "sw": "Swahili",
    "gl": "Galician",
    "mr": "Marathi",
    "pa": "Punjabi",
    "si": "Sinhala",
    "km": "Khmer",
    "sn": "Shona",
    "yo": "Yoruba",
    "so": "Somali",
    "af": "Afrikaans",
    "oc": "Occitan",
    "ka": "Georgian",
    "be": "Belarusian",
    "tg": "Tajik",
    "sd": "Sindhi",
    "gu": "Gujarati",
    "am": "Amharic",
    "yi": "Yiddish",
    "lo": "Lao",
    "uz": "Uzbek",
    "fo": "Faroese",
    "ht": "Haitian Creole",
    "ps": "Pashto",
    "tk": "Turkmen",
    "nn": "Nynorsk",
    "mt": "Maltese",
    "sa": "Sanskrit",
    "lb": "Luxembourgish",
    "my": "Myanmar",
    "bo": "Tibetan",
    "tl": "Tagalog",
    "mg": "Malagasy",
    "as": "Assamese",
    "tt": "Tatar",
    "haw": "Hawaiian",
    "ln": "Lingala",
    "ha": "Hausa",
    "ba": "Bashkir",
    "jw": "Javanese",
    "su": "Sundanese",
}


class WhisperXEngine:
    """WhisperX transcription engine with word-level timestamps.

    This engine uses:
    - WhisperX large-v3-turbo for transcription (6x faster than large-v3)
    - Wav2Vec2 for forced phoneme alignment (frame-accurate word boundaries)
    - Pyannote VAD for voice activity detection
    """

    def __init__(
        self,
        cache_dir: str = "/models/hf_cache",
    ):
        """Initialize WhisperX engine.

        Args:
            cache_dir: Path to HuggingFace cache directory (for models)
        """
        self.cache_dir = Path(cache_dir)

        # Models (loaded in load_models)
        self.whisper_model = None
        self.alignment_model = None
        self.alignment_metadata = None

        self._loaded = False
        self.device = None
        self.compute_type = None

    def load_models(self) -> None:
        """Load WhisperX and alignment models from Modal Volume.

        This should be called once per container startup (via @modal.enter()).

        Raises:
            RuntimeError: If model loading fails
        """
        if self._loaded:
            logger.info("whisperx_engine.already_loaded")
            return

        try:
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

            # Set cache directories (HuggingFace models)
            os.environ["HF_HUB_CACHE"] = str(self.cache_dir)
            os.environ["HF_HOME"] = str(self.cache_dir)

            # Determine device and compute type
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.compute_type = "float16" if self.device == "cuda" else "int8"

            logger.info(
                "whisperx_engine.loading",
                cache_dir=str(self.cache_dir),
                device=self.device,
                compute_type=self.compute_type,
            )

            # ================================================================
            # 1. Load WhisperX model (large-v3-turbo)
            # ================================================================
            logger.info("whisperx_engine.loading_whisper", model="large-v3-turbo")

            # WhisperX loads from HF_HOME cache
            self.whisper_model = whisperx.load_model(
                "large-v3-turbo",
                device=self.device,
                compute_type=self.compute_type,
            )

            logger.info("whisperx_engine.whisper_loaded")

            # ================================================================
            # 2. Load alignment model (Wav2Vec2 for English)
            # ================================================================
            # Note: We pre-load English alignment model
            # Other languages will be loaded on-demand
            logger.info("whisperx_engine.loading_alignment", language="en")

            # Alignment model also loads from HF_HOME cache
            self.alignment_model, self.alignment_metadata = whisperx.load_align_model(
                language_code="en",
                device=self.device,
            )

            logger.info(
                "whisperx_engine.alignment_loaded",
                language="en",
                metadata=self.alignment_metadata,
            )

            # ================================================================
            # Mark as loaded
            # ================================================================
            self._loaded = True

            logger.info(
                "whisperx_engine.loaded",
                device=self.device,
                compute_type=self.compute_type,
                whisper_loaded=self.whisper_model is not None,
                alignment_loaded=self.alignment_model is not None,
            )

        except Exception as e:
            logger.error("whisperx_engine.load_failed", error=str(e))
            raise RuntimeError(f"Failed to load WhisperX models: {e}") from e

    def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Transcribe audio with word-level timestamps.

        Args:
            audio_path: Path to audio file (16kHz mono WAV)
            language: Language code (e.g., "en", "es"). Auto-detected if None.

        Returns:
            Dictionary containing:
                - text: Full transcribed text
                - segments: List of segments with word-level timestamps
                - language: Detected or specified language code

        Raises:
            RuntimeError: If transcription or alignment fails
            ValueError: If language is invalid
        """
        if not self._loaded:
            raise RuntimeError("Models not loaded. Call load_models() first.")

        if language and language not in SUPPORTED_LANGUAGES:
            raise ValueError(
                f"Unsupported language: {language}. "
                f"Supported languages: {', '.join(list(SUPPORTED_LANGUAGES.keys())[:10])}..."
            )

        try:
            import whisperx

            logger.info(
                "whisperx_engine.transcribe.start",
                audio_path=audio_path,
                language=language or "auto",
            )

            # ================================================================
            # 1. Load audio
            # ================================================================
            audio = whisperx.load_audio(audio_path)

            # ================================================================
            # 2. Transcribe with WhisperX
            # ================================================================
            # batch_size: Number of audio chunks to process in parallel
            # Higher = faster but more VRAM usage
            batch_size = 16  # Good balance for A10G (24GB VRAM)

            result = self.whisper_model.transcribe(
                audio,
                batch_size=batch_size,
                language=language,  # None for auto-detect
            )

            detected_language = result.get("language", language or "en")

            logger.info(
                "whisperx_engine.transcribe.complete",
                segments=len(result.get("segments", [])),
                language=detected_language,
            )

            # ================================================================
            # 3. Align with Wav2Vec2 for word-level timestamps
            # ================================================================
            logger.info("whisperx_engine.align.start", language=detected_language)

            # Load alignment model for detected language (if different from English)
            alignment_model = self.alignment_model
            alignment_metadata = self.alignment_metadata

            if detected_language != "en":
                logger.info(
                    "whisperx_engine.loading_alignment",
                    language=detected_language,
                )
                # Load language-specific alignment model (from cache)
                alignment_model, alignment_metadata = whisperx.load_align_model(
                    language_code=detected_language,
                    device=self.device,
                )

            # Perform forced alignment
            result = whisperx.align(
                result["segments"],
                alignment_model,
                alignment_metadata,
                audio,
                self.device,
                return_char_alignments=False,  # Word-level only
            )

            logger.info(
                "whisperx_engine.align.complete",
                segments=len(result.get("segments", [])),
                has_words=any(
                    "words" in seg for seg in result.get("segments", [])
                ),
            )

            # ================================================================
            # 4. Extract full text and segments
            # ================================================================
            full_text = " ".join(
                seg.get("text", "").strip()
                for seg in result.get("segments", [])
            ).strip()

            return {
                "text": full_text,
                "segments": result.get("segments", []),
                "language": detected_language,
            }

        except Exception as e:
            logger.error("whisperx_engine.transcribe.failed", error=str(e))
            raise RuntimeError(f"Transcription failed: {e}") from e

    def get_supported_languages(self) -> Dict[str, str]:
        """Get dictionary of supported languages.

        Returns:
            Dictionary mapping language codes to language names
        """
        return SUPPORTED_LANGUAGES.copy()

    def is_gpu_available(self) -> bool:
        """Check if GPU is available.

        Returns:
            True if GPU is available and loaded
        """
        return self._loaded and self.device == "cuda"

    def is_alignment_available(self) -> bool:
        """Check if alignment model is loaded.

        Returns:
            True if alignment model is loaded
        """
        return self._loaded and self.alignment_model is not None
