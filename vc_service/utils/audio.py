"""Audio utilities — re-exported from shared.audio."""

from shared.audio import (
    stitch_audio_chunks,
    crossfade_pcm16,
    generate_silence,
    normalize_audio,
    wrap_wav,
    estimate_duration,
    AudioValidationResult,
    validate_audio_duration,
    validate_audio_quality,
    validate_reference_audio,
    validate_source_audio,
)

__all__ = [
    "stitch_audio_chunks",
    "crossfade_pcm16",
    "generate_silence",
    "normalize_audio",
    "wrap_wav",
    "estimate_duration",
    "AudioValidationResult",
    "validate_audio_duration",
    "validate_audio_quality",
    "validate_reference_audio",
    "validate_source_audio",
]
