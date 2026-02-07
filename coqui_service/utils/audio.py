"""Audio processing utilities for TTS synthesis.

Retrofitted from story_reels tts_v2/pipeline for Modal API use.
Includes stitching, normalization, WAV wrapping, and validation.
"""

import io
import wave
import tempfile
import struct
from array import array
from typing import List, Tuple, Optional
from dataclasses import dataclass


def stitch_audio_chunks(
    chunks: List[bytes],
    sample_rate: int = 22050,
    sample_width: int = 2,
    channels: int = 1,
    crossfade_ms: int = 40,
    silence_ms: int = 0,
) -> bytes:
    """Stitch audio chunks with optional crossfade and silence.

    Args:
        chunks: List of PCM audio chunks
        sample_rate: Sample rate in Hz
        sample_width: Bytes per sample (2 = 16-bit)
        channels: Number of audio channels (1 = mono)
        crossfade_ms: Crossfade duration in milliseconds
        silence_ms: Silence duration between chunks

    Returns:
        Combined PCM audio bytes
    """
    if not chunks:
        return b""

    if len(chunks) == 1:
        return chunks[0]

    combined = chunks[0]
    silence = generate_silence(silence_ms, sample_rate, sample_width, channels)

    for chunk in chunks[1:]:
        if crossfade_ms > 0 and sample_width == 2:  # Only support 16-bit crossfade
            fade_samples = int(sample_rate * (crossfade_ms / 1000)) * channels
            combined = crossfade_pcm16(combined, chunk, fade_samples=fade_samples)
            if silence:
                combined += silence
        else:
            combined = combined + silence + chunk

    return combined


def crossfade_pcm16(left: bytes, right: bytes, fade_samples: int) -> bytes:
    """Crossfade two PCM16 audio chunks.

    Args:
        left: First audio chunk
        right: Second audio chunk
        fade_samples: Number of samples to crossfade

    Returns:
        Crossfaded audio bytes
    """
    if fade_samples <= 0:
        return left + right

    left_samples = array("h")
    right_samples = array("h")
    left_samples.frombytes(left)
    right_samples.frombytes(right)

    # Adjust fade if chunks are too short
    fade_samples = min(fade_samples, len(left_samples), len(right_samples))
    if fade_samples == 0:
        return left + right

    output = array("h")
    output.extend(left_samples[:-fade_samples])

    # Crossfade region
    for idx in range(fade_samples):
        left_sample = left_samples[-fade_samples + idx]
        right_sample = right_samples[idx]
        left_gain = (fade_samples - idx) / fade_samples
        right_gain = idx / fade_samples
        mixed = int(left_sample * left_gain + right_sample * right_gain)
        # Clamp to 16-bit range
        mixed = max(-32768, min(32767, mixed))
        output.append(mixed)

    output.extend(right_samples[fade_samples:])
    return output.tobytes()


def generate_silence(
    duration_ms: int,
    sample_rate: int,
    sample_width: int,
    channels: int,
) -> bytes:
    """Generate silence of specified duration.

    Args:
        duration_ms: Duration in milliseconds
        sample_rate: Sample rate in Hz
        sample_width: Bytes per sample
        channels: Number of channels

    Returns:
        Silent PCM audio bytes
    """
    if duration_ms <= 0:
        return b""

    total_frames = int(sample_rate * (duration_ms / 1000))
    total_samples = total_frames * channels
    return b"\x00" * (total_samples * sample_width)


def normalize_audio(
    audio: bytes,
    sample_width: int = 2,
    channels: int = 1,
    target_peak: float = 0.95,
) -> bytes:
    """Peak-normalize PCM audio.

    Args:
        audio: PCM audio bytes
        sample_width: Bytes per sample (2 = 16-bit)
        channels: Number of channels
        target_peak: Target peak level (0.0-1.0)

    Returns:
        Normalized PCM audio bytes
    """
    if not audio or sample_width != 2:  # Only support 16-bit
        return audio

    samples = array("h")
    samples.frombytes(audio)

    if not samples:
        return audio

    # Find peak
    peak = max(abs(sample) for sample in samples)
    if peak == 0:
        return audio

    # Calculate scale factor
    scale = (target_peak * 32767) / peak

    # Apply normalization
    for idx, sample in enumerate(samples):
        scaled = int(sample * scale)
        # Clamp to 16-bit range
        samples[idx] = max(-32768, min(32767, scaled))

    return samples.tobytes()


def wrap_wav(
    audio_pcm: bytes,
    sample_rate: int = 22050,
    sample_width: int = 2,
    channels: int = 1,
) -> bytes:
    """Wrap raw PCM bytes into a WAV container.

    Args:
        audio_pcm: Raw PCM audio bytes
        sample_rate: Sample rate in Hz
        sample_width: Bytes per sample
        channels: Number of channels

    Returns:
        WAV file bytes
    """
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(sample_width)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_pcm)
    return buffer.getvalue()


def estimate_duration(
    audio: bytes,
    sample_rate: int,
    sample_width: int,
    channels: int,
) -> float:
    """Estimate audio duration in seconds.

    Args:
        audio: PCM audio bytes
        sample_rate: Sample rate in Hz
        sample_width: Bytes per sample
        channels: Number of channels

    Returns:
        Duration in seconds
    """
    if not audio or sample_rate <= 0 or sample_width <= 0 or channels <= 0:
        return 0.0

    frames = len(audio) / (sample_width * channels)
    return frames / sample_rate


# ============================================================================
# Audio Validation Utilities
# ============================================================================

@dataclass
class AudioValidationResult:
    """Result of audio validation."""

    is_valid: bool
    duration: Optional[float] = None
    sample_rate: Optional[int] = None
    channels: Optional[int] = None
    error_message: Optional[str] = None
    warning_message: Optional[str] = None


def validate_audio_duration(
    audio_bytes: bytes,
    min_duration: float = 3.0,
    max_duration: float = 30.0,
    optimal_min: float = 6.0,
    optimal_max: float = 10.0,
) -> AudioValidationResult:
    """Validate audio duration for voice cloning.

    Args:
        audio_bytes: Audio file bytes (WAV format)
        min_duration: Minimum acceptable duration (seconds)
        max_duration: Maximum acceptable duration (seconds)
        optimal_min: Optimal minimum duration (seconds)
        optimal_max: Optimal maximum duration (seconds)

    Returns:
        AudioValidationResult with duration info and validation status
    """
    try:
        # Write to temp file and analyze with wave module
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=True) as f:
            f.write(audio_bytes)
            f.flush()

            try:
                with wave.open(f.name, 'rb') as wav:
                    frames = wav.getnframes()
                    rate = wav.getframerate()
                    channels = wav.getnchannels()
                    duration = frames / float(rate)

                    # Check minimum duration
                    if duration < min_duration:
                        return AudioValidationResult(
                            is_valid=False,
                            duration=duration,
                            sample_rate=rate,
                            channels=channels,
                            error_message=f"Audio too short ({duration:.1f}s). Minimum: {min_duration}s for voice cloning",
                        )

                    # Check maximum duration
                    if duration > max_duration:
                        return AudioValidationResult(
                            is_valid=False,
                            duration=duration,
                            sample_rate=rate,
                            channels=channels,
                            error_message=f"Audio too long ({duration:.1f}s). Maximum: {max_duration}s",
                        )

                    # Check if within optimal range
                    warning = None
                    if duration < optimal_min or duration > optimal_max:
                        warning = f"Audio duration ({duration:.1f}s) is acceptable but {optimal_min}-{optimal_max}s is optimal for best quality"

                    return AudioValidationResult(
                        is_valid=True,
                        duration=duration,
                        sample_rate=rate,
                        channels=channels,
                        warning_message=warning,
                    )

            except wave.Error as e:
                return AudioValidationResult(
                    is_valid=False,
                    error_message=f"Invalid WAV file: {str(e)}",
                )

    except Exception as e:
        return AudioValidationResult(
            is_valid=False,
            error_message=f"Audio validation failed: {str(e)}",
        )


def validate_audio_quality(
    audio_bytes: bytes,
    min_sample_rate: int = 16000,
    preferred_sample_rate: int = 22050,
) -> AudioValidationResult:
    """Validate audio quality for voice cloning.

    Args:
        audio_bytes: Audio file bytes (WAV format)
        min_sample_rate: Minimum acceptable sample rate
        preferred_sample_rate: Preferred sample rate for best quality

    Returns:
        AudioValidationResult with quality info
    """
    try:
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=True) as f:
            f.write(audio_bytes)
            f.flush()

            try:
                with wave.open(f.name, 'rb') as wav:
                    sample_rate = wav.getframerate()
                    channels = wav.getnchannels()
                    sample_width = wav.getsampwidth()

                    # Check sample rate
                    if sample_rate < min_sample_rate:
                        return AudioValidationResult(
                            is_valid=False,
                            sample_rate=sample_rate,
                            channels=channels,
                            error_message=f"Sample rate too low ({sample_rate}Hz). Minimum: {min_sample_rate}Hz",
                        )

                    # Generate warnings for non-optimal settings
                    warnings = []
                    if sample_rate < preferred_sample_rate:
                        warnings.append(f"Sample rate {sample_rate}Hz is below optimal {preferred_sample_rate}Hz")

                    if channels > 1:
                        warnings.append(f"Audio is {channels}-channel (stereo). Mono is preferred for voice cloning")

                    if sample_width < 2:
                        warnings.append(f"Audio is {sample_width*8}-bit. 16-bit or higher recommended")

                    warning_msg = "; ".join(warnings) if warnings else None

                    return AudioValidationResult(
                        is_valid=True,
                        sample_rate=sample_rate,
                        channels=channels,
                        warning_message=warning_msg,
                    )

            except wave.Error as e:
                return AudioValidationResult(
                    is_valid=False,
                    error_message=f"Invalid WAV file: {str(e)}",
                )

    except Exception as e:
        return AudioValidationResult(
            is_valid=False,
            error_message=f"Quality validation failed: {str(e)}",
        )


def validate_reference_audio(
    audio_bytes: bytes,
    max_size_mb: float = 10.0,
) -> AudioValidationResult:
    """Comprehensive validation for reference audio files.

    Args:
        audio_bytes: Audio file bytes
        max_size_mb: Maximum file size in MB

    Returns:
        AudioValidationResult with comprehensive validation
    """
    # Check file size
    size_mb = len(audio_bytes) / (1024 * 1024)
    if size_mb > max_size_mb:
        return AudioValidationResult(
            is_valid=False,
            error_message=f"File too large ({size_mb:.1f}MB). Maximum: {max_size_mb}MB",
        )

    # Validate duration
    duration_result = validate_audio_duration(audio_bytes)
    if not duration_result.is_valid:
        return duration_result

    # Validate quality
    quality_result = validate_audio_quality(audio_bytes)
    if not quality_result.is_valid:
        return quality_result

    # Combine warnings
    warnings = []
    if duration_result.warning_message:
        warnings.append(duration_result.warning_message)
    if quality_result.warning_message:
        warnings.append(quality_result.warning_message)

    warning_msg = "; ".join(warnings) if warnings else None

    return AudioValidationResult(
        is_valid=True,
        duration=duration_result.duration,
        sample_rate=quality_result.sample_rate,
        channels=quality_result.channels,
        warning_message=warning_msg,
    )
