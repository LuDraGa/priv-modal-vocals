"""Audio processing utilities for TTS synthesis.

Retrofitted from story_reels tts_v2/pipeline for Modal API use.
Includes stitching, normalization, and WAV wrapping.
"""

import io
import wave
from array import array
from typing import List


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
