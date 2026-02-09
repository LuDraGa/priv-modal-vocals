"""Audio processing utilities for WhisperX STT service."""

import io
import tempfile
import os
from pathlib import Path
from typing import Tuple
import structlog

logger = structlog.get_logger()


def convert_to_wav(audio_bytes: bytes, source_format: str = "auto") -> Tuple[str, float]:
    """Convert audio bytes to WAV format suitable for WhisperX.

    WhisperX requires 16kHz mono WAV format. This function accepts any
    audio format (MP3, M4A, FLAC, WAV, etc.) and converts it using ffmpeg.

    Args:
        audio_bytes: Input audio file bytes
        source_format: Source format hint (e.g., "mp3", "m4a"). Use "auto" to detect.

    Returns:
        Tuple of (wav_file_path, duration_seconds)

    Raises:
        RuntimeError: If audio conversion fails
        ValueError: If audio_bytes is empty
    """
    if not audio_bytes:
        raise ValueError("audio_bytes cannot be empty")

    try:
        # Create temp file for input audio
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{source_format}") as input_file:
            input_file.write(audio_bytes)
            input_path = input_file.name

        # Create temp file for output WAV
        output_fd, output_path = tempfile.mkstemp(suffix=".wav")
        os.close(output_fd)  # Close fd, ffmpeg will open it

        logger.info(
            "audio_utils.convert.start",
            input_size=len(audio_bytes),
            source_format=source_format,
        )

        # Convert to 16kHz mono WAV using ffmpeg
        # -y: overwrite output file
        # -i: input file
        # -ar 16000: 16kHz sample rate (WhisperX requirement)
        # -ac 1: mono (1 channel)
        # -c:a pcm_s16le: 16-bit PCM encoding
        import subprocess

        result = subprocess.run(
            [
                "ffmpeg",
                "-y",  # Overwrite output
                "-i", input_path,  # Input file
                "-ar", "16000",  # 16kHz sample rate
                "-ac", "1",  # Mono
                "-c:a", "pcm_s16le",  # 16-bit PCM
                output_path,  # Output file
            ],
            capture_output=True,
            text=True,
            check=True,
        )

        # Get audio duration from ffmpeg output
        duration = _extract_duration_from_ffmpeg_output(result.stderr)

        logger.info(
            "audio_utils.convert.complete",
            output_path=output_path,
            duration=duration,
        )

        # Clean up input file
        try:
            os.remove(input_path)
        except OSError:
            logger.warning("audio_utils.cleanup_failed", path=input_path)

        return output_path, duration

    except subprocess.CalledProcessError as e:
        logger.error(
            "audio_utils.convert.failed",
            error=str(e),
            stderr=e.stderr,
        )
        # Clean up temp files
        if 'input_path' in locals():
            try:
                os.remove(input_path)
            except OSError:
                pass
        if 'output_path' in locals():
            try:
                os.remove(output_path)
            except OSError:
                pass

        raise RuntimeError(f"Audio conversion failed: {e.stderr}") from e

    except Exception as e:
        logger.error("audio_utils.convert.failed", error=str(e))
        raise RuntimeError(f"Audio conversion failed: {e}") from e


def _extract_duration_from_ffmpeg_output(stderr: str) -> float:
    """Extract audio duration from ffmpeg stderr output.

    Args:
        stderr: ffmpeg stderr output

    Returns:
        Duration in seconds, or 0.0 if not found
    """
    try:
        # Look for "Duration: HH:MM:SS.ms" in ffmpeg output
        for line in stderr.split('\n'):
            if 'Duration:' in line:
                # Example: "Duration: 00:01:23.45, start: 0.000000, bitrate: 128 kb/s"
                duration_str = line.split('Duration:')[1].split(',')[0].strip()
                # Parse HH:MM:SS.ms
                hours, minutes, seconds = duration_str.split(':')
                total_seconds = (
                    int(hours) * 3600 +
                    int(minutes) * 60 +
                    float(seconds)
                )
                return total_seconds
    except Exception as e:
        logger.warning("audio_utils.duration_parse_failed", error=str(e))

    return 0.0


def cleanup_temp_file(file_path: str) -> None:
    """Clean up temporary audio file.

    Args:
        file_path: Path to temporary file
    """
    if file_path and os.path.exists(file_path):
        try:
            os.remove(file_path)
            logger.debug("audio_utils.cleanup", path=file_path)
        except OSError as e:
            logger.warning("audio_utils.cleanup_failed", path=file_path, error=str(e))
