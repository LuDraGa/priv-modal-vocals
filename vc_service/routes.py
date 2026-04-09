"""FastAPI route handlers for OpenVoice v2 Voice Conversion API."""

from typing import List
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import Response
import structlog

from vc_service.models import (
    HealthResponse,
    ErrorResponse,
    ErrorDetail,
    APIInfoResponse,
    APIEndpointInfo,
)
from vc_service.engine import VCEngine
from shared.audio import (
    validate_source_audio,
    validate_reference_audio,
    normalize_audio,
    wrap_wav,
    estimate_duration,
)

logger = structlog.get_logger()


def create_routes(app: FastAPI, engine: VCEngine) -> None:
    """Register all API routes on the FastAPI application.

    Args:
        app: FastAPI application instance
        engine: VCEngine instance
    """

    @app.get("/health", response_model=HealthResponse)
    async def health_check():
        """Health check endpoint."""
        return HealthResponse(
            status="healthy",
            model_loaded=engine._loaded,
            version="0.1.0",
        )

    @app.get("/api-info", response_model=APIInfoResponse)
    async def api_info():
        """Get API usage documentation for all endpoints."""
        return APIInfoResponse(
            service="OpenVoice v2 Voice Conversion API",
            version="0.1.0",
            endpoints=[
                APIEndpointInfo(
                    endpoint="/voice-convert",
                    method="POST",
                    description="Convert source audio into a target voice while preserving content, emotion, and rhythm",
                    inputs={
                        "source_audio": "file (required) — audio to convert (WAV/MP3/M4A, max 50MB, any duration)",
                        "reference_audio": "file[] (1-3 files, required) — target voice samples (WAV, 3-30s each, max 10MB each; 6-10s optimal)",
                    },
                    outputs={
                        "content_type": "audio/wav",
                        "headers": {
                            "X-Sample-Rate": "24000 (OpenVoice v2 native output rate)",
                            "X-Duration-Sec": "Duration of converted audio in seconds",
                            "X-Engine": "openvoice_v2",
                            "X-Mode": "voice_conversion",
                            "X-Reference-Count": "Number of reference files used",
                            "X-Validation-Warnings": "Audio quality warnings, if any",
                        },
                    },
                    example=(
                        'curl -X POST https://[ENDPOINT]/voice-convert '
                        '-F "source_audio=@source.wav" '
                        '-F "reference_audio=@ref.wav" '
                        '--output converted.wav'
                    ),
                ),
                APIEndpointInfo(
                    endpoint="/health",
                    method="GET",
                    description="Check service health and model load status",
                    inputs={},
                    outputs={
                        "content_type": "application/json",
                        "fields": {
                            "status": "healthy/unhealthy",
                            "model_loaded": "boolean",
                            "version": "API version",
                        },
                    },
                    example="curl https://[ENDPOINT]/health | jq",
                ),
            ],
        )

    @app.post("/voice-convert")
    async def voice_convert(
        source_audio: UploadFile = File(..., description="Audio to convert (WAV/MP3/M4A, max 50MB)"),
        reference_audio: List[UploadFile] = File(
            ...,
            description="Target voice reference audio (1-3 files, WAV, 3-30s each, max 10MB each)",
        ),
    ):
        """Convert the voice in source_audio to match the tone of reference_audio.

        Preserves all content, emotion, rhythm, and speaking style from the source.
        Only the voice timbre is changed to match the reference voice.

        Args:
            source_audio: The audio whose voice will be converted
            reference_audio: 1-3 audio clips of the target voice (3-30s each, WAV preferred)

        Returns:
            WAV audio: source content spoken in the reference voice (24kHz mono)
        """
        try:
            # --- Validate reference count ---
            if not reference_audio:
                raise HTTPException(
                    status_code=400,
                    detail="At least one reference audio file is required",
                )
            if len(reference_audio) > 3:
                raise HTTPException(
                    status_code=400,
                    detail="Maximum 3 reference audio files allowed",
                )

            # --- Read and validate source audio ---
            source_bytes = await source_audio.read()
            source_validation = validate_source_audio(source_bytes, max_size_mb=50.0)
            if not source_validation.is_valid:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid source audio: {source_validation.error_message}",
                )

            # --- Read and validate reference audio files ---
            ref_bytes_list: List[bytes] = []
            validation_warnings: List[str] = []

            for idx, audio_file in enumerate(reference_audio):
                audio_bytes = await audio_file.read()
                validation = validate_reference_audio(audio_bytes, max_size_mb=10.0)

                if not validation.is_valid:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Reference audio {idx + 1} invalid: {validation.error_message}",
                    )

                if validation.warning_message:
                    validation_warnings.append(f"File {idx + 1}: {validation.warning_message}")

                ref_bytes_list.append(audio_bytes)

                logger.info(
                    "voice_convert.ref_validated",
                    file_index=idx,
                    duration=validation.duration,
                    sample_rate=validation.sample_rate,
                    channels=validation.channels,
                )

            logger.info(
                "voice_convert.request",
                source_size=len(source_bytes),
                ref_count=len(ref_bytes_list),
            )

            # --- Run voice conversion ---
            pcm_bytes = engine.convert(
                source_audio_bytes=source_bytes,
                reference_audio_bytes=ref_bytes_list,
            )

            # --- Post-process and wrap ---
            normalized = normalize_audio(pcm_bytes)
            wav_bytes = wrap_wav(normalized, sample_rate=engine.sample_rate)
            duration = estimate_duration(
                normalized,
                sample_rate=engine.sample_rate,
                sample_width=2,
                channels=1,
            )

            headers = {
                "X-Sample-Rate": str(engine.sample_rate),
                "X-Duration-Sec": f"{duration:.2f}",
                "X-Engine": "openvoice_v2",
                "X-Mode": "voice_conversion",
                "X-Reference-Count": str(len(ref_bytes_list)),
            }

            if validation_warnings:
                headers["X-Validation-Warnings"] = "; ".join(validation_warnings)

            logger.info(
                "voice_convert.complete",
                audio_size=len(wav_bytes),
                duration_sec=duration,
            )

            return Response(content=wav_bytes, media_type="audio/wav", headers=headers)

        except HTTPException:
            raise

        except Exception as e:
            logger.error("voice_convert.failed", error=str(e))
            raise HTTPException(
                status_code=500,
                detail=ErrorResponse(
                    error=ErrorDetail(
                        code="voice_conversion_failed",
                        message=f"Voice conversion failed: {str(e)}",
                    )
                ).model_dump(),
            )
