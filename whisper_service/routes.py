"""FastAPI routes for WhisperX STT API."""

from typing import Optional
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
import structlog

from whisper_service.models import (
    TranscribeResponse,
    HealthResponse,
    LanguagesResponse,
    LanguageInfo,
    Segment,
    WordSegment,
    APIInfoResponse,
    APIEndpointInfo,
)
from whisper_service.utils.audio_utils import convert_to_wav, cleanup_temp_file

logger = structlog.get_logger()


def create_routes(app: FastAPI, engine) -> None:
    """Register all API routes.

    Args:
        app: FastAPI application instance
        engine: WhisperXEngine instance
    """

    # ========================================================================
    # Health Check Endpoint
    # ========================================================================
    @app.get(
        "/health",
        response_model=HealthResponse,
        summary="Health Check",
        description="Check if the STT service is running and models are loaded",
        tags=["Status"],
    )
    async def health_check():
        """Health check endpoint."""
        return HealthResponse(
            status="healthy",
            model="large-v3-turbo",
            gpu_available=engine.is_gpu_available(),
            alignment_available=engine.is_alignment_available(),
        )

    # ========================================================================
    # Languages Endpoint
    # ========================================================================
    @app.get(
        "/languages",
        response_model=LanguagesResponse,
        summary="Get Supported Languages",
        description="Get list of all languages supported by WhisperX",
        tags=["Info"],
    )
    async def get_languages():
        """Get list of supported languages."""
        languages_dict = engine.get_supported_languages()

        languages = [
            LanguageInfo(code=code, name=name)
            for code, name in languages_dict.items()
        ]

        # Sort by language name
        languages.sort(key=lambda x: x.name)

        return LanguagesResponse(
            languages=languages,
            total=len(languages),
        )

    # ========================================================================
    # Transcribe Endpoint
    # ========================================================================
    @app.post(
        "/transcribe",
        response_model=TranscribeResponse,
        summary="Transcribe Audio",
        description=(
            "Transcribe audio file with word-level timestamps for karaoke-style highlighting. "
            "Supports all common audio formats (WAV, MP3, M4A, FLAC, etc.). "
            "Language is auto-detected by default, or can be specified manually."
        ),
        tags=["Transcription"],
    )
    async def transcribe_audio(
        file: UploadFile = File(
            ...,
            description="Audio file to transcribe (WAV, MP3, M4A, FLAC, etc.)"
        ),
        language: Optional[str] = Form(
            None,
            description="Language code (e.g., 'en', 'es', 'fr'). Auto-detected if not provided.",
            example="en",
        ),
    ):
        """Transcribe audio file with word-level timestamps.

        Args:
            file: Audio file upload
            language: Optional language code

        Returns:
            TranscribeResponse with full text and word-level segments

        Raises:
            HTTPException: If transcription fails
        """
        wav_path = None
        try:
            # ================================================================
            # 1. Read uploaded file
            # ================================================================
            logger.info(
                "routes.transcribe.start",
                filename=file.filename,
                content_type=file.content_type,
                language=language or "auto",
            )

            audio_bytes = await file.read()

            if not audio_bytes:
                raise HTTPException(
                    status_code=400,
                    detail="Uploaded file is empty",
                )

            # ================================================================
            # 2. Convert to WAV (16kHz mono)
            # ================================================================
            # Extract file extension for format hint
            source_format = "auto"
            if file.filename:
                parts = file.filename.rsplit(".", 1)
                if len(parts) == 2:
                    source_format = parts[1].lower()

            wav_path, duration = convert_to_wav(audio_bytes, source_format)

            logger.info(
                "routes.transcribe.converted",
                wav_path=wav_path,
                duration=duration,
            )

            # ================================================================
            # 3. Transcribe with WhisperX
            # ================================================================
            result = engine.transcribe(
                audio_path=wav_path,
                language=language,
            )

            # ================================================================
            # 4. Format response
            # ================================================================
            segments = []
            for seg in result["segments"]:
                # Extract word-level timestamps
                words = []
                if "words" in seg:
                    for word in seg["words"]:
                        words.append(
                            WordSegment(
                                word=word.get("word", ""),
                                start=word.get("start", 0.0),
                                end=word.get("end", 0.0),
                                score=word.get("score", 0.0),
                            )
                        )

                segments.append(
                    Segment(
                        text=seg.get("text", "").strip(),
                        start=seg.get("start", 0.0),
                        end=seg.get("end", 0.0),
                        words=words,
                    )
                )

            response = TranscribeResponse(
                text=result["text"],
                segments=segments,
                language=result["language"],
                duration=duration,
            )

            logger.info(
                "routes.transcribe.complete",
                text_length=len(result["text"]),
                segments=len(segments),
                total_words=sum(len(s.words) for s in segments),
                language=result["language"],
                duration=duration,
            )

            return response

        except HTTPException:
            raise

        except ValueError as e:
            logger.error("routes.transcribe.validation_error", error=str(e))
            raise HTTPException(status_code=400, detail=str(e))

        except Exception as e:
            logger.error("routes.transcribe.failed", error=str(e))
            raise HTTPException(
                status_code=500,
                detail=f"Transcription failed: {str(e)}",
            )

        finally:
            # Clean up temporary WAV file
            if wav_path:
                cleanup_temp_file(wav_path)

    # ========================================================================
    # API Info Endpoint
    # ========================================================================
    @app.get(
        "/api-info",
        response_model=APIInfoResponse,
        summary="API Documentation",
        description="Get comprehensive API documentation for all endpoints",
        tags=["Info"],
    )
    async def api_info():
        """Get API usage documentation for all endpoints."""
        return APIInfoResponse(
            service="WhisperX STT API",
            version="0.1.0",
            endpoints=[
                APIEndpointInfo(
                    endpoint="/transcribe",
                    method="POST",
                    description="Transcribe audio with word-level timestamps for karaoke-style highlighting",
                    inputs={
                        "file": "Audio file (WAV, MP3, M4A, FLAC, etc.) - multipart/form-data",
                        "language": "Optional language code (en, es, fr, etc.) - auto-detected if not provided",
                    },
                    outputs={
                        "text": "Full transcribed text",
                        "segments": "List of segments with word-level timestamps (each word has: word, start, end, score)",
                        "language": "Detected or specified language code",
                        "duration": "Audio duration in seconds",
                    },
                    example=(
                        'curl -X POST "https://[ENDPOINT]/transcribe" \\\n'
                        '  -F "file=@audio.mp3" \\\n'
                        '  -F "language=en" | jq'
                    ),
                ),
                APIEndpointInfo(
                    endpoint="/languages",
                    method="GET",
                    description="Get list of all 99+ supported languages",
                    outputs={
                        "languages": "Array of {code: str, name: str} objects",
                        "total": "Total number of supported languages",
                    },
                    example='curl "https://[ENDPOINT]/languages" | jq',
                ),
                APIEndpointInfo(
                    endpoint="/health",
                    method="GET",
                    description="Service health check with GPU and model status",
                    outputs={
                        "status": "Service status (healthy/unhealthy)",
                        "model": "Loaded Whisper model (large-v3-turbo)",
                        "gpu_available": "GPU availability (true/false)",
                        "alignment_available": "Alignment model availability (true/false)",
                    },
                    example='curl "https://[ENDPOINT]/health" | jq',
                ),
                APIEndpointInfo(
                    endpoint="/api-info",
                    method="GET",
                    description="Get this API documentation",
                    outputs={
                        "service": "Service name",
                        "version": "API version",
                        "endpoints": "Array of endpoint documentation objects",
                    },
                    example='curl "https://[ENDPOINT]/api-info" | jq',
                ),
            ],
        )
