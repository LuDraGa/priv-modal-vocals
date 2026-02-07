"""FastAPI route handlers for Coqui TTS API."""

from typing import List
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import Response
import structlog

from coqui_service.models import (
    TTSRequest,
    VoiceCloneRequest,
    SpeakersResponse,
    HealthResponse,
    ErrorResponse,
    ErrorDetail,
    APIInfoResponse,
    APIEndpointInfo,
)
from coqui_service.engine import TTSEngine
from coqui_service.utils.speaker_cache import SpeakerCache
from coqui_service.utils.chunker import chunk_text
from coqui_service.utils.audio import (
    stitch_audio_chunks,
    normalize_audio,
    wrap_wav,
    estimate_duration,
    validate_reference_audio,
)

logger = structlog.get_logger()


def create_routes(app: FastAPI, engine: TTSEngine, speaker_cache: SpeakerCache, volume) -> None:
    """Create and register all API routes.

    Args:
        app: FastAPI application instance
        engine: TTS engine instance
        speaker_cache: Speaker cache instance
        volume: Modal Volume instance
    """

    @app.get("/health", response_model=HealthResponse)
    async def health_check():
        """Health check endpoint."""
        try:
            speakers = engine.get_speakers()
            return HealthResponse(
                status="healthy",
                model_loaded=engine._loaded,
                speakers_available=len(speakers),
                version="0.1.0",
            )
        except Exception as e:
            logger.error("health_check.failed", error=str(e))
            raise HTTPException(status_code=503, detail="Service unhealthy")

    @app.get("/api-info", response_model=APIInfoResponse)
    async def api_info():
        """Get API usage documentation for all endpoints."""
        return APIInfoResponse(
            service="Coqui TTS & Voice Cloning API",
            version="0.1.0",
            endpoints=[
                APIEndpointInfo(
                    endpoint="/tts",
                    method="POST",
                    description="Generate speech from text using built-in speakers",
                    inputs={
                        "text": "string (1-5000 chars) - Text to synthesize",
                        "speaker_id": "string - Built-in speaker name (e.g., 'Claribel Dervla')",
                        "language": "string - Language code (en, es, fr, de, it, pt, pl, tr, ru, nl, cs, ar, zh-cn, ja, hu, ko, hi)",
                        "speed": "float (0.7-1.3) - Speech speed multiplier (default: 1.0)",
                        "output_format": "string - Audio format (default: wav)"
                    },
                    outputs={
                        "content_type": "audio/wav",
                        "headers": {
                            "X-Sample-Rate": "Sample rate in Hz (e.g., 24000)",
                            "X-Duration-Sec": "Audio duration in seconds",
                            "X-Engine": "TTS engine used (coqui_xtts)",
                            "X-Speaker": "Speaker name used",
                            "X-Chunks": "Number of text chunks processed"
                        }
                    },
                    example='curl -X POST https://[ENDPOINT]/tts -H "Content-Type: application/json" -d \'{"text": "Hello world", "speaker_id": "Claribel Dervla", "language": "en"}\' --output audio.wav'
                ),
                APIEndpointInfo(
                    endpoint="/speakers",
                    method="GET",
                    description="List all available built-in speakers (58 speakers)",
                    inputs={
                        "refresh": "boolean (optional) - Force rebuild speaker cache (default: false)"
                    },
                    outputs={
                        "content_type": "application/json",
                        "fields": {
                            "speakers": "List of speaker names",
                            "count": "Total number of speakers (58)",
                            "last_updated": "ISO timestamp of cache update",
                            "cache_age_days": "Age of cached data in days"
                        }
                    },
                    example="curl https://[ENDPOINT]/speakers | jq"
                ),
                APIEndpointInfo(
                    endpoint="/voice-clone",
                    method="POST",
                    description="Clone a voice from reference audio and generate speech",
                    inputs={
                        "text": "string (1-5000 chars) - Text to synthesize",
                        "language": "string - Language code (same as /tts)",
                        "reference_audio": "file[] - 1-5 audio files (WAV/MP3/M4A, 3-30s each, <10MB, 6-10s optimal)"
                    },
                    outputs={
                        "content_type": "audio/wav",
                        "headers": {
                            "X-Sample-Rate": "Sample rate in Hz",
                            "X-Duration-Sec": "Audio duration in seconds",
                            "X-Engine": "TTS engine (coqui_xtts)",
                            "X-Mode": "voice_clone",
                            "X-Reference-Count": "Number of reference files used",
                            "X-Validation-Warnings": "Audio quality warnings (if any)"
                        }
                    },
                    example='curl -X POST https://[ENDPOINT]/voice-clone -F "text=Hello from cloned voice" -F "language=en" -F "reference_audio=@voice.wav" --output cloned.wav'
                ),
                APIEndpointInfo(
                    endpoint="/health",
                    method="GET",
                    description="Check service health and model status",
                    inputs={},
                    outputs={
                        "content_type": "application/json",
                        "fields": {
                            "status": "healthy/unhealthy",
                            "model_loaded": "boolean",
                            "speakers_available": "number of speakers",
                            "version": "API version"
                        }
                    },
                    example="curl https://[ENDPOINT]/health | jq"
                )
            ]
        )

    @app.get("/speakers", response_model=SpeakersResponse)
    async def list_speakers(refresh: bool = False):
        """List available speakers with caching.

        Args:
            refresh: Force cache refresh if True

        Returns:
            SpeakersResponse with speakers list and metadata
        """
        try:
            result = await speaker_cache.get_speakers(
                tts_model=engine.tts,
                volume=volume,
                force_refresh=refresh,
            )

            return SpeakersResponse(
                speakers=result["speakers"],
                count=result["count"],
                last_updated=result["last_updated"],
                cache_age_days=result["cache_age_days"],
            )

        except Exception as e:
            logger.error("list_speakers.failed", error=str(e))
            raise HTTPException(
                status_code=500,
                detail=ErrorResponse(
                    error=ErrorDetail(
                        code="speaker_list_failed",
                        message=f"Failed to retrieve speaker list: {str(e)}",
                    )
                ).model_dump(),
            )

    @app.post("/tts")
    async def text_to_speech(request: TTSRequest):
        """Synthesize speech from text using built-in speaker.

        Args:
            request: TTS request with text, speaker_id, language

        Returns:
            WAV audio file
        """
        try:
            logger.info(
                "tts.request",
                speaker=request.speaker_id,
                language=request.language,
                text_len=len(request.text),
            )

            # Chunk text for optimal quality (XTTS warns at 250 chars)
            chunks = chunk_text(request.text, max_chars=200, max_words=60)

            if not chunks:
                raise HTTPException(status_code=400, detail="Empty text input")

            logger.info("tts.chunking", chunk_count=len(chunks))

            # Synthesize each chunk
            audio_chunks = []
            for idx, chunk in enumerate(chunks):
                logger.debug("tts.chunk", index=idx, text=chunk[:50])
                pcm = engine.synthesize_builtin(
                    text=chunk,
                    speaker_id=request.speaker_id,
                    language=request.language,
                )
                audio_chunks.append(pcm)

            # Stitch chunks together
            if len(audio_chunks) > 1:
                combined_pcm = stitch_audio_chunks(
                    audio_chunks,
                    sample_rate=engine.sample_rate,
                    crossfade_ms=40,
                )
            else:
                combined_pcm = audio_chunks[0]

            # Normalize audio
            normalized_pcm = normalize_audio(combined_pcm)

            # Wrap in WAV container
            wav_bytes = wrap_wav(
                normalized_pcm,
                sample_rate=engine.sample_rate,
            )

            # Calculate duration
            duration = estimate_duration(
                normalized_pcm,
                sample_rate=engine.sample_rate,
                sample_width=2,
                channels=1,
            )

            logger.info(
                "tts.complete",
                audio_size=len(wav_bytes),
                duration_sec=duration,
                chunks=len(audio_chunks),
            )

            return Response(
                content=wav_bytes,
                media_type="audio/wav",
                headers={
                    "X-Sample-Rate": str(engine.sample_rate),
                    "X-Duration-Sec": f"{duration:.2f}",
                    "X-Engine": "coqui_xtts",
                    "X-Speaker": request.speaker_id,
                    "X-Chunks": str(len(audio_chunks)),
                },
            )

        except ValueError as e:
            # Invalid speaker or language
            logger.warning("tts.invalid_input", error=str(e))
            speakers = engine.get_speakers()
            raise HTTPException(
                status_code=400,
                detail=ErrorResponse(
                    error=ErrorDetail(
                        code="invalid_speaker",
                        message=str(e),
                        valid_options=speakers[:10],  # First 10 speakers as hint
                    )
                ).model_dump(),
            )

        except Exception as e:
            logger.error("tts.failed", error=str(e))
            raise HTTPException(
                status_code=500,
                detail=ErrorResponse(
                    error=ErrorDetail(
                        code="synthesis_failed",
                        message=f"TTS synthesis failed: {str(e)}",
                    )
                ).model_dump(),
            )

    @app.post("/voice-clone")
    async def voice_clone(
        text: str = Form(..., min_length=1, max_length=5000),
        language: str = Form(default="en"),
        reference_audio: List[UploadFile] = File(...),
    ):
        """Synthesize speech using voice cloning.

        Supports multiple reference audio files for improved quality through interpolation.

        Args:
            text: Text to synthesize
            language: Language code
            reference_audio: List of reference audio files (WAV, MP3, M4A)
                           Multiple files improve cloning quality (1-5 files recommended)

        Returns:
            WAV audio file in cloned voice
        """
        try:
            logger.info(
                "voice_clone.request",
                language=language,
                text_len=len(text),
                ref_audio_count=len(reference_audio),
            )

            # Validate number of reference files
            if not reference_audio:
                raise HTTPException(
                    status_code=400,
                    detail="At least one reference audio file is required",
                )

            if len(reference_audio) > 5:
                raise HTTPException(
                    status_code=400,
                    detail="Maximum 5 reference audio files allowed",
                )

            # Read and validate all reference audio files
            ref_audio_bytes_list = []
            validation_warnings = []

            for idx, audio_file in enumerate(reference_audio):
                # Read file
                audio_bytes = await audio_file.read()

                # Validate reference audio
                validation = validate_reference_audio(audio_bytes, max_size_mb=10.0)

                if not validation.is_valid:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Reference audio {idx+1} invalid: {validation.error_message}",
                    )

                # Collect warnings
                if validation.warning_message:
                    validation_warnings.append(f"File {idx+1}: {validation.warning_message}")

                ref_audio_bytes_list.append(audio_bytes)

                logger.info(
                    "voice_clone.audio_validated",
                    file_index=idx,
                    duration=validation.duration,
                    sample_rate=validation.sample_rate,
                    channels=validation.channels,
                )

            # Chunk text
            chunks = chunk_text(text, max_chars=200, max_words=60)

            if not chunks:
                raise HTTPException(status_code=400, detail="Empty text input")

            logger.info("voice_clone.chunking", chunk_count=len(chunks))

            # Synthesize each chunk with voice cloning
            audio_chunks = []
            for idx, chunk in enumerate(chunks):
                logger.debug("voice_clone.chunk", index=idx, text=chunk[:50])
                pcm = engine.synthesize_clone(
                    text=chunk,
                    reference_audio_bytes=ref_audio_bytes_list,
                    language=language,
                )
                audio_chunks.append(pcm)

            # Stitch chunks
            if len(audio_chunks) > 1:
                combined_pcm = stitch_audio_chunks(
                    audio_chunks,
                    sample_rate=engine.sample_rate,
                    crossfade_ms=40,
                )
            else:
                combined_pcm = audio_chunks[0]

            # Normalize audio
            normalized_pcm = normalize_audio(combined_pcm)

            # Wrap in WAV container
            wav_bytes = wrap_wav(
                normalized_pcm,
                sample_rate=engine.sample_rate,
            )

            # Calculate duration
            duration = estimate_duration(
                normalized_pcm,
                sample_rate=engine.sample_rate,
                sample_width=2,
                channels=1,
            )

            logger.info(
                "voice_clone.complete",
                audio_size=len(wav_bytes),
                duration_sec=duration,
                chunks=len(audio_chunks),
                ref_audio_count=len(ref_audio_bytes_list),
            )

            # Build response headers
            headers = {
                "X-Sample-Rate": str(engine.sample_rate),
                "X-Duration-Sec": f"{duration:.2f}",
                "X-Engine": "coqui_xtts",
                "X-Mode": "voice_clone",
                "X-Chunks": str(len(audio_chunks)),
                "X-Reference-Count": str(len(ref_audio_bytes_list)),
            }

            # Add validation warnings if any
            if validation_warnings:
                headers["X-Validation-Warnings"] = "; ".join(validation_warnings)

            return Response(
                content=wav_bytes,
                media_type="audio/wav",
                headers=headers,
            )

        except HTTPException:
            raise

        except Exception as e:
            logger.error("voice_clone.failed", error=str(e))
            raise HTTPException(
                status_code=500,
                detail=ErrorResponse(
                    error=ErrorDetail(
                        code="voice_clone_failed",
                        message=f"Voice cloning failed: {str(e)}",
                    )
                ).model_dump(),
            )
