"""FastAPI route handlers for Dia 1.6B TTS API."""

from __future__ import annotations

import json
import tempfile
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import structlog
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response

from dia16_service.constants import (
    DIA16_MODEL_NAME,
    DIA16_REFERENCE_MAX_SEC,
    DIA16_REFERENCE_MIN_SEC,
    DIA16_REFERENCE_OPTIMAL_MAX_SEC,
    DIA16_REFERENCE_OPTIMAL_MIN_SEC,
    DIA16_SAMPLE_RATE,
)
from dia16_service.engine import Dia16Engine
from dia16_service.models import (
    APIEndpointInfo,
    APIInfoResponse,
    DeleteVoiceProfileResponse,
    DialogueRequest,
    ErrorDetail,
    ErrorResponse,
    HealthResponse,
    PredefinedVoiceProfile,
    PredefinedVoiceProfilesResponse,
    TTSRequest,
    VoiceProfileCreateResponse,
    VoiceProfileDetail,
    VoiceProfilesResponse,
    VoiceProfileSummary,
)
from dia16_service.profile_store import PredefinedVoiceStore, VoiceProfile, VoiceProfileStore
from shared.audio import (
    AudioValidationResult,
    normalize_audio,
    validate_audio_duration,
    validate_audio_quality,
    wrap_wav,
)

logger = structlog.get_logger()


def create_routes(
    app: FastAPI,
    engine: Dia16Engine,
    profile_store: VoiceProfileStore,
    volume,
    predefined_voice_store: Optional[PredefinedVoiceStore] = None,
) -> None:
    """Register Dia 1.6B API routes."""
    predefined_voice_store = predefined_voice_store or PredefinedVoiceStore()

    def resolve_prompt_profile(profile_id: str) -> tuple[str, str]:
        try:
            profile = profile_store.get_profile(profile_id)
            return profile.reference_audio_path, profile.reference_transcript
        except KeyError:
            profile = predefined_voice_store.get_profile(profile_id)
            return profile["reference_audio_path"], _ensure_reference_transcript_tag(
                profile["reference_transcript"]
            )

    @app.get("/health", response_model=HealthResponse)
    async def health_check():
        """Check service health and model load status."""
        return HealthResponse(
            status="healthy",
            model_loaded=engine._loaded,
            model=DIA16_MODEL_NAME,
            gpu=engine.device or "unknown",
            version="0.1.0",
        )

    @app.get("/api-info", response_model=APIInfoResponse)
    async def api_info():
        """Return self-documenting API usage information."""
        return APIInfoResponse(
            service="Dia 1.6B High-Fidelity TTS API",
            version="0.1.0",
            model=DIA16_MODEL_NAME,
            notes=[
                "Dia16 is English-only, high-fidelity batch WAV generation at 44.1 kHz.",
                "The current official Dia 1.6B model card says English generation only; profile language is restricted to en.",
                "This is a separate service from Dia2 and uses separate voice profiles.",
                "The 43 predefined profiles are preloaded third-party audio prompts from devnen/Dia-TTS-Server, not native Nari model voices.",
                "Voice conditioning uses reference_audio plus reference_transcript prepended to the target script.",
                "For best cloning, reference_audio must be 5-20 seconds and reference_transcript must start with [S1].",
                "cfg_scale is accepted for cross-service compatibility but the Transformers Dia16 path may ignore it.",
                "Realtime streaming and voice conversion are not provided by this service.",
            ],
            endpoints=[
                APIEndpointInfo(
                    endpoint="/health",
                    method="GET",
                    description="Check service health, lazy model-load state, and runtime device.",
                    inputs={},
                    outputs={
                        "content_type": "application/json",
                        "fields": {
                            "status": "healthy/unhealthy",
                            "model_loaded": "false until the first generation request lazy-loads Dia16",
                            "model": DIA16_MODEL_NAME,
                            "gpu": "not_loaded/cuda/cpu/unknown",
                            "version": "service version",
                        },
                    },
                    example="curl https://[ENDPOINT]/health | jq",
                ),
                APIEndpointInfo(
                    endpoint="/tts",
                    method="POST",
                    description="Generate simple single-speaker speech from text, optionally using a saved Dia16 voice profile.",
                    inputs={
                        "text": "string, 1-5000 chars, required",
                        "voice_profile_id": "optional Dia16 voice profile id",
                        "predefined_voice_id": "optional preloaded voice id from GET /predefined-voice-profiles",
                        "style": "optional short nonverbal/style hint, prepended conservatively",
                        "temperature": "float, default 0.8",
                        "top_k": "integer, default 50",
                        "top_p": "float, default 0.95",
                        "cfg_scale": "float, accepted for compatibility",
                        "max_new_tokens": "integer, default 1024; roughly controls max output length",
                        "seed": "optional integer",
                    },
                    outputs={"content_type": "audio/wav", "headers": _audio_header_docs()},
                    example=(
                        'curl -X POST https://[ENDPOINT]/tts '
                        '-H "Content-Type: application/json" '
                        '-d \'{"text":"Hello from Dia sixteen.","max_new_tokens":512}\' --output dia16.wav'
                    ),
                ),
                APIEndpointInfo(
                    endpoint="/dialogue",
                    method="POST",
                    description="Generate Dia-native [S1]/[S2] dialogue.",
                    inputs={
                        "script": "string with [S1]/[S2] tags, required",
                        "speaker_profiles": "optional object mapping S1/S2 to saved Dia16 profile ids",
                        "temperature": "float, default 0.8",
                        "top_k": "integer, default 50",
                        "top_p": "float, default 0.95",
                        "cfg_scale": "float, accepted for compatibility",
                        "max_new_tokens": "integer, default 1024",
                        "seed": "optional integer",
                    },
                    outputs={"content_type": "audio/wav", "headers": _audio_header_docs()},
                    example=(
                        'curl -X POST https://[ENDPOINT]/dialogue '
                        '-H "Content-Type: application/json" '
                        '-d \'{"script":"[S1] Hi.\\n[S2] Hello.","max_new_tokens":512}\' --output dialogue.wav'
                    ),
                ),
                APIEndpointInfo(
                    endpoint="/tts-with-upload",
                    method="POST",
                    description="One-shot Dia16 voice-conditioned TTS without saving a profile.",
                    inputs={
                        "text": "form string, 1-5000 chars, required",
                        "reference_audio": "WAV file, 5-20s; 5-10s recommended for cloning quality",
                        "reference_transcript": "exact transcript, required, must start with [S1]; prepended to target text",
                        "temperature/top_k/top_p/cfg_scale/max_new_tokens/seed": "optional generation controls",
                    },
                    outputs={"content_type": "audio/wav", "headers": _audio_header_docs()},
                    example=(
                        'curl -X POST https://[ENDPOINT]/tts-with-upload '
                        '-F "text=Hello in this voice." '
                        '-F "reference_transcript=[S1] Reference words." '
                        '-F "reference_audio=@voice.wav" --output conditioned.wav'
                    ),
                ),
                APIEndpointInfo(
                    endpoint="/voice-profiles",
                    method="POST",
                    description="Create a reusable Dia16 voice profile from reference audio and labels.",
                    inputs={
                        "name": "form string, required",
                        "reference_audio": "WAV file, 5-20s; 5-10s recommended for cloning quality",
                        "reference_transcript": "exact transcript, required, must start with [S1]; stored for future conditioning",
                        "gender/accent/language/style_tags/use_case/quality_rating/notes": "profile metadata",
                        "consent_confirmed": "boolean, must be true",
                    },
                    outputs={"content_type": "application/json"},
                    example=(
                        'curl -X POST https://[ENDPOINT]/voice-profiles '
                        '-F "name=Expressive Narrator" -F "gender=female" '
                        '-F "accent=Indian English" -F "language=en" '
                        '-F "style_tags=expressive,dialogue,narrator" '
                        '-F "reference_transcript=[S1] Welcome to this walkthrough." '
                        '-F "consent_confirmed=true" -F "reference_audio=@voice.wav"'
                    ),
                ),
                APIEndpointInfo(
                    endpoint="/voice-profiles",
                    method="GET",
                    description="List saved reusable Dia16 voice profiles.",
                    inputs={},
                    outputs={
                        "content_type": "application/json",
                        "fields": {
                            "profiles": "profile summaries with id, name, labels, and created_at",
                            "count": "number of profiles",
                        },
                    },
                    example="curl https://[ENDPOINT]/voice-profiles | jq",
                ),
                APIEndpointInfo(
                    endpoint="/predefined-voice-profiles",
                    method="GET",
                    description="List preloaded Dia16 predefined voice prompt profiles.",
                    inputs={},
                    outputs={
                        "content_type": "application/json",
                        "fields": {
                            "profiles": "preloaded profiles with id, name, source, transcript, and reference paths",
                            "count": "number of preloaded profiles; expected 43 after running the downloader",
                        },
                    },
                    example="curl https://[ENDPOINT]/predefined-voice-profiles | jq",
                ),
                APIEndpointInfo(
                    endpoint="/voice-profiles/{id}",
                    method="GET",
                    description="Inspect one saved Dia16 voice profile.",
                    inputs={"id": "voice profile id returned by POST/GET /voice-profiles"},
                    outputs={"content_type": "application/json"},
                    example="curl https://[ENDPOINT]/voice-profiles/[PROFILE_ID] | jq",
                ),
                APIEndpointInfo(
                    endpoint="/voice-profiles/{id}",
                    method="DELETE",
                    description="Delete one saved Dia16 voice profile and its stored reference audio.",
                    inputs={"id": "voice profile id to delete"},
                    outputs={
                        "content_type": "application/json",
                        "fields": {"deleted": "true when deletion succeeds", "id": "deleted profile id"},
                    },
                    example="curl -X DELETE https://[ENDPOINT]/voice-profiles/[PROFILE_ID]",
                ),
            ],
        )

    @app.post("/tts")
    async def text_to_speech(request: TTSRequest):
        """Generate single-speaker speech."""
        try:
            prompt_audio_path = None
            prompt_transcript = None
            if request.voice_profile_id:
                prompt_audio_path, prompt_transcript = resolve_prompt_profile(
                    request.voice_profile_id
                )
            elif request.predefined_voice_id:
                prompt_audio_path, prompt_transcript = resolve_prompt_profile(
                    request.predefined_voice_id
                )

            result = engine.generate(
                script=_single_speaker_script(request.text, request.style),
                temperature=request.temperature,
                top_k=request.top_k,
                top_p=request.top_p,
                cfg_scale=request.cfg_scale,
                max_new_tokens=request.max_new_tokens,
                seed=request.seed,
                prompt_audio_path=prompt_audio_path,
                prompt_transcript=prompt_transcript,
            )
            return _audio_response(
                result=result,
                mode="tts",
                voice_profile_id=request.voice_profile_id,
                predefined_voice_id=request.predefined_voice_id,
            )

        except KeyError:
            raise _profile_not_found(request.voice_profile_id or request.predefined_voice_id or "")
        except Exception as e:
            logger.error("dia16_tts.failed", error=str(e))
            raise _server_error("tts_failed", f"Dia16 TTS failed: {e}")

    @app.post("/dialogue")
    async def dialogue(request: DialogueRequest):
        """Generate multi-speaker dialogue."""
        try:
            prompt_audio_path = None
            prompt_transcript = None
            if request.speaker_profiles:
                first_profile_id = request.speaker_profiles.get("S1") or next(
                    iter(request.speaker_profiles.values())
                )
                prompt_audio_path, prompt_transcript = resolve_prompt_profile(first_profile_id)

            result = engine.generate(
                script=_ensure_dialogue_script(request.script),
                temperature=request.temperature,
                top_k=request.top_k,
                top_p=request.top_p,
                cfg_scale=request.cfg_scale,
                max_new_tokens=request.max_new_tokens,
                seed=request.seed,
                prompt_audio_path=prompt_audio_path,
                prompt_transcript=prompt_transcript,
            )
            return _audio_response(
                result=result,
                mode="dialogue",
                voice_profile_id=",".join(sorted(request.speaker_profiles.values())) or None,
            )

        except KeyError as e:
            raise _profile_not_found(str(e).strip("'"))
        except Exception as e:
            logger.error("dia16_dialogue.failed", error=str(e))
            raise _server_error("dialogue_failed", f"Dia16 dialogue generation failed: {e}")

    @app.post("/tts-with-upload")
    async def tts_with_upload(
        text: str = Form(..., min_length=1, max_length=5000),
        reference_transcript: str = Form(..., min_length=1, max_length=2000),
        reference_audio: UploadFile = File(...),
        temperature: float = Form(default=0.8),
        top_k: int = Form(default=50),
        top_p: float = Form(default=0.95),
        cfg_scale: float = Form(default=2.0),
        max_new_tokens: int = Form(default=1024),
        seed: Optional[int] = Form(default=None),
    ):
        """Generate one-shot conditioned speech from an uploaded reference voice."""
        temp_path = None
        try:
            audio_bytes = await reference_audio.read()
            validation = _validate_dia16_reference_audio(audio_bytes)
            if not validation.is_valid:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid reference audio: {validation.error_message}",
                )
            clean_reference_transcript = _validate_reference_transcript(reference_transcript)

            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
                temp_file.write(audio_bytes)
                temp_path = temp_file.name

            result = engine.generate(
                script=_single_speaker_script(text),
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                cfg_scale=cfg_scale,
                max_new_tokens=max_new_tokens,
                seed=seed,
                prompt_audio_path=temp_path,
                prompt_transcript=clean_reference_transcript,
            )
            return _audio_response(result=result, mode="tts_upload")

        except HTTPException:
            raise
        except Exception as e:
            logger.error("dia16_tts_upload.failed", error=str(e))
            raise _server_error("tts_upload_failed", f"Dia16 upload-conditioned TTS failed: {e}")
        finally:
            if temp_path:
                Path(temp_path).unlink(missing_ok=True)

    @app.post("/voice-profiles", response_model=VoiceProfileCreateResponse)
    async def create_voice_profile(
        name: str = Form(..., min_length=1, max_length=120),
        reference_transcript: str = Form(..., min_length=1, max_length=2000),
        reference_audio: UploadFile = File(...),
        gender: Optional[str] = Form(default=None),
        accent: Optional[str] = Form(default=None),
        language: str = Form(default="en"),
        style_tags: str = Form(default=""),
        use_case: Optional[str] = Form(default=None),
        quality_rating: Optional[int] = Form(default=None),
        notes: Optional[str] = Form(default=None),
        consent_confirmed: bool = Form(default=False),
    ):
        """Create a reusable Dia16 voice profile."""
        try:
            if language.lower() != "en":
                raise HTTPException(status_code=400, detail="Dia16 v1 only supports English profiles")
            if not consent_confirmed:
                raise HTTPException(
                    status_code=400,
                    detail="consent_confirmed must be true before storing voice data",
                )
            if quality_rating is not None and not 1 <= quality_rating <= 5:
                raise HTTPException(status_code=400, detail="quality_rating must be 1-5")

            audio_bytes = await reference_audio.read()
            validation = _validate_dia16_reference_audio(audio_bytes)
            if not validation.is_valid:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid reference audio: {validation.error_message}",
                )
            clean_reference_transcript = _validate_reference_transcript(reference_transcript)

            profile = profile_store.create_profile(
                name=name,
                reference_audio=audio_bytes,
                reference_transcript=clean_reference_transcript,
                gender=gender,
                accent=accent,
                language=language,
                style_tags=_parse_style_tags(style_tags),
                use_case=use_case,
                quality_rating=quality_rating,
                notes=notes,
                consent_confirmed=consent_confirmed,
                reference_duration_sec=validation.duration,
                reference_sample_rate=validation.sample_rate,
            )
            volume.commit()

            return VoiceProfileCreateResponse(
                id=profile.id,
                name=profile.name,
                created_at=datetime.fromisoformat(profile.created_at),
            )

        except HTTPException:
            raise
        except Exception as e:
            logger.error("dia16_profile.create_failed", error=str(e))
            raise _server_error("profile_create_failed", f"Failed to create voice profile: {e}")

    @app.get("/voice-profiles", response_model=VoiceProfilesResponse)
    async def list_voice_profiles():
        """List saved Dia16 voice profiles."""
        profiles = [_summary(profile) for profile in profile_store.list_profiles()]
        return VoiceProfilesResponse(profiles=profiles, count=len(profiles))

    @app.get("/predefined-voice-profiles", response_model=PredefinedVoiceProfilesResponse)
    async def list_predefined_voice_profiles():
        """List preloaded Dia16 voice prompt profiles."""
        profiles = [
            PredefinedVoiceProfile(**profile)
            for profile in predefined_voice_store.list_profiles()
        ]
        return PredefinedVoiceProfilesResponse(profiles=profiles, count=len(profiles))

    @app.get("/voice-profiles/{profile_id}", response_model=VoiceProfileDetail)
    async def get_voice_profile(profile_id: str):
        """Get one saved Dia16 voice profile."""
        try:
            return _detail(profile_store.get_profile(profile_id))
        except KeyError:
            raise _profile_not_found(profile_id)

    @app.delete("/voice-profiles/{profile_id}", response_model=DeleteVoiceProfileResponse)
    async def delete_voice_profile(profile_id: str):
        """Delete one saved Dia16 voice profile and its reference audio."""
        try:
            profile_store.delete_profile(profile_id)
            volume.commit()
            return DeleteVoiceProfileResponse(deleted=True, id=profile_id)
        except KeyError:
            raise _profile_not_found(profile_id)


def _single_speaker_script(text: str, style: Optional[str] = None) -> str:
    clean_text = text.strip()
    if clean_text.startswith("[S1]") or clean_text.startswith("[S2]"):
        return clean_text
    if style:
        clean_text = f"({style.strip()}) {clean_text}"
    return f"[S1] {clean_text}"


def _ensure_dialogue_script(script: str) -> str:
    clean_script = script.strip()
    if "[S1]" not in clean_script and "[S2]" not in clean_script:
        return f"[S1] {clean_script}"
    return clean_script


def _validate_reference_transcript(reference_transcript: str) -> str:
    clean_transcript = reference_transcript.strip()
    if not clean_transcript.startswith("[S1]"):
        raise HTTPException(
            status_code=400,
            detail="reference_transcript must be the exact reference-audio transcript and start with [S1]",
        )
    return clean_transcript


def _ensure_reference_transcript_tag(reference_transcript: str) -> str:
    clean_transcript = reference_transcript.strip()
    if clean_transcript.startswith("[S1]") or clean_transcript.startswith("[S2]"):
        return clean_transcript
    return f"[S1] {clean_transcript}"


def _validate_dia16_reference_audio(audio_bytes: bytes):
    size_mb = len(audio_bytes) / (1024 * 1024)
    max_size_mb = 10.0
    if size_mb > max_size_mb:
        return AudioValidationResult(
            is_valid=False,
            error_message=f"File too large ({size_mb:.1f}MB). Maximum: {max_size_mb}MB",
        )

    duration_result = validate_audio_duration(
        audio_bytes,
        min_duration=DIA16_REFERENCE_MIN_SEC,
        max_duration=DIA16_REFERENCE_MAX_SEC,
        optimal_min=DIA16_REFERENCE_OPTIMAL_MIN_SEC,
        optimal_max=DIA16_REFERENCE_OPTIMAL_MAX_SEC,
    )
    if not duration_result.is_valid:
        return duration_result

    quality_result = validate_audio_quality(
        audio_bytes,
        min_sample_rate=16000,
        preferred_sample_rate=DIA16_SAMPLE_RATE,
    )
    if not quality_result.is_valid:
        return quality_result

    warnings = []
    if duration_result.warning_message:
        warnings.append(duration_result.warning_message)
    if quality_result.warning_message:
        warnings.append(quality_result.warning_message)

    duration_result.sample_rate = quality_result.sample_rate
    duration_result.channels = quality_result.channels
    duration_result.warning_message = "; ".join(warnings) if warnings else None
    return duration_result


def _audio_response(
    result,
    mode: str,
    voice_profile_id: Optional[str] = None,
    predefined_voice_id: Optional[str] = None,
) -> Response:
    normalized_pcm = normalize_audio(result.pcm_bytes)
    wav_bytes = wrap_wav(normalized_pcm, sample_rate=result.sample_rate)
    headers = {
        "X-Sample-Rate": str(result.sample_rate),
        "X-Duration-Sec": f"{result.duration_sec:.2f}",
        "X-Compute-Sec": f"{result.compute_sec:.2f}",
        "X-Engine": "dia16",
        "X-Model": result.model_name,
        "X-Mode": mode,
    }
    if voice_profile_id:
        headers["X-Voice-Profile-Id"] = voice_profile_id
    if predefined_voice_id:
        headers["X-Predefined-Voice-Id"] = predefined_voice_id
    return Response(content=wav_bytes, media_type="audio/wav", headers=headers)


def _audio_header_docs() -> dict:
    return {
        "X-Sample-Rate": "Output sample rate in Hz; Dia16 should return 44100",
        "X-Duration-Sec": "Generated audio duration",
        "X-Compute-Sec": "Model generation wall-clock seconds",
        "X-Engine": "dia16",
        "X-Model": DIA16_MODEL_NAME,
        "X-Mode": "tts/dialogue/tts_upload",
        "X-Voice-Profile-Id": "Returned when profile conditioning was used",
        "X-Predefined-Voice-Id": "Returned when a preloaded predefined voice was used",
    }


def _parse_style_tags(raw_tags: str) -> List[str]:
    if not raw_tags:
        return []
    try:
        parsed = json.loads(raw_tags)
        if isinstance(parsed, list):
            return [str(tag).strip() for tag in parsed if str(tag).strip()]
    except json.JSONDecodeError:
        pass
    return [tag.strip() for tag in raw_tags.split(",") if tag.strip()]


def _summary(profile: VoiceProfile) -> VoiceProfileSummary:
    return VoiceProfileSummary(
        id=profile.id,
        name=profile.name,
        gender=profile.gender,
        accent=profile.accent,
        language=profile.language,
        style_tags=profile.style_tags,
        use_case=profile.use_case,
        quality_rating=profile.quality_rating,
        created_at=datetime.fromisoformat(profile.created_at),
    )


def _detail(profile: VoiceProfile) -> VoiceProfileDetail:
    return VoiceProfileDetail(
        **_summary(profile).model_dump(),
        reference_transcript=profile.reference_transcript,
        reference_audio_path=profile.reference_audio_path,
        reference_duration_sec=profile.reference_duration_sec,
        reference_sample_rate=profile.reference_sample_rate,
        notes=profile.notes,
        consent_confirmed=profile.consent_confirmed,
    )


def _profile_not_found(profile_id: str) -> HTTPException:
    return HTTPException(
        status_code=404,
        detail=ErrorResponse(
            error=ErrorDetail(
                code="voice_profile_not_found",
                message=f"Dia16 voice profile not found: {profile_id}",
            )
        ).model_dump(),
    )


def _server_error(code: str, message: str) -> HTTPException:
    return HTTPException(
        status_code=500,
        detail=ErrorResponse(error=ErrorDetail(code=code, message=message)).model_dump(),
    )
