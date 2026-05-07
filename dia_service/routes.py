"""FastAPI route handlers for Dia2 TTS API."""

from __future__ import annotations

import json
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import structlog
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response

from dia_service.constants import DIA_MODEL_NAME
from dia_service.engine import DiaEngine
from dia_service.models import (
    APIEndpointInfo,
    APIInfoResponse,
    DeleteVoiceProfileResponse,
    DialogueRequest,
    ErrorDetail,
    ErrorResponse,
    HealthResponse,
    TTSRequest,
    VoiceProfileCreateResponse,
    VoiceProfileDetail,
    VoiceProfilesResponse,
    VoiceProfileSummary,
)
from dia_service.profile_store import VoiceProfile, VoiceProfileStore
from shared.audio import normalize_audio, validate_reference_audio, wrap_wav

logger = structlog.get_logger()


def create_routes(app: FastAPI, engine: DiaEngine, profile_store: VoiceProfileStore, volume) -> None:
    """Register Dia2 API routes."""

    @app.get("/health", response_model=HealthResponse)
    async def health_check():
        """Check service health and model load status."""
        return HealthResponse(
            status="healthy",
            model_loaded=engine._loaded,
            model=DIA_MODEL_NAME,
            gpu=engine.device or "unknown",
            version="0.1.0",
        )

    @app.get("/api-info", response_model=APIInfoResponse)
    async def api_info():
        """Return self-documenting API usage information."""
        return APIInfoResponse(
            service="Dia2 Expressive TTS API",
            version="0.1.0",
            model=DIA_MODEL_NAME,
            notes=[
                "Dia2 v1 here is English-only batch WAV generation.",
                "Dia2 has no built-in fixed voice catalog; /voice-profiles creates one.",
                "Voice conversion remains in vc_service; realtime streaming is deferred.",
                "Saved profiles require consent_confirmed=true.",
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
                            "model_loaded": "false until the first generation request lazy-loads Dia2",
                            "model": DIA_MODEL_NAME,
                            "gpu": "not_loaded/cuda/cpu/unknown",
                            "version": "service version",
                        },
                    },
                    example="curl https://[ENDPOINT]/health | jq",
                ),
                APIEndpointInfo(
                    endpoint="/tts",
                    method="POST",
                    description="Generate simple single-speaker speech from text.",
                    inputs={
                        "text": "string, 1-5000 chars, required",
                        "voice_profile_id": "optional saved Dia2 voice profile id",
                        "style": "optional short nonverbal/style hint, used conservatively",
                        "temperature": "float, default 0.8",
                        "top_k": "integer, default 50",
                        "cfg_scale": "float, default 2.0",
                        "seed": "optional integer",
                    },
                    outputs={
                        "content_type": "audio/wav",
                        "headers": _audio_header_docs(),
                    },
                    example=(
                        'curl -X POST https://[ENDPOINT]/tts '
                        '-H "Content-Type: application/json" '
                        '-d \'{"text":"Hello from Dia2."}\' --output dia.wav'
                    ),
                ),
                APIEndpointInfo(
                    endpoint="/dialogue",
                    method="POST",
                    description="Generate Dia2-native [S1]/[S2] dialogue.",
                    inputs={
                        "script": "string with [S1]/[S2] tags, required",
                        "speaker_profiles": "optional object mapping S1/S2 to saved profile ids",
                        "temperature": "float, default 0.8",
                        "top_k": "integer, default 50",
                        "cfg_scale": "float, default 2.0",
                        "seed": "optional integer",
                    },
                    outputs={
                        "content_type": "audio/wav",
                        "headers": _audio_header_docs(),
                    },
                    example=(
                        'curl -X POST https://[ENDPOINT]/dialogue '
                        '-H "Content-Type: application/json" '
                        '-d \'{"script":"[S1] Hi.\\n[S2] Hello."}\' --output dialogue.wav'
                    ),
                ),
                APIEndpointInfo(
                    endpoint="/tts-with-upload",
                    method="POST",
                    description="One-shot voice-conditioned TTS without saving a profile.",
                    inputs={
                        "text": "form string, 1-5000 chars, required",
                        "reference_audio": "WAV file, 3-30s, 6-10s optimal",
                        "reference_transcript": "form string, stored for audit/logging but Dia2 currently transcribes prefix audio internally",
                        "temperature/top_k/cfg_scale/seed": "optional generation controls",
                    },
                    outputs={
                        "content_type": "audio/wav",
                        "headers": _audio_header_docs(),
                    },
                    example=(
                        'curl -X POST https://[ENDPOINT]/tts-with-upload '
                        '-F "text=Hello in this voice." '
                        '-F "reference_transcript=Reference words." '
                        '-F "reference_audio=@voice.wav" --output conditioned.wav'
                    ),
                ),
                APIEndpointInfo(
                    endpoint="/voice-profiles",
                    method="POST",
                    description="Create a reusable Dia2 voice profile from reference audio and labels.",
                    inputs={
                        "name": "form string, required",
                        "reference_audio": "WAV file, 3-30s, 6-10s optimal",
                        "reference_transcript": "form string, required",
                        "gender/accent/language/style_tags/use_case/quality_rating/notes": "profile metadata",
                        "consent_confirmed": "boolean, must be true",
                    },
                    outputs={"content_type": "application/json"},
                    example=(
                        'curl -X POST https://[ENDPOINT]/voice-profiles '
                        '-F "name=Warm Narrator" -F "gender=female" '
                        '-F "accent=Indian English" -F "language=en" '
                        '-F "style_tags=warm,calm,narrator" '
                        '-F "reference_transcript=Welcome to this walkthrough." '
                        '-F "consent_confirmed=true" -F "reference_audio=@voice.wav"'
                    ),
                ),
                APIEndpointInfo(
                    endpoint="/voice-profiles",
                    method="GET",
                    description="List saved reusable voice profiles.",
                    inputs={},
                    outputs={
                        "content_type": "application/json",
                        "fields": {
                            "profiles": "profile summaries with id, name, gender, accent, language, style_tags, use_case, quality_rating, created_at",
                            "count": "number of profiles",
                        },
                    },
                    example="curl https://[ENDPOINT]/voice-profiles | jq",
                ),
                APIEndpointInfo(
                    endpoint="/voice-profiles/{id}",
                    method="GET",
                    description="Inspect one saved voice profile, including transcript and reference audio metadata.",
                    inputs={
                        "id": "voice profile id returned by POST /voice-profiles or GET /voice-profiles"
                    },
                    outputs={
                        "content_type": "application/json",
                        "fields": {
                            "id/name": "profile identity",
                            "reference_transcript": "stored transcript supplied at profile creation",
                            "reference_audio_path": "Modal Volume path for the stored reference WAV",
                            "reference_duration_sec/reference_sample_rate": "reference audio metadata when available",
                            "gender/accent/language/style_tags/use_case/quality_rating/notes": "caller-supplied labels",
                            "consent_confirmed": "whether storage consent was confirmed",
                        },
                    },
                    example="curl https://[ENDPOINT]/voice-profiles/[PROFILE_ID] | jq",
                ),
                APIEndpointInfo(
                    endpoint="/voice-profiles/{id}",
                    method="DELETE",
                    description="Delete one saved voice profile and its stored reference audio.",
                    inputs={
                        "id": "voice profile id to delete; use when a profile is bad, mistaken, or consent is revoked"
                    },
                    outputs={
                        "content_type": "application/json",
                        "fields": {
                            "deleted": "true when deletion succeeds",
                            "id": "deleted profile id",
                        },
                    },
                    example="curl -X DELETE https://[ENDPOINT]/voice-profiles/[PROFILE_ID]",
                ),
            ],
        )

    @app.post("/tts")
    async def text_to_speech(request: TTSRequest):
        """Generate simple single-speaker speech."""
        try:
            prefix_speaker_1 = None
            if request.voice_profile_id:
                prefix_speaker_1 = profile_store.get_profile(request.voice_profile_id).reference_audio_path

            script = _single_speaker_script(request.text, request.style)
            result = engine.generate(
                script=script,
                temperature=request.temperature,
                top_k=request.top_k,
                cfg_scale=request.cfg_scale,
                seed=request.seed,
                prefix_speaker_1=prefix_speaker_1,
            )

            return _audio_response(
                result=result,
                mode="tts",
                voice_profile_id=request.voice_profile_id,
            )

        except KeyError:
            raise _profile_not_found(request.voice_profile_id or "")
        except Exception as e:
            logger.error("dia_tts.failed", error=str(e))
            raise _server_error("tts_failed", f"Dia2 TTS failed: {e}")

    @app.post("/dialogue")
    async def dialogue(request: DialogueRequest):
        """Generate multi-speaker dialogue."""
        try:
            prefix_paths = _speaker_prefix_paths(request.speaker_profiles, profile_store)
            result = engine.generate(
                script=_ensure_dialogue_script(request.script),
                temperature=request.temperature,
                top_k=request.top_k,
                cfg_scale=request.cfg_scale,
                seed=request.seed,
                prefix_speaker_1=prefix_paths.get("S1"),
                prefix_speaker_2=prefix_paths.get("S2"),
            )

            return _audio_response(
                result=result,
                mode="dialogue",
                voice_profile_id=",".join(sorted(request.speaker_profiles.values())) or None,
            )

        except KeyError as e:
            raise _profile_not_found(str(e).strip("'"))
        except Exception as e:
            logger.error("dia_dialogue.failed", error=str(e))
            raise _server_error("dialogue_failed", f"Dia2 dialogue generation failed: {e}")

    @app.post("/tts-with-upload")
    async def tts_with_upload(
        text: str = Form(..., min_length=1, max_length=5000),
        reference_transcript: str = Form(..., min_length=1, max_length=2000),
        reference_audio: UploadFile = File(...),
        temperature: float = Form(default=0.8),
        top_k: int = Form(default=50),
        cfg_scale: float = Form(default=2.0),
        seed: Optional[int] = Form(default=None),
    ):
        """Generate one-shot conditioned speech from an uploaded reference voice."""
        del reference_transcript  # Dia2 currently performs its own prefix transcription.

        temp_path = None
        try:
            audio_bytes = await reference_audio.read()
            validation = validate_reference_audio(audio_bytes, max_size_mb=10.0)
            if not validation.is_valid:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid reference audio: {validation.error_message}",
                )

            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
                temp_file.write(audio_bytes)
                temp_path = temp_file.name

            result = engine.generate(
                script=_single_speaker_script(text),
                temperature=temperature,
                top_k=top_k,
                cfg_scale=cfg_scale,
                seed=seed,
                prefix_speaker_1=temp_path,
            )

            return _audio_response(result=result, mode="tts_upload")

        except HTTPException:
            raise
        except Exception as e:
            logger.error("dia_tts_upload.failed", error=str(e))
            raise _server_error("tts_upload_failed", f"Dia2 upload-conditioned TTS failed: {e}")
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
        """Create a reusable voice profile."""
        try:
            if language.lower() != "en":
                raise HTTPException(status_code=400, detail="Dia2 v1 only supports English profiles")
            if not consent_confirmed:
                raise HTTPException(
                    status_code=400,
                    detail="consent_confirmed must be true before storing voice data",
                )
            if quality_rating is not None and not 1 <= quality_rating <= 5:
                raise HTTPException(status_code=400, detail="quality_rating must be 1-5")

            audio_bytes = await reference_audio.read()
            validation = validate_reference_audio(audio_bytes, max_size_mb=10.0)
            if not validation.is_valid:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid reference audio: {validation.error_message}",
                )

            profile = profile_store.create_profile(
                name=name,
                reference_audio=audio_bytes,
                reference_transcript=reference_transcript,
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
            logger.error("dia_profile.create_failed", error=str(e))
            raise _server_error("profile_create_failed", f"Failed to create voice profile: {e}")

    @app.get("/voice-profiles", response_model=VoiceProfilesResponse)
    async def list_voice_profiles():
        """List saved voice profiles."""
        profiles = [_summary(profile) for profile in profile_store.list_profiles()]
        return VoiceProfilesResponse(profiles=profiles, count=len(profiles))

    @app.get("/voice-profiles/{profile_id}", response_model=VoiceProfileDetail)
    async def get_voice_profile(profile_id: str):
        """Get one saved voice profile."""
        try:
            return _detail(profile_store.get_profile(profile_id))
        except KeyError:
            raise _profile_not_found(profile_id)

    @app.delete("/voice-profiles/{profile_id}", response_model=DeleteVoiceProfileResponse)
    async def delete_voice_profile(profile_id: str):
        """Delete one saved voice profile and its reference audio."""
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


def _speaker_prefix_paths(
    speaker_profiles: Dict[str, str],
    profile_store: VoiceProfileStore,
) -> Dict[str, str]:
    paths = {}
    for speaker, profile_id in speaker_profiles.items():
        paths[speaker] = profile_store.get_profile(profile_id).reference_audio_path
    return paths


def _audio_response(result, mode: str, voice_profile_id: Optional[str] = None) -> Response:
    normalized_pcm = normalize_audio(result.pcm_bytes)
    wav_bytes = wrap_wav(normalized_pcm, sample_rate=result.sample_rate)
    headers = {
        "X-Sample-Rate": str(result.sample_rate),
        "X-Duration-Sec": f"{result.duration_sec:.2f}",
        "X-Compute-Sec": f"{result.compute_sec:.2f}",
        "X-Engine": "dia2",
        "X-Model": DIA_MODEL_NAME,
        "X-Mode": mode,
    }
    if voice_profile_id:
        headers["X-Voice-Profile-Id"] = voice_profile_id
    return Response(content=wav_bytes, media_type="audio/wav", headers=headers)


def _audio_header_docs() -> dict:
    return {
        "X-Sample-Rate": "Output sample rate in Hz",
        "X-Duration-Sec": "Generated audio duration",
        "X-Compute-Sec": "Model generation wall-clock seconds",
        "X-Engine": "dia2",
        "X-Model": DIA_MODEL_NAME,
        "X-Mode": "tts/dialogue/tts_upload",
        "X-Voice-Profile-Id": "Returned when profile conditioning was used",
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
                message=f"Voice profile not found: {profile_id}",
            )
        ).model_dump(),
    )


def _server_error(code: str, message: str) -> HTTPException:
    return HTTPException(
        status_code=500,
        detail=ErrorResponse(error=ErrorDetail(code=code, message=message)).model_dump(),
    )
