"""Pydantic models for the Dia2 TTS API."""

from datetime import datetime
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator

from dia_service.constants import DEFAULT_DIA_MODEL_SIZE


class GenerationControls(BaseModel):
    """Common generation controls accepted by Dia2 endpoints."""

    model_size: Literal["1b", "2b"] = Field(
        default=DEFAULT_DIA_MODEL_SIZE,
        description="Dia2 checkpoint size. 2b is experimental and more expensive.",
    )
    temperature: float = Field(default=0.8, gt=0.0, le=2.0)
    top_k: int = Field(default=50, ge=1, le=500)
    cfg_scale: float = Field(default=2.0, gt=0.0, le=10.0)
    seed: Optional[int] = Field(default=None, ge=0)


class TTSRequest(GenerationControls):
    """Simple low-input text-to-speech request."""

    text: str = Field(..., min_length=1, max_length=5000)
    voice_profile_id: Optional[str] = Field(default=None, min_length=1, max_length=128)
    style: Optional[str] = Field(default=None, max_length=120)


class DialogueRequest(GenerationControls):
    """Multi-speaker Dia2 dialogue request."""

    script: str = Field(..., min_length=1, max_length=7000)
    speaker_profiles: Dict[str, str] = Field(default_factory=dict)

    @field_validator("speaker_profiles")
    @classmethod
    def validate_speaker_profiles(cls, value: Dict[str, str]) -> Dict[str, str]:
        valid_speakers = {"S1", "S2"}
        invalid = sorted(set(value) - valid_speakers)
        if invalid:
            raise ValueError(f"Only S1 and S2 speaker profile mappings are supported: {invalid}")
        return value


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether Dia2 is loaded")
    model: str = Field(..., description="Model checkpoint name")
    gpu: str = Field(..., description="Runtime GPU/device")
    version: str = Field(..., description="Service version")


class VoiceProfileCreateResponse(BaseModel):
    """Response returned after creating a reusable voice profile."""

    id: str
    name: str
    created_at: datetime


class VoiceProfileSummary(BaseModel):
    """Compact voice profile listing entry."""

    id: str
    name: str
    gender: Optional[str] = None
    accent: Optional[str] = None
    language: str = "en"
    style_tags: List[str] = Field(default_factory=list)
    use_case: Optional[str] = None
    quality_rating: Optional[int] = None
    created_at: datetime


class VoiceProfileDetail(VoiceProfileSummary):
    """Full voice profile metadata."""

    reference_transcript: str
    reference_audio_path: str
    reference_duration_sec: Optional[float] = None
    reference_sample_rate: Optional[int] = None
    notes: Optional[str] = None
    consent_confirmed: bool


class VoiceProfilesResponse(BaseModel):
    """Voice profile list response."""

    profiles: List[VoiceProfileSummary]
    count: int


class DeleteVoiceProfileResponse(BaseModel):
    """Delete voice profile response."""

    deleted: bool
    id: str


class ErrorDetail(BaseModel):
    """Standard error payload."""

    code: str
    message: str
    valid_options: Optional[List[str]] = None


class ErrorResponse(BaseModel):
    """Standard error response."""

    error: ErrorDetail


class APIEndpointInfo(BaseModel):
    """Information about a single API endpoint."""

    endpoint: str
    method: str
    description: str
    inputs: dict
    outputs: dict
    example: Optional[str] = None


class APIInfoResponse(BaseModel):
    """API documentation response."""

    service: str
    version: str
    model: str
    endpoints: List[APIEndpointInfo]
    notes: List[str] = Field(default_factory=list)
