"""Pydantic models for the Dia 1.6B TTS API."""

from datetime import datetime
from typing import Dict, List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


class GenerationControls(BaseModel):
    """Common Dia 1.6B generation controls."""

    temperature: float = Field(default=0.8, gt=0.0, le=2.0)
    top_k: int = Field(default=50, ge=1, le=500)
    top_p: float = Field(default=0.95, gt=0.0, le=1.0)
    cfg_scale: float = Field(default=2.0, gt=0.0, le=10.0)
    max_new_tokens: int = Field(default=1024, ge=64, le=4096)
    seed: Optional[int] = Field(default=None, ge=0)


class TTSRequest(GenerationControls):
    """Simple low-input text-to-speech request."""

    text: str = Field(..., min_length=1, max_length=5000)
    voice_profile_id: Optional[str] = Field(default=None, min_length=1, max_length=128)
    predefined_voice_id: Optional[str] = Field(default=None, min_length=1, max_length=180)
    style: Optional[str] = Field(default=None, max_length=120)

    @field_validator("predefined_voice_id")
    @classmethod
    def normalize_predefined_voice_id(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        return value.strip() or None

    @field_validator("voice_profile_id")
    @classmethod
    def normalize_voice_profile_id(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        return value.strip() or None

    @model_validator(mode="after")
    def validate_single_profile_source(self):
        if self.voice_profile_id and self.predefined_voice_id:
            raise ValueError("Use either voice_profile_id or predefined_voice_id, not both")
        return self


class DialogueRequest(GenerationControls):
    """Multi-speaker Dia 1.6B dialogue request."""

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

    status: str
    model_loaded: bool
    model: str
    gpu: str
    version: str


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


class PredefinedVoiceProfile(BaseModel):
    """Preloaded Dia16 voice prompt profile."""

    id: str
    name: str
    reference_audio_path: str
    transcript_path: str
    reference_transcript: str
    source: str
    language: str = "en"


class PredefinedVoiceProfilesResponse(BaseModel):
    """Predefined voice profile list response."""

    profiles: List[PredefinedVoiceProfile]
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
