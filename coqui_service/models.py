"""Pydantic models for API requests and responses."""

from typing import List, Optional
from pydantic import BaseModel, Field, field_validator
from datetime import datetime


# Request Models

class TTSRequest(BaseModel):
    """Request model for text-to-speech synthesis."""

    text: str = Field(..., min_length=1, max_length=5000, description="Text to synthesize")
    speaker_id: str = Field(..., description="Built-in speaker name (e.g., 'Aaron Dreschner')")
    language: str = Field(default="en", description="Language code (en, es, fr, de, etc.)")
    speed: float = Field(default=1.0, ge=0.7, le=1.3, description="Speech speed multiplier")
    output_format: str = Field(default="wav", description="Audio output format")

    @field_validator("language")
    @classmethod
    def validate_language(cls, v: str) -> str:
        supported = ["en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru", "nl", "cs", "ar", "zh-cn", "ja", "hu", "ko", "hi"]
        if v not in supported:
            raise ValueError(f"Unsupported language: {v}. Supported: {', '.join(supported)}")
        return v


class VoiceCloneRequest(BaseModel):
    """Request model for voice cloning synthesis."""

    text: str = Field(..., min_length=1, max_length=5000, description="Text to synthesize")
    language: str = Field(default="en", description="Language code")
    output_format: str = Field(default="wav", description="Audio output format")

    # Note: reference_audio will be handled as multipart/form-data file upload

    @field_validator("language")
    @classmethod
    def validate_language(cls, v: str) -> str:
        supported = ["en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru", "nl", "cs", "ar", "zh-cn", "ja", "hu", "ko", "hi"]
        if v not in supported:
            raise ValueError(f"Unsupported language: {v}. Supported: {', '.join(supported)}")
        return v


# Response Models

class SpeakersResponse(BaseModel):
    """Response model for speaker list endpoint."""

    speakers: List[str] = Field(..., description="List of available speaker names")
    count: int = Field(..., description="Total number of speakers")
    last_updated: datetime = Field(..., description="When speaker list was last refreshed")
    cache_age_days: int = Field(..., description="Age of cached data in days")


class HealthResponse(BaseModel):
    """Response model for health check endpoint."""

    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether TTS model is loaded")
    speakers_available: int = Field(..., description="Number of speakers available")
    version: str = Field(..., description="Service version")


class ErrorDetail(BaseModel):
    """Error detail model."""

    code: str = Field(..., description="Error code")
    message: str = Field(..., description="Human-readable error message")
    valid_options: Optional[List[str]] = Field(None, description="List of valid options (if applicable)")
    request_id: Optional[str] = Field(None, description="Request ID for debugging")


class ErrorResponse(BaseModel):
    """Standard error response model."""

    error: ErrorDetail
