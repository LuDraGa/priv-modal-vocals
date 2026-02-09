"""Pydantic models for WhisperX STT API."""

from typing import List, Optional
from pydantic import BaseModel, Field


# ============================================================================
# Word-level Segment Models
# ============================================================================

class WordSegment(BaseModel):
    """Individual word with precise timestamps."""

    word: str = Field(..., description="The transcribed word")
    start: float = Field(..., description="Start time in seconds")
    end: float = Field(..., description="End time in seconds")
    score: float = Field(..., description="Confidence score (0-1)")


class Segment(BaseModel):
    """Sentence-level segment containing words."""

    text: str = Field(..., description="Full segment text")
    start: float = Field(..., description="Segment start time in seconds")
    end: float = Field(..., description="Segment end time in seconds")
    words: List[WordSegment] = Field(..., description="Word-level timestamps")


# ============================================================================
# Request Models
# ============================================================================

class TranscribeRequest(BaseModel):
    """Request model for transcription (multipart/form-data)."""

    language: Optional[str] = Field(
        None,
        description="Language code (e.g., 'en', 'es', 'fr'). Auto-detected if not provided.",
        example="en"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "language": "en"
            }
        }


# ============================================================================
# Response Models
# ============================================================================

class TranscribeResponse(BaseModel):
    """Response model for transcription with word-level timestamps."""

    text: str = Field(..., description="Full transcribed text")
    segments: List[Segment] = Field(..., description="Segments with word-level timestamps")
    language: str = Field(..., description="Detected or specified language code")
    duration: float = Field(..., description="Audio duration in seconds")

    class Config:
        json_schema_extra = {
            "example": {
                "text": "Hello world, this is a test.",
                "segments": [
                    {
                        "text": "Hello world, this is a test.",
                        "start": 0.0,
                        "end": 2.5,
                        "words": [
                            {"word": "Hello", "start": 0.0, "end": 0.5, "score": 0.98},
                            {"word": "world,", "start": 0.5, "end": 1.0, "score": 0.97},
                            {"word": "this", "start": 1.0, "end": 1.3, "score": 0.99},
                            {"word": "is", "start": 1.3, "end": 1.5, "score": 0.98},
                            {"word": "a", "start": 1.5, "end": 1.6, "score": 0.97},
                            {"word": "test.", "start": 1.6, "end": 2.5, "score": 0.96},
                        ]
                    }
                ],
                "language": "en",
                "duration": 2.5
            }
        }


class LanguageInfo(BaseModel):
    """Information about a supported language."""

    code: str = Field(..., description="ISO 639-1 language code")
    name: str = Field(..., description="Language name in English")


class LanguagesResponse(BaseModel):
    """Response model for supported languages list."""

    languages: List[LanguageInfo] = Field(..., description="List of supported languages")
    total: int = Field(..., description="Total number of supported languages")


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(..., description="Service status", example="healthy")
    model: str = Field(..., description="Loaded Whisper model", example="large-v3-turbo")
    gpu_available: bool = Field(..., description="GPU availability")
    alignment_available: bool = Field(..., description="Alignment model availability")
