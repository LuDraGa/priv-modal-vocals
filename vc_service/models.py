"""Pydantic models for Voice Conversion API requests and responses."""

from typing import List, Optional
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Response model for health check endpoint."""

    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether VC model is loaded")
    version: str = Field(..., description="Service version")


class ErrorDetail(BaseModel):
    """Error detail model."""

    code: str = Field(..., description="Error code")
    message: str = Field(..., description="Human-readable error message")
    valid_options: Optional[List[str]] = Field(None, description="Valid options, if applicable")


class ErrorResponse(BaseModel):
    """Standard error response model."""

    error: ErrorDetail


class APIEndpointInfo(BaseModel):
    """Information about a single API endpoint."""

    endpoint: str = Field(..., description="Endpoint path")
    method: str = Field(..., description="HTTP method")
    description: str = Field(..., description="One-line description")
    inputs: dict = Field(..., description="Input parameters and types")
    outputs: dict = Field(..., description="Output format and headers")
    example: Optional[str] = Field(None, description="Example curl command")


class APIInfoResponse(BaseModel):
    """Response model for API info endpoint."""

    service: str = Field(..., description="Service name")
    version: str = Field(..., description="API version")
    endpoints: List[APIEndpointInfo] = Field(..., description="List of available endpoints")
