"""
Pydantic schemas for Tetrax ASR REST and WebSocket payloads.
"""

from pydantic import BaseModel, Field
from typing import Optional, List


class Segment(BaseModel):
    id: int
    start: float
    end: float
    text: str


class UsageInfo(BaseModel):
    audio_seconds: float


class TranscriptionResponse(BaseModel):
    id: str = Field(..., description="Unique transcription task identifier")
    model: str = Field(..., description="Model identifier used for transcription")
    language: str = Field(default="en", description="Detected or specified language")
    text: str = Field(..., description="Full transcribed text")
    duration: float = Field(..., description="Total audio duration in seconds")
    segments: List[Segment] = Field(default_factory=list, description="Timestamped transcription segments")
    processing_time: Optional[float] = Field(default=None, description="Time taken to process in seconds")
    rtf: Optional[float] = Field(default=None, description="Real-Time Factor (processing_time / duration)")
    usage: UsageInfo = Field(..., description="Billable usage metrics")


class ModelInfo(BaseModel):
    id: str
    name: str
    base_checkpoint: str
    is_finetuned: bool
    adapter_path: Optional[str] = None
    language_focus: str
    architecture: str
    device: str
    compute_type: str


class ModelsListResponse(BaseModel):
    active_model: str
    models: List[ModelInfo]


class StreamEvent(BaseModel):
    event: str  # 'ready', 'interim', 'final', 'speech_start', 'speech_end', 'error'
    text: Optional[str] = None
    start: Optional[float] = None
    end: Optional[float] = None
    duration: Optional[float] = None
    confidence: Optional[float] = None
    detail: Optional[str] = None
