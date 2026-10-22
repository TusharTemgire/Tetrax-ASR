"""
Transcription REST Route: POST /v1/audio/transcriptions
Supports WAV, MP3, M4A, OGG, FLAC, WEBM file uploads.
"""

import uuid
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Request
from typing import Optional

from tetrax_asr.api.schemas.audio import TranscriptionResponse, Segment, UsageInfo

router = APIRouter(prefix="/v1/audio", tags=["Audio"])


@router.post("/transcriptions", response_model=TranscriptionResponse)
async def transcribe_audio(
    request: Request,
    file: UploadFile = File(..., description="Audio file (WAV, MP3, M4A, WEBM, FLAC)"),
    model: Optional[str] = Form(default="tetrax-stt-marathi-hindi"),
    language: Optional[str] = Form(default="mr"),
    timestamps: Optional[bool] = Form(default=True),
):
    engine = getattr(request.app.state, "engine", None)
    if not engine:
        raise HTTPException(status_code=503, detail="Inference engine is not initialized.")

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty audio file uploaded.")

    try:
        res = engine.transcribe(
            audio_input=content,
            language=language if language != "auto" else None,
            return_timestamps=timestamps if timestamps is not None else True
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")

    task_id = f"tr_{uuid.uuid4().hex[:12]}"
    segments = [
        Segment(
            id=s["id"],
            start=s["start"],
            end=s["end"],
            text=s["text"]
        )
        for s in res.get("segments", [])
    ]

    model_display_name = "Tetrax-ASR-Open-764M-A19M" if engine.is_finetuned else f"{model} (base)"

    return TranscriptionResponse(
        id=task_id,
        model=model_display_name,
        language=res.get("language", language or "mr"),
        text=res.get("text", ""),
        duration=res.get("duration", 0.0),
        segments=segments,
        processing_time=res.get("processing_time"),
        rtf=res.get("rtf"),
        usage=UsageInfo(audio_seconds=res.get("duration", 0.0))
    )
