"""
Real-Time WebSocket Streaming Route: WS /v1/audio/stream
Receives live audio chunks and emits interim and final transcription events.
"""

import uuid
import json
import base64
import numpy as np
import logging
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from tetrax_asr.inference.streaming import StreamingSession

logger = logging.getLogger("TetraxASR.WebSocket")
router = APIRouter(prefix="/v1/audio", tags=["Streaming"])


@router.websocket("/stream")
async def websocket_stream(websocket: WebSocket, language: str = "mr"):
    await websocket.accept()
    session_id = f"wss_{uuid.uuid4().hex[:8]}"

    engine = getattr(websocket.app.state, "engine", None)
    if not engine:
        await websocket.send_json({
            "event": "error",
            "detail": "Inference engine is not ready on server."
        })
        await websocket.close()
        return

    session = StreamingSession(session_id=session_id, engine=engine, language=language)

    await websocket.send_json({
        "event": "ready",
        "session_id": session_id,
        "model": "Tetrax-ASR-Open-764M-A19M" if engine.is_finetuned else f"{engine.base_model_id} (base)",
        "language": language,
        "sample_rate": 16000,
        "message": f"Connected to Tetrax Real-Time ASR Stream ({'Marathi' if language == 'mr' else 'Hindi'})."
    })

    try:
        while True:
            message = await websocket.receive()

            chunk_np: np.ndarray = None

            # Handle binary audio frame
            if "bytes" in message and message["bytes"]:
                raw_bytes = message["bytes"]
                # Decode 16-bit PCM (Int16) to Float32 [-1.0, 1.0]
                try:
                    int16_arr = np.frombuffer(raw_bytes, dtype=np.int16)
                    chunk_np = int16_arr.astype(np.float32) / 32768.0
                except Exception:
                    # In case raw float32 was sent
                    chunk_np = np.frombuffer(raw_bytes, dtype=np.float32)

            # Handle JSON / text frame
            elif "text" in message and message["text"]:
                try:
                    payload = json.loads(message["text"])
                    action = payload.get("action", "")

                    if action == "stop":
                        final_res = session.finish()
                        if final_res:
                            await websocket.send_json(final_res)
                        await websocket.send_json({"event": "completed", "session_id": session_id})
                        break

                    elif action == "audio" and "data" in payload:
                        raw_bytes = base64.b64decode(payload["data"])
                        int16_arr = np.frombuffer(raw_bytes, dtype=np.int16)
                        chunk_np = int16_arr.astype(np.float32) / 32768.0

                    elif action == "ping":
                        await websocket.send_json({"event": "pong"})
                        continue
                except Exception as ex:
                    logger.warning(f"Error parsing text frame: {ex}")
                    continue

            if chunk_np is not None and len(chunk_np) > 0:
                events = session.add_audio_chunk(chunk_np)
                for evt in events:
                    await websocket.send_json(evt)

    except WebSocketDisconnect:
        logger.info(f"WebSocket client disconnected: {session_id}")
    except Exception as e:
        logger.error(f"WebSocket error in {session_id}: {e}")
        try:
            await websocket.send_json({"event": "error", "detail": str(e)})
        except Exception:
            pass
    finally:
        final_res = session.finish()
        if final_res:
            try:
                await websocket.send_json(final_res)
            except Exception:
                pass
