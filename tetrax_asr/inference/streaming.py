"""
Tetrax Real-Time Audio Streaming Pipeline
Manages continuous chunk reception over WebSockets, energy-based VAD (Voice Activity Detection),
sliding-window decoding, and interim / final caption generation.
"""

import time
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
import logging
from tetrax_asr.config import audio_config

logger = logging.getLogger("TetraxASR.Streamer")


class StreamingSession:
    """
    State tracker for an active real-time streaming WebSocket connection.
    """
    def __init__(self, session_id: str, engine):
        self.session_id = session_id
        self.engine = engine
        self.sample_rate = audio_config.sample_rate
        self.silence_threshold = audio_config.silence_threshold_energy
        self.silence_commit_sec = audio_config.silence_duration_commit

        # Audio buffers
        self.audio_buffer: List[np.ndarray] = []
        self.total_samples_received: int = 0
        self.last_speech_time: float = time.time()
        self.last_interim_time: float = 0.0
        self.is_speaking: bool = False
        self.last_committed_text: str = ""
        self.interim_interval_sec: float = 0.5  # Generate interim captions every 500ms

        # Timing offsets
        self.start_wall_time = time.time()
        self.committed_duration_sec: float = 0.0

    def add_audio_chunk(self, chunk: np.ndarray) -> List[Dict[str, Any]]:
        """
        Process an incoming chunk of 16kHz float32 audio.
        Returns a list of event dictionaries (e.g., 'interim', 'final').
        """
        events = []
        if chunk.ndim > 1:
            chunk = np.mean(chunk, axis=-1)
        chunk = chunk.astype(np.float32)

        # Calculate energy (RMS)
        energy = float(np.sqrt(np.mean(chunk**2))) if len(chunk) > 0 else 0.0
        now = time.time()

        self.audio_buffer.append(chunk)
        self.total_samples_received += len(chunk)
        current_buffer_samples = sum(len(c) for c in self.audio_buffer)
        current_buffer_duration = current_buffer_samples / float(self.sample_rate)

        # VAD detection
        if energy > self.silence_threshold:
            if not self.is_speaking:
                self.is_speaking = True
                events.append({
                    "event": "speech_start",
                    "timestamp": round(now - self.start_wall_time, 2)
                })
            self.last_speech_time = now
        else:
            silence_duration = now - self.last_speech_time
            # Commit segment if silence duration exceeded and we have enough audio
            if self.is_speaking and silence_duration >= self.silence_commit_sec and current_buffer_duration >= audio_config.min_speech_duration:
                final_event = self._commit_buffer()
                if final_event:
                    events.append(final_event)
                self.is_speaking = False

        # Forced commit if buffer gets too long (e.g., > 12 seconds) to avoid high memory/latency
        if current_buffer_duration >= 12.0:
            final_event = self._commit_buffer()
            if final_event:
                events.append(final_event)
            self.is_speaking = False

        # Periodic interim transcription during active speech
        if self.is_speaking and (now - self.last_interim_time) >= self.interim_interval_sec and current_buffer_duration >= 0.8:
            full_audio = np.concatenate(self.audio_buffer)
            interim_res = self.engine.transcribe(full_audio, language="en", return_timestamps=False)
            text = interim_res.get("text", "").strip()
            if text:
                events.append({
                    "event": "interim",
                    "text": text,
                    "duration": round(current_buffer_duration, 2),
                    "timestamp": round(now - self.start_wall_time, 2)
                })
            self.last_interim_time = now

        return events

    def _commit_buffer(self) -> Optional[Dict[str, Any]]:
        """Transcribes accumulated buffer and emits committed 'final' event."""
        if not self.audio_buffer:
            return None

        full_audio = np.concatenate(self.audio_buffer)
        duration = len(full_audio) / float(self.sample_rate)

        # Clear buffer
        self.audio_buffer.clear()

        if duration < audio_config.min_speech_duration:
            return None

        res = self.engine.transcribe(full_audio, language="en", return_timestamps=True)
        text = res.get("text", "").strip()

        if not text:
            return None

        start_time = self.committed_duration_sec
        end_time = start_time + duration
        self.committed_duration_sec = end_time
        self.last_committed_text = text

        return {
            "event": "final",
            "text": text,
            "start": round(start_time, 2),
            "end": round(end_time, 2),
            "duration": round(duration, 2),
            "rtf": res.get("rtf", 0.0),
            "processing_time": res.get("processing_time", 0.0)
        }

    def finish(self) -> Optional[Dict[str, Any]]:
        """Flushes remaining audio on stream end."""
        return self._commit_buffer()
