"""
Test Script: Real-Time Audio Streaming Client over WebSocket
Streams audio from a local WAV file in real-time chunks (256ms) to ws://localhost:8000/v1/audio/stream
and prints live interim & final captions.
"""

import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import time
import asyncio
import soundfile as sf
import numpy as np
import websockets
import json

CHUNK_DURATION = 0.256  # 256ms chunks
TARGET_SR = 16000


async def stream_audio_file(file_path: str, ws_url: str = "ws://localhost:8000/v1/audio/stream"):
    path = Path(file_path)
    if not path.exists():
        print(f"Error: Audio file not found at {file_path}")
        return

    audio, sr = sf.read(str(path), dtype="float32")
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    if sr != TARGET_SR:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)
        sr = TARGET_SR

    chunk_size = int(TARGET_SR * CHUNK_DURATION)
    total_chunks = int(np.ceil(len(audio) / chunk_size))
    audio_duration = len(audio) / float(sr)

    print(f"Connecting to Tetrax ASR WebSocket: {ws_url}...")
    print(f"Streaming file: {path.name} ({audio_duration:.2f}s, {total_chunks} chunks)")
    print("-" * 60)

    async with websockets.connect(ws_url) as ws:
        # Task to receive and print responses
        async def receive_events():
            try:
                async for message in ws:
                    data = json.loads(message)
                    event = data.get("event")

                    if event == "ready":
                        print(f"✅ Engine Ready: {data.get('model')} (Session: {data.get('session_id')})\n")
                    elif event == "interim":
                        # Print interim in-place with carriage return
                        sys.stdout.write(f"\r\033[93m[Interim ~{data.get('duration')}s]: {data.get('text')}\033[0m")
                        sys.stdout.flush()
                    elif event == "final":
                        # Print finalized line
                        sys.stdout.write(f"\r\033[92m[Final {data.get('start')}s - {data.get('end')}s]: {data.get('text')}\033[0m\n")
                        sys.stdout.flush()
                    elif event == "completed":
                        print("\nStream completed by server.")
                        break
            except asyncio.CancelledError:
                pass
            except Exception as e:
                print(f"\nReceiver ended: {e}")

        recv_task = asyncio.create_task(receive_events())

        # Send audio in real-time pace
        for i in range(total_chunks):
            start = i * chunk_size
            end = min(start + chunk_size, len(audio))
            chunk = audio[start:end]

            # Convert to 16-bit PCM bytes
            int16_chunk = (chunk * 32767).astype(np.int16)
            await ws.send(int16_chunk.tobytes())

            # Real-time pacing
            await asyncio.sleep(CHUNK_DURATION * 0.95)

        # Notify server to flush and stop
        await ws.send(json.dumps({"action": "stop"}))
        await asyncio.sleep(1.0)
        recv_task.cancel()

    print("-" * 60)
    print("Streaming session finished.")


def main():
    import argparse
    import io
    from tetrax_asr.data.svarah_loader import load_raw_svarah

    parser = argparse.ArgumentParser(description="Tetrax WebSocket Audio Streaming Client")
    parser.add_argument("--file", type=str, default="", help="Path to WAV audio file (optional)")
    parser.add_argument("--url", type=str, default="ws://localhost:8000/v1/audio/stream", help="WebSocket URL")
    args = parser.parse_args()

    fpath = Path(args.file) if args.file else None

    # If no file specified or not found, stream sample from the local Svarah dataset in data/
    if not fpath or not fpath.exists():
        print("No audio file passed. Loading first sample directly from Svarah dataset in data/...")
        dataset = load_raw_svarah()
        sample = dataset[0]
        temp_wav = Path("temp_svarah_stream.wav")
        raw_bytes = sample["audio_filepath"]["bytes"]
        temp_wav.write_bytes(raw_bytes)
        fpath = temp_wav
        print(f"Sample Ground Truth: \"{sample['text']}\"")

    try:
        asyncio.run(stream_audio_file(str(fpath), args.url))
    finally:
        # Cleanup temporary audio if created
        if fpath and fpath.name == "temp_svarah_stream.wav" and fpath.exists():
            fpath.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
