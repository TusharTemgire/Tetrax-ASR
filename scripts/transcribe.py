"""
CLI Script to transcribe Marathi and Hindi audio using fine-tuned TETRAX-ASR-MODEL-769M.

Usage:
    python scripts/transcribe.py --audio "path/to/audio.wav" --lang mr
    python scripts/transcribe.py --audio "path/to/hindi_sample.mp3" --lang hi
"""

import sys
import time
import argparse
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tetrax_asr.config import model_config
from tetrax_asr.inference.engine import TetraxWhisperEngine


def main():
    parser = argparse.ArgumentParser(
        description="Transcribe audio with fine-tuned TETRAX-ASR Whisper Medium (769M)"
    )
    parser.add_argument(
        "--audio",
        "-a",
        type=str,
        required=True,
        help="Path to the WAV, MP3, or FLAC audio file to transcribe."
    )
    parser.add_argument(
        "--lang",
        "-l",
        type=str,
        default="mr",
        choices=["mr", "hi"],
        help="Target language code: 'mr' (Marathi) or 'hi' (Hindi). Default: mr"
    )
    parser.add_argument(
        "--model_dir",
        "-m",
        type=str,
        default=model_config.adapter_path,
        help="Path to the fine-tuned LoRA checkpoint directory."
    )
    parser.add_argument(
        "--timestamps",
        action="store_true",
        help="Whether to output word/segment timestamps."
    )

    args = parser.parse_args()
    audio_path = Path(args.audio)

    if not audio_path.exists():
        print(f"❌ Error: Audio file not found at: {audio_path}")
        sys.exit(1)

    print("=" * 65)
    print("🎙️  TETRAX ASR — Tetrax-ASR-Open-764M-A19M")
    print(f"📁 Model Checkpoint : {args.model_dir}")
    print(f"🎵 Audio File        : {audio_path.name}")
    print(f"🌐 Target Language  : {'Marathi (mr)' if args.lang == 'mr' else 'Hindi (hi)'}")
    print("=" * 65)

    print("\n⏳ Initializing Whisper Medium + LoRA engine (this may take a few seconds)...")
    start_load = time.time()
    engine = TetraxWhisperEngine(adapter_path=args.model_dir)
    print(f"✅ Engine ready in {time.time() - start_load:.2f}s (Fine-tuned adapter attached: {engine.is_finetuned})\n")

    print("🔊 Transcribing audio...")
    start_transcribe = time.time()
    result = engine.transcribe_file(
        str(audio_path),
        language=args.lang,
        return_timestamps=args.timestamps
    )
    duration = time.time() - start_transcribe

    print("\n" + "=" * 65)
    print("📝 TRANSCRIPTION RESULT:")
    print("=" * 65)
    print(result.get("text", "").strip())
    print("=" * 65)
    print(f"⚡ Processing Time : {duration:.2f} seconds")
    print(f"⏱️ Audio Duration  : {result.get('duration', 'N/A')} seconds")
    print(f"📊 RTF (Real-Time Factor): {result.get('rtf', 'N/A')}")
    print("=" * 65)


if __name__ == "__main__":
    main()
