"""
Tetrax ASR Configuration Module
Centralized configuration for model paths, inference engine, audio settings, and API server.
"""

from pathlib import Path
from pydantic import BaseModel
import os
import torch

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
TETRAX_DIR = BASE_DIR / "tetrax_asr"
DATA_DIR = BASE_DIR / "data"
CHECKPOINTS_DIR = BASE_DIR / "checkpoints"
NOTEBOOKS_DIR = BASE_DIR / "notebooks"

# Ensure directories exist
for p in [DATA_DIR, CHECKPOINTS_DIR, NOTEBOOKS_DIR]:
    p.mkdir(parents=True, exist_ok=True)


class AudioConfig(BaseModel):
    sample_rate: int = 16000
    chunk_size_samples: int = 4096  # ~256ms audio chunks for streaming
    min_speech_duration: float = 0.5  # Min speech to trigger transcription
    max_speech_duration: float = 30.0  # Max Whisper window (30s)
    silence_threshold_energy: float = 0.008  # VAD energy threshold
    silence_duration_commit: float = 0.8  # Seconds of silence to commit final segment


def get_default_adapter_path() -> str:
    """Finds the trained LoRA adapter directory automatically."""
    candidates = [
        BASE_DIR / "Tetrax-ASR-Open-764M-A19M",
        CHECKPOINTS_DIR / "Tetrax-ASR-Open-764M-A19M",
        BASE_DIR / "TETRAX-ASR_MODEL-769M",
        BASE_DIR / "TETRAX-ASR-WHISPER-MEDIUM-769M",
        CHECKPOINTS_DIR / "TETRAX-ASR_MODEL-769M",
        CHECKPOINTS_DIR / "TETRAX-ASR-WHISPER-MEDIUM-769M",
        CHECKPOINTS_DIR / "tetrax-whisper-medium-marathi-hindi",
    ]
    for c in candidates:
        if c.exists() and (c / "adapter_config.json").exists():
            return str(c)
    return str(CHECKPOINTS_DIR / "tetrax-whisper-medium-marathi-hindi")


class ModelConfig(BaseModel):
    base_model_id: str = "openai/whisper-medium"
    fallback_model_id: str = "openai/whisper-small"
    adapter_path: str = get_default_adapter_path()
    language: str = "mr"  # Default primary language (supports 'mr', 'hi')
    supported_languages: list[str] = ["mr", "hi"]
    task: str = "transcribe"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type: str = "float16" if torch.cuda.is_available() else "float32"
    use_faster_whisper: bool = False  # Set True if ctranslate2 model is converted


class APIConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: list[str] = ["*"]
    max_upload_size_mb: int = 50
    api_key: str = os.getenv("TETRAX_API_KEY", "trx_voice_test_key")


class TrainingConfig(BaseModel):
    dataset_name: str = "ai4bharat/IndicVoices"
    languages: list[str] = ["mr", "hi"]
    dataset_split: str = "train"
    train_ratio: float = 0.80
    val_ratio: float = 0.10
    test_ratio: float = 0.10
    random_seed: int = 42
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    learning_rate: float = 1e-4
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 1  # Fits Whisper-Medium (769M) safely on 15GB T4 GPU
    gradient_accumulation_steps: int = 8  # Effective batch size = 8
    gradient_checkpointing: bool = True  # Critical for T4 GPU VRAM management
    fp16: bool = torch.cuda.is_available()
    output_dir: str = str(CHECKPOINTS_DIR / "tetrax-whisper-medium-marathi-hindi")


audio_config = AudioConfig()
model_config = ModelConfig()
api_config = APIConfig()
training_config = TrainingConfig()
