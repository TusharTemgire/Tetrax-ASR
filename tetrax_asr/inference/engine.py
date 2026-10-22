"""
Tetrax Whisper Inference Engine
Supports standard Transformers + PEFT (LoRA fine-tuned) and Faster-Whisper (CTranslate2).
Handles file path, audio bytes, and numpy array audio inputs.
"""

import io
import time
import os
from pathlib import Path
from typing import Dict, Any, Optional, Union
import warnings
import logging
import torch
import numpy as np
import soundfile as sf
import librosa

# Force offline loading - strictly prevent any outbound HTTP pings to Hugging Face
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

# Silence noisy HTTP connection and cache checking logs
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)

# Suppress informational generation deprecation notices
warnings.filterwarnings("ignore", message=".*max_new_tokens.*")
warnings.filterwarnings("ignore", message=".*forced_decoder_ids.*")
warnings.filterwarnings("ignore", message=".*clean_up_tokenization_spaces.*")
warnings.filterwarnings("ignore", message=".*generation_config.*")

try:
    from transformers.utils import logging as hf_logging
    hf_logging.set_verbosity_error()
    hf_logging.disable_progress_bar()
except Exception:
    pass

from tetrax_asr.config import model_config, audio_config, BASE_DIR, CHECKPOINTS_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TetraxASR.Inference")


class TetraxWhisperEngine:
    def __init__(
        self,
        base_model_id: Optional[str] = None,
        adapter_path: Optional[str] = None,
        device: Optional[str] = None,
        use_faster_whisper: Optional[bool] = None,
    ):
        self.base_model_id = base_model_id or model_config.base_model_id
        self.adapter_path = adapter_path or model_config.adapter_path
        self.device = device or model_config.device
        self.use_faster_whisper = (
            use_faster_whisper
            if use_faster_whisper is not None
            else model_config.use_faster_whisper
        )

        self.processor = None
        self.model = None
        self.fw_model = None
        self.is_finetuned = False
        self._load_engine()

    def _load_engine(self):
        """Loads Whisper model and attaches PEFT adapter if present."""
        if self.use_faster_whisper:
            try:
                from faster_whisper import WhisperModel
                logger.info(f"Loading Faster-Whisper model: {self.base_model_id} on {self.device}...")
                compute_type = "float16" if self.device == "cuda" else "int8"
                self.fw_model = WhisperModel(
                    self.base_model_id,
                    device=self.device,
                    compute_type=compute_type
                )
                logger.info("Faster-Whisper engine ready.")
                return
            except Exception as e:
                logger.warning(f"Could not initialize Faster-Whisper ({e}), falling back to PyTorch Transformers.")
        # Check if user has generated a 100% standalone merged model
        merged_path = BASE_DIR / "Tetrax-ASR-Merged-764M"
        is_merged = (merged_path / "model.safetensors").exists()
        if is_merged:
            logger.info(f"Found standalone merged model at: {merged_path}")
            self.base_model_id = str(merged_path)

        from transformers import WhisperProcessor, WhisperForConditionalGeneration

        logger.info(f"Loading WhisperProcessor for {self.base_model_id} (offline)...")
        try:
            self.processor = WhisperProcessor.from_pretrained(
                self.base_model_id,
                language=model_config.language,
                task=model_config.task,
                local_files_only=True,
            )
        except Exception:
            self.processor = WhisperProcessor.from_pretrained(
                self.base_model_id,
                language=model_config.language,
                task=model_config.task,
            )

        torch_dtype = torch.float16 if self.device == "cuda" else torch.float32
        logger.info(f"Loading Whisper weights ({self.base_model_id}) on {self.device} ({torch_dtype})...")

        try:
            self.model = WhisperForConditionalGeneration.from_pretrained(
                self.base_model_id,
                torch_dtype=torch_dtype,
                low_cpu_mem_usage=True,
                local_files_only=True,
            ).to(self.device)
        except Exception as e:
            logger.warning(f"Could not load local weights directly ({e}), checking fallback...")
            try:
                self.model = WhisperForConditionalGeneration.from_pretrained(
                    self.base_model_id,
                    torch_dtype=torch_dtype,
                    low_cpu_mem_usage=True,
                ).to(self.device)
            except Exception as e2:
                logger.warning(f"Failed loading {self.base_model_id} ({e2}), falling back to {model_config.fallback_model_id}")
                self.base_model_id = model_config.fallback_model_id
                self.processor = WhisperProcessor.from_pretrained(
                    self.base_model_id,
                    language=model_config.language,
                    task=model_config.task,
                    local_files_only=True,
                )
                self.model = WhisperForConditionalGeneration.from_pretrained(
                    self.base_model_id,
                    torch_dtype=torch_dtype,
                    low_cpu_mem_usage=True,
                    local_files_only=True,
                ).to(self.device)

        if is_merged:
            self.is_finetuned = True
            self.model.eval()
            if hasattr(self.model, "generation_config") and self.model.generation_config is not None:
                self.model.generation_config.max_length = None
            logger.info("Tetrax Whisper engine initialized with standalone merged model.")
            return

        # Check for LoRA adapter checkpoint
        candidate_paths = [
            self.adapter_path,
            str(BASE_DIR / "Tetrax-ASR-Open-764M-A19M"),
            str(CHECKPOINTS_DIR / "Tetrax-ASR-Open-764M-A19M"),
            str(BASE_DIR / "TETRAX-ASR_MODEL-769M"),
            str(BASE_DIR / "TETRAX-ASR-WHISPER-MEDIUM-769M"),
            str(CHECKPOINTS_DIR / "TETRAX-ASR_MODEL-769M"),
            str(CHECKPOINTS_DIR / "TETRAX-ASR-WHISPER-MEDIUM-769M"),
            str(CHECKPOINTS_DIR / "tetrax-whisper-medium-marathi-hindi"),
            str(Path(self.adapter_path).parent / "Tetrax-ASR-Open-764M-A19M") if self.adapter_path else None,
            str(Path(self.adapter_path).parent / "TETRAX-ASR_MODEL-769M") if self.adapter_path else None,
            str(Path(self.adapter_path).parent / "TETRAX-ASR-WHISPER-MEDIUM-769M") if self.adapter_path else None,
        ]
        chosen_adapter = None
        for p in candidate_paths:
            if p and os.path.exists(p) and os.path.exists(os.path.join(p, "adapter_config.json")):
                chosen_adapter = p
                break

        if chosen_adapter:
            try:
                from peft import PeftModel
                logger.info(f"Loading fine-tuned LoRA adapter from: {chosen_adapter}...")
                self.model = PeftModel.from_pretrained(self.model, chosen_adapter, local_files_only=True)
                self.is_finetuned = True
                self.adapter_path = chosen_adapter
                logger.info("Successfully attached Tetrax Marathi + Hindi Whisper-Medium LoRA fine-tuned weights!")
            except Exception as e:
                logger.error(f"Error loading LoRA adapter ({e}), continuing with base model.")
                self.is_finetuned = False
        else:
            logger.info("No fine-tuned LoRA adapter found at configured path. Running with base model.")
            self.is_finetuned = False

        self.model.eval()
        if hasattr(self.model, "generation_config") and self.model.generation_config is not None:
            self.model.generation_config.max_length = None
        logger.info("Tetrax Whisper engine initialized and ready.")

    def load_audio_array(self, audio_input: Union[str, bytes, io.BytesIO, np.ndarray], target_sr: int = 16000) -> np.ndarray:
        """
        Converts various audio inputs (file path, raw bytes, BytesIO, or numpy array)
        into a 16kHz float32 mono numpy array.
        """
        if isinstance(audio_input, np.ndarray):
            audio = audio_input.astype(np.float32)
            if audio.ndim > 1:
                audio = np.mean(audio, axis=-1)
            return audio

        if isinstance(audio_input, (bytes, bytearray)):
            audio_input = io.BytesIO(audio_input)

        try:
            # Try soundfile first
            audio, sr = sf.read(audio_input, dtype="float32")
            if audio.ndim > 1:
                audio = np.mean(audio, axis=1)
            if sr != target_sr:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
            return audio
        except Exception:
            # Fallback to librosa
            if isinstance(audio_input, io.BytesIO):
                audio_input.seek(0)
            audio, _ = librosa.load(audio_input, sr=target_sr, mono=True)
            return audio

    def transcribe(
        self,
        audio_input: Union[str, bytes, io.BytesIO, np.ndarray],
        language: Optional[str] = "mr",
        return_timestamps: bool = True,
    ) -> Dict[str, Any]:
        """
        Transcribes audio and returns text, language, duration, segments, and latency metrics.
        Supports Marathi ('mr'), Hindi ('hi'), or None (for automatic language detection).
        """
        t_start = time.perf_counter()
        audio = self.load_audio_array(audio_input, target_sr=audio_config.sample_rate)
        duration = len(audio) / float(audio_config.sample_rate)

        if duration < 0.1:
            return {
                "text": "",
                "language": language or "mr",
                "duration": duration,
                "segments": [],
                "processing_time": 0.0,
                "rtf": 0.0
            }

        if self.use_faster_whisper and self.fw_model:
            segments_gen, info = self.fw_model.transcribe(
                audio,
                language=language,
                beam_size=5,
                word_timestamps=return_timestamps
            )
            detected_lang = getattr(info, "language", language) or language
            segments = []
            full_text_parts = []
            for idx, s in enumerate(segments_gen):
                segments.append({
                    "id": idx,
                    "start": round(s.start, 2),
                    "end": round(s.end, 2),
                    "text": s.text.strip()
                })
                full_text_parts.append(s.text.strip())
            full_text = " ".join(full_text_parts).strip()
            final_lang = detected_lang
        else:
            # PyTorch Transformers inference
            input_features = self.processor(
                audio,
                sampling_rate=audio_config.sample_rate,
                return_tensors="pt"
            ).input_features

            if self.device == "cuda":
                input_features = input_features.to(self.device).half()
            else:
                input_features = input_features.to(self.device)

            gen_kwargs = {
                "max_new_tokens": 440,
                "return_timestamps": return_timestamps,
            }
            if language:
                gen_kwargs["language"] = language
                gen_kwargs["task"] = "transcribe"

            with torch.no_grad():
                predicted_ids = self.model.generate(input_features, **gen_kwargs)

            full_text = self.processor.batch_decode(
                predicted_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0].strip()
            final_lang = language or "auto"

            # Timestamp segment extraction
            segments = [{
                "id": 0,
                "start": 0.0,
                "end": round(duration, 2),
                "text": full_text
            }]

        t_end = time.perf_counter()
        processing_time = round(t_end - t_start, 3)
        rtf = round(processing_time / max(duration, 0.001), 3)

        return {
            "text": full_text,
            "language": final_lang,
            "duration": round(duration, 2),
            "segments": segments,
            "processing_time": processing_time,
            "rtf": rtf
        }
