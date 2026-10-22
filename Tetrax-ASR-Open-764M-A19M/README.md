---
base_model: openai/whisper-medium
library_name: peft
pipeline_tag: automatic-speech-recognition
language:
- mr
- hi
tags:
- automatic-speech-recognition
- whisper
- indic
- marathi
- hindi
- lora
- peft
- tetrax-asr
license: apache-2.0
---

# 🎙️ Tetrax-ASR-Open-764M-A19M

**Tetrax-ASR-Open-764M-A19M** is a high-accuracy bilingual speech recognition model adapted for **Marathi (`mr`)** and **Hindi (`hi`)** in Devanagari script.

It builds upon **`openai/whisper-medium` (764M base parameters)** and fine-tunes **18.87M active LoRA adapter parameters** on conversational speech from the **AI4Bharat IndicVoices** dataset.

---

## 📊 Parameter & Architecture Specifications

Following open-weights model sizing conventions (e.g. *Muse-Spark-Open-816B-A42B*):

| Specification | Metric / Value |
|---|---|
| **Model Name** | `Tetrax-ASR-Open-764M-A19M` |
| **Base Backbone** | `openai/whisper-medium` |
| **Total Base Parameters** | **763,857,920 (~764M)** |
| **Active / Trainable Parameters (A)** | **18,874,368 (~19M)** (via PEFT / LoRA) |
| **Total Effective Parameters** | **782,732,288 (~783M)** |
| **LoRA Target Modules** | `q_proj`, `k_proj`, `v_proj`, `out_proj` (Encoder & Decoder Attention) |
| **LoRA Hyperparameters** | `r = 32`, `alpha = 64`, `dropout = 0.05` |
| **Precision** | FP16 mixed precision |
| **Target Languages** | Marathi (`mr`) & Hindi (`hi`) |
| **Sampling Rate** | 16,000 Hz (16kHz mono) |
| **Adapter Storage Size** | **~75.6 MB** (`adapter_model.safetensors`) |

---

## 🚀 Quickstart Inference

### 1. Using the Tetrax CLI
```bash
# Transcribe Marathi speech:
python scripts/transcribe.py --audio "sample_mr.wav" --lang mr

# Transcribe Hindi speech:
python scripts/transcribe.py --audio "sample_hi.mp3" --lang hi
```

### 2. Native Transformers + PEFT Python API
```python
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from peft import PeftModel
import soundfile as sf
import librosa

BASE_MODEL = "openai/whisper-medium"
ADAPTER_PATH = "Tetrax-ASR-Open-764M-A19M"

processor = WhisperProcessor.from_pretrained(BASE_MODEL, language="mr", task="transcribe")
model = WhisperForConditionalGeneration.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
)
model = PeftModel.from_pretrained(model, ADAPTER_PATH)
model.eval()

# Load audio
audio, sr = sf.read("sample.wav", dtype="float32")
if sr != 16000:
    audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)

inputs = processor(audio, sampling_rate=16000, return_tensors="pt").input_features
if torch.cuda.is_available():
    inputs = inputs.half().cuda()

with torch.no_grad():
    predicted_ids = model.generate(inputs, language="mr", task="transcribe")

transcript = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
print("Transcript:", transcript)
```

### 3. FastAPI Real-Time WebSocket Streaming Server
```bash
python -m uvicorn tetrax_asr.api.main:app --host 0.0.0.0 --port 8000
```
Open [http://localhost:8000](http://localhost:8000) for the live microphone streaming console.

---

## 🏋️ Training Details

- **Dataset**: AI4Bharat IndicVoices (Marathi & Hindi conversational speech)
- **Loss Function**: Cross-Entropy Loss with label smoothing & sequence padding mask
- **Optimizer**: AdamW (`lr = 1e-4`, `warmup_steps = 50`)
- **Collator**: Sequence-to-Sequence padding with 448-token maximum positional sequence clamp
- **Hardware**: Google Colab Tesla T4 GPU (FP16 mixed precision, `batch_size = 4`, `gradient_accumulation_steps = 2`)