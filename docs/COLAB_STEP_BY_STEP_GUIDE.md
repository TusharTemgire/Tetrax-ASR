# 🎙️ Google Colab Step-by-Step Training & Usage Guide
## Model: `Tetrax-ASR-Open-764M-A19M` (Marathi + Hindi Bilingual)

This guide provides a complete, step-by-step walkthrough to train **`openai/whisper-medium` (769M parameters)** on **AI4Bharat IndicVoices (Marathi & Hindi)** using Google Colab's Free T4 GPU, save the model as **`Tetrax-ASR-Open-764M-A19M.zip`** directly to your Google Drive, and run it locally in your Tetrax ASR project.

---

## 📊 Whisper Architecture & Parameter Sizes

Whisper models come in different parameter / token capacity sizes:

| Model Tier | Total Parameters | Precision | Memory (FP16) | Target Languages | Fit on Colab T4 (15GB)? |
|---|---|---|---|---|---|
| **Whisper Base** | **~74M** | FP16 | ~150 MB | Multilingual (99+ langs) | ✅ Yes (very lightweight) |
| **Whisper Small** | **~244M (~250M)** | FP16 | ~500 MB | Multilingual / Indian English | ✅ Yes (~4 GB VRAM) |
| **Whisper Medium** *(Current)* | **~769M (~0.8B)** | FP16 | ~1.5 GB | **Marathi (`mr`) + Hindi (`hi`)** | ✅ **Yes with LoRA & Grad Checkpointing** |
| **Whisper Large-v3** | **~1.55B (~1.5B)** | FP16 | ~3.1 GB | Multilingual Benchmark | ⚠️ Requires A100 / High-RAM GPU |

### Why Whisper Medium 769M with LoRA?
- **High Accuracy on Devanagari Scripts**: Captures the phonetic, morphological, and dialectal nuances of spoken Marathi and Hindi far better than Base or Small models.
- **Trains only ~8.3M Parameters (~1.1%)**: The 769M core transformer layers are frozen, while low-rank matrices (`r=32`, `lora_alpha=64`) adapt attention layers (`q_proj`, `v_proj`, `k_proj`, `out_proj`).
- **Compact Output**: Produces an adapter ZIP of only **~80 MB** instead of an unwieldy 3 GB full model!

---

## 🚀 Part 1: Colab Training Step-by-Step

### Step 1: Open Google Colab and Select T4 GPU
1. Go to [Google Colab](https://colab.research.google.com).
2. Go to **Runtime > Change runtime type**.
3. Under **Hardware accelerator**, select **T4 GPU**.
4. Click **Save**.
5. Check GPU in a code cell:
   ```python
   !nvidia-smi
   ```
   You should see `Tesla T4` with ~15,360 MiB (~15 GB) VRAM.

---

### Step 2: Upload the Fine-Tuning Notebook
1. In Colab, click **File > Upload notebook**.
2. Select the notebook from your local project:
   ```text
   Tetrax-ASR/notebooks/Tetrax_ASR_Colab_FineTuning.ipynb
   ```
3. The notebook will open with all cells pre-configured.

---

### Step 3: Run the Cells Sequentially

#### 1. Install Fine-Tuning Dependencies (Step 1)
```bash
!pip install -q transformers datasets peft accelerate evaluate jiwer soundfile librosa bitsandbytes tensorboard
```

#### 2. Mount Google Drive (Step 2)
```python
from google.colab import drive
drive.mount('/content/drive')
```
*Colab will ask for permission to access your Google Drive. Click "Connect to Google Drive".*

#### 3. Load IndicVoices Dataset from Drive (Step 3)
Because you already downloaded the dataset:
- **Hindi**: 83 parquet files (46.41 GB) in `/content/drive/MyDrive/Tetrax-ASR/data/hindi/`
- **Marathi**: 73 parquet files (34.96 GB) in `/content/drive/MyDrive/Tetrax-ASR/data/marathi/`

The notebook automatically detects these files and loads a balanced subset:
```python
PARQUET_FILES_PER_LANG = 2   # Loads 2 Marathi + 2 Hindi files (~1.5 GB, ~10-20 hrs audio)
```

> [!TIP]
> **Why 2 files per language?**
> Free Google Colab provides 12.7 GB of System RAM. Loading 2 files per language (~5,000–8,000 samples) prevents Colab from running out of RAM while providing enough audio for LoRA adaptation.

#### 4. Audio Preprocessing & Tokenization (Steps 4 & 5)
- Converts audio to 16kHz log-Mel spectrograms.
- Dynamically conditions language tokens per sample:
  - `<|mr|>` for Marathi
  - `<|hi|>` for Hindi
- Tokenizes target text into Devanagari token sequences.

#### 5. Model & LoRA Setup (Step 7)
- Loads `openai/whisper-medium` (769M) in FP16.
- Activates **Gradient Checkpointing** (`model.gradient_checkpointing_enable()`).
- Injects LoRA trainable parameters into `q_proj`, `v_proj`, `k_proj`, `out_proj`.

#### 6. Start Training Loop (Step 8)
- `per_device_train_batch_size = 4` (Parallel batch processing; uses ~9-10 GB of the 15 GB VRAM, 3.5x faster throughput!)
- `gradient_accumulation_steps = 2` (Effective batch size = `4 * 2 = 8`)
- `max_steps = 500` (Optimized for LoRA adaptation; finishes in ~15-20 minutes, well within Colab's 2h limit)
- `dataloader_num_workers = 2` & `dataloader_pin_memory = True` (Background audio pre-loading)
- `learning_rate = 1e-4`, `fp16 = True`, `gradient_checkpointing = True`
- Evaluates both **WER** and **CER** (Character Error Rate for Devanagari text).

---

### Step 4: Training Completion & Google Drive Export (Step 9)

When training finishes, Step 9 runs automatically and displays:

```text
========================================================================
🎉 MODEL SUCCESSFULLY TRAINED AND SAVED!
  • Model Name    : Tetrax-ASR-Open-764M-A19M
  • Parameter Size: 769M Parameters (~0.8B Backbone, ~8.3M Trainable LoRA)
  • Languages     : Marathi ('mr') + Hindi ('hi') [Devanagari text]
  • Saved to Drive: /content/drive/MyDrive/Tetrax-ASR/checkpoints/Tetrax-ASR-Open-764M-A19M.zip
  • ZIP File Size : ~80 MB
========================================================================
```

Your fine-tuned model is permanently preserved in your **Google Drive**:
```text
My Drive/
└── Tetrax-ASR/
    └── checkpoints/
        ├── Tetrax-ASR-Open-764M-A19M.zip
        └── Tetrax-ASR-Open-764M-A19M/
            ├── adapter_config.json
            ├── adapter_model.safetensors
            ├── tokenizer_config.json
            └── vocab.json
```

---

## 🛠️ Part 2: How to Use the Trained Model

### Method 1: Place the Checkpoint in Your Local Project

1. Open your Google Drive in a web browser.
2. Go to `Tetrax-ASR/checkpoints/` and download `Tetrax-ASR-Open-764M-A19M.zip`.
3. Extract the ZIP into your local repository under `checkpoints/`:
   ```text
   Tetrax-ASR/
   └── checkpoints/
       └── Tetrax-ASR-Open-764M-A19M/
           ├── adapter_config.json
           ├── adapter_model.safetensors
           ├── tokenizer_config.json
           └── ...
   ```

---

### Method 2: Transcribe Directly in Python

You can use the fine-tuned model directly inside any Python script:

```python
from tetrax_asr.inference.engine import TetraxWhisperEngine

# 1. Initialize the engine (automatically detects Tetrax-ASR-Open-764M-A19M)
engine = TetraxWhisperEngine()

# 2. Transcribe a Marathi audio file
result_mr = engine.transcribe("sample_marathi.wav", language="mr")
print("Language:", result_mr["language"])
print("Marathi Transcript:", result_mr["text"])
print("Audio Duration:", result_mr["duration"], "seconds")
print("Latency:", result_mr["processing_time"], "seconds")

# 3. Transcribe a Hindi audio file
result_hi = engine.transcribe("sample_hindi.wav", language="hi")
print("Hindi Transcript:", result_hi["text"])

# 4. Auto-detect language (Marathi or Hindi automatically identified!)
result_auto = engine.transcribe("unknown_speech.wav", language=None)
print("Detected Language:", result_auto["language"])
print("Transcript:", result_auto["text"])
```

---

### Method 3: Run the FastAPI Server & REST API

Start the high-performance Tetrax ASR backend:

```bash
python -m uvicorn tetrax_asr.api.main:app --host 0.0.0.0 --port 8000
```

#### Transcribe via cURL:
```bash
# Transcribe Marathi audio
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -F "file=@marathi_audio.wav" \
  -F "language=mr"

# Transcribe Hindi audio
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -F "file=@hindi_audio.mp3" \
  -F "language=hi"
```

#### Response Format:
```json
{
  "id": "tr_9a8b7c6d5e4f",
  "model": "tetrax-stt-marathi-hindi",
  "language": "mr",
  "text": "नमस्कार, आजच्या बैठकीमध्ये आपले स्वागत आहे.",
  "duration": 4.12,
  "segments": [
    {
      "id": 0,
      "start": 0.0,
      "end": 4.12,
      "text": "नमस्कार, आजच्या बैठकीमध्ये आपले स्वागत आहे."
    }
  ],
  "processing_time": 0.38,
  "rtf": 0.092,
  "usage": {
    "audio_seconds": 4.12
  }
}
```

#### Transcribe via Python Requests:
```python
import requests

url = "http://localhost:8000/v1/audio/transcriptions"
files = {"file": open("speech.wav", "rb")}
data = {"language": "mr"}

response = requests.post(url, files=files, data=data)
print(response.json()["text"])
```

---

### Method 4: Interactive Web Console & Real-Time Live Streaming

1. Start the server:
   ```bash
   python -m uvicorn tetrax_asr.api.main:app --host 0.0.0.0 --port 8000
   ```
2. Open your web browser:
   👉 **[http://localhost:8000](http://localhost:8000)**
3. **Live Streaming Tab**:
   - Select your language (`Marathi` or `Hindi`).
   - Click **"Start Live Speech Stream"**.
   - Speak into your microphone to view live Devanagari interim captions and finalized sentences streamed over WebSockets (`WS /v1/audio/stream`).
4. **File Transcription Tab**:
   - Drag and drop any recorded WAV, MP3, M4A, or FLAC file to view timestamped transcription.
