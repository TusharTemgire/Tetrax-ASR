# 🎙️ Tetrax ASR — Real-Time Indian-English Speech Intelligence

Tetrax ASR is a real-time speech-to-text intelligence system adapted for Indian-accented English, built for integration into Tetrax CPaaS. It combines an adapted Whisper model fine-tuned using **PEFT / LoRA (Low-Rank Adaptation)** on the **AI4Bharat Svarah** dataset, a **FastAPI backend with low-latency WebSocket streaming**, and an **interactive Voice Intelligence Console**.

---

## 🌟 Key Capabilities

- **Real-Time Streaming Speech Recognition (`WS /v1/audio/stream`)**:
  - Live microphone / audio streaming with chunked sliding-window decoding (256ms audio packets).
  - Energy-based Voice Activity Detection (VAD) with silence commit triggers.
  - Generates instant **interim hypotheses** and finalized **committed text segments** in real-time.
- **REST Transcription API (`POST /v1/audio/transcriptions`)**:
  - Batch transcription for WAV, MP3, M4A, WEBM, and FLAC files.
  - Word & sentence timestamps, duration metrics, and usage accounting.
- **Trained on AI4Bharat Svarah**:
  - Optimized for Indian English accents across 19 Indian states and native languages (Hindi, Marathi, Tamil, Telugu, Bengali, Kannada, etc.).
- **Cloud & Local Training Ready**:
  - 📘 **[Google Colab Training Guide](docs/COLAB_TRAINING_GUIDE.md)**: Free T4 GPU fine-tuning in ~15-30 minutes.
  - 🚀 **[RunPod Training & Hosting Guide](docs/RUNPOD_TRAINING_GUIDE.md)**: Dedicated cloud GPU (RTX 3090/4090) fine-tuning in ~10 minutes and public API hosting.

---

## 📂 Project Directory Structure

```
Tetrax-ASR/
├── docs/                                     # In-depth training & deployment guides
│   ├── COLAB_TRAINING_GUIDE.md               # Complete Colab setup & execution guide
│   └── RUNPOD_TRAINING_GUIDE.md              # RunPod setup, training & cloud API hosting
├── checkpoints/                              # Saved LoRA checkpoints
│   └── tetrax-whisper-small-indian-en/       # Fine-tuned adapter weights
├── data/                                     # AI4Bharat Svarah dataset
│   ├── test-00000-of-00003.parquet           # Svarah Parquet Part 1 (~365MB)
│   ├── test-00001-of-00003.parquet           # Svarah Parquet Part 2 (~360MB)
│   └── test-00002-of-00003.parquet           # Svarah Parquet Part 3 (~370MB)
├── notebooks/
│   └── Tetrax_ASR_Colab_FineTuning.ipynb     # Google Colab fine-tuning notebook
├── scripts/
│   ├── download_svarah.py                    # Dataset explorer & statistics CLI
│   └── test_realtime_stream.py               # Real-time WebSocket streaming tester
├── tetrax_asr/
│   ├── api/
│   │   ├── main.py                           # FastAPI application entry point
│   │   ├── routes/
│   │   │   ├── transcriptions.py             # POST /v1/audio/transcriptions
│   │   │   ├── stream.py                     # WS /v1/audio/stream
│   │   │   ├── models.py                     # GET /v1/models
│   │   │   └── health.py                     # GET /v1/health
│   │   └── schemas/                          # Pydantic schemas
│   ├── data/
│   │   ├── svarah_loader.py                  # Local & remote Svarah dataset loader
│   │   └── collator.py                       # Whisper Data Collator with padding
│   ├── inference/
│   │   ├── engine.py                         # Whisper inference engine (PyTorch/CTranslate2)
│   │   └── streaming.py                      # Real-time streaming & VAD session manager
│   ├── training/
│   │   ├── prepare_dataset.py                # Mel-spectrogram & tokenizer preprocessing
│   │   ├── train_whisper.py                  # Local fine-tuning script
│   │   ├── evaluate.py                       # WER benchmark evaluation script
│   │   └── configs/train_config.yaml         # Training hyperparameters
│   └── web/
│       ├── static/                           # Web console CSS & JavaScript
│       └── templates/index.html              # Tetrax Voice Intelligence Console
├── requirements.txt                          # Python dependencies
└── README.md
```

---

## 🚀 Quickstart: Running Locally

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Start Tetrax ASR Server
```bash
python -m uvicorn tetrax_asr.api.main:app --host 0.0.0.0 --port 8000
```

### 3. Open the Interactive Web Console
Open your browser at:
👉 **[http://localhost:8000](http://localhost:8000)**

- **Real-Time Streaming Tab**: Click *"Start Live Speech Stream"* and speak into your microphone to see live interim & final Indian-English captions with live audio waveforms.
- **File Transcription Tab**: Drag & drop any recorded audio file (WAV, MP3, M4A, WEBM, FLAC) to test full transcription with timestamped breakdown.

---

## ⚡ Training in Google Colab

You can fine-tune `openai/whisper-small` with LoRA on Google Colab (Free T4 GPU):

1. Go to [Google Colab](https://colab.research.google.com).
2. Choose **File > Upload notebook** and upload:
   `notebooks/Tetrax_ASR_Colab_FineTuning.ipynb`
3. Switch runtime: **Runtime > Change runtime type > T4 GPU**.
4. Run the cells sequentially:
   - **Step 1-2**: Installs dependencies and mounts Google Drive.
   - **Step 3-5**: Loads Svarah dataset (directly from Hugging Face or uploaded `test-*.parquet`), tokenizes transcripts, and computes 80-channel log-Mel spectrograms.
   - **Step 6-8**: Attaches LoRA adapter (Rank=32, Alpha=64) to Whisper-small and fine-tunes using `Seq2SeqTrainer` with Word Error Rate (WER) evaluation.
   - **Step 9**: Exports the trained adapter to `tetrax-whisper-small-indian-en.zip`.
5. Download `tetrax-whisper-small-indian-en.zip`, extract it, and place it into:
   ```
   Tetrax-ASR/checkpoints/tetrax-whisper-small-indian-en/
   ```
6. Restart or run the local Tetrax ASR server. It will automatically detect and mount the fine-tuned adapter!

> For full details, see the 📘 **[Google Colab Fine-Tuning Guide](docs/COLAB_TRAINING_GUIDE.md)**.

---

## ⚡ Training & Hosting on RunPod

For high-speed training (RTX 3090/4090) in **~10 minutes** or hosting a public cloud endpoint:

1. Launch a Pod with **RTX 3090 / 4090** on [RunPod](https://www.runpod.io) using the PyTorch template.
2. Clone repository & install dependencies:
   ```bash
   git clone <YOUR_REPO_URL> Tetrax-ASR
   cd Tetrax-ASR && pip install -r requirements.txt peft
   ```
3. Run training with batch size 16:
   ```bash
   python -m tetrax_asr.training.train_whisper --base_model openai/whisper-small --epochs 3 --batch_size 16 --lr 1e-4
   ```
4. Host the live streaming server with public URL:
   ```bash
   python -m uvicorn tetrax_asr.api.main:app --host 0.0.0.0 --port 8000
   ```

> For step-by-step RunPod deployment, port exposure, and SSH setup, see the 🚀 **[RunPod Training & Hosting Guide](docs/RUNPOD_TRAINING_GUIDE.md)**.

---

## 🎯 How to Use Your Fine-Tuned Model (Post-Training Guide)

Once your model has finished training in Google Colab or RunPod, follow these steps to use it:

### Step 1: Place Your Model Checkpoint
1. Download `tetrax-whisper-small-indian-en.zip` from Colab/Drive.
2. Extract it into your project under `checkpoints/`:
   ```
   Tetrax-ASR/
   └── checkpoints/
       └── tetrax-whisper-small-indian-en/
           ├── adapter_config.json
           ├── adapter_model.safetensors
           ├── tokenizer_config.json
           ├── special_tokens_map.json
           └── vocab.json
   ```
3. Whenever the Tetrax ASR service starts, it **automatically detects and attaches** your fine-tuned LoRA weights.

---

### Step 2: Choose How You Want to Use It

#### Method 1: Interactive Web Console (Zero Code)
1. Start the server:
   ```bash
   python -m uvicorn tetrax_asr.api.main:app --host 0.0.0.0 --port 8000
   ```
2. Open **[http://localhost:8000](http://localhost:8000)**.
3. Check the top header badge: it will display `Tetrax Indian-En (LoRA Active)`.
4. **Real-time Live Audio**: Switch to the *Real-Time Streaming* tab, click *Start Live Speech Stream*, and speak into your microphone.
5. **File Transcription**: Switch to the *File Transcription* tab and drag-and-drop call recordings or meeting files.

---

#### Method 2: Direct Python Code (In Your Own Scripts/Apps)
You can directly import and use the fine-tuned engine in any Python script without starting an HTTP server:

```python
from tetrax_asr.inference.engine import TetraxWhisperEngine

# 1. Initialize engine (loads base Whisper + your fine-tuned LoRA adapter)
engine = TetraxWhisperEngine()

# 2. Transcribe any audio file (WAV, MP3, M4A, WEBM, FLAC)
result = engine.transcribe("path/to/meeting.wav", language="en", return_timestamps=True)

# 3. Access results
print("Full Transcript:\n", result["text"])
print(f"Duration: {result['duration']}s | Processing Time: {result['processing_time']}s | RTF: {result['rtf']}")

# 4. Access timestamped segments
for seg in result["segments"]:
    print(f"[{seg['start']}s -> {seg['end']}s]: {seg['text']}")
```

---

#### Method 3: Via REST API (`POST /v1/audio/transcriptions`)
Send audio files from any programming language or tool:

**cURL**:
```bash
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -F "file=@customer_call.wav" \
  -F "model=tetrax-stt-indian-en" \
  -F "timestamps=true"
```

**Python (`requests`)**:
```python
import requests

url = "http://localhost:8000/v1/audio/transcriptions"
with open("customer_call.wav", "rb") as f:
    response = requests.post(
        url,
        files={"file": f},
        data={"model": "tetrax-stt-indian-en", "timestamps": "true"}
    )

data = response.json()
print("Transcript:", data["text"])
print("Duration:", data["duration"], "seconds")
```

**JavaScript / Node.js (`fetch`)**:
```javascript
const formData = new FormData();
formData.append('file', audioBlob, 'recording.wav');
formData.append('model', 'tetrax-stt-indian-en');

const res = await fetch('http://localhost:8000/v1/audio/transcriptions', {
  method: 'POST',
  body: formData
});
const data = await res.json();
console.log('Transcript:', data.text);
```

---

#### Method 4: Real-Time Streaming via WebSocket (`WS /v1/audio/stream`)
For live meetings, IVR phone calls, or voice agents:

1. Connect to `ws://localhost:8000/v1/audio/stream`.
2. Stream raw 16kHz 16-bit PCM chunks (e.g. 256ms audio packets).
3. The server emits real-time interim preview text and finalized committed sentence segments with timestamps.

Run the included real-time streaming test client:
```bash
# Streams audio from your local Svarah dataset in real-time pace:
python scripts/test_realtime_stream.py

# Or stream your own audio file:
python scripts/test_realtime_stream.py --file my_recording.wav
```

---

#### Method 5: Measure Accuracy Improvement (WER Benchmark)
Compare your fine-tuned model against the baseline on held-out test audio:

```bash
python -m tetrax_asr.training.evaluate \
    --base_model openai/whisper-small \
    --adapter_path checkpoints/tetrax-whisper-small-indian-en \
    --num_samples 100
```
This prints the exact Word Error Rate (WER) reduction achieved by your fine-tuned model!

---

## 📡 API Specification

### 1. Real-Time Streaming WebSocket
- **Endpoint**: `ws://localhost:8000/v1/audio/stream`
- **Audio Format**: Raw PCM 16-bit 16,000 Hz Mono chunks (or Float32 chunks)
- **Client Protocol**:
  - Connect WebSocket
  - Send binary audio chunks (e.g. 256ms / 4096 samples per chunk)
  - Send `{"action": "stop"}` when speaking ends
- **Server Events**:
  ```json
  {"event": "ready", "session_id": "wss_a1b2c3d4", "model": "tetrax-stt-indian-en"}
  {"event": "interim", "text": "welcome to the", "duration": 1.25}
  {"event": "final", "text": "Welcome to the meeting.", "start": 0.0, "end": 2.4, "duration": 2.4}
  ```

### 2. File Upload Transcription (REST)
- **Endpoint**: `POST /v1/audio/transcriptions`
- **Request**:
  ```bash
  curl -X POST http://localhost:8000/v1/audio/transcriptions \
    -F "file=@your_audio.wav" \
    -F "model=tetrax-stt-indian-en" \
    -F "language=en" \
    -F "timestamps=true"
  ```
- **Response**:
  ```json
  {
    "id": "tr_9f81a7b2c019",
    "model": "tetrax-stt-indian-en",
    "language": "en",
    "text": "Our district Hasana was the land of the great Hoysala dynasty who ruled over this region from 11th to 14th century CE.",
    "duration": 8.76,
    "segments": [
      {
        "id": 0,
        "start": 0.0,
        "end": 8.76,
        "text": "Our district Hasana was the land of the great Hoysala dynasty who ruled over this region from 11th to 14th century CE."
      }
    ],
    "processing_time": 0.84,
    "rtf": 0.096,
    "usage": {
      "audio_seconds": 8.76
    }
  }
  ```

---

## 🧪 Testing CLI Tools

### Test Real-Time WebSocket Streaming:
```bash
# Streams directly from the Svarah parquet dataset in data/
python scripts/test_realtime_stream.py

# Or stream any custom audio file:
python scripts/test_realtime_stream.py --file path/to/your_audio.wav
```

### Run Benchmark WER Evaluation:
```bash
python -m tetrax_asr.training.evaluate --base_model openai/whisper-small --num_samples 50
```
