# 🎙️ Google Colab Fine-Tuning Guide (Step-by-Step)
## Model: `TETRAX-ASR-WHISPER-MEDIUM-769M` (Marathi + Hindi Bilingual)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com)
![Whisper Medium](https://img.shields.io/badge/Model-Whisper--Medium%20(769M)-blue.svg)
![LoRA](https://img.shields.io/badge/Fine--Tuning-PEFT%20LoRA-orange.svg)
![Languages](https://img.shields.io/badge/Languages-Marathi%20%7C%20Hindi-green.svg)
![Hardware](https://img.shields.io/badge/Hardware-Google%20Colab%20T4%20(15GB)-purple.svg)

This guide provides a comprehensive, step-by-step walkthrough to fine-tune **`openai/whisper-medium` (769M parameters)** for **Marathi (`mr`) and Hindi (`hi`)** speech recognition on **Google Colab's Free T4 GPU**, automatically package the trained model adapter (`.zip`), and deploy it into your Tetrax ASR pipeline.

---

## 📑 Table of Contents

- [Architecture & Parameter Breakdown](#-architecture--parameter-breakdown)
- [Prerequisites](#-prerequisites)
- [Google Drive Directory Structure](#-google-drive-directory-structure)
- [Step-by-Step Colab Execution](#-step-by-step-colab-execution)
  - [Step 1: Open Colab & Select T4 GPU Runtime](#step-1-open-colab--select-t4-gpu-runtime)
  - [Step 2: Upload the Notebook](#step-2-upload-the-notebook)
  - [Step 3: Install Dependencies (Step 1 Cell)](#step-3-install-dependencies-step-1-cell)
  - [Step 4: Mount Google Drive (Step 2 Cell)](#step-4-mount-google-drive-step-2-cell)
  - [Step 5: Load IndicVoices Dataset (Step 3 Cell)](#step-5-load-indicvoices-dataset-step-3-cell)
  - [Step 6: Initialize Whisper Processor (Step 4 Cell)](#step-6-initialize-whisper-processor-step-4-cell)
  - [Step 7: Preprocessing & Dynamic Tokenization (Step 5 Cell)](#step-7-preprocessing--dynamic-tokenization-step-5-cell)
  - [Step 8: Setup Data Collator & Metrics (Step 6 Cell)](#step-8-setup-data-collator--metrics-step-6-cell)
  - [Step 9: Load Whisper Medium & Configure LoRA (Step 7 Cell)](#step-9-load-whisper-medium--configure-lora-step-7-cell)
  - [Step 10: Start Training with Seq2SeqTrainer (Step 8 Cell)](#step-10-start-training-with-seq2seqtrainer-step-8-cell)
  - [Step 11: Export Model & Download (Step 9 Cell)](#step-11-export-model--download-step-9-cell)
- [Deploying the Trained Model in Tetrax ASR](#-deploying-the-trained-model-in-tetrax-asr)
- [Troubleshooting & FAQs](#-troubleshooting--faqs)

---

## 📊 Architecture & Parameter Breakdown

| Specification | Configuration |
|---|---|
| **Base Model** | [`openai/whisper-medium`](https://huggingface.co/openai/whisper-medium) |
| **Total Parameters** | **~769M (~0.8B)** |
| **Trainable Parameters** | **~8.38M (~1.09%)** via PEFT / LoRA |
| **LoRA Target Modules** | `q_proj`, `v_proj`, `k_proj`, `out_proj` (Attention Projections) |
| **LoRA Rank / Alpha** | `r = 32`, `lora_alpha = 64`, `dropout = 0.05` |
| **Target Languages** | **Marathi (`mr`)** & **Hindi (`hi`)** (Devanagari script) |
| **Precision** | FP16 mixed precision |
| **Memory Optimization** | `gradient_checkpointing = True`, `batch_size = 1`, `grad_accum = 8` |
| **VRAM Consumption** | **~8.2 GB** (fits comfortably within Colab T4's 15 GB VRAM) |
| **Output Model Size** | **~80 MB** (portable adapter archive vs ~3 GB full checkpoint) |

---

## 🧰 Prerequisites

1. A **Google Account** with access to [Google Colab](https://colab.research.google.com).
2. ~3 to 5 GB of free space in your **Google Drive** for dataset files and the trained adapter output.
3. The fine-tuning notebook:
   [`notebooks/Tetrax_ASR_Colab_FineTuning.ipynb`](file:///d:/Tushar_repos/Tetrax-ASR/notebooks/Tetrax_ASR_Colab_FineTuning.ipynb)

---

## 📁 Google Drive Directory Structure

Before starting, you can organize your Google Drive with your downloaded IndicVoices parquet files as follows:

```text
My Drive/
└── Tetrax-ASR/
    ├── data/
    │   ├── marathi/
    │   │   ├── 001.parquet
    │   │   └── ...
    │   └── hindi/
    │       ├── 001.parquet
    │       └── ...
    └── checkpoints/
        └── TETRAX-ASR-WHISPER-MEDIUM-769M.zip   <-- (Auto-saved here after training)
```

> [!NOTE]
> If you do not have parquet files placed in Google Drive, the notebook automatically falls back to streaming / downloading directly from Hugging Face Hub (`ai4bharat/IndicVoices`).

---

## 🚀 Step-by-Step Colab Execution

### Step 1: Open Colab & Select T4 GPU Runtime
1. Open [Google Colab](https://colab.research.google.com).
2. Click **Runtime** > **Change runtime type**.
3. Under **Hardware accelerator**, select **T4 GPU**.
4. Click **Save**.
5. Check your GPU allocation:
   ```python
   !nvidia-smi
   ```
   *Expected output: Tesla T4 with ~15,360 MiB (~15 GB) VRAM.*

---

### Step 2: Upload the Notebook
1. In Colab, click **File** > **Upload notebook**.
2. Browse to your local project directory and select:
   ```text
   Tetrax-ASR/notebooks/Tetrax_ASR_Colab_FineTuning.ipynb
   ```
3. The notebook will open with all 9 execution steps configured.

---

### Step 3: Install Dependencies (Step 1 Cell)
Run the first code cell to install the required libraries:
```bash
!pip install -q transformers datasets peft accelerate evaluate jiwer soundfile librosa bitsandbytes tensorboard
# Colab pre-installs an outdated torchao (0.10.0) which conflicts with PEFT (requires >=0.16.0)
!pip uninstall -y torchao
```

---

### Step 4: Mount Google Drive (Step 2 Cell)
Connect Colab to your Google Drive so datasets and trained checkpoints persist across sessions:
```python
from google.colab import drive
import os

drive.mount('/content/drive')

DRIVE_DATA_DIR = "/content/drive/MyDrive/Tetrax-ASR/data"
CHECKPOINT_DIR = "/content/drive/MyDrive/Tetrax-ASR/checkpoints/tetrax-whisper-medium-marathi-hindi"
os.makedirs(DRIVE_DATA_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
```
*When prompted, click **Connect to Google Drive** and grant access.*

---

### Step 5: Load IndicVoices Dataset (Step 3 Cell)
This cell scans your Google Drive for Marathi and Hindi parquet files:
```python
PARQUET_FILES_PER_LANG = 2   # Loads 2 Marathi + 2 Hindi parquet files (~1-2 GB)
MAX_SAMPLES_PER_LANG = None  # Optional sample cap
```
- Creates an **80% Train / 10% Validation / 10% Test** stratified split across Marathi and Hindi.
- Uses `PARQUET_FILES_PER_LANG = 2` to stay well within Colab's 12.7 GB system RAM while giving ~15,000+ balanced audio samples.

---

### Step 6: Initialize Whisper Processor (Step 4 Cell)
Initializes the multilingual `WhisperProcessor` for `openai/whisper-medium`:
```python
from transformers import WhisperProcessor

MODEL_ID = "openai/whisper-medium"
processor = WhisperProcessor.from_pretrained(MODEL_ID, language="mr", task="transcribe")
```

---

### Step 7: Preprocessing & Dynamic Tokenization (Step 5 Cell)
Extracts 80-channel log-Mel spectrograms and conditions the tokenizer on language prefix tokens dynamically (`<|mr|>` for Marathi, `<|hi|>` for Hindi):

```python
# Handles binary bytes, decoded arrays, and disk filepaths seamlessly
tokenized_train = dataset_dict["train"].map(
    prepare_bilingual_dataset, 
    remove_columns=dataset_dict["train"].column_names, 
    num_proc=2
)
tokenized_val = dataset_dict["validation"].map(
    prepare_bilingual_dataset, 
    remove_columns=dataset_dict["validation"].column_names, 
    num_proc=2
)
```

> [!TIP]
> The notebook includes a robust audio extractor (`extract_raw_audio`) that automatically handles Parquet raw byte buffers, pre-decoded arrays, and disk paths, preventing column format errors.

---

### Step 8: Setup Data Collator & Metrics (Step 6 Cell)
Configures sequence-to-sequence padding and loads evaluation metrics:
- **WER (Word Error Rate)**: Standard word-level recognition accuracy.
- **CER (Character Error Rate)**: Critical for Devanagari script where phonetic character fidelity matters.

---

### Step 9: Load Whisper Medium & Configure LoRA (Step 7 Cell)
Loads the 769M model in FP16 precision, enables gradient checkpointing, and injects LoRA adapters:
```python
from peft import LoraConfig, get_peft_model

model.gradient_checkpointing_enable()

peft_config = LoraConfig(
    r=32,
    lora_alpha=64,
    target_modules=["q_proj", "v_proj", "k_proj", "out_proj"],
    lora_dropout=0.05,
    bias="none"
)
model = get_peft_model(model, peft_config)
```
- Trains **8.38M parameters (1.09%)** while freezing the remaining ~760M backbone parameters.

---

### Step 10: Start Training with Seq2SeqTrainer (Step 8 Cell)
Launches the memory-optimized fine-tuning loop:
```python
training_args = Seq2SeqTrainingArguments(
    output_dir="/content/TETRAX-ASR-WHISPER-MEDIUM-769M",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,     # Effective batch size = 8
    learning_rate=1e-4,
    warmup_steps=100,
    num_train_epochs=3,
    eval_strategy="steps",
    fp16=True,
    gradient_checkpointing=True,
    save_total_limit=2,
    metric_for_best_model="wer",
    ...
)
trainer.train()
```

---

### Step 11: Export Model & Download (Step 9 Cell)
When training finishes, this cell automatically:
1. Saves the trained LoRA adapter weights and tokenizer files.
2. Compresses the checkpoint into **`TETRAX-ASR-WHISPER-MEDIUM-769M.zip`** (~80 MB).
3. Saves a permanent copy to your Google Drive under `MyDrive/Tetrax-ASR/checkpoints/`.
4. Triggers an automatic browser download of the `.zip` archive.

---

## 💻 Deploying the Trained Model in Tetrax ASR

Once you have downloaded the `.zip` from Colab or Google Drive:

### 1. Extract Checkpoint into Local Project
Extract the contents into your project's `checkpoints/` directory:
```text
Tetrax-ASR/
└── checkpoints/
    └── TETRAX-ASR-WHISPER-MEDIUM-769M/
        ├── adapter_config.json
        ├── adapter_model.safetensors
        ├── special_tokens_map.json
        ├── tokenizer_config.json
        └── vocab.json
```

### 2. Transcribe in Python
```python
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from peft import PeftModel

BASE_MODEL = "openai/whisper-medium"
ADAPTER_PATH = "./checkpoints/TETRAX-ASR-WHISPER-MEDIUM-769M"

# Load processor and base model
processor = WhisperProcessor.from_pretrained(BASE_MODEL, language="mr", task="transcribe")
model = WhisperForConditionalGeneration.from_pretrained(
    BASE_MODEL, 
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
)

# Attach LoRA adapter
model = PeftModel.from_pretrained(model, ADAPTER_PATH)
model.eval()

# Transcribe Marathi or Hindi audio
import librosa
audio, sr = librosa.load("sample.wav", sr=16000)
inputs = processor(audio, sampling_rate=16000, return_tensors="pt").input_features

if torch.cuda.is_available():
    inputs = inputs.half().cuda()

with torch.no_grad():
    predicted_ids = model.generate(inputs, language="mr", task="transcribe")

transcript = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
print("Transcript:", transcript)
```

### 3. Run with Tetrax FastAPI Server
```bash
python -m tetrax_asr.api.server
```
Visit `http://localhost:8000/docs` to test real-time bilingual transcription endpoints.

---

## ❓ Troubleshooting & FAQs

### Q1: Colab runs out of GPU memory (CUDA OOM)
- Ensure `gradient_checkpointing_enable()` is executed in Step 7.
- Keep `per_device_train_batch_size = 1` and rely on `gradient_accumulation_steps = 8` for effective batch size.
- Ensure `fp16 = True` is set.

### Q2: Colab runs out of System RAM during dataset loading
- Colab Free tier provides ~12.7 GB of System RAM.
- Keep `PARQUET_FILES_PER_LANG = 2` (or `1`) in Step 3. Do not set `PARQUET_FILES_PER_LANG = None` on the free tier unless you have Colab Pro High-RAM.

### Q3: "Unrecognized audio column format" error during mapping
- Ensure you are using the updated notebook which includes `extract_raw_audio`. It handles raw binary bytes in Parquet files, nested dictionaries, and file paths automatically.

### Q4: "ImportError: Found an incompatible version of torchao"
- Google Colab pre-installs an outdated `torchao==0.10.0`, whereas recent `peft` versions require `torchao >= 0.16.0`.
- Even though Whisper LoRA does not use `torchao` quantization, `peft` checks for it.
- **Immediate Fix (without restarting kernel)**:
  Run this cell in Colab:
  ```python
  !pip uninstall -y torchao
  import peft.import_utils
  peft.import_utils.is_torchao_available = lambda: False
  ```
- Re-run Step 7, and it will load the model without errors.

### Q5: Preventing Colab Timeout / Disconnect during training
- Keep the Colab browser tab open and active.
- Training ~15,000 samples for 3 epochs with LoRA takes approximately 1 to 2 hours on a T4 GPU.

---

## 📚 Related Documentation

- 📘 [Colab Detailed Step-by-Step Guide](../docs/COLAB_STEP_BY_STEP_GUIDE.md)
- 📗 [Local Fine-Tuning & Architecture Guide](../docs/COLAB_TRAINING_GUIDE.md)
- 📙 [RunPod Cloud Training Guide](../docs/RUNPOD_TRAINING_GUIDE.md)
