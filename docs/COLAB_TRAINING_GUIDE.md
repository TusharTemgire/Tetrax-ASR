# 🎙️ Google Colab Fine-Tuning Guide — Tetrax ASR (Marathi + Hindi Bilingual)

This guide walks you through fine-tuning **Whisper Medium (769M)** for **Marathi (`mr`) + Hindi (`hi`)** speech recognition on the **AI4Bharat IndicVoices** dataset using **Google Colab's Free T4 GPU** with **PEFT / LoRA (Low-Rank Adaptation)**.

---

## 📋 Overview & Requirements

| Item | Details |
|---|---|
| **Hardware** | Free Google Colab **T4 GPU** (15 GB VRAM) |
| **Model** | `openai/whisper-medium` (769M multilingual parameters) |
| **Languages** | **Marathi (`mr`)** and **Hindi (`hi`)** (Devanagari script output) |
| **Adaptation** | PEFT LoRA (`r=32`, `lora_alpha=64`, trains ~1.1% of parameters, ~8.3M params) |
| **Memory Optimization** | `batch_size=1`, `gradient_accumulation_steps=8`, `gradient_checkpointing=True`, `fp16=True` |
| **Dataset** | AI4Bharat IndicVoices (Marathi & Hindi conversational & read speech) |
| **Evaluation Metrics** | **WER** (Word Error Rate) and **CER** (Character Error Rate for Devanagari text) |
| **Output** | Fine-tuned LoRA adapter ZIP (`~80 MB`), auto-saved to Google Drive |

---

## 📁 Google Drive Dataset Setup

You can place your IndicVoices dataset files manually into your Google Drive:

```text
My Drive/
└── Tetrax-ASR/
    ├── data/
    │   ├── marathi/
    │   │   └── *.parquet  (or indicvoices_marathi.parquet)
    │   └── hindi/
    │       └── *.parquet  (or indicvoices_hindi.parquet)
    └── checkpoints/
        └── tetrax-whisper-medium-marathi-hindi.zip  (auto-exported after training)
```

> [!TIP]
> The notebook automatically scans `/content/drive/MyDrive/Tetrax-ASR/data/` for:
> - Subdirectories: `data/marathi/` and `data/hindi/`
> - Language-tagged parquet files: `*marathi*.parquet` and `*hindi*.parquet`
> - Direct Hub fallback: If no files are placed in Drive, it fetches from `ai4bharat/IndicVoices` automatically.

---

## 🚀 Step-by-Step Training Walkthrough

### Step 1: Open Google Colab & Select T4 GPU Runtime
1. Open [Google Colab](https://colab.research.google.com).
2. Go to **Runtime > Change runtime type**.
3. Under **Hardware accelerator**, select **T4 GPU**.
4. Click **Save**.
5. Verify GPU is active:
   ```python
   !nvidia-smi
   ```

---

### Step 2: Upload the Fine-Tuning Notebook
1. In Colab, click **File > Upload notebook**.
2. Select the notebook from your local repository:
   ```
   Tetrax-ASR/notebooks/Tetrax_ASR_Colab_FineTuning.ipynb
   ```
3. The notebook will open with all cells pre-configured for Whisper Medium and IndicVoices.

---

### Step 3: Run the Notebook Cells

#### 1. Mount Google Drive (Step 2 in Notebook)
```python
from google.colab import drive
drive.mount('/content/drive')
```

#### 2. Load IndicVoices Dataset (Step 3 in Notebook)
- Detects your manual Marathi and Hindi parquet files from `/content/drive/MyDrive/Tetrax-ASR/data/`.
- Merges them into a balanced bilingual pool and creates a stratified **80% Train / 10% Validation / 10% Test** split.

#### 3. Bilingual Audio Preprocessing & Dynamic Tokenization (Steps 4 & 5)
- Extracts 80-channel log-Mel spectrograms from 16kHz audio.
- Dynamically sets language conditioning tokens per sample:
  - `<|mr|>` for Marathi
  - `<|hi|>` for Hindi
- Tokenizes Devanagari text into label token IDs.

#### 4. Model & LoRA Setup (Step 7)
- Loads `openai/whisper-medium` in FP16 precision.
- **Enables Gradient Checkpointing**: Vital to prevent CUDA Out-Of-Memory (OOM) on Colab's 15GB T4 GPU.
- Applies LoRA adapter to attention projection layers (`q_proj`, `v_proj`, `k_proj`, `out_proj`).

#### 5. Seq2Seq Training Loop (Step 8)
- `per_device_train_batch_size: 1`
- `gradient_accumulation_steps: 8` (effective batch size = 8)
- `learning_rate: 1e-4`
- `gradient_checkpointing: True`
- `fp16: True`
- Evaluates both **WER** and **CER** at regular step intervals.

---

### Step 4: Automatic Export to Google Drive

At the end of training, the final cell:
1. Saves the fine-tuned LoRA adapter weights and processor to `/content/tetrax-whisper-medium-marathi-hindi`.
2. Compresses the adapter into `tetrax-whisper-medium-marathi-hindi.zip`.
3. Copies both the ZIP and uncompressed folder directly to your **Google Drive**:
   ```
   MyDrive/Tetrax-ASR/checkpoints/tetrax-whisper-medium-marathi-hindi.zip
   ```
4. Triggers a direct browser download as a local backup.

---

### Step 5: Deploy to Your Local Tetrax ASR Project

1. Unzip `tetrax-whisper-medium-marathi-hindi.zip`.
2. Move the extracted folder into your project under `checkpoints/`:
   ```text
   Tetrax-ASR/
   └── checkpoints/
       └── tetrax-whisper-medium-marathi-hindi/
           ├── adapter_config.json
           ├── adapter_model.safetensors
           ├── tokenizer_config.json
           └── ...
   ```
3. Start the Tetrax ASR server:
   ```bash
   python -m uvicorn tetrax_asr.api.main:app --host 0.0.0.0 --port 8000
   ```
4. **Automatic Detection**: On startup, Tetrax ASR automatically detects `checkpoints/tetrax-whisper-medium-marathi-hindi/`, attaches the fine-tuned LoRA weights to `openai/whisper-medium`, and serves both Marathi and Hindi transcriptions with full Devanagari output!

---

## 💡 Troubleshooting & T4 GPU Tips

- **CUDA Out of Memory (OOM)**:
  - Ensure `gradient_checkpointing=True` is enabled.
  - Keep `per_device_train_batch_size=1`. If you need larger effective batches, increase `gradient_accumulation_steps=16`.
- **Devanagari Transcript Evaluation**:
  - Always check both **CER** and **WER**. CER helps assess if character/spelling predictions are phonetically accurate even when minor word variations occur.
