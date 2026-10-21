# 🎙️ Google Colab Fine-Tuning Guide — Tetrax ASR

This guide walks you through fine-tuning **Whisper-Small** (or Whisper-Base) on the **AI4Bharat Svarah** dataset using **Google Colab's Free T4 GPU** with **PEFT / LoRA (Low-Rank Adaptation)**.

---

## 📋 Overview & Requirements

| Item | Details |
|---|---|
| **Hardware** | Free Google Colab **T4 GPU** (16 GB VRAM) |
| **Model** | `openai/whisper-small` (244M parameters) or `openai/whisper-base` (74M parameters) |
| **Adaptation** | PEFT LoRA (Trains only ~1.5% of parameters, ~4 GB VRAM usage) |
| **Dataset** | AI4Bharat Svarah (6,656 samples / ~9.6 hours of Indian English audio) |
| **Training Time** | ~15 – 30 minutes on free T4 GPU |
| **Output** | Lightweight LoRA adapter folder (`~25 MB`), easy to download and deploy |

---

## 🚀 Step-by-Step Training Walkthrough

### Step 1: Open Google Colab & Select GPU Runtime
1. Open [Google Colab](https://colab.research.google.com).
2. Go to **Runtime > Change runtime type**.
3. Under **Hardware accelerator**, select **T4 GPU**.
4. Click **Save**.
5. Verify GPU is active in a code cell:
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
3. The notebook will open with all cells pre-configured and ready to run.

---

### Step 3: Dataset Loading Options in Colab

The notebook supports **two ways** to load the dataset:

#### Option A: Direct Hugging Face Load (Zero File Upload Needed — Recommended)
- By default, the notebook will automatically fetch `ai4bharat/Svarah` from Hugging Face Hub.
- No file upload or download from your computer is needed.

#### Option B: Use Your Local Parquet Files
If you prefer using the 3 parquet files from your local `data/` folder (`test-00000-of-00003.parquet`, etc.):
1. In the Colab file explorer (left sidebar icon), drag and drop the 3 `test-*.parquet` files into `/content/`.
2. OR, if they are stored in your Google Drive, mount Drive:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```
   and place them in `/content/drive/MyDrive/Tetrax-ASR/data/`.
3. The notebook automatically detects these parquet files and loads them instantly without re-downloading!

---

### Step 4: Run the Training Cells

Run each cell in order:

1. **Install Dependencies**:
   ```bash
   !pip install -q transformers datasets peft accelerate evaluate jiwer soundfile librosa bitsandbytes tensorboard
   ```
2. **Dataset Split**:
   - Creates a stratified **80% Train / 10% Validation / 10% Test** split across the primary native languages (Nepali, Kannada, Urdu, Malayalam, Tamil, Telugu, Hindi, Marathi, etc.) to ensure balanced accent coverage.
3. **Feature Extraction & Tokenization**:
   - Computes 80-channel log-Mel spectrograms from 16kHz audio.
   - Tokenizes English text into label IDs.
4. **LoRA Adapter Setup**:
   - Attaches low-rank trainable matrices to attention projection layers (`q_proj`, `v_proj`, `k_proj`, `out_proj`).
   - Trainable parameters: **~3.5M out of 244M (1.4%)**.
5. **Seq2SeqTrainer Loop**:
   - `learning_rate`: `1e-4`
   - `per_device_train_batch_size`: `8`
   - `gradient_accumulation_steps`: `2` (effective batch size = 16)
   - `fp16`: `True`
   - `evaluation_strategy`: `"steps"` with Word Error Rate (WER) tracking.

---

### Step 5: Automatic ZIP & Google Drive Export

At the end of training, the final cell automatically packages the fine-tuned model into a `.zip` archive and saves it directly to your **Google Drive**:

```python
import shutil
import os
from google.colab import files

OUTPUT_DIR = "/content/tetrax-whisper-small-indian-en"
ZIP_NAME = "tetrax-whisper-small-indian-en.zip"

# 1. Save adapter & processor
model.save_pretrained(OUTPUT_DIR)
processor.save_pretrained(OUTPUT_DIR)

# 2. Create ZIP archive
shutil.make_archive("tetrax-whisper-small-indian-en", "zip", OUTPUT_DIR)

# 3. Automatically save ZIP and folder to Google Drive
DRIVE_DIR = "/content/drive/MyDrive/Tetrax-ASR/checkpoints"
os.makedirs(DRIVE_DIR, exist_ok=True)
shutil.copy2(ZIP_NAME, os.path.join(DRIVE_DIR, ZIP_NAME))
shutil.copytree(OUTPUT_DIR, os.path.join(DRIVE_DIR, "tetrax-whisper-small-indian-en"), dirs_exist_ok=True)
print(f"💾 Saved ZIP directly to Google Drive: {DRIVE_DIR}/{ZIP_NAME}")

# 4. Trigger direct browser download
files.download(ZIP_NAME)
```

Your fine-tuned model is safely preserved in:
- **Google Drive**: `MyDrive/Tetrax-ASR/checkpoints/tetrax-whisper-small-indian-en.zip`
- **Direct browser download**: `tetrax-whisper-small-indian-en.zip` (~25 MB)

---

### Step 6: Deploy to Your Local Tetrax ASR Project

1. Unzip the downloaded file:
   ```
   tetrax-whisper-small-indian-en.zip
   ```
2. Move the extracted folder into your project under `checkpoints/`:
   ```
   Tetrax-ASR/
   └── checkpoints/
       └── tetrax-whisper-small-indian-en/
           ├── adapter_config.json
           ├── adapter_model.safetensors
           ├── tokenizer_config.json
           └── ...
   ```
3. Start the local Tetrax ASR server:
   ```bash
   python -m uvicorn tetrax_asr.api.main:app --host 0.0.0.0 --port 8000
   ```
4. **Automatic Detection**: On startup, Tetrax ASR automatically detects `checkpoints/tetrax-whisper-small-indian-en/`, mounts the LoRA adapter, and activates fine-tuned Indian-English mode for both WebSocket live streaming and REST endpoints!

---

## 💡 Colab Tips & Troubleshooting

- **Prevent Disconnects**: Keep the Colab browser tab open while training runs.
- **Out of Memory (OOM)**: If you switch to `openai/whisper-medium` and hit memory limits, reduce `per_device_train_batch_size=4` and increase `gradient_accumulation_steps=4`.
- **Free Colab Limits**: Colab Free provides ~3-5 hours of continuous T4 GPU time per session, which is plenty for 3 epochs of LoRA fine-tuning.
