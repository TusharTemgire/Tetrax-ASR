# ⚡ RunPod Cloud GPU Training & Deployment Guide — Tetrax ASR

This guide explains how to fine-tune and deploy **Tetrax ASR** on **RunPod** using dedicated cloud GPUs (e.g., NVIDIA RTX 3090, RTX 4090, or A5000).

---

## 🌟 Why Use RunPod?

| Advantage | RunPod Benefits |
|---|---|
| **Ultra-Fast Training** | RTX 4090 (24 GB VRAM) completes 3 epochs in **~8 – 12 minutes** |
| **No Disconnects** | Runs in persistent cloud background (`tmux`), unaffected by browser closures |
| **Bigger Models** | Ample VRAM to train larger models like `openai/whisper-medium` (769M) or `whisper-large-v3` |
| **Low Cost** | Pay-as-you-go (~$0.22 – $0.34/hr for RTX 3090 / 4090) |
| **Public Deployment** | Can serve the live FastAPI real-time WebSocket API on a public HTTPS URL |

---

## 🚀 Step 1: Deploy a Pod on RunPod

1. Log in to [RunPod](https://www.runpod.io) and navigate to **Pods**.
2. Click **Deploy Pod** and select **Secure Cloud** or **Community Cloud**:
   - **Recommended GPU**: **NVIDIA RTX 3090 (24 GB)** or **RTX 4090 (24 GB)** (~$0.25 - $0.34/hr).
   - Alternatively: **RTX A4000 (16 GB)** (~$0.18/hr).
3. **Select Template**:
   - Choose the official PyTorch template:  
     `RunPod PyTorch 2.1.0` (or `runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04`).
4. **Storage Configuration**:
   - **Container Disk**: `40 GB`
   - **Volume Disk (persistent)**: `20 GB` (mounted at `/workspace`)
5. **Port Forwarding**:
   - Under *Customize Pod*, ensure HTTP ports are exposed:
     - `8000` (FastAPI / WebSocket server)
     - `8888` (JupyterLab)
6. Click **Deploy On-Demand**.

---

## 💻 Step 2: Connect to Your Pod

Once the pod status shows **Running**:
1. Click **Connect**.
2. You can access via:
   - **JupyterLab** (recommended for easy file uploads and terminal access).
   - **Web Terminal** (browser-based bash prompt).
   - **SSH** (via standard terminal).

---

## 📦 Step 3: Clone Repository & Setup Environment

Open the terminal in `/workspace`:

```bash
cd /workspace

# 1. Clone your Tetrax-ASR repository
git clone <YOUR_GIT_REPO_URL> Tetrax-ASR
cd Tetrax-ASR

# 2. Install dependencies
pip install -r requirements.txt
pip install peft
```

---

## 📁 Step 4: Dataset Setup on RunPod

You have **two options** for the dataset:

### Option A: Use Local Parquet Files (Already in your repo)
If you upload your local parquet files:
1. In JupyterLab, navigate to `/workspace/Tetrax-ASR/data/`.
2. Drag and drop your 3 files (`test-00000-of-00003.parquet`, etc.).
3. The loader will automatically detect and load them instantly from disk.

### Option B: Automatic Hugging Face Hub Download
If you do not upload parquet files, the training scripts automatically download `ai4bharat/Svarah` directly from Hugging Face Hub with zero manual file transfers.

---

## 🏋️ Step 5: Launch Training in Background (`tmux`)

Always run training inside a `tmux` session so training continues even if you close your laptop or browser:

```bash
# 1. Start a persistent tmux session
tmux new -s tetrax_train

# 2. Enter project folder
cd /workspace/Tetrax-ASR

# 3. Launch LoRA fine-tuning
python -m tetrax_asr.training.train_whisper \
    --base_model openai/whisper-small \
    --epochs 3 \
    --batch_size 16 \
    --lr 1e-4 \
    --output_dir ./checkpoints/tetrax-whisper-small-indian-en
```

> **Tip for 24 GB GPUs (RTX 3090 / 4090)**:  
> Because you have 24 GB VRAM, `--batch_size 16` runs at peak throughput, cutting training time down to **~10 minutes**!

### Tmux Cheatsheet:
- **Detach from session** (leaves training running in background): Press `Ctrl + B`, then press `D`.
- **Re-attach later**:
  ```bash
  tmux attach -t tetrax_train
  ```

---

## 📊 Step 6: Benchmark & Evaluate WER

Once training completes, evaluate the Word Error Rate (WER) improvement against the baseline:

```bash
python -m tetrax_asr.training.evaluate \
    --base_model openai/whisper-small \
    --adapter_path ./checkpoints/tetrax-whisper-small-indian-en \
    --num_samples 100
```

---

## 🌐 Step 7: Host the Real-Time API Directly on RunPod

You can run the Tetrax ASR live streaming server directly on your cloud GPU:

```bash
# Start server bound to all interfaces
python -m uvicorn tetrax_asr.api.main:app --host 0.0.0.0 --port 8000
```

### Accessing the Web Console & API from the Internet:
1. In the RunPod dashboard, go to your Pod and click **Connect**.
2. Look for **HTTP Services (Port 8000)**.
3. Click the generated proxy URL:
   `https://<YOUR_POD_ID>-8000.proxy.runpod.net`
4. The **Tetrax Voice Intelligence Console** will load live with GPU acceleration! You can test real-time microphone streaming and REST transcription directly from your browser.

---

## 📥 Step 8: Download Model Checkpoints to Local PC

When finished, package the fine-tuned adapter:

```bash
cd /workspace/Tetrax-ASR/checkpoints
tar -czvf tetrax-whisper-small-indian-en.tar.gz tetrax-whisper-small-indian-en/
```

- **Via JupyterLab**: Right-click `tetrax-whisper-small-indian-en.tar.gz` and click **Download**.
- **Via SCP**:
  ```bash
  scp -P <PORT> root@<IP>:/workspace/Tetrax-ASR/checkpoints/tetrax-whisper-small-indian-en.tar.gz ./
  ```
- Extract the archive into your local machine's `Tetrax-ASR/checkpoints/` folder.

---

## 🛑 Step 9: Stop or Terminate Pod

To avoid incurring costs after training is done:
- Go to **RunPod Dashboard > Pods**.
- Click **Stop** (preserves your volume disk so you can restart anytime).
- OR click **Terminate** if you have already downloaded your checkpoints.
