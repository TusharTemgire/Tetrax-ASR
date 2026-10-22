"""
Script to merge fine-tuned LoRA adapter weights directly into the base Whisper model.
Produces a 100% standalone, self-contained model folder with no PEFT dependency.

Usage:
    python scripts/merge_lora.py
"""

import sys
import os
import time
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from peft import PeftModel
from tetrax_asr.config import model_config, BASE_DIR

BASE_MODEL = "openai/whisper-medium"
ADAPTER_DIR = model_config.adapter_path
OUTPUT_DIR = str(BASE_DIR / "Tetrax-ASR-Merged-764M")


def merge_and_export():
    print("=" * 65)
    print("🔄 TETRAX ASR — Merging LoRA Adapter into Full Standalone Model")
    print(f"📁 Base Model     : {BASE_MODEL}")
    print(f"📁 Adapter Source : {ADAPTER_DIR}")
    print(f"📁 Destination    : {OUTPUT_DIR}")
    print("=" * 65)

    if not os.path.exists(ADAPTER_DIR):
        print(f"❌ Error: Adapter path does not exist: {ADAPTER_DIR}")
        return

    print("\n1. Loading base Whisper-Medium model into RAM (offline)...")
    start = time.time()
    base_model = WhisperForConditionalGeneration.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
        local_files_only=True,
    )
    processor = WhisperProcessor.from_pretrained(BASE_MODEL, local_files_only=True)
    print(f"   Base model loaded in {time.time() - start:.2f}s")

    print("\n2. Attaching fine-tuned LoRA adapter...")
    start = time.time()
    peft_model = PeftModel.from_pretrained(base_model, ADAPTER_DIR, local_files_only=True)
    print(f"   LoRA adapter attached in {time.time() - start:.2f}s")

    print("\n3. Merging LoRA weights directly into base layers (merge_and_unload)...")
    start = time.time()
    merged_model = peft_model.merge_and_unload()
    print(f"   Weights fused in {time.time() - start:.2f}s")

    print(f"\n4. Saving standalone merged model to: {OUTPUT_DIR}...")
    start = time.time()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    merged_model.save_pretrained(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)
    print(f"   Full model saved in {time.time() - start:.2f}s")

    print("\n" + "=" * 65)
    print("✅ MERGE COMPLETE!")
    print(f"Your 100% self-contained standalone model is ready at:")
    print(f"👉 {OUTPUT_DIR}")
    print("=" * 65)


if __name__ == "__main__":
    merge_and_export()
