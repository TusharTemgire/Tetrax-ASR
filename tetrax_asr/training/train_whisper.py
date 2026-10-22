"""
Tetrax Whisper Small Fine-Tuning Script using LoRA (PEFT)
Designed for both local GPU training and Google Colab execution.
Supports both local parquet dataset in data/ and Hugging Face Hub.
"""

import os
import argparse
from pathlib import Path
import torch
import numpy as np
import evaluate
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
from peft import LoraConfig, get_peft_model
import logging

from tetrax_asr.data.indicvoices_loader import prepare_indicvoices_dataset_dict, extract_audio_array
from tetrax_asr.data.collator import DataCollatorSpeechSeq2SeqWithPadding
from tetrax_asr.config import model_config, training_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TetraxASR.Train")


def train(
    base_model_id: str = "openai/whisper-medium",
    output_dir: str = "./checkpoints/tetrax-whisper-medium-marathi-hindi",
    num_train_epochs: int = 3,
    batch_size: int = 1,
    gradient_accumulation_steps: int = 8,
    learning_rate: float = 1e-4,
    lora_r: int = 32,
    lora_alpha: int = 64,
    fp16: bool = True,
    languages: list[str] = None,
    drive_dir: str = None,
):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    target_langs = languages or ["mr", "hi"]

    logger.info(f"Loading processor for multilingual {base_model_id}...")
    processor = WhisperProcessor.from_pretrained(
        base_model_id,
        language="mr",
        task="transcribe"
    )

    # 1. Prepare IndicVoices Marathi + Hindi Dataset
    logger.info(f"Loading IndicVoices dataset for {target_langs} and splitting...")
    dataset_dict = prepare_indicvoices_dataset_dict(
        languages=target_langs,
        train_ratio=training_config.train_ratio,
        val_ratio=training_config.val_ratio,
        test_ratio=training_config.test_ratio,
        seed=training_config.random_seed
    )

    def prepare_dataset_batch(batch):
        # Prioritize audio struct with actual payload (bytes/array) over plain string path
        audio_col = None
        for key in ["audio", "audio_filepath", "audio_path", "file"]:
            val = batch.get(key)
            if isinstance(val, dict) and (val.get("bytes") or val.get("array") is not None or val.get("path")):
                audio_col = val
                break
            elif isinstance(val, (str, bytes)) and (isinstance(val, bytes) or os.path.exists(str(val))):
                audio_col = val
                break
            elif audio_col is None and val is not None:
                audio_col = val

        audio_arr, sr = extract_audio_array(audio_col, target_sr=16000)
        batch["input_features"] = processor.feature_extractor(
            audio_arr,
            sampling_rate=sr
        ).input_features[0]

        # Condition decoder tokens on the sample language ('mr' or 'hi')
        sample_lang = batch.get("language", "mr")
        if sample_lang in (0, "0"):
            sample_lang = "mr"
        elif sample_lang in (1, "1"):
            sample_lang = "hi"
        processor.tokenizer.set_prefix_tokens(language=sample_lang, task="transcribe")
        text_content = batch.get("text") or batch.get("normalized_text") or ""
        batch["labels"] = processor.tokenizer(text_content).input_ids
        return batch

    logger.info("Mapping features and tokenizing labels on train and validation splits...")
    train_data = dataset_dict["train"].map(
        prepare_dataset_batch,
        remove_columns=dataset_dict["train"].column_names,
        desc="Tokenizing Train Data"
    )
    val_data = dataset_dict["validation"].map(
        prepare_dataset_batch,
        remove_columns=dataset_dict["validation"].column_names,
        desc="Tokenizing Validation Data"
    )

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    # 2. Metric Computation: WER and CER (Character Error Rate for Devanagari text)
    wer_metric = evaluate.load("wer")
    cer_metric = evaluate.load("cer")

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids

        # Unpack if returned as tuple
        if isinstance(pred_ids, tuple):
            pred_ids = pred_ids[0]
        if isinstance(label_ids, tuple):
            label_ids = label_ids[0]

        pred_ids = np.array(pred_ids)
        label_ids = np.array(label_ids)

        pad_id = processor.tokenizer.pad_token_id or 0
        vocab_size = len(processor.tokenizer)

        # Replace -100 padding index and any out-of-bounds IDs to prevent OverflowError in Rust tokenizer
        pred_ids = np.where((pred_ids >= 0) & (pred_ids < vocab_size), pred_ids, pad_id)
        label_ids = np.where((label_ids >= 0) & (label_ids < vocab_size), label_ids, pad_id)

        pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        wer = 100 * wer_metric.compute(predictions=pred_str, references=label_str)
        cer = 100 * cer_metric.compute(predictions=pred_str, references=label_str)
        return {"wer": wer, "cer": cer}

    # 3. Model & LoRA Setup (Whisper Medium with Gradient Checkpointing)
    logger.info(f"Loading base model: {base_model_id}...")
    model = WhisperForConditionalGeneration.from_pretrained(
        base_model_id,
        torch_dtype=torch.float16 if fp16 and torch.cuda.is_available() else torch.float32,
        low_cpu_mem_usage=True
    )

    model.config.use_cache = False
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []

    # Enable gradient checkpointing to fit Whisper Medium (769M) into 15GB T4 GPU
    if torch.cuda.is_available():
        model.gradient_checkpointing_enable()

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=["q_proj", "v_proj", "k_proj", "out_proj"],
        lora_dropout=0.05,
        bias="none"
    )

    logger.info("Applying LoRA adapter to Whisper Medium...")
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 4. Training Arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        warmup_steps=100,
        num_train_epochs=num_train_epochs,
        eval_strategy="steps",
        fp16=fp16 and torch.cuda.is_available(),
        per_device_eval_batch_size=batch_size,
        generation_max_length=225,
        save_steps=100,
        eval_steps=100,
        logging_steps=25,
        gradient_checkpointing=True,
        report_to=["tensorboard"],
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        save_total_limit=2,
        remove_unused_columns=False,
        label_names=["labels"],
    )

    import inspect
    trainer_kwargs = {
        "args": training_args,
        "model": model,
        "train_dataset": train_data,
        "eval_dataset": val_data,
        "data_collator": data_collator,
        "compute_metrics": compute_metrics,
    }
    if "processing_class" in inspect.signature(Seq2SeqTrainer.__init__).parameters:
        trainer_kwargs["processing_class"] = processor.feature_extractor
    else:
        trainer_kwargs["tokenizer"] = processor.feature_extractor

    trainer = Seq2SeqTrainer(**trainer_kwargs)

    logger.info("Starting Whisper LoRA fine-tuning...")
    trainer.train()

    # 5. Save Adapter, Processor, and Create ZIP Archive
    import shutil
    logger.info(f"Saving fine-tuned LoRA weights and processor to {output_dir}...")
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    logger.info("Training complete and adapter successfully saved!")

    # 6. Create ZIP Archive of the model
    zip_path = Path(str(output_path) + ".zip")
    logger.info(f"Creating ZIP archive of model: {zip_path}...")
    shutil.make_archive(str(output_path), "zip", output_dir)
    zip_size_mb = zip_path.stat().st_size / (1024 * 1024) if zip_path.exists() else 0.0
    logger.info(f"Successfully created {zip_path.name} ({zip_size_mb:.2f} MB)")

    # 7. Save to Google Drive if available
    drive_target = None
    default_colab_drive = Path("/content/drive/MyDrive/Tetrax-ASR/checkpoints")
    if drive_dir and Path(drive_dir).parent.exists():
        drive_target = Path(drive_dir)
    elif default_colab_drive.parent.exists():
        drive_target = default_colab_drive

    if drive_target:
        try:
            drive_target.mkdir(parents=True, exist_ok=True)
            # Copy zip to Drive
            drive_zip_dest = drive_target / zip_path.name
            shutil.copy2(str(zip_path), str(drive_zip_dest))
            logger.info(f"Saved model ZIP to Google Drive: {drive_zip_dest}")

            # Also copy folder to Drive
            drive_folder_dest = drive_target / output_path.name
            shutil.copytree(output_dir, str(drive_folder_dest), dirs_exist_ok=True)
            logger.info(f"Saved model folder to Google Drive: {drive_folder_dest}")
        except Exception as e:
            logger.warning(f"Could not save to Google Drive ({e}). Zip is preserved locally at {zip_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Whisper Medium LoRA on IndicVoices (Marathi + Hindi)")
    parser.add_argument("--base_model", type=str, default="openai/whisper-medium")
    parser.add_argument("--output_dir", type=str, default="./checkpoints/tetrax-whisper-medium-marathi-hindi")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--drive_dir", type=str, default="/content/drive/MyDrive/Tetrax-ASR/checkpoints", help="Google Drive destination directory")
    args = parser.parse_args()

    train(
        base_model_id=args.base_model,
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        drive_dir=args.drive_dir,
    )
