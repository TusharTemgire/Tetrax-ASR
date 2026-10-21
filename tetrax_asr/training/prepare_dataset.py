"""
Dataset Preparation Script for Whisper Fine-Tuning
Loads AI4Bharat Svarah (from local data/ directory or Hugging Face),
extracts Mel-spectrogram features, tokenizes target text, and saves processed dataset.
"""

import os
from pathlib import Path
from transformers import WhisperProcessor
import logging

from tetrax_asr.data.svarah_loader import prepare_svarah_dataset_dict, extract_audio_array
from tetrax_asr.config import model_config, training_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TetraxASR.PrepareData")


def prepare_dataset_for_whisper(
    model_id: str = "openai/whisper-small",
    save_preprocessed_dir: str = "./data/svarah_prepared",
    num_proc: int = 1
):
    logger.info(f"Loading WhisperProcessor for {model_id}...")
    processor = WhisperProcessor.from_pretrained(
        model_id,
        language=model_config.language,
        task=model_config.task
    )

    logger.info("Loading and splitting Svarah dataset...")
    dataset_dict = prepare_svarah_dataset_dict(
        train_ratio=training_config.train_ratio,
        val_ratio=training_config.val_ratio,
        test_ratio=training_config.test_ratio,
        seed=training_config.random_seed
    )

    def prepare_sample(batch):
        # 1. Load audio and compute log-Mel spectrogram input features
        audio_arr, sr = extract_audio_array(batch["audio_filepath"], target_sr=16000)
        batch["input_features"] = processor.feature_extractor(
            audio_arr,
            sampling_rate=sr
        ).input_features[0]

        # 2. Encode target text to label token IDs
        batch["labels"] = processor.tokenizer(batch["text"]).input_ids
        return batch

    logger.info("Extracting log-Mel features and tokenizing transcripts...")
    prepared_dataset = dataset_dict.map(
        prepare_sample,
        remove_columns=dataset_dict["train"].column_names,
        num_proc=num_proc,
        desc="Preprocessing Svarah dataset"
    )

    if save_preprocessed_dir:
        out_path = Path(save_preprocessed_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        prepared_dataset.save_to_disk(str(out_path))
        logger.info(f"Successfully saved preprocessed dataset to {out_path}")

    return prepared_dataset


if __name__ == "__main__":
    prepare_dataset_for_whisper()
