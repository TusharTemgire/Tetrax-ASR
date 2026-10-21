"""
Evaluation and Benchmark Script for Tetrax ASR
Evaluates Base Whisper vs Tetrax Fine-Tuned LoRA model on Svarah benchmark.
Calculates Word Error Rate (WER) and Character Error Rate (CER).
"""

import os
import argparse
import evaluate
from tqdm import tqdm
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from peft import PeftModel
import torch
import logging

from tetrax_asr.data.svarah_loader import prepare_svarah_dataset_dict, extract_audio_array
from tetrax_asr.config import model_config, training_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TetraxASR.Evaluate")


def evaluate_model(
    base_model_id: str = "openai/whisper-small",
    adapter_path: str = "./checkpoints/tetrax-whisper-small-indian-en",
    num_samples: int = 50,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    wer_metric = evaluate.load("wer")
    cer_metric = evaluate.load("cer")

    processor = WhisperProcessor.from_pretrained(base_model_id, language="en", task="transcribe")
    torch_dtype = torch.float16 if device == "cuda" else torch.float32

    logger.info(f"Loading Base Model: {base_model_id} on {device}...")
    base_model = WhisperForConditionalGeneration.from_pretrained(
        base_model_id,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True
    ).to(device)
    base_model.eval()

    has_adapter = os.path.exists(adapter_path) and os.path.exists(os.path.join(adapter_path, "adapter_config.json"))
    ft_model = None
    if has_adapter:
        logger.info(f"Loading Fine-Tuned Model with Adapter: {adapter_path}...")
        ft_model = PeftModel.from_pretrained(base_model, adapter_path).to(device)
        ft_model.eval()
    else:
        logger.warning(f"No adapter found at {adapter_path}. Only evaluating base model.")

    # Load Svarah test split
    logger.info("Loading Svarah test split...")
    dataset_dict = prepare_svarah_dataset_dict(
        train_ratio=training_config.train_ratio,
        val_ratio=training_config.val_ratio,
        test_ratio=training_config.test_ratio,
        seed=training_config.random_seed
    )
    test_ds = dataset_dict["test"]
    if num_samples > 0:
        test_ds = test_ds.select(range(min(num_samples, len(test_ds))))

    logger.info(f"Evaluating on {len(test_ds)} test audio samples...")

    references = []
    base_predictions = []
    ft_predictions = []

    forced_decoder_ids = processor.get_decoder_prompt_ids(language="en", task="transcribe")

    for sample in tqdm(test_ds, desc="Evaluating"):
        ref_text = sample["text"]
        audio_arr, sr = extract_audio_array(sample["audio_filepath"], target_sr=16000)

        input_features = processor(audio_arr, sampling_rate=sr, return_tensors="pt").input_features
        if device == "cuda":
            input_features = input_features.to(device).half()
        else:
            input_features = input_features.to(device)

        with torch.no_grad():
            # Base generation
            pred_base_ids = base_model.generate(
                input_features,
                forced_decoder_ids=forced_decoder_ids,
                max_new_tokens=448
            )
            base_pred = processor.batch_decode(pred_base_ids, skip_special_tokens=True)[0]
            base_predictions.append(base_pred)

            if ft_model:
                pred_ft_ids = ft_model.generate(
                    input_features,
                    forced_decoder_ids=forced_decoder_ids,
                    max_new_tokens=448
                )
                ft_pred = processor.batch_decode(pred_ft_ids, skip_special_tokens=True)[0]
                ft_predictions.append(ft_pred)

        references.append(ref_text)

    # Compute metrics
    base_wer = 100 * wer_metric.compute(predictions=base_predictions, references=references)
    base_cer = 100 * cer_metric.compute(predictions=base_predictions, references=references)

    print("\n" + "=" * 55)
    print("           TETRAX ASR BENCHMARK RESULTS")
    print("=" * 55)
    print(f"Dataset: AI4Bharat Svarah (Samples: {len(test_ds)})")
    print(f"Base Model: {base_model_id}")
    print(f"  • Base WER: {base_wer:.2f}%")
    print(f"  • Base CER: {base_cer:.2f}%")

    if ft_model:
        ft_wer = 100 * wer_metric.compute(predictions=ft_predictions, references=references)
        ft_cer = 100 * cer_metric.compute(predictions=ft_predictions, references=references)
        wer_reduction = ((base_wer - ft_wer) / base_wer) * 100 if base_wer > 0 else 0.0

        print(f"\nTetrax Fine-Tuned Model (LoRA Adapted):")
        print(f"  • Fine-Tuned WER: {ft_wer:.2f}%")
        print(f"  • Fine-Tuned CER: {ft_cer:.2f}%")
        print(f"  • Relative Error Reduction: {wer_reduction:+.2f}%")
    print("=" * 55 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, default="openai/whisper-small")
    parser.add_argument("--adapter_path", type=str, default="./checkpoints/tetrax-whisper-small-indian-en")
    parser.add_argument("--num_samples", type=int, default=50)
    args = parser.parse_args()

    evaluate_model(
        base_model_id=args.base_model,
        adapter_path=args.adapter_path,
        num_samples=args.num_samples
    )
