"""
AI4Bharat Svarah Dataset Inspection and Download CLI
Loads dataset metadata from Hugging Face, inspects Indian accents distribution,
and optionally downloads a slice or complete split to local disk.
"""

import sys
import argparse
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tetrax_asr.data.svarah_loader import load_raw_svarah, split_svarah_dataset, get_dataset_stats


def main():
    parser = argparse.ArgumentParser(description="Inspect and prepare AI4Bharat Svarah dataset")
    parser.add_argument("--streaming", action="store_true", help="Stream dataset without downloading all 1GB")
    parser.add_argument("--save_dir", type=str, default="", help="Optional directory to save splits")
    args = parser.parse_args()

    print("=" * 60)
    print("      AI4Bharat Svarah Dataset Explorer (Tetrax ASR)")
    print("=" * 60)

    print("\nFetching dataset metadata from Hugging Face: ai4bharat/Svarah...")
    raw_dataset = load_raw_svarah(split="test", streaming=args.streaming)

    if not args.streaming:
        print(f"Total samples: {len(raw_dataset)}")
        stats = get_dataset_stats(raw_dataset)
        print(f"Total Audio Duration: {stats['total_audio_hours']} Hours")
        print(f"Average Clip Length: {stats['mean_duration_sec']} Seconds")

        print("\nPrimary Native Language Backgrounds of Speakers:")
        for lang, count in sorted(stats["primary_languages"].items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"  • {lang}: {count} samples")

        print("\nTop Native States Represented:")
        for state, count in sorted(stats["states"].items(), key=lambda x: x[1], reverse=True)[:8]:
            print(f"  • {state}: {count} samples")

        if args.save_dir:
            from tetrax_asr.data.svarah_loader import prepare_svarah_dataset_dict
            print(f"\nSaving stratified train/val/test splits to {args.save_dir}...")
            prepare_svarah_dataset_dict(save_dir=args.save_dir)
            print("Splits successfully saved to disk.")
    else:
        print("Streaming mode active: taking first sample...")
        sample = next(iter(raw_dataset))
        print("Sample transcript:", sample["text"])
        print("Sample native language:", sample.get("primary_language"))
        print("Sample state:", sample.get("native_place_state"))

    print("\nDone.")


if __name__ == "__main__":
    main()
